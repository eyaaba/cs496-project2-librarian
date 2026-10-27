
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pymupdf as fitz
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

INDEX_PATH = Path("data/index.json")
DOCS_PATH = Path("data/documents")
CACHE_PATH = Path("data/embedding_cache")


def client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    return OpenAI()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def chunk_text(
    text: str,
    max_words: int = 260,
    overlap: int = 45
) -> list[str]:
    tokens = text.split()

    if max_words <= overlap:
        raise ValueError("max_words must exceed overlap")

    step = max_words - overlap

    return [
        " ".join(tokens[i:i + max_words])
        for i in range(0, len(tokens), step)
        if tokens[i:i + max_words]
    ]


def load_corpus() -> list[dict]:
    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    chunks = []

    for path in sorted(DOCS_PATH.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as document:
                pages = [
                    (i + 1, page.get_text("text"))
                    for i, page in enumerate(document)
                ]

        elif path.suffix.lower() in (".txt", ".md"):
            pages = [(1, path.read_text(encoding="utf-8-sig"))]

        else:
            continue

        source = path.relative_to(DOCS_PATH).as_posix()
        source_id = source.replace("/", "__")

        for page_number, text in pages:
            for j, passage in enumerate(chunk_text(text)):
                if len(passage.strip()) < 30:
                    continue

                chunks.append({
                    "id": f"{source_id}-p{page_number}-c{j}",
                    "source": source,
                    "page": page_number,
                    "text": passage,
                })

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    result_vectors = []
    api_client = None

    for offset in range(0, len(texts), 80):
        batch = texts[offset:offset + 80]
        payload = json.dumps({"model": model, "texts": batch}, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        cache_file = CACHE_PATH / f"{digest}.json"

        if cache_file.exists():
            vectors = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            if api_client is None:
                api_client = client()
            response = api_client.embeddings.create(model=model, input=batch)
            vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            if len(vectors) != len(batch):
                raise RuntimeError("Incorrect embedding count returned by API")
            temp_file = cache_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(vectors), encoding="utf-8")
            temp_file.replace(cache_file)

        if len(vectors) != len(batch) or any(len(v) != 1536 for v in vectors) and model == "text-embedding-3-small":
            raise RuntimeError("Embedding cache mismatch")
        result_vectors.extend(vectors)
        print(f"Processed {min(offset + 80, len(texts))}/{len(texts)} chunks")
    return result_vectors


def build_index() -> int:
    chunks = load_corpus()
    if not chunks:
        raise RuntimeError("No documents found in data/documents/")
    print(f"Indexing {len(chunks)} chunks...")
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count mismatch")
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = INDEX_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(chunks), encoding="utf-8")
    temp_path.replace(INDEX_PATH)
    load_search_index.cache_clear()
    print("Index saved successfully.")
    return len(chunks)


@lru_cache(maxsize=2)
def load_search_index(index_mtime_ns: int, index_size: int):
    """Reuse index, BM25 and vectors until the index file is replaced."""
    chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not chunks:
        return chunks, None, np.empty((0, 0), dtype=np.float32), np.array([])
    bm25 = BM25Okapi([words(chunk["text"]) or [""] for chunk in chunks])
    matrix = np.asarray([chunk["vector"] for chunk in chunks], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    return chunks, bm25, matrix, norms

@lru_cache(maxsize=1)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve(query: str, k: int = 5, mode: str = "hybrid", source_prefix: str | None = None) -> list[dict]:
    if mode not in ("bm25", "vector", "hybrid"):
        raise ValueError("mode must be bm25, vector, or hybrid")
    if k <= 0 or not query.strip():
        return []
    if not INDEX_PATH.exists():
        raise RuntimeError("Build the index first")

    stat = INDEX_PATH.stat()
    chunks, bm25, matrix, norms = load_search_index(stat.st_mtime_ns, stat.st_size)
    if not chunks:
        return []

    bm_scores = np.asarray(bm25.get_scores(words(query)), dtype=np.float64)
    if mode == "bm25":
        scores = bm_scores
    else:
        qvec = np.asarray(embed_texts([query])[0], dtype=np.float32)
        if qvec.shape[0] != matrix.shape[1]:
            raise RuntimeError("Query embedding dimensions don't match the index")
        cosine = (matrix @ qvec) / (norms * np.linalg.norm(qvec) + 1e-12)
        if mode == "vector":
            scores = cosine
        else:
            def zero_based_ranks(values):
                positions = np.empty(len(values), dtype=np.int64)
                positions[np.argsort(-values, kind="stable")] = np.arange(len(values))
                return positions

            br = zero_based_ranks(bm_scores)
            vr = zero_based_ranks(cosine)
            scores = 1.0 / (60 + br + 1) + 1.0 / (60 + vr + 1)
    indices = np.argsort(-scores, kind="stable")

    if source_prefix is not None:
        indices = [
            i for i in indices
            if chunks[int(i)]["source"].startswith(source_prefix)
        ]

    indices = indices[:k]

    return [{key: value for key, value in chunks[int(i)].items() if key != "vector"}
            for i in indices]


def rerank(query: str, candidates: list[dict], k: int = 5) -> list[dict]:
    if not candidates or k <= 0:
        return []

    model = load_reranker()

    pairs = [
        (query, passage["text"])
        for passage in candidates
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [passage for passage, score in ranked[:k]]



def retrieve_reranked(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    source_prefix: str | None = None,
) -> list[dict]:
    if not query.strip() or k <= 0:
        return []

    bm25_candidates = retrieve(
        query,
        k=candidate_k,
        mode="bm25",
        source_prefix=source_prefix,
    )

    vector_candidates = retrieve(
        query,
        k=candidate_k,
        mode="vector",
        source_prefix=source_prefix,
    )

    unique_candidates = {}

    for passage in bm25_candidates + vector_candidates:
        unique_candidates[passage["id"]] = passage

    candidates = list(unique_candidates.values())

    return rerank(query, candidates, k=k)

def answer(question: str, mode: str = "hybrid") -> tuple[str, list[dict]]:
    if mode == "hybrid_rerank":
        passages = retrieve_reranked(question, k=5, candidate_k=20)
    else:
        passages = retrieve(question, mode=mode)
    context = "\n\n".join(
        f"[{p['id']}] Source: {p['source']} page {p['page']}\n{p['text']}"
        for p in passages
    )
    messages = [
        {"role": "system", "content": (
            "You are a source-grounded research assistant. Only use the supplied passages. "
            "Cite the exact passage ID in square brackets for every factual claim. "
            "If the passages do not contain enough evidence, say "
            "'I don't know based on the provided documents.' Never invent citations."
        )},
        {"role": "user", "content": f"PASSAGES:\n{context}\n\nQUESTION:\n{question}"},
    ]

    response = client().chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content or "", passages

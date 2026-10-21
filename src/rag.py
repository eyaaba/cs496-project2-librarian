"""Minimal inspectable RAG baseline. Replace/add reranker and hybrid fusion for final project."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import fitz
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()
INDEX_PATH = Path('data/index.json')
DOCS_PATH = Path('data/documents')

def client() -> OpenAI:
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY in .env or deployment secrets')
    return OpenAI()

def words(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

def chunk_text(text: str, max_words: int = 260, overlap: int = 45) -> list[str]:
    tokens = text.split()
    if max_words <= overlap: raise ValueError('max_words must exceed overlap')
    step = max_words - overlap
    return [' '.join(tokens[i:i + max_words]) for i in range(0, len(tokens), step) if tokens[i:i + max_words]]

def load_corpus() -> list[dict]:
    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    out = []

    for p in sorted(DOCS_PATH.rglob('*')):
        if not p.is_file():
            continue

        if p.suffix.lower() == '.pdf':
            with fitz.open(p) as doc:
                pages = [
                    (n + 1, page.get_text('text'))
                    for n, page in enumerate(doc)
                ]

        elif p.suffix.lower() in ('.txt', '.md'):
            pages = [(1, p.read_text(encoding='utf-8-sig'))]

        else:
            continue

        # Preserve the relative path for unique source identifiers
        relative_path = p.relative_to(DOCS_PATH).as_posix()
        source_id = relative_path.replace('/', '__')

        for page, txt in pages:
            for j, chunk in enumerate(chunk_text(txt)):
                if len(chunk.strip()) < 30:
                    continue

                out.append({
                    'id': f'{source_id}-p{page}-c{j}',
                    'source': relative_path,
                    'page': page,
                    'text': chunk
                })

    return out

def embed_texts(texts: list[str]) -> list[list[float]]:
    c = client()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    cache_dir = Path("data/embedding_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for offset in range(0, len(texts), 80):
        batch_number = offset // 80
        cache_file = cache_dir / f"batch_{batch_number}.json"

        if cache_file.exists():
            vectors = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            batch = texts[offset:offset + 80]
            response = c.embeddings.create(
                model=model,
                input=batch
            )
            vectors = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            cache_file.write_text(json.dumps(vectors), encoding="utf-8")

        results.extend(vectors)
        print(f"Embedded {min(offset + 80, len(texts))}/{len(texts)} chunks")

    return results

def build_index() -> int:
    chunks = load_corpus()
    if not chunks: raise RuntimeError('Add authorized PDFs/TXT/MD files to data/documents/')
    vectors = embed_texts([c['text'] for c in chunks])
    for c, vec in zip(chunks, vectors): c['vector'] = vec
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks), encoding='utf-8')
    return len(chunks)

def retrieve(query: str, k: int = 5, mode: str = 'hybrid') -> list[dict]:
    if not INDEX_PATH.exists(): raise RuntimeError('Build the index first')
    chunks = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
    if not chunks: return []
    bm25 = BM25Okapi([words(c['text']) or [''] for c in chunks])
    bm = np.asarray(bm25.get_scores(words(query)), dtype=float)
    qvec = np.asarray(embed_texts([query])[0], dtype=float)
    matrix = np.asarray([c['vector'] for c in chunks], dtype=float)
    cosine = (matrix @ qvec)/(np.linalg.norm(matrix, axis=1)*np.linalg.norm(qvec)+1e-12)
    # Rank fusion allows incomparable BM25 and cosine scores to be combined.
    def rank(vals):
        positions = np.empty(len(vals),dtype=int)
        positions[np.argsort(-vals)] = np.arange(len(vals))
        return positions
    br, vr = rank(bm), rank(cosine)
    if mode == 'bm25': scores = -br
    elif mode == 'vector': scores = -vr
    else: scores = 1/(60+br)+1/(60+vr)
    indices = np.argsort(-scores)[:k]
    return [{key:v for key,v in chunks[int(i)].items() if key != 'vector'} for i in indices]

def answer(question: str, mode: str = 'hybrid') -> tuple[str,list[dict]]:
    chunks = retrieve(question, mode=mode)
    context = '\n\n'.join(f"[{c['id']}] Source: {c['source']} page {c['page']}\n{c['text']}" for c in chunks)
    messages = [
        {'role':'system','content': 'You are a source-grounded research assistant. Only use the supplied passages. Cite the exact passage ID in square brackets for every factual claim. If the passages do not contain enough evidence, say "I don\'t know based on the provided documents." Never invent citations.'},
        {'role':'user','content':f'PASSAGES:\n{context}\n\nQUESTION:\n{question}'}
    ]
    result = client().chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL','gpt-4o-mini'),messages=messages,temperature=0)
    return result.choices[0].message.content or '', chunks

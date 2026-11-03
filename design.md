# System Design — The Librarian

## 1. Overview

The Librarian is a Retrieval-Augmented Generation (RAG) system developed for CS496 AI Engineering. It answers Python programming questions using retrieved passages from the official Python documentation.

The system combines lexical search, semantic retrieval, optional neural reranking, and source-grounded answer generation. A Streamlit interface provides access to the retrieval modes, generated answers, and supporting evidence.

## 2. Document Corpus

**Source:** Official Python 3.14.0 Documentation  
**Website:** https://docs.python.org/3.14/

| Property | Value |
|---|---|
| Documents | 522 |
| Indexed passages | 8,772 |
| Chunk size | 260 words |
| Chunk overlap | 45 words |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1,536 |
| Storage | Persistent JSON index |

The corpus covers Python language features, syntax, standard-library modules, and programming concepts. The intended users are university students learning Python and developers consulting its documentation.

The corpus is collected through `download_corpus.py`. Document source metadata is retained during indexing to support retrieval and evidence display.

The documentation is publicly available under the applicable Python documentation licensing terms.

## 3. Document Processing and Indexing

The offline indexing pipeline performs the following operations:

1. Load documentation files and their source metadata.
2. Divide documents into overlapping text passages.
3. Assign unique identifiers to the passages.
4. Generate embeddings using OpenAI `text-embedding-3-small`.
5. Cache embeddings using content-based SHA-256 identifiers.
6. Persist passage text, metadata, and embeddings in `data/index.json`.

A 260-word chunk size was selected as the implemented configuration, balancing passage specificity with sufficient context for question answering. The 45-word overlap reduces the likelihood of losing information at chunk boundaries.

### Storage Choice

The system uses a local JSON index instead of a dedicated vector database.

This design simplifies development, inspection, persistence, and reproducibility for a fixed documentation corpus.

However, JSON storage has relatively high memory and loading overhead. The distributed index is approximately 276 MB, making deployment resource requirements an important limitation.

The prebuilt index is distributed through a versioned GitHub Release:

https://github.com/eyaaba/cs496-project2-librarian/releases/tag/v1.0-index

The Streamlit application downloads this index when it is absent and validates its SHA-256 checksum.

## 4. Retrieval Architecture

The system implements four retrieval configurations.

| Retrieval mode | Implementation |
|---|---|
| BM25 | Lexical matching using Rank-BM25 |
| Vector | Semantic similarity using OpenAI embeddings |
| Hybrid | BM25 and vector results combined using Reciprocal Rank Fusion |
| Hybrid + Cross-Encoder | Hybrid retrieval followed by neural reranking |

### Hybrid Retrieval

BM25 identifies passages with relevant keywords and technical terms, while vector search identifies passages with semantic similarity to the query.

Reciprocal Rank Fusion combines the two rankings without requiring their raw similarity scores to share a common scale.

The hybrid system retrieves candidates from both methods, merges their rankings, and selects the highest-ranked passages.

### Cross-Encoder Reranking

The optional reranking stage uses:

`cross-encoder/ms-marco-MiniLM-L6-v2`

The Cross-Encoder jointly evaluates queries and candidate passages to reorder results.

This stage introduces additional CPU inference latency. Evaluation showed no aggregate Hit@5 improvement over hybrid retrieval alone.

Consequently, hybrid retrieval without reranking is the application's default configuration.

### Index Caching and Filtering

The implementation caches the loaded search index and associated retrieval structures in memory.

Source-prefix metadata filtering is also supported.

## 5. Question Answering

The online RAG pipeline follows these stages:

1. Receive a question through Streamlit.
2. Rewrite the question when appropriate using GPT-4o-mini.
3. Retrieve relevant documentation passages using the selected retrieval mode.
4. Construct a grounded generation prompt containing the retrieved evidence.
5. Generate an answer using GPT-4o-mini.
6. Extract and validate citation identifiers against the retrieved passages.
7. Display the answer and supporting source passages.

When the available documentation does not support an answer, the system is instructed to abstain rather than invent information.

Citation validation verifies that references correspond to recognized source identifiers. It does not independently prove that every generated claim is supported by its citation.

## 6. Evaluation

The retrieval dataset contains 100 question-answer examples:

- Development set: 20 questions
- Held-out test set: 80 questions

### Retrieval Ablation

| Method | Hit@5 |
|---|---:|
| BM25 | 76% |
| Vector | 62% |
| Hybrid | 84% |
| Hybrid + Cross-Encoder | 84% |

Hybrid retrieval outperformed both individual retrieval strategies on this dataset.

On the held-out test set, hybrid retrieval and hybrid retrieval with reranking each achieved 69/80 hits, or 86.25% Hit@5.

These measurements evaluate supporting-passage retrieval rather than complete answer correctness.

### Preliminary Generation Evaluation

An initial generation review covered 10 supported questions and five unsupported questions.

Nine supported-question runs aligned with the expected reference answers. One encountered a citation-validation error during the original evaluation. All five unsupported questions produced the intended abstention response.

This limited review does not establish complete claim-level faithfulness.

## 7. Technical Limitations

- JSON index loading requires substantial memory.
- Reranking introduces significant CPU latency.
- Complex tables, code blocks, and cross-references can reduce retrieval reliability.
- Citation-ID validation cannot guarantee factual faithfulness.
- The evaluation dataset has limited difficulty diversity.
- Retrieval accuracy does not directly measure generated-answer correctness.
- The system depends on authorized OpenAI API access for question answering.
- Cloud deployment requires sufficient resources for the index and machine-learning dependencies.

## 8. Implementation References

- `src/rag.py` — indexing, retrieval, reranking, and generation
- `app.py` — Streamlit interface and index initialization
- `download_corpus.py` — corpus acquisition
- `eval/questions.jsonl` — evaluation dataset
- `eval/compare_retrieval.py` — retrieval ablation
- `eval/benchmark_optimization.py` — optimization benchmark
- `optimisation.md` — optimization analysis
- `postmortem.md` — technical failures and lessons learned
# The Librarian — CS496 Project 2

**Retrieval-Augmented Generation for Python Documentation**

The Librarian is a Retrieval-Augmented Generation (RAG) application that answers Python programming questions using relevant passages from the official Python 3.14 documentation. The system combines lexical and semantic retrieval, optional reranking, and source-grounded answer generation.

## Team

- Eya Ben Ammar
- Khalil Ben Omrane
- Yosr Maalej
- Mohamed Ben Haj Salah



## Live Application

**[The Librarian — Streamlit](https://the-librarian-cs496-project2.streamlit.app/)**

The application is hosted on Streamlit Community Cloud. OpenAI-powered question answering requires a configured, authorized API key.

**[GitHub Repository](https://github.com/eyaaba/cs496-project2-librarian)**

## Corpus

The project uses the official Python 3.14.0 documentation, available at https://docs.python.org/3.14/.

| Property | Value |
|---|---|
| Source | Official Python 3.14.0 documentation |
| Documents | 522 |
| Indexed passages | 8,772 |
| Chunk size | 260 words |
| Chunk overlap | 45 words |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1,536 |
| Index format | Persistent JSON |

The documentation was selected to support students and developers looking for answers about Python syntax, language features, and standard-library functionality.

## System Architecture

The system consists of two pipelines.

### Document Processing

1. Download Python documentation.
2. Extract and load document text with source metadata.
3. Split documents into overlapping passages.
4. Generate embeddings using OpenAI.
5. Cache generated embeddings to avoid repeated processing.
6. Store passages, metadata, and embeddings in a persistent index.

### Question Answering

1. Receive a question through Streamlit.
2. Rewrite the query when appropriate.
3. Retrieve supporting documentation passages.
4. Optionally rerank the retrieved candidates.
5. Generate an answer using GPT-4o-mini and retrieved context.
6. Validate source citation identifiers.
7. Display the answer alongside the retrieved evidence.

The application supports four retrieval configurations:

| Mode | Description |
|---|---|
| BM25 | Keyword-based retrieval |
| Vector | Embedding similarity search |
| Hybrid | BM25 + vector search using Reciprocal Rank Fusion |
| Hybrid + Cross-Encoder | Hybrid retrieval followed by neural reranking |

**Hybrid retrieval is the default configuration.**

The optional Cross-Encoder uses `cross-encoder/ms-marco-MiniLM-L6-v2`.

## Technology Stack

- Python 3.12
- Streamlit
- OpenAI API
- GPT-4o-mini
- OpenAI `text-embedding-3-small`
- Rank-BM25
- NumPy
- Sentence Transformers
- PyTorch
- PyMuPDF

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eyaaba/cs496-project2-librarian.git
cd cs496-project2-librarian
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure the API Key

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_authorized_api_key
```

The `.env` file is excluded from version control.

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The application automatically downloads the prebuilt search index when `data/index.json` is absent. The downloaded file is verified using SHA-256.

The prebuilt index is available under [GitHub Releases — v1.0-index](https://github.com/eyaaba/cs496-project2-librarian/releases/tag/v1.0-index).

### Reproducing the Corpus

```bash
python download_corpus.py
```

This downloads the Python documentation used by the indexing pipeline. Index generation and embedding utilities are implemented in `src/rag.py`.

Regenerating embeddings requires an authorized OpenAI API key.

## Evaluation

The retrieval benchmark uses **100 question-answer examples**, divided into 20 development questions and 80 held-out test questions.

### Retrieval Ablation

| Method | Hit@5 |
|---|---:|
| BM25 | 76% |
| Vector | 62% |
| Hybrid | 84% |
| Hybrid + Cross-Encoder | 84% |

Hybrid retrieval achieved the highest aggregate Hit@5, matching the reranked configuration without requiring Cross-Encoder inference.

### Optimization Results

The optimization experiment compared Hybrid + Cross-Encoder against Hybrid alone on the same 80 held-out test questions.

| Metric | Hybrid + Cross-Encoder | Hybrid |
|---|---:|---:|
| Hit@5 | 86.25% | 86.25% |
| Correct retrieval hits | 69/80 | 69/80 |
| Median retrieval latency | Approximately 3.2 s | 50.0 ms |
| p95 retrieval latency | Approximately 3–4 s | 64.5 ms |

Removing the reranking stage reduced measured warm-cache p95 retrieval latency by approximately **98%**, with **0 percentage-point change in Hit@5**.

These measurements concern the retrieval stage, not full end-to-end answer generation.

### Answer Generation Evaluation

An initial answer-generation review covered 10 supported questions and five unsupported questions.

- 9 of 10 supported-question runs aligned with their reference expectations.
- One supported-question run encountered a citation-validation failure.
- All five unsupported questions produced the intended abstention response.

The citation parsing issue was subsequently corrected in the local implementation.

This sample-based review is separate from the 100-question retrieval benchmark.

## Evaluation Files

| File | Purpose |
|---|---|
| `eval/questions.jsonl` | Evaluation dataset |
| `eval/validate_dataset.py` | Dataset validation |
| `eval/compare_retrieval.py` | Retrieval ablation |
| `eval/benchmark_optimization.py` | Before-and-after latency comparison |
| `eval/evaluate_answers.py` | Answer-generation review |

## Documentation

| File | Description |
|---|---|
| `design.md` | Architecture, corpus, and design decisions |
| `optimisation.md` | Optimization experiment and measurements |
| `postmortem.md` | Technical failures, corrections, and lessons learned |
| `eval/task4_results.md` | Retrieval implementation and evaluation notes |

## Limitations

- Hit@5 measures supporting-passage retrieval rather than complete answer correctness.
- Citation-ID validation does not independently establish factual faithfulness.
- Cross-Encoder reranking increases CPU inference latency.
- Large JSON indexes increase memory and startup requirements.
- Technical documentation containing complex tables, code blocks, or cross-references can introduce retrieval errors.
- OpenAI-powered answering requires authorized API access.

## Attribution

The documentation corpus is derived from the official Python documentation. Copyright and licensing requirements of the Python Software Foundation and other applicable contributors must be respected.
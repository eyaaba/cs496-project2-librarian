# Optimization Report — The Librarian

## 1. Objective

The optimization objective for CS496 Project 2 is to reduce cost per request or p95 latency by at least 50%, while limiting the loss in accuracy to a maximum of two percentage points.

The experiment evaluates whether removing Cross-Encoder reranking improves retrieval efficiency without reducing supporting-passage retrieval accuracy.

## 2. Experimental Design

Two configurations were compared:

**Baseline:** Hybrid retrieval with Cross-Encoder reranking

**Optimized:** Hybrid retrieval without Cross-Encoder reranking

Both configurations combine BM25 and embedding-based vector search using Reciprocal Rank Fusion (RRF).

The baseline additionally applies the `cross-encoder/ms-marco-MiniLM-L6-v2` model.

### Evaluation Setup

| Property | Configuration |
|---|---|
| Corpus | Official Python 3.14.0 documentation |
| Indexed passages | 8,772 |
| Evaluation questions | 100 |
| Development split | 20 |
| Held-out test split | 80 |
| Retrieval accuracy metric | Hit@5 |
| Performance metrics | p50 and p95 retrieval latency |
| Execution environment | Local CPU-based environment |
| Index | Loaded and cached in memory |

The optimization comparison uses the same 80-question held-out test set.

The latency measurements concern retrieval operations, not complete question rewriting and answer generation.

## 3. Retrieval Ablation

The following results were obtained across all 100 evaluation questions:

| Method | Hit@5 | Median retrieval latency | p95 retrieval latency |
|---|---:|---:|---:|
| BM25 | 76% | 37.2 ms | 51.8 ms |
| Vector | 62% | 55.6 ms | 77.6 ms |
| Hybrid | 84% | 45.3 ms | 60.5 ms |
| Hybrid + Cross-Encoder | 84% | 3,112.2 ms | 3,826.0 ms |

These results show that hybrid retrieval improved passage retrieval accuracy compared with either BM25 or vector search alone.

The Cross-Encoder did not improve aggregate Hit@5, despite introducing substantially higher CPU processing latency.

The table represents the initial full-dataset retrieval comparison. A separate optimization benchmark was performed on the held-out test split.

## 4. Before-and-After Optimization

The final optimization benchmark compared the baseline and optimized configurations over the same 80 held-out questions with a warm, cached search index.

| Metric | Baseline: Hybrid + Reranker | Optimized: Hybrid |
|---|---:|---:|
| Test questions | 80 | 80 |
| Hit@5 | 69/80 | 69/80 |
| Retrieval accuracy | 86.25% | 86.25% |
| Median retrieval latency | Approximately 3.2 s | 50.0 ms |
| p95 retrieval latency | Several seconds | 64.5 ms |
| Accuracy change | — | 0 percentage points |

The benchmark indicated an approximately 98% reduction in p95 retrieval latency.

The optimization retained the same Hit@5 on the identical held-out evaluation set.

The exact baseline latency from the separate optimization benchmark is not reproduced here; the ablation table above reports the recorded full-dataset baseline measurements.

## 5. Optimization Analysis

Cross-Encoder reranking evaluates query-passage pairs using a neural model.

Although this can improve ranking quality, it adds inference computation proportional to the number of candidates evaluated.

For the Python documentation dataset, hybrid retrieval already produced effective rankings through the combination of lexical and semantic evidence.

Removing Cross-Encoder reranking eliminated this additional processing stage without changing observed aggregate Hit@5.

The optimization therefore favored a simpler retrieval configuration supported by measured results.

## 6. Additional Efficiency Measures

The system also incorporates several implementation-level efficiency measures.

### Embedding Caching

Document embeddings are cached using content-derived identifiers to avoid unnecessary repeated embedding generation.

### Persistent Search Index

Passage metadata and embeddings are stored in a persistent index, avoiding re-indexing during normal operation.

### In-Memory Index Caching

Retrieval structures are reused between questions, reducing repeated loading and initialization work.

### Prebuilt Deployment Index

A versioned GitHub Release provides the generated index separately from the Git repository.

This avoids rebuilding document embeddings during cloud application startup.

### Optional Reranking

Reranking remains available as an experimental retrieval option, while standard hybrid retrieval is the default.

These mechanisms improve operational efficiency, but their individual cost and latency effects were not isolated in the main optimization comparison.

## 7. Scope and Limitations

The observed approximately 98% p95 reduction applies to **warm-cache retrieval latency**.

It does not establish an equivalent reduction in total chatbot response time.

The measurements exclude some or all of the following operations:

- Question rewriting with an OpenAI model
- Network communication
- Answer generation
- Cold-start index and model initialization
- Cloud application startup

The accuracy metric is Hit@5, which indicates whether a labeled supporting passage was retrieved among the top five results.

It is not a direct measure of factual answer correctness or claim-level faithfulness.

The experiment also does not establish a measured decrease in OpenAI API cost per generated answer.

## 8. Conclusion

Removing Cross-Encoder reranking reduced measured warm-cache retrieval p95 latency by approximately 98%, while preserving the same 86.25% Hit@5 on the 80-question held-out test set.

The result exceeds the required 50% latency-reduction threshold at the retrieval stage, with no observed loss in retrieval Hit@5.

The optimized hybrid configuration was selected as the default because it offers a more favorable retrieval accuracy–latency trade-off.

Full end-to-end response latency and answer correctness remain separate evaluation dimensions.
# Project Postmortem — The Librarian

## 1. Overview

The Librarian is a Retrieval-Augmented Generation (RAG) system developed over the official Python 3.14 documentation.

The project combines document processing, OpenAI embeddings, BM25 keyword retrieval, semantic search, Reciprocal Rank Fusion, optional Cross-Encoder reranking, grounded answer generation, citation validation, evaluation, and a Streamlit application.

This postmortem records technical problems encountered during development, their impact, the implemented corrective actions, and remaining limitations.

## 2. Problems and Resolutions

### 2.1 Large Documentation Corpus

**Problem**

The official documentation produced thousands of indexed passages and a large embedding index.

**Impact**

The generated index was unsuitable for regular Git tracking, and rebuilding it during each deployment would involve repeated processing and OpenAI embedding requests.

**Resolution**

The project implemented persistent embedding storage and caching. The generated index was excluded from Git and distributed separately through GitHub Releases.

The released index is approximately 276 MB and is downloaded automatically by the Streamlit application when absent.

**Lesson Learned**

Large generated artifacts should be managed separately from source code. Reusable indexes make deployment more practical and reduce repeated API usage.

### 2.2 Corpus Consistency

**Problem**

Early development used a combination of starter documents and Python documentation files. Subsequent cleanup changed the number of documents and indexed passages.

**Impact**

Earlier corpus statistics no longer described the final evaluated index.

**Resolution**

Unnecessary documents were removed, and the corpus was re-indexed.

The final recorded configuration contains 522 documentation files and 8,772 passages.

**Lesson Learned**

Evaluation results should be associated with a specific corpus version, and corpus statistics must be updated after cleaning or re-indexing.

### 2.3 Citation Parsing Error

**Problem**

The original citation extraction logic interpreted certain bracketed Python documentation expressions as citation references.

For example:

`socket.send(bytes[, flags])`

**Impact**

Ordinary technical syntax could trigger citation-validation errors, causing a generated answer to be rejected incorrectly.

**Resolution**

Citation extraction was restricted to identifiers matching the project's document-source naming format.

The correction was implemented in the local RAG source code. The original failing evaluation run remains recorded rather than being treated as a successful result.

**Lesson Learned**

Citation parsers for technical documentation must account for brackets, parameters, optional arguments, and code notation.

### 2.4 Expensive Cross-Encoder Reranking

**Problem**

Cross-Encoder reranking added several seconds of CPU processing to retrieval.

**Impact**

The slower retrieval configuration did not improve overall Hit@5 compared with hybrid search alone.

**Resolution**

Four retrieval configurations were evaluated, including BM25, vector retrieval, hybrid RRF, and hybrid RRF with Cross-Encoder reranking.

On the 80 held-out test questions, both hybrid configurations achieved 69/80 Hit@5.

Removing reranking reduced measured warm-cache p95 retrieval latency by approximately 98%, without changing the observed Hit@5.

Hybrid retrieval was therefore selected as the default configuration.

**Lesson Learned**

Additional model complexity should be justified through measured improvements rather than assumptions about model sophistication.

### 2.5 Retrieval and Generation Evaluation

**Problem**

Retrieving a relevant source passage does not guarantee that the generated answer is factually correct or properly supported.

**Impact**

Retrieval-only results could overstate the quality of the complete RAG system.

**Resolution**

A 100-question dataset was prepared for retrieval evaluation.

A separate preliminary generation review covered 10 supported questions and five unsupported questions.

Nine supported-question runs aligned with their reference expectations, and one encountered a citation-validation error. The five unsupported questions produced the intended abstention responses.

**Lesson Learned**

Retrieval accuracy, answer correctness, citation accuracy, and abstention behavior should be evaluated separately.

A valid citation identifier does not automatically establish that the cited passage supports the associated claim.

### 2.6 Cloud Deployment Complexity

**Problem**

The application requires a substantial documentation index and machine-learning dependencies, including PyTorch and Sentence Transformers.

**Impact**

Deploying the application involves package installation, additional storage, memory requirements, and index initialization.

**Resolution**

The application was configured for Streamlit Community Cloud using Python 3.12.

A prebuilt index downloader was implemented, with SHA-256 verification to detect corrupted downloads.

Deployment logs confirmed successful dependency installation and server startup. Complete index initialization and end-to-end cloud question answering were not established by those logs alone.

**Lesson Learned**

Successful local execution and successful server startup are not sufficient to establish that an application works correctly in production-like conditions.

## 3. Evaluation Summary

The 100-question retrieval ablation produced the following results:

| Method | Hit@5 |
|---|---:|
| BM25 | 76% |
| Vector | 62% |
| Hybrid | 84% |
| Hybrid + Cross-Encoder | 84% |

On the held-out 80-question test set, both hybrid configurations achieved 86.25% Hit@5.

The optimization experiment showed approximately 98% lower warm-cache retrieval p95 latency after removing Cross-Encoder reranking, with no observed Hit@5 reduction.

This optimization result does not imply the same reduction in total response time.

## 4. Remaining Limitations

### Evaluation Coverage

The evaluation dataset has limited difficulty diversity, and the completed generation review covers only a small sample.

### Factual Faithfulness

Citation-ID validation cannot fully determine whether every generated claim is supported by the cited passage
# Design decisions (DRAFT)
Corpus: Not yet selected/approved.
Chunking: 260 words, 45-word overlap per PDF page (baseline to tune).
Embedding: OpenAI text-embedding-3-small (subject to key availability).
Index: Local JSON vectors (baseline); requires production alternative as corpus grows.
Keyword: BM25; vector: cosine; fusion: reciprocal rank fusion.
Reranker: TODO.
Known failures: tables, code blocks, PDF reading order, cross-references, OCR-only PDFs, missing evidence.

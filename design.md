# Design decisions (DRAFT)
Corpus: Not yet selected/approved.
Chunking: 260 words, 45-word overlap per PDF page (baseline to tune).
Embedding: OpenAI text-embedding-3-small (subject to key availability).
Index: Local JSON vectors (baseline); requires production alternative as corpus grows.
Keyword: BM25; vector: cosine; fusion: reciprocal rank fusion.
Reranker: TODO.
Known failures: tables, code blocks, PDF reading order, cross-references, OCR-only PDFs, missing evidence.

## Corpus Selection

### Selected Corpus
Official Python 3.14.0 Documentation

### Source
https://docs.python.org/3.14/

### Corpus Statistics
- 528 documentation files
- Approximately 13.65 MiB of text
- 8,954 document chunks
- Chunk size: 260 words
- Chunk overlap: 45 words

### Intended Users
University students learning Python programming.

### Real User Validation
Students will test the chatbot by asking questions
about Python concepts, syntax, and libraries.
User feedback collection is pending.

### Usage Permission
The Python documentation is distributed under
the Python Software Foundation License.
Required license and copyright notices will
be preserved.

### Evaluation Plan
At least 100 verifiable question-answer pairs
will be created with supporting source passages.

### Approval
Status: Pending professor approval.

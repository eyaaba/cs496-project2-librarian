# CS496 — Project 2: The Librarian

**Status:** WIP starter prototype (NOT a completed submission).

## What works in this baseline
- Parse PDFs, TXT, Markdown into page-aware chunks
- OpenAI embedding and persistent JSON index
- BM25, vector and simple reciprocal-rank-fusion hybrid retrieval
- Grounded OpenAI generation with passage IDs
- Streamlit interface to index documents and ask questions

## Setup (Python 3.10+)
```bash
python -m pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add the professor-authorized `OPENAI_API_KEY`. Never commit `.env` or paste secrets into Bloomi/GitHub. Check allowable models and credits first.

## Run
```bash
streamlit run app.py
```
Add legally usable PDFs to `data/documents/`, then click **Build / rebuild index** in the sidebar (requires API credit). Ask a question.

## IMPORTANT limitations / next steps
- Get professor approval for a REAL hundreds-of-pages corpus; collect real user feedback and 100+ verified Q&A with supporting evidence.
- No reranker yet; no metadata filtering, refusal threshold or question rewrite yet.
- Citations are prompt-instructed, not independently verified; add citation validation and faithfulness eval.
- Index is a JSON file (not a production vector DB); change for large corpus and justify.
- No deployment, held-out evaluation, ablation numbers, optimization or cost estimates yet.
- Hybrid fusion is provisional baseline, not experimentally tuned.
- Keep confidential docs out of public git. Verify data permissions before upload to any cloud/API.

## Contributions
- Eya: RAG & OpenAI integration
- khalil: corpus & indexing
- Yosr: evaluation & optimization
- Mohamed: app & deployment
- Shared: corpus approval, final report, testing and demo

## Current Progress

### Milestone 1 — Local RAG Prototype (Completed)

- GitHub repository created
- Python environment configured
- OpenAI API integrated
- PDF extraction and chunking tested
- Search index successfully built
- First source-grounded question answered with citation

Test question:
"What are the five requirements for a good corpus?"

Result:
Correct answer citing project2-librarian.pdf, page 5.

### Next Milestone

- Choose and obtain approval for a real document corpus
- Index the approved documents
- Create verified evaluation questions
- Improve hybrid retrieval and add reranking


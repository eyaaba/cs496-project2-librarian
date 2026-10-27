# CS496 — Project 2: The Librarian

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from documents.

## Team Members

- Eya Ben Ammar
- Khalil Ben Omrane
- Yosr Maalej
- Mohamed Ben HajSalah

## Project Overview

The Librarian is a RAG application that retrieves relevant information from documents and generates answers using the OpenAI API.

Our selected corpus is the official Python 3.14 documentation. The application aims to help students find reliable answers to Python programming questions, with references to the original documentation.

## Technologies

- Python
- OpenAI API
- Streamlit
- PyMuPDF
- BM25
- NumPy

## Installation

**1. Install the dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure the OpenAI API**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
```

**3. Download the corpus**

```bash
python download_corpus.py
```

This downloads the official Python 3.14.0 documentation into `data/documents/`.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local Streamlit URL and select **Build / rebuild index** to generate the document embeddings and create the searchable index.

Once indexing is complete, enter a question to retrieve information from the Python documentation.

An authorized OpenAI API key is required for indexing and answer generation.

## Current Progress

- RAG application implemented with Streamlit and OpenAI
- Official Python 3.14 documentation collected (528 files)
- 8,954 document chunks generated and indexed
- OpenAI embeddings generated (1,536 dimensions)
- BM25, vector search, and hybrid retrieval implemented
- Embedding caching and persistent index implemented
- Corpus download script added for reproducibility
- Local retrieval and question-answering tests completed

## Remaining Work

- Implement reranking and improve retrieval performance
- Validate citations and handle unsupported questions
- Prepare 100+ verified question-answer pairs
- Evaluate retrieval and generation quality
- Measure and optimize cost and latency
- Deploy the application and collect user feedback
- Complete the final report and documentation

## Contributions

| Member | Responsibility |
|---|---|
| Eya | RAG pipeline and OpenAI integration |
| Khalil | Document processing and indexing |
| Yosr | Evaluation and optimization
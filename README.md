# CS496 — Project 2: The Librarian

A Retrieval-Augmented Generation (RAG) application that answers questions using information from a collection of documents.


## Team Members

- Eya Ben Ammar
- khalil Ben Omrane
- Yosr Maalej
- Mohamed Ben HajSalah

## Project Overview

Our project uses RAG to retrieve relevant information from documents and generate answers using the OpenAI API.

We selected the official Python 3.14 documentation as our document collection. The application is designed to help students find information about Python programming.

## Technologies

- Python
- OpenAI API
- Streamlit
- PyMuPDF
- BM25
- NumPy

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file containing the OpenAI API key:

```env
OPENAI_API_KEY=your_api_key
```

## Running the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL, build the document index, and ask a question.

## Current Progress

- Basic RAG application implemented using Streamlit and OpenAI
- Official Python 3.14 documentation collected (528 files)
- 8,954 document chunks indexed using OpenAI embeddings
- BM25, vector search and hybrid retrieval implemented
- Embedding cache and persistent index implemented
- Local retrieval and question-answering tests completed

## Remaining Work

- Add reranking and improve retrieval performance
- Validate citations and unsupported-question handling
- Build the 100+ question evaluation dataset
- Run retrieval and generation evaluations
- Optimize cost and latency
- Deploy the application and collect user feedback
- Complete the final report

## Contributions

| Member | Responsibility |
|---|---|
| Eya | RAG pipeline and OpenAI integration |
| khalil | Document processing and indexing |
| Yosr | Evaluation and optimization |
| Mohamed | Application and deployment |
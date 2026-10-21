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

- GitHub repository and Python environment configured
- Basic RAG application implemented and tested
- OpenAI API connected
- Document extraction and chunking implemented
- 528 Python documentation files collected
- 8,954 document chunks generated
- Initial embedding test completed successfully

## Remaining Work

- Complete indexing of the Python documentation
- Improve document retrieval and add reranking
- Prepare the evaluation dataset
- Measure accuracy and performance
- Optimize and deploy the application

## Contributions

| Member | Responsibility |
|---|---|
| Eya | RAG pipeline and OpenAI integration |
| khalil | Document processing and indexing |
| Yosr | Evaluation and optimization |
| Mohamed | Application and deployment |
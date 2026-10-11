# Thaif Document RAG

An AI-powered PDF question-answering application using Retrieval-Augmented Generation (RAG), semantic search, and NVIDIA-hosted language models.

## Features
- Upload PDF documents
- Extract text and split it into chunks
- Generate text embeddings using NVIDIA Nemotron
- Retrieve relevant passages using cosine similarity
- Generate answers based on retrieved document content
- Simple desktop GUI built with Tkinter

## Technologies
- Python
- NVIDIA API
- Nemotron Embeddings
- OpenAI Python SDK
- NumPy
- PyPDF
- Tkinter

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Create a `.env` file in the project directory and add:

   `NVIDIA_API_KEY=your_api_key_here`

4. Run the application:

   `python main.py`

## Security
Never upload your `.env` file or expose your API key. Use your own NVIDIA API key to run the application.

## Current Limitations
Scanned PDFs without selectable text are not supported in this version.
# Multi-Agent Document Intelligence Platform

A FastAPI-based document intelligence platform that performs document ingestion, semantic retrieval, question answering, summarization, and compliance validation using a multi-agent RAG architecture.

## Features

- Document ingestion from text files
- Text chunking with overlap
- Local HuggingFace embeddings
- FAISS vector database for semantic search
- Retriever Agent for relevant chunk retrieval
- Gemini-based Answer Agent with fallback handling
- Summarizer Agent with local fallback summary
- Compliance Checker Agent with rule-based fallback
- Neon PostgreSQL integration for metadata, query logs, and compliance reports
- REST API using FastAPI

## Tech Stack

- Python
- FastAPI
- LangChain
- HuggingFace Sentence Transformers
- FAISS
- Gemini API
- PostgreSQL
- Neon
- Psycopg2
- Uvicorn

## Project Architecture

```text
User
 |
FastAPI Backend
 |
 |-- Ingestion Module
 |     Loads document text
 |
 |-- Chunking Module
 |     Splits document into overlapping chunks
 |
 |-- HuggingFace Embeddings
 |     Converts chunks into vectors locally
 |
 |-- FAISS Vector Store
 |     Stores vectors for semantic search
 |
 |-- Retriever Agent
 |     Retrieves relevant chunks
 |
 |-- Answer Agent
 |     Uses Gemini for RAG-based answering
 |
 |-- Summarizer Agent
 |     Generates document summaries
 |
 |-- Compliance Agent
 |     Validates policy/compliance requirements
 |
 |-- Neon PostgreSQL
       Stores document metadata, query logs, and compliance reports
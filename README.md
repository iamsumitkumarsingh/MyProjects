# 🚀 Git Cheat Code Assistant - RAG-based Chatbot

This is an AI-powered **RAG-based (Retrieval-Augmented Generation) chatbot** built using **FastAPI, LangChain, LangGraph, and Flask**. It allows users to ask **Git-related questions**, retrieves relevant documentation, and generates AI-powered answers.

---

## 📌 **Project Overview**

### ✅ Features:
- **Document Ingestion**: Loads PDF files, splits them into chunks, and creates embeddings.
- **Vector Search**: Retrieves similar documents using **FAISS / ChromaDB**.
- **RAG Pipeline**: Uses a **retrieval-augmented generation** approach to improve responses.
- **Query Expansion**: Rewrites user queries for better retrieval.
- **Reranking**: Sorts retrieved documents using a **Cross-Encoder model**.
- **Flask Web Interface**: Provides a simple UI for user interaction.

---

## 🛠️ **Tech Stack**
- **Python** (Primary Language)
- **LangChain** (Document processing & retrieval)
- **LangGraph** (Agent-based execution workflow)
- **FastAPI** (Backend for API endpoints)
- **Flask** (Frontend integration)
- **ChromaDB / FAISS** (Vector store for document retrieval)
- **HuggingFace Embeddings** (Text embeddings)
- **Cross-Encoder (MS-MARCO)** (Document reranking)
- **Groq LLM** (AI-powered response generation)

---

## 📂 **Project Structure**

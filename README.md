# DocMind — Production-Grade RAG Pipeline with Observability
[Live ](http://docmind-rag.duckdns.org/)
## 🚀 Overview

DocMind is a production-style Retrieval Augmented Generation (RAG) system that allows users to upload documents and ask questions over them using an LLM-powered pipeline. It goes beyond a basic RAG implementation by adding reranking, observability, and deployment readiness.

---

## ✨ Key Features

* 📄 Multi-format document support (PDF, DOCX, TXT, CSV, MD)
* 🔍 Semantic search using embeddings
* 🎯 Reranking using cross-encoder model for better relevance
* 🤖 LLM-based answer generation (Groq / LLaMA models)
* 📊 Observability with Langfuse (traces, prompts, responses)
* 🌐 Streamlit-based interactive UI
* ☁️ Production deployment on AWS EC2
* 🔐 Nginx reverse proxy setup
* ⚡ Fast inference pipeline

---

## 🏗️ System Architecture

User → Streamlit UI → Document Upload → Text Extraction → Chunking → Embeddings → Vector Search → Reranking → LLM (Groq) → Response

With observability:
All steps are logged and traced using Langfuse.

---

## 🧰 Tech Stack

### Backend / AI

* Python
* LangChain / custom RAG pipeline
* SentenceTransformers (embeddings)
* Cross-encoder reranker (MiniLM)
* Groq LLM (LLaMA 3.1)

### Vector Store

* FAISS / Chroma (depending on config)

### Frontend

* Streamlit

### Deployment

* AWS EC2 (Ubuntu)
* Nginx (reverse proxy)
* DuckDNS / custom domain

### Observability

* Langfuse
* Logging (system logs / debug logs)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/fatima-firdouse/Production-Grade-RAG-Pipeline-with-Observability
cd DocMind
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
# 🔑 LLM / Inference
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key

# 📊 Observability (Langfuse)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# 🧠 Tracking / Debugging
USER_AGENT=rag-observability-pipeline/1.0
```

---

## ▶️ Run the Application

### Local

```bash
streamlit run app.py
```

### Production (EC2)

```bash
nohup streamlit run app.py --server.port 8501 &
```

---

## 🌐 Nginx Configuration

Used as reverse proxy to expose Streamlit app:

* Handles HTTP routing
* Manages client size limits
* Improves security layer

---

## 📊 Observability

DocMind integrates Langfuse to track:

* User queries
* Retrieved documents
* Prompt sent to LLM
* Model responses
* Latency and performance

This helps in debugging and improving retrieval quality.

---

## ⚠️ Known Challenges & Fixes

### 1. Large file upload issues

* Problem: 413 Request Entity Too Large
* Fix: Increased `client_max_body_size` in Nginx

### 2. Slow retrieval

* Fix: Added reranking layer

### 3. Poor answer quality

* Fix: Improved chunking strategy + reranker

---

## 📌 Future Improvements

* Add caching layer (Redis)
* Add authentication system
* Support multi-user sessions
* Improve chunking with semantic splitting
* Add evaluation dashboard for RAG quality

---

## 🧠 What makes this project different

Unlike basic RAG systems, DocMind includes:

* Production deployment setup
* Observability layer
* Reranking for accuracy boost
* Real-world file handling constraints
* End-to-end pipeline monitoring

---

## 👩‍💻 Author

Built as part of an AI engineering learning journey focused on production-ready LLM systems.

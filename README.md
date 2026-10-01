# DocMind — Production-Grade RAG Pipeline with Observability

⚡ [Live Demo](https://docmind-rag-pipeline.streamlit.app/)

## 🚀 Overview
DocMind is a production-style Retrieval Augmented Generation (RAG) system that allows users to upload documents and ask questions over them using an LLM-powered pipeline. It goes beyond a basic RAG implementation by adding hybrid retrieval, reranking, and full observability.

## ✨ Key Features
- 📄 Multi-format document support (PDF, DOCX, TXT, CSV, MD)
- 🔍 Hybrid retrieval — BM25 keyword search + semantic embedding search
- 🎯 Reranking using a cross-encoder model (MiniLM) for better relevance
- 🤖 LLM-based answer generation (Groq / LLaMA 3.1)
- 📊 Observability with Langfuse (traces, prompts, responses, latency)
- 🌐 Streamlit-based interactive UI
- ⚡ Fast inference pipeline

## 🏗️ System Architecture
```
User → Streamlit UI → Document Upload → Text Extraction → Chunking → Embeddings → Hybrid Retrieval (BM25 + Semantic) → Reranking → LLM (Groq) → Response
```
Every step is logged and traced using Langfuse.

## 🧰 Tech Stack
**Backend / AI**
- Python, LangChain
- SentenceTransformers (`BAAI/bge-small-en-v1.5`) for embeddings
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- BM25 for keyword-based retrieval
- Groq LLM (LLaMA 3.1)

**Vector Store**
- ChromaDB (persistent, cosine similarity, HNSW index)

**Frontend**
- Streamlit

**Deployment**
- Streamlit Community Cloud
- *(Originally self-hosted on AWS EC2 with Nginx as reverse proxy; migrated to Streamlit Community Cloud for continuous, maintenance-free hosting.)*

**Observability**
- Langfuse

## ⚙️ Installation
```bash
git clone https://github.com/fatima-firdouse/Production-Grade-RAG-Pipeline-with-Observability
cd Production-Grade-RAG-Pipeline-with-Observability
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For offline evaluation (Ragas-based), also install:
```bash
pip install -r requirements-dev.txt
```

## 🔐 Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
USER_AGENT=rag-observability-pipeline/1.0
```

## ▶️ Run Locally
```bash
streamlit run app.py
```

## 📊 Observability
DocMind integrates Langfuse to track user queries, retrieved documents, prompts sent to the LLM, responses, and per-call latency — useful for debugging and improving retrieval quality over time.

## ⚠️ Known Challenges & Fixes
1. **Large file uploads** — Hit `413 Request Entity Too Large` during the original Nginx-based deployment; fixed via `client_max_body_size`.
2. **Slow / imprecise retrieval** — Semantic search alone missed exact-keyword matches; added BM25 hybrid retrieval plus a cross-encoder reranking layer.
3. **Poor answer quality on long documents** — Improved by tuning the chunking strategy.

## 📌 Future Improvements
- Add caching layer (Redis)
- Add authentication system
- Support multi-user sessions
- Add a live evaluation dashboard for RAG quality (building on the existing Ragas-based offline evaluation)

## 🧠 What Makes This Project Different
- Hybrid retrieval, not just vector search
- Full observability layer (not just logs)
- Offline evaluation with Ragas, tracked separately from the production dependency set
- Real-world file handling constraints, solved and documented

## 👩‍💻 Author
Fatima Firdouse — B.Tech Artificial Intelligence & Data Science
📧 fatimafirdouse011@gmail.com

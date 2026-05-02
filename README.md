# ⚖️ AI Law Advisor

> **Intelligent Indian Constitutional Law Assistant** — Powered by RAG, Fine-tuned LLM, and Semantic Search

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/Callmeyash11/Ai_Law_advisor)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/yashjaiwal/Ai_Law_Advisor)
[![Model](https://img.shields.io/badge/🤖%20Model-legal--clean--model-orange?style=for-the-badge)](https://huggingface.co/Callmeyash11/legal-clean-model)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📌 Table of Contents

- [About](#about)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [Hackathon](#hackathon)
- [Disclaimer](#disclaimer)

---

## 🎯 About

**AI Law Advisor** is a production-grade, domain-specific AI assistant for **Indian Constitutional Law**. It combines a fine-tuned Large Language Model (LLM) with a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-grounded answers from the Constitution of India.

Unlike general-purpose chatbots, this system:
- **Never hallucinates** — answers are strictly grounded in retrieved constitutional text
- **Remembers context** — maintains conversation history per session
- **Scales cleanly** — decoupled FastAPI backend + Streamlit frontend

> Built for **HackDiwas 3.0** by **Team Gradient Gang** 🏆 (Top 17 Teams)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📜 Constitution-grounded answers | Retrieves relevant chunks from the Constitution of India |
| 🧠 No hallucination | Strict context filtering with FAISS similarity scores |
| 💬 Chat memory | Per-session conversation history via SQLite |
| ⚡ FastAPI backend | Production-grade REST API with health checks |
| 🎨 Streamlit frontend | Clean, responsive chat UI |
| 🔍 Semantic search | FAISS vector store with MiniLM embeddings |
| 🐳 Docker support | Containerized deployment |
| 🤗 HF Spaces ready | Auto-deploy on HuggingFace Spaces |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Browser                        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Streamlit Frontend (Port 7860)               │
│                     app.py                               │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP REST
┌─────────────────────▼───────────────────────────────────┐
│              FastAPI Backend (Port 8000)                  │
│                     api.py                               │
└──────┬──────────────┬──────────────────┬────────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼──────┐ ┌────────▼───────┐
│  RAG Engine │ │   Memory   │ │  Model Loader  │
│rag_engine.py│ │ memory.py  │ │model_loader.py │
└──────┬──────┘ └─────┬──────┘ └────────────────┘
       │              │
┌──────▼──────┐ ┌─────▼──────┐
│ Vector Store│ │  SQLite DB │
│vectorstore.py│ │  chat.db   │
└──────┬──────┘ └────────────┘
       │
┌──────▼──────┐
│ FAISS Index │
│ Constitution│
│    of India │
└─────────────┘
```

---

## 📁 Project Structure

```
Ai_Law_Advisor/
│
├── 📄 app.py               # Streamlit UI — Chat interface
├── 📄 api.py               # FastAPI Backend — REST endpoints
├── 📄 rag_engine.py        # RAG Pipeline — Retrieval + Generation
├── 📄 model_loader.py      # LLM Loading + Inference
├── 📄 vectorstore.py       # FAISS Index Build + Semantic Search
├── 📄 memory.py            # Chat Memory — SQLite CRUD
├── 📄 prompt.py            # Prompt Engineering + Output Cleaning
├── 📄 config.py            # Centralized Configuration
│
├── 🐳 Dockerfile           # Docker Image
├── 🐳 docker-compose.yml   # Multi-service Orchestration
├── 📄 start.sh             # Startup Script (FastAPI + Streamlit)
├── 📄 requirements.txt     # Python Dependencies
├── 📄 .gitignore           # Git Ignore Rules
├── 📄 .dockerignore        # Docker Ignore Rules
│
└── 📁 data/
    └── the_constitution_of_india.pdf   # Source Document
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- 8GB+ RAM (for local LLM)

### 1. Clone

```bash
git clone https://github.com/yashjaiwal/Ai_Law_Advisor
cd Ai_Law_Advisor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file:

```env
MODEL_NAME=Callmeyash11/legal-clean-model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
INDEX_PATH=storage/faiss_index
DB_PATH=chat.db
PDF_PATH=data/the_constitution_of_india.pdf
TOP_K=5
```

### 4. Run

```bash
# Terminal 1 — FastAPI Backend
python api.py

# Terminal 2 — Streamlit Frontend
streamlit run app.py --server.port 8501
```

> ⚠️ **Note:** On first run, FAISS index auto-builds from the PDF — takes ~2 minutes.

Open browser: `http://localhost:8501`

---

## 🐳 Docker Setup

### Build & Run

```bash
docker-compose up --build
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `ai_law_api` | 8000 | FastAPI Backend |
| `ai_law_streamlit` | 7860 | Streamlit Frontend |

### Stop

```bash
docker-compose down
```

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "ready": true,
  "index_ready": true,
  "message": "✅ Fully ready!"
}
```

---

### Ask Question

```http
POST /ask
```

**Request Body:**
```json
{
  "question": "What is Article 21?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "Article 21 provides protection of life and personal liberty...",
  "source_found": true,
  "session_id": "abc123"
}
```

---

### Get Chat History

```http
GET /history/{session_id}
```

**Response:**
```json
{
  "session_id": "abc123",
  "history": [
    {"role": "user", "content": "What is Article 21?"},
    {"role": "assistant", "content": "Article 21 provides..."}
  ],
  "total_messages": 2
}
```

---

### Clear Session

```http
DELETE /session/{session_id}
```

---

### New Session

```http
POST /session/new
```

**Response:**
```json
{
  "session_id": "new-uuid-here"
}
```

---

## 🧠 Model Details

| Component | Value |
|-----------|-------|
| **LLM** | `Callmeyash11/legal-clean-model` |
| **Base Architecture** | Llama 3.2 3B Instruct |
| **Fine-tuning Method** | QLoRA (4-bit) |
| **Training Data** | Saul-Instruct-v1 + LawInstruct |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (L2 distance) |
| **Context Window** | 2048 tokens |
| **Max Generation** | 150 tokens |

---

## 📊 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Streamlit | 1.32.0 |
| **Backend** | FastAPI | 0.110.0 |
| **LLM Framework** | HuggingFace Transformers | 4.44.0 |
| **RAG Framework** | LangChain | 0.1.16 |
| **Vector Store** | FAISS | 1.8.0 |
| **Embeddings** | Sentence Transformers | 2.7.0 |
| **Memory** | SQLite | Built-in |
| **Containerization** | Docker + Compose | Latest |
| **Deployment** | HuggingFace Spaces | — |

---

## 🏆 Hackathon

| Field | Details |
|-------|---------|
| **Event** | HackDiwas 3.0 |
| **Team Name** | Gradient Gang |
| **Team ID** | UUHD3046 |
| **Project** | AI Law Adversary |
| **Result** | 🏆 Top 17 Teams Selected |

---

## 🛠️ Configuration

All configuration is in `config.py` — auto-detects environment:

```python
# Auto-detects: Colab → Docker → HuggingFace Spaces
BASE_DIR = "/app"              # Docker
BASE_DIR = "/home/user/app"    # HF Spaces  
BASE_DIR = "/content/..."      # Google Colab
```

---

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**.

- ❌ Not a substitute for professional legal advice
- ❌ Not verified by legal professionals
- ✅ Always consult a qualified lawyer for legal matters
- ✅ Answers based solely on retrieved constitutional text

---

## 👨‍💻 Author

**Yash Jaiwal**

[![GitHub](https://img.shields.io/badge/GitHub-yashjaiwal-black?style=flat&logo=github)](https://github.com/yashjaiwal)
[![HuggingFace](https://img.shields.io/badge/🤗-Callmeyash11-orange?style=flat)](https://huggingface.co/Callmeyash11)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ for HackDiwas 3.0 | Team Gradient Gang</sub>
</div>

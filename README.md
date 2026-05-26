<p align="center">
  <h1 align="center">📚 DocuMind</h1>
  <p align="center">
    <strong>Production-Grade RAG Pipeline for Intelligent Document Q&A</strong>
  </p>
  <p align="center">
    <a href="https://medium.com/@aakashyadav1607/documind-building-a-production-grade-rag-system-that-actually-works-09b47e76f9b6">📝 Read the Blog</a>
    &nbsp;·&nbsp;
    <a href="#-quick-start">🚀 Quick Start</a>
    &nbsp;·&nbsp;
    <a href="#-architecture">🏗️ Architecture</a>
    &nbsp;·&nbsp;
    <a href="#-benchmarks">📊 Benchmarks</a>
  </p>
</p>

---

Most RAG systems work in tutorials. DocuMind is built to work in production.

This isn't another "query → retrieve top-k → send to LLM → answer" wrapper. DocuMind is an **end-to-end RAG system** with hybrid search, cross-encoder reranking, graph-based self-correction, automated evaluation, per-query cost tracking, and full Docker deployment — designed to handle real enterprise documents like financial reports, 10-Ks, and annual filings.

> **Why is this different?** Every design decision — from the retrieval strategy to the hallucination detection — was made to solve a specific production failure I encountered while building this system. [Read the full story →](https://medium.com/@aakashyadav1607/documind-building-a-production-grade-rag-system-that-actually-works-09b47e76f9b6)

---

## ✨ Key Features

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| 🔍 **Sequential Hybrid Search** | Dense vector retrieval → local BM25 on top-50 candidates → RRF fusion | Gets both semantic meaning AND exact keyword matches without the O(N) BM25 bottleneck |
| 🎯 **Cohere Reranking** | Cross-encoder reranker scores all candidates for true relevance | Solves "lost in the middle" — puts the best chunk at rank #1 |
| 🔄 **LangGraph Self-Correction** | Stateful graph workflow with hallucination detection and conditional re-retrieval | Catches bad answers before the user sees them |
| ⚡ **Score-Based Skip-Grading** | High-confidence reranker results bypass LLM grading entirely | Grading latency: 3.6s → <300ms |
| 🧵 **Parallel LLM Grading** | ThreadPoolExecutor grades ambiguous docs concurrently | No more sequential API calls for multi-doc grading |
| 📄 **LlamaParse Ingestion** | Vision-LM converts PDFs to Markdown, preserving table structure | Financial tables stay intact — no more shredded rows |
| 💰 **Per-Query Cost Tracking** | Every LLM call records tokens, model, and cost in real-time | Know exactly what each query costs ($0.0006 avg) |
| 📊 **RAGAS Evaluation Pipeline** | Automated faithfulness, relevancy, precision, recall scoring | Prove your pipeline quality with numbers, not vibes |
| 🐳 **Docker Compose Deployment** | One command spins up Qdrant + FastAPI + Streamlit | Clone → `docker compose up` → working system |
| 🔌 **Decoupled API** | FastAPI backend serves any frontend via REST | Plug into any UI, mobile app, or third-party service |

---

## 🏗️ Architecture

```
                          ┌─────────────────┐
                          │   User Query     │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Sequential Hybrid Search   │
                    │                              │
                    │  1. Dense (Qdrant) → Top 50  │
                    │  2. BM25 on 50 candidates    │
                    │  3. RRF Fusion               │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Cohere Reranker v3         │
                    │   (Cross-encoder scoring)    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │        LangGraph RAG Workflow            │
              │                                         │
              │  ┌──────────┐  ┌───────────┐  ┌──────┐ │
              │  │  Grade   │→ │ Generate  │→ │Check │ │
              │  │  (skip/  │  │ (Groq     │  │Halluc│ │
              │  │ parallel)│  │  LLaMA)   │  │inate │ │
              │  └──────────┘  └───────────┘  └──┬───┘ │
              │                                  │     │
              │         ↻ Re-retrieve if flagged │     │
              └──────────────────────────────────┘     │
                                   │
                    ┌──────────────▼──────────────┐
                    │  Response + Sources + Cost   │
                    │  + Latency + Hallucination   │
                    └─────────────────────────────┘
```

### 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq (LLaMA 3.1 8B / 3.3 70B) with automatic fallback |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, zero API cost) |
| **Vector DB** | Qdrant (HNSW indexing, cosine similarity) |
| **Reranker** | Cohere Rerank v3 (cross-encoder) |
| **Orchestration** | LangGraph (stateful graph with conditional edges) |
| **Ingestion** | LlamaParse (markdown-aware) + PyMuPDF fallback |
| **API** | FastAPI with Pydantic V2 validation |
| **UI** | Streamlit (calls FastAPI via HTTP) |
| **Evaluation** | RAGAS (faithfulness, relevancy, precision, recall) |
| **Observability** | structlog + custom CostTracker + tiktoken |
| **Deployment** | Docker Compose (Qdrant + API + UI) |

---

## 📊 Benchmarks

### RAGAS Evaluation Scores

| Metric | Score | What It Measures |
|--------|-------|-----------------|
| Faithfulness | 0.47 → **improving** | Are answers grounded in retrieved context? |
| Answer Relevancy | **0.80** | Do answers address the question? |
| Context Precision | **0.78** | Are retrieved chunks relevant? |
| Context Recall | **1.00** ✅ | Did retrieval find everything needed? |

### Latency Breakdown

| Stage | Before Optimization | After Optimization |
|-------|--------------------|--------------------|
| Embedding Model Load | 24,600ms | **0ms** (singleton + pre-warm) |
| Hybrid Search | 1,500ms | **480ms** (sequential approach) |
| Cohere Reranking | 2,790ms | 2,790ms (API-bound) |
| Document Grading | 3,597ms | **<300ms** (skip-grading + parallel) |
| LLM Generation | 3,008ms | 3,008ms (API-bound) |
| Hallucination Check | 3,339ms | 1,658ms (removed truncation) |
| **Total (first query)** | **~38,000ms** | **~6,900ms** |

### Cost Per Query

| Model | Avg Cost |
|-------|----------|
| LLaMA 3.1 8B (grading + hallucination) | $0.0001 |
| LLaMA 3.3 70B (generation) | $0.0005 |
| **Total per query** | **~$0.0006** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop ([install](https://www.docker.com/products/docker-desktop/))
- API Keys: [Groq](https://console.groq.com/keys), [Cohere](https://dashboard.cohere.com/api-keys) (free tier works)

### Option 1: Docker (Recommended) 🐳

```bash
# Clone the repo
git clone https://github.com/Aak07/documind.git
cd documind

# Create your .env file
cp .env.example .env
# Fill in your GROQ_API_KEY and COHERE_API_KEY

# Start everything
docker compose up --build -d

# Verify all services are running
docker compose ps

# Open the UI
# Streamlit: http://localhost:8501
# FastAPI Docs: http://localhost:8000/docs
# Qdrant Dashboard: http://localhost:6333/dashboard
```

### Option 2: Local Development 💻

```powershell
# Clone and enter
git clone https://github.com/Aak07/documind.git
cd documind

# Create virtual environment
python -m venv env
.\env\Scripts\Activate   # Windows
# source env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Fill in your API keys

# Start Qdrant
docker run -d --name documind-qdrant-cont -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Ingest documents
python -m src.ingestion.ingest --dir data/sample_docs/

# Start the API (Terminal 1)
uvicorn src.api.main:app --reload --port 8000

# Start the UI (Terminal 2)
streamlit run ui/app.py

# Open http://localhost:8501 in your browser
```

---

## 📁 Project Structure

```
documind/
├── src/
│   ├── config.py                    # Pydantic V2 settings (env vars, model config)
│   ├── ingestion/
│   │   ├── loader.py                # LlamaParse + PyMuPDF factory loader
│   │   ├── chunker.py               # Markdown-aware + recursive chunking
│   │   ├── embedder.py              # Thread-safe singleton HuggingFace embedder
│   │   ├── store.py                 # Qdrant collection management + batch upsert
│   │   └── ingest.py                # CLI ingestion pipeline
│   ├── retrieval/
│   │   ├── hybrid_search.py         # Dense → BM25 → RRF (sequential hybrid)
│   │   ├── reranker.py              # Cohere v3 reranker with graceful fallback
│   │   └── retriever.py             # Orchestrates search → rerank → top-k
│   ├── generation/
│   │   ├── state.py                 # LangGraph TypedDict state schema
│   │   ├── prompts.py               # Versioned prompt templates
│   │   ├── nodes.py                 # Graph nodes with cost tracking + parallel grading
│   │   └── graph.py                 # LangGraph workflow with self-correction loop
│   ├── evaluation/
│   │   ├── ragas_eval.py            # RAGAS pipeline (uses Groq, not OpenAI)
│   │   └── benchmark.py             # Timestamped benchmark runner
│   ├── observability/
│   │   ├── cost_tracker.py          # Per-call token counting + cost estimation
│   │   ├── latency.py               # Async-safe timing decorator
│   │   └── logger.py                # structlog (console dev / JSON prod)
│   └── api/
│       ├── main.py                  # FastAPI app with lifespan pre-warming
│       └── schemas.py               # Pydantic V2 request/response models
├── ui/
│   └── app.py                       # Streamlit chat UI (calls FastAPI via HTTP)
├── eval/
│   └── golden_dataset.json          # 22 Q&A pairs for NVIDIA financial reports
├── docker-compose.yml               # Qdrant + API + UI with bind mounts
├── Dockerfile                       # Pre-baked HF model, Python 3.11-slim
├── make.ps1                         # PowerShell task runner
└── requirements.txt                 # Pinned dependencies
```

---

## 🔧 API Reference

Once the API is running at `http://localhost:8000`, interactive docs are at `/docs`.

### Query Documents
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVIDIA Q3 fiscal 2026 revenue?"}'
```

**Response:**
```json
{
  "question": "What was NVIDIA Q3 fiscal 2026 revenue?",
  "answer": "The record revenue for Q3 fiscal 2026 was $57.0 billion. [NVIDIAAn 2025.pdf, Page 1]",
  "sources": [...],
  "is_hallucination": false,
  "retry_count": 0,
  "latency_ms": {
    "retrieval": 2751,
    "grading": 291,
    "generation": 2211,
    "hallucination_check": 1658
  },
  "cost_usd": 0.000637
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + Qdrant connection status |
| `POST` | `/ingest` | Upload and ingest a document (PDF/TXT) |
| `POST` | `/query` | Run a question through the RAG pipeline |
| `GET` | `/metrics` | Collection statistics |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 📈 Running Evaluation

DocuMind includes an automated RAGAS evaluation pipeline with a golden dataset of 22 NVIDIA financial document Q&A pairs:

```powershell
# Run RAGAS evaluation
python -m src.evaluation.benchmark

# Results saved to eval/eval_results/benchmark_YYYYMMDD_HHMMSS.json
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
COHERE_API_KEY=your_cohere_key_here
HUGGINGFACE_API_TOKEN=hf_your_token_here     # Optional but recommended
LLAMA_CLOUD_API_KEY=your_llamaparse_key      # Optional (falls back to PyMuPDF)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=documind
```

---

## 🛠️ Development Commands

Using PowerShell task runner (`make.ps1`):

```powershell
.\make.ps1 setup          # Install dependencies
.\make.ps1 ingest         # Ingest sample documents
.\make.ps1 api            # Start FastAPI server
.\make.ps1 ui             # Start Streamlit UI
.\make.ps1 eval           # Run RAGAS benchmark
.\make.ps1 docker-up      # Build and start all containers
.\make.ps1 docker-down    # Stop all containers
.\make.ps1 docker-logs    # Tail all container logs
.\make.ps1 status         # Check container status
.\make.ps1 clean          # Remove __pycache__ files
```

---

## 🎯 What Makes This Production-Grade

This isn't a list of buzzwords. Each of these solved a real failure I encountered:

| Problem I Hit | How DocuMind Solves It |
|--------------|----------------------|
| Financial tables got shredded by text splitters | LlamaParse converts to Markdown; MarkdownTextSplitter preserves structure |
| BM25 over full corpus = 35s latency | Sequential hybrid: BM25 runs on 50 dense candidates only |
| Embedding model reloads on every query (24s) | Thread-safe singleton with app-startup pre-warming |
| LLM grading 3 docs sequentially = 3.6s | Score-based skip-grading + ThreadPoolExecutor parallel calls |
| Hallucination checker flagged correct answers | Rewrote prompt for paraphrase tolerance; removed context truncation |
| No visibility into why answers were wrong | Per-query cost, token, and latency tracking on every LLM call |
| Works on my machine but breaks everywhere else | Docker Compose with health checks and bind mounts |
| API tightly coupled to Streamlit UI | FastAPI backend serves any client; Streamlit is just one consumer |

---

## 📝 Blog Post

For the full engineering story — including the latency profiling, the hallucination debugging saga, and the architecture tradeoffs — read the detailed blog post:

**[DocuMind: Building a Production-Grade RAG System That Actually Works →](https://medium.com/@aakashyadav1607/documind-building-a-production-grade-rag-system-that-actually-works-09b47e76f9b6)**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🧠 by <a href="https://linkedin.com/in/aakash-yadav007">Aakash Yadav</a>
</p>

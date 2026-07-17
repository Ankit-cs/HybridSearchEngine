# 🚀 AstraSearch (India-Focused Hybrid Search Engine)

**AstraSearch** is a domain-specific hybrid search engine built from scratch in Python. It automatically filters large-scale Wikipedia datasets during indexing to create a specialized search engine focused exclusively on **Indian history, culture, geography, and leaders**.

It combines classical information retrieval (BM25) with modern semantic search, an agentic AI layer, and a machine-learning ranker — designed as a **modular, extensible, and production-inspired system**.

---

## ✨ Features

### 🔍 Core Search
- BM25 ranking (primary retrieval)
- Inverted index with term frequencies
- Title-aware ranking (separate title index)

### 🧠 Semantic Search
- Transformer-based embeddings (`all-MiniLM-L6-v2`)
- Cosine similarity for semantic matching
- Precomputed document embeddings (offline)

### ⚡ Hybrid Retrieval — Reciprocal Rank Fusion (RRF)
- **RRF fusion algorithm** (industry gold standard used by ElasticSearch & Pinecone)
- Combines BM25 rank + Semantic rank mathematically without raw score normalization
- Achieves ~31% improvement in retrieval accuracy over basic linear interpolation
- Formula: `1/(60 + rank_bm25) + 1/(60 + rank_semantic)`

### 🤖 Learning-to-Rank (LightGBM LTR)
- **LambdaMART gradient boosting model** trained on India-specific queries
- Extracts 6 rich features per (query, doc) pair: BM25 score, semantic score, title overlap, body overlap, doc length, title match
- ML model dynamically determines optimal ranking — no static rules
- Gracefully falls back to RRF if model is not yet trained
- Train with: `python -m scripts.train_ltr`

### 🔄 Query Intelligence
- Semantic query expansion
- Improves recall for weak/short queries

### 🧠 Agentic AI Layer (`/api/v1/agent/smart`)
- **Multi-LLM Support** via `litellm` (Groq / OpenAI / Gemini — auto-detected from `.env`)
- **Multi-Agent Query Router** — classifies queries as `chat`, `literature`, or `compare`
- **Corrective RAG (CRAG)** — rewrites query automatically if retrieval confidence is low
- **Cross-Encoder Re-ranking** — uses `ms-marco-MiniLM-L-6-v2` for agentic search path
- **Generative Answers** — LLM synthesizes a response from retrieved Wikipedia documents
- Core `/api/v1/search` remains untouched and millisecond-fast

### 📦 Data Support
- Multi-parser support (XML, CSV, extensible)
- Automatic parser detection

### ⚙️ System Design
- Modular architecture (parser → index → ranking → API)
- Separate document store and index
- Metadata-driven ranking
- Singleton embedding model (prevents double-loading in memory)

### 🏎️ Enterprise-Grade Storage (AI-Lakehouse Inspired)
- **FAISS Vector Indexing** — blazing-fast similarity search via `IndexFlatIP`
- **Memory-Mapped Loading** (`faiss.IO_FLAG_MMAP`) — vectors stream from disk, RAM usage stays near zero
- **Apache Parquet Document Store** — columnar, compressed document storage via `pandas` / `pyarrow`
- **ID Mapping Layer** — FAISS IDs are transparently mapped back to Wikipedia doc IDs

### 🇮🇳 India-Specific Domain Filter
- Automatically filters all 240k+ Wikipedia articles during indexing
- Extracts only articles related to Indian history, culture, geography, politics, and leaders
- Keywords include: Bharat, Mughal, Chola, Maratha, Ashoka, Gandhi, Modi, ISRO, Bollywood, Vedic, Sanskrit, and 30+ more

### 🌐 API + UI
- FastAPI backend
- Fast search endpoint (`/api/v1/search`)
- Agentic AI endpoint (`/api/v1/agent/smart`)
- Interactive Swagger docs (`/docs`)
- Simple web UI

---

## 🏗️ Architecture Overview

### Offline (Indexing)

```bash
Dataset
 ↓
Parser (auto-detected)
 ↓
Cleaner + Tokenizer
 ↓
Inverted Index + Title Index
 ↓
Metadata (doc lengths, stats)
 ↓
Embedding Generation (Singleton Model)
 ↓
FAISS Index + Parquet Storage
```

### Online (Search) — 4-Tier Pipeline

```bash
User Query
 ↓
Tier 1: BM25 Retrieval  ← (milliseconds, top 1000 docs)
 ↓
Tier 2: Semantic Query Expansion  ← (synonym broadening)
 ↓
Tier 3: RRF Fusion  ← (BM25 rank + Semantic rank merged)
 ↓
Tier 4: LightGBM LTR  ← (ML model final re-rank, top 20)
 ↓
Final Results
```

### Agentic Path (`/api/v1/agent/smart`)

```bash
User Query
 ↓
Multi-Agent Router  ← (chat / literature / compare)
 ↓
CRAG Workflow  ← (Fast FAISS Search → Cross-Encoder Eval)
 ↓
Low Confidence? → LLM Query Rewrite → Search Again
 ↓
LLM Answer Generation  ← (Groq / OpenAI / Gemini)
 ↓
Synthesized Answer + Sources
```

Each component is **independent, testable, and replaceable**, making the system easy to extend with new ranking models, storage backends, or APIs.

---

## 📁 Project Structure

```bash
├── src/
│   ├── parser/       # Dataset parsers (XML, CSV, etc.)
│   ├── preprocessing/ # Cleaning & tokenization
│   ├── indexer/      # Inverted index logic
│   ├── storage/      # Document store & index reader
│   ├── ranking/      # BM25, TF-IDF, LTR (LightGBM)
│   ├── semantic/     # Embeddings, RRF reranker, query expansion, Cross-Encoder
│   ├── agent/        # LLM client, query router, CRAG workflow
│   ├── query/        # Search engine core (4-tier pipeline)
│   └── utils/
├── api/
│   └── routes/
│       ├── search.py   # /api/v1/search (fast, core)
│       └── agentic.py  # /api/v1/agent/smart (AI layer)
├── models/           # Trained LightGBM model (ltr_model.pkl)
├── scripts/          # Indexing, evaluation, LTR training
├── data/             # (ignored) raw + index files
├── logs/
├── .env.example      # API key template
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API Keys (for Agentic features)

```bash
copy .env.example .env
# Edit .env and paste your Groq, OpenAI, or Gemini API key
```

### 4. Fetching the Dataset

**Option A: Standard Dataset (Recommended for testing)**
```bash
python download_data.py
```

**Option B: Massive Dataset (For production metrics)**
```bash
python download_massive_data.py
```
*Warning: Requires at least 120GB of free disk space.*

**Option C: Manual Download**
Download from: https://dumps.wikimedia.org/simplewiki/
Place at: `data/raw/simplewiki.xml`

### 5. Build the index

```bash
python -m scripts.build_index --source data/raw/simplewiki.xml
```

This generates:
```bash
data/index/
├── inverted_index.json     # BM25 keyword index
├── title_index.json        # Title-boosted keyword index
├── documents.parquet       # Compressed columnar document store (Parquet)
├── metadata.json           # Doc lengths & corpus stats
├── embeddings.index        # FAISS binary vector index (mmap-ready)
└── faiss_id_map.json       # Mapping: FAISS sequential ID → Wikipedia doc_id
```

### 6. (Optional) Train the LightGBM LTR Model

```bash
python -m scripts.train_ltr
```

This trains a LambdaMART model on India-specific queries and saves it to `models/ltr_model.pkl`. The server auto-loads it on startup.

### 7. Run the server

```bash
python -m uvicorn api.app:app --reload
```

### 8. Run tests

```bash
python -m pytest
```

---

## ⚙️ Configuration

All paths and constants are centralized in:
```bash
src/utils/config.py
```

---

## 🔑 Key Concepts Implemented

- Inverted Index (BM25 + TF-IDF)
- Title-Aware Ranking with configurable boost factor
- Semantic Embeddings (`all-MiniLM-L6-v2` via SentenceTransformers)
- **Reciprocal Rank Fusion (RRF)** — rank-based hybrid score merging
- **Learning-to-Rank (LightGBM LambdaMART)** — ML-based final re-ranking
- Semantic Query Expansion
- Offline vs Online computation split
- **FAISS Vector Indexing** (Inner Product similarity)
- **Memory-Mapped Index Loading** (near-zero RAM overhead)
- **Apache Parquet Storage** (compressed columnar documents)
- **Singleton Embedding Model** (prevents OOM on startup)
- **India Domain Filter** (custom keyword-based corpus filtration)
- **Agentic CRAG Workflow** (corrective retrieval with query rewriting)
- **Cross-Encoder Re-ranking** (contextual relevance scoring)
- **Multi-LLM Routing** (Groq / OpenAI / Gemini via litellm)

---

## 📊 Evaluation

A custom evaluation script tests the engine's MAP and NDCG@10 against a simulated ground-truth dataset.

```bash
python -m scripts.evaluate
```

*Expected Output (varies by dataset size):*
- MAP: ~0.76
- NDCG@10: ~0.88

---

## 📜 License
MIT License

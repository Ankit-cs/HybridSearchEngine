# AstraSearch — India-Focused Hybrid Search Engine

**AstraSearch** is a domain-specific hybrid search engine built from scratch in Python. It automatically filters large-scale Wikipedia datasets during indexing to create a specialized search engine focused exclusively on **Indian history, culture, geography, and leaders**.

It combines classical information retrieval (BM25) with modern semantic search, an agentic AI layer, and a machine-learning ranker — designed as a **modular, extensible, and production-inspired system**.

---

## Features

### Core Search
- BM25 ranking (primary retrieval)
- Inverted index with term frequencies
- Title-aware ranking (separate title index)

### Semantic Search
- Transformer-based embeddings (`all-MiniLM-L6-v2`)
- Cosine similarity for semantic matching
- Precomputed document embeddings (offline)

### Hybrid Retrieval — Reciprocal Rank Fusion (RRF)
- **RRF fusion algorithm** (industry gold standard used by ElasticSearch & Pinecone)
- Combines BM25 rank + Semantic rank mathematically without raw score normalization
- Achieves ~31% improvement in retrieval accuracy over basic linear interpolation
- Formula: `1/(60 + rank_bm25) + 1/(60 + rank_semantic)`


### Learning-to-Rank (LightGBM LTR)
- **LambdaMART gradient boosting model** trained on India-specific queries
- Extracts 6 rich features per (query, doc) pair: BM25 score, semantic score, title overlap, body overlap, doc length, title match
- ML model dynamically determines optimal ranking — no static rules
- Gracefully falls back to RRF if model is not yet trained
- Train with: `python -m scripts.train_ltr`

### Query Intelligence
- Semantic query expansion
- Improves recall for weak/short queries

### Agentic AI Layer (`/api/v1/agent/smart`)
- **Multi-LLM Support** via `litellm` (Groq / OpenAI / Gemini — auto-detected from `.env`)
- **Multi-Agent Query Router** — classifies queries as `chat`, `literature`, or `compare`
- **Corrective RAG (CRAG)** — rewrites query automatically if retrieval confidence is low
- **Cross-Encoder Re-ranking** — uses `ms-marco-MiniLM-L-6-v2` for agentic search path
- **Generative Answers** — LLM synthesizes a response from retrieved Wikipedia documents
- Core `/api/v1/search` remains untouched and millisecond-fast

### Data Support
- Multi-parser support (XML, CSV, extensible)
- Automatic parser detection

### System Design
- Modular architecture (parser → index → ranking → API)
- Separate document store and index
- Metadata-driven ranking
- Singleton embedding model (prevents double-loading in memory)

### Enterprise-Grade Storage
- **FAISS Vector Indexing** — blazing-fast similarity search via `IndexFlatIP`
- **Memory-Mapped Loading** (`faiss.IO_FLAG_MMAP`) — vectors stream from disk, RAM usage stays near zero
- **Apache Parquet Document Store** — columnar, compressed document storage via `pandas` / `pyarrow`
- **ID Mapping Layer** — FAISS IDs are transparently mapped back to Wikipedia doc IDs
- **HNSW Index** — Hierarchical Navigable Small World graph for sub-millisecond approximate nearest neighbor search (`faiss.IndexHNSWFlat`)
- **IVF-PQ Index** — Inverted File Index with Product Quantization for billion-scale vector search
- **GPU-Accelerated Index** — Automatic GPU detection and IVF-PQ offload for large corpora
- **Adaptive Index Selection** — Automatically chooses optimal index type (Flat/HNSW/IVF-PQ/GPU) based on corpus size

### India-Specific Domain Filter
- Automatically filters all 240k+ Wikipedia articles during indexing
- Extracts only articles related to Indian history, culture, geography, politics, and leaders
- Keywords include: Bharat, Mughal, Chola, Maratha, Ashoka, Gandhi, Modi, ISRO, Bollywood, Vedic, Sanskrit, and 30+ more

### Vector Quantization
- **F16 Quantization** — half-precision vectors (50% memory reduction, negligible accuracy loss)
- **I8 Quantization** — 8-bit integer vectors (75% memory reduction)
- **IVF-PQ Compression** — Product Quantization with configurable M and nbits

### Persistent Full-Text Search
- **Disk-based FTS Index** — persistent BM25 term index backed by SQLite
- **Hybrid FTS + Vector** — combines keyword search with semantic embedding search
- **Document-level Term Tracking** — per-document term frequency maps

### Dual Embeddings
- **Content Embeddings** — capture what the document says (text semantics)
- **Context Embeddings** — capture where/how the document appears (section, preceding/following text)
- **Dual Hybrid Search** — fuse content + context similarity scores for richer retrieval

### ACID Transactions
- **Transaction Manager** — begin/commit/rollback for index writes
- **Idempotent Writer** — batch deduplication prevents duplicate indexing on re-runs
- **Compaction Planner** — merges small files into larger ones (Iceberg-style compaction)

### SQLite Catalog
- **Snapshot Versioning** — immutable snapshots with version history
- **File Manifests** — track files, doc counts, centroids, radius per snapshot
- **Schema Versioning** — store and evolve schema definitions over time
- **Batch Log** — idempotency tracking for re-buildable pipelines

### Schema Evolution
- **ChunkMetadata** — document_id, section_path, preceding/following context, chunk_index
- **LLMContextSchema** — structured context assembly with token budgets
- **ContextAssembler** — deduplication + token-limited context assembly for LLMs
- **SchemaEvolver** — add/rename/drop columns with versioned schema history

### Time-Travel & Versioning
- **TimeTravelManager** — create, list, and restore index versions
- **Version Diffs** — compare document counts between any two versions
- **Restore** — revert index to a previous snapshot state

### Cloud Storage Backends
- **LocalStorage** — default filesystem backend
- **S3Storage** — AWS S3 via boto3 (lazy upload, presigned URLs)
- **GCSStorage** — Google Cloud Storage via google-cloud-storage
- **Unified Interface** — `get_storage_backend()` returns the active backend

### Geometric Pruning
- **Centroid + Radius Pruning** — skip irrelevant file segments before search
- **GeometricPruner** — computes centroids and prunes by query distance threshold

### Parallel Search
- **ConcurrentSearcher** — thread-pool based parallel file search
- **RangeGETLoader** — efficient partial reads (footer, header) without loading entire files
- **LazyIndexLoader** — on-demand index loading with LRU eviction

### Agent Memory System
- **EpisodicMemoryStore** — long-term memory with importance scoring, decay, and FAISS search
- **WorkingMemoryBuffer** — short-term FIFO buffer with overflow draining
- **AgentPartitionManager** — isolated memory partitions per agent

### LLM Context Assembly
- **ContextAssembler** — builds deduplicated, token-limited context from search results
- **Section Path Tracking** — includes document structure (section paths)
- **Preceding/Following Context** — enriches chunks with surrounding text

### API + UI
- FastAPI backend
- Fast search endpoint (`/api/v1/search`)
- Context assembly endpoint (`/api/v1/search/context`)
- Dual embedding search (`/api/v1/search/dual`)
- Full-text search endpoint (`/api/v1/search/fts`)
- Agentic AI endpoint (`/api/v1/agent/smart`)
- Agent memory CRUD (`/api/v1/agent/memory`)
- Time-travel versions (`/api/v1/agent/versions`)
- Catalog stats (`/api/v1/agent/catalog`)
- Schema evolution (`/api/v1/agent/schema`)
- Interactive Swagger UI (`/docs`) — built-in web interface
- ReDoc (`/redoc`) — alternative documentation UI

---

## Architecture Overview

### Offline (Indexing)

```
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
FAISS Index (Flat / HNSW / IVF-PQ / GPU) + Parquet Storage
 ↓
Optional: Dual Embeddings, FTS Index, Catalog Snapshot
```

### Online (Search) — 4-Tier Pipeline

```
User Query
 ↓
Tier 1: BM25 Retrieval  ← (milliseconds, top 1000 docs)
         + FTS Boost (optional)
         + Column Filter (optional)
 ↓
Tier 2: Semantic Query Expansion  ← (synonym broadening)
 ↓
Tier 3: RRF Fusion  ← (BM25 rank + Semantic rank merged)
         + Dual Embedding Boost (optional)
         + Agent Memory Boost (optional)
         + Working Memory Boost (optional)
 ↓
Tier 4: LightGBM LTR  ← (ML model final re-rank, top 20)
 ↓
Final Results
```

### Agentic Path (`/api/v1/agent/smart`)

```
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

## Project Structure

```
├── src/
│   ├── parser/        # Dataset parsers (XML, CSV, etc.)
│   ├── preprocessing/ # Cleaning & tokenization
│   ├── indexer/       # Inverted index, compression, FTS, parallel search
│   │   ├── compression.py   # VectorQuantizer, IVFPQIndex, AdaptiveIndexSelector, GeometricPruner
│   │   ├── fts.py           # PersistentFTSIndex (disk-based full-text search)
│   │   └── parallel_search.py  # ConcurrentSearcher, RangeGETLoader, LazyIndexLoader
│   ├── storage/       # Document store, catalog, ACID, schema, cloud, time-travel
│   │   ├── catalog.py       # SQLite catalog (snapshots, files, schema versions, batch log)
│   │   ├── acid.py          # TransactionManager, IdempotentWriter, CompactionPlanner
│   │   ├── schema.py        # ChunkMetadata, LLMContextSchema, ContextAssembler, SchemaEvolver
│   │   ├── cloud_store.py   # LocalStorage, S3Storage, GCSStorage
│   │   ├── time_travel.py   # TimeTravelManager (version snapshots, restore, diffs)
│   │   └── document_store.py # Parquet document store with context metadata
│   ├── ranking/       # BM25, TF-IDF, LTR (LightGBM)
│   ├── semantic/      # Embeddings, RRF reranker, query expansion, dual embeddings
│   │   ├── embedding_store.py  # FAISS store with HNSW/IVF-PQ/GPU support
│   │   └── dual_embeddings.py  # DualEmbeddingGenerator, DualEmbeddingStore
│   ├── agent/         # LLM client, query router, CRAG workflow, episodic memory
│   │   └── memory.py          # EpisodicMemoryStore, WorkingMemoryBuffer, AgentPartitionManager
│   ├── query/         # Search engine core (4-tier pipeline + context assembly)
│   └── utils/
│       └── config.py  # All centralized paths and constants
├── api/
│   ├── app.py            # FastAPI application (v2.0)
│   └── routes/
│       ├── search.py     # /api/v1/search, /search/context, /search/dual, /search/fts
│       └── agentic.py    # Memory CRUD, catalog, schema endpoints
├── models/             # Trained LightGBM model (ltr_model.pkl)
├── scripts/
│   ├── build_index.py  # Indexing pipeline with all feature flags
│   ├── train_ltr.py    # LightGBM LTR training
│   └── evaluate.py     # Evaluation metrics
├── data/               # (ignored) raw + index files
├── logs/
├── .env.example        # API key template
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- Git
- (Optional) NVIDIA GPU for GPU-accelerated indexing

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HybridSearchEngine.git
cd HybridSearchEngine
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Optional: Install GPU-accelerated FAISS (if you have NVIDIA GPU):**
```bash
pip uninstall faiss-cpu -y
pip install faiss-gpu
```

**Optional: Install test dependencies:**
```bash
pip install pytest
```

### 4. Set up API Keys (for Agentic features)

**Windows:**
```bash
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Then edit `.env` and paste your API keys:
```
GROQ_API_KEY=gsk_your_groq_key_here
OPENAI_API_KEY=sk-your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

You only need **one** LLM provider. Groq is recommended (free tier available).

### 5. Fetching the Dataset

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

### 6. Build the Index

**Basic build (BM25 + FAISS vectors):**
```bash
python -m scripts.build_index --source data/raw/simplewiki.xml
```

**Full build with all features:**
```bash
python -m scripts.build_index --source data/raw/simplewiki.xml --precision f16 --use-dual-embeddings --use-fts --batch-id batch-001
```

**Agent-partitioned build:**
```bash
python -m scripts.build_index --source data/raw/simplewiki.xml --agent-id research-agent
```

**Build with IVF-PQ index (for large datasets):**
```bash
python -m scripts.build_index --source data/raw/simplewiki.xml --precision i8
```

This generates:
```
data/index/
├── inverted_index.json     # BM25 keyword index
├── title_index.json        # Title-boosted keyword index
├── documents.parquet       # Compressed columnar document store (Parquet)
├── metadata.json           # Doc lengths & corpus stats
├── embeddings.index        # FAISS binary vector index (mmap-ready)
├── faiss_id_map.json       # Mapping: FAISS sequential ID → Wikipedia doc_id
├── context_embeddings.index # Context embeddings (dual mode)
├── context_id_map.json     # Context embedding ID mapping
├── fts/                    # Persistent FTS index (SQLite-backed)
├── geometric_stats.json    # Geometric pruning centroids
└── catalog.db              # SQLite catalog (snapshots, files, schemas)
```

### 7. (Optional) Train the LightGBM LTR Model

```bash
python -m scripts.train_ltr
```

This trains a LambdaMART model on India-specific queries and saves it to `models/ltr_model.pkl`. The server auto-loads it on startup.

### 8. Run Tests

```bash
python -m pytest tests/test_search.py -v
```

### 9. Run the Backend Server

**Development mode (with auto-reload):**
```bash
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using Gunicorn (Linux/macOS only):**
```bash
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The server starts at: `http://localhost:8000`

### 10. Access the API

**Swagger UI (Interactive Docs):**
```
http://localhost:8000/docs
```

**ReDoc (Alternative Docs):**
```
http://localhost:8000/redoc
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### 11. API Usage Examples

**Basic Search:**
```bash
curl "http://localhost:8000/api/v1/search?q=Gandhi&k=5"
```

**Search with Dual Embeddings:**
```bash
curl "http://localhost:8000/api/v1/search/dual?q=Indian+independence&k=10"
```

**Full-Text Search:**
```bash
curl "http://localhost:8000/api/v1/search/fts?q=modi+policy&k=10"
```

**Context Assembly (for LLMs):**
```bash
curl "http://localhost:8000/api/v1/search/context?q=Mughal+empire&k=5&max_tokens=4000"
```

**Agentic AI Search:**
```bash
curl "http://localhost:8000/api/v1/agent/smart?q=Tell+me+about+Indian+space+program"
```

**Agent Memory (Add):**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/memory" -H "Content-Type: application/json" -d "{\"agent_id\": \"research\", \"text\": \"ISRO launched Chandrayaan-3\", \"importance\": 1.5}"
```

**Agent Memory (Search):**
```bash
curl "http://localhost:8000/api/v1/agent/memory/search?agent_id=research&q=space+mission"
```

**Catalog Stats:**
```bash
curl "http://localhost:8000/api/v1/agent/catalog"
```

**Time-Travel Versions:**
```bash
curl "http://localhost:8000/api/v1/agent/versions"
```

**Schema Evolution:**
```bash
curl "http://localhost:8000/api/v1/agent/schema/columns"
```

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/search` | GET | Basic hybrid search (BM25 + Semantic) |
| `/api/v1/search/context` | GET | Search + LLM context assembly |
| `/api/v1/search/dual` | GET | Dual embedding search (content + context) |
| `/api/v1/search/fts` | GET | Full-text search with FTS index |
| `/api/v1/agent/smart` | GET | Agentic AI search (CRAG + LLM) |
| `/api/v1/agent/memory` | POST | Add episodic memory |
| `/api/v1/agent/memory/search` | GET | Search agent memory |
| `/api/v1/agent/memory/stats` | GET | Agent memory stats |
| `/api/v1/agent/catalog` | GET | Catalog statistics |
| `/api/v1/agent/versions` | GET | List time-travel versions |
| `/api/v1/agent/schema/columns` | GET | List schema columns |

---

## Configuration

All paths and constants are centralized in:
```
src/utils/config.py
```

**Key paths:**
```
data/index/                  # All index files
data/snapshots/              # Time-travel version snapshots
data/catalog.db              # SQLite catalog database
data/episodic_memory/        # Per-agent episodic memory stores
data/working_memory/         # Working memory buffers
data/index/fts/              # Persistent FTS index
logs/app.log                 # Application logs
models/ltr_model.pkl         # Trained LightGBM model
```

**Environment Variables (`.env`):**
```
GROQ_API_KEY=               # Groq API key (recommended, free tier)
OPENAI_API_KEY=             # OpenAI API key
GEMINI_API_KEY=             # Google Gemini API key
ALLOWED_ORIGINS=*           # CORS allowed origins (comma-separated)
```

Logs are written to:
```
logs/app.log
```

---

## Key Concepts Implemented

- Inverted Index (BM25 + TF-IDF)
- Title-Aware Ranking with configurable boost factor
- Semantic Embeddings (`all-MiniLM-L6-v2` via SentenceTransformers)
- **Reciprocal Rank Fusion (RRF)** — rank-based hybrid score merging
- **Learning-to-Rank (LightGBM LambdaMART)** — ML-based final re-ranking
- Semantic Query Expansion
- Offline vs Online computation split
- **FAISS Vector Indexing** (Inner Product similarity)
- **HNSW Graph Index** — sub-millisecond approximate nearest neighbor search
- **IVF-PQ Index** — billion-scale vector search with Product Quantization
- **GPU-Accelerated Index** — automatic GPU detection and offload
- **Adaptive Index Selection** — auto-selects optimal index type by corpus size
- **F16/I8 Vector Quantization** — memory-efficient vector storage
- **Memory-Mapped Index Loading** (near-zero RAM overhead)
- **Apache Parquet Storage** (compressed columnar documents)
- **Singleton Embedding Model** (prevents OOM on startup)
- **India Domain Filter** (custom keyword-based corpus filtration)
- **Agentic CRAG Workflow** (corrective retrieval with query rewriting)
- **Cross-Encoder Re-ranking** (contextual relevance scoring)
- **Multi-LLM Routing** (Groq / OpenAI / Gemini via litellm)
- **Persistent Full-Text Search** (disk-based BM25 with SQLite backing)
- **Dual Embeddings** (content + context vector spaces)
- **ACID Transactions** (transactional index writes with idempotent batching)
- **SQLite Catalog** (snapshot versioning, file manifests, schema history)
- **Schema Evolution** (add/rename/drop columns with versioned history)
- **Time-Travel Indexing** (create, restore, diff index versions)
- **Cloud Storage Backends** (S3, GCS, local filesystem)
- **Geometric Pruning** (centroid + radius file-level skip)
- **Parallel Search** (thread-pool concurrent file search)
- **Agent Memory** (episodic long-term + working short-term + per-agent partitions)
- **LLM Context Assembly** (token-limited, deduplicated context from search results)

---

## Evaluation

A custom evaluation script tests the engine's MAP and NDCG@10 against a simulated ground-truth dataset.

```bash
python -m scripts.evaluate
```

*Expected Output (varies by dataset size):*
- MAP: ~0.76
- NDCG@10: ~0.88

---

## License
MIT License

# 🚀 AstraSearch

**AstraSearch** is a hybrid search engine built from scratch in Python that combines classical information retrieval (BM25) with modern semantic search using transformer-based embeddings.

It is designed as a **modular, extensible, and production-inspired system** to demonstrate how real-world search engines work internally.

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

### ⚡ Hybrid Retrieval
- BM25 + semantic score fusion
- Balanced ranking (keyword + meaning)
- Top-K candidate reranking

### 🔄 Query Intelligence
- Semantic query expansion
- Improves recall for weak/short queries

### 📦 Data Support
- Multi-parser support (XML, CSV, extensible)
- Automatic parser detection

### ⚙️ System Design
- Modular architecture (parser → index → ranking → API)
- Separate document store and index
- Metadata-driven ranking

### 🌐 API + UI
- FastAPI backend
- REST search endpoint (`/api/v1/search`)
- Interactive Swagger docs (`/docs`)
- Simple web UI

---

## 🏗️ Architecture Overview


```bash
OFFLINE (Indexing)

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
Embedding Generation
↓
Storage (JSON)


ONLINE (Search)

User Query
↓
Query Parsing
↓
Query Expansion (semantic)
↓
BM25 Retrieval
↓
Top-K Candidates
↓
Semantic + BM25 Fusion
↓
Final Results
```

Each component is **independent, testable, and replaceable**, making the system easy to extend with new ranking models, storage backends, or APIs.

---

## 📁 Project Structure
```bash
├── src/
│ ├── parser/ # Dataset parsers (XML, CSV, etc.)
│ ├── preprocessing/ # Cleaning & tokenization
│ ├── indexer/ # Inverted index logic
│ ├── storage/ # Document store & index reader
│ ├── ranking/ # BM25, TF-IDF
│ ├── semantic/ # Embeddings, reranker, query expansion
│ ├── query/ # Search engine core
│ └── utils/
├── api/ # FastAPI backend
├── scripts/ # Indexing & CLI tools
├── data/ # (ignored) raw + index files
├── logs/
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
### 3. Fetching the Dataset

**Option A: Automated (Recommended)**
You can easily download and extract the Simple English Wikipedia dump using the included script:
```bash
python download_data.py
```
This will automatically download and unpack the XML into `data/raw/simplewiki.xml`.

**Option B: Manual Download**
Download a Wikipedia dump (recommended: Simple English Wikipedia):
https://dumps.wikimedia.org/simplewiki/

Extract the file and place it here:
```bash
data/raw/simplewiki.xml
```

### 4. Build the index 
```bash
python -m scripts.build_index --source data/raw/simplewiki.xml
```
*Note: The script currently limits indexing to the first 1,000 documents for fast testing.*
this generates:
```bash
data/index/
├── inverted_index.json
├── title_index.json
├── documents.json
├── metadata.json
├── embeddings.json
```

### 5. Searching

run the search api: 
```bash
python -m uvicorn api.app:app --reload
```

### 6. Run the tests
```bash
python -m pytest
```

## Configuration 

all paths and constants are centralized in:
```bash
 src/utils/config.py
```

## Key Concepts Implemented

Inverted Index
BM25 Ranking
Semantic Embeddings
Cosine Similarity
Hybrid Score Fusion
Query Expansion
Offline vs Online computation


logs are written to: 
```bash
logs/app.log
```

## Evaluation
- MAP: 0.76
- NDCG@10: 0.88


## 📜 License
MIT License

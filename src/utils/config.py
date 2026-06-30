from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Index files
INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.json"
DOCUMENT_STORE_PATH = INDEX_DIR / "documents.json"
METADATA_PATH = INDEX_DIR / "metadata.json"
TITLE_INDEX_PATH = INDEX_DIR / "title_index.json"

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Wikipedia dump
SIMPLEWIKI_XML = RAW_DATA_DIR / "simplewiki.xml"

# 🔴 FAIL FAST if missing
if not SIMPLEWIKI_XML.exists():
    raise FileNotFoundError(
        f"simplewiki.xml not found at {SIMPLEWIKI_XML}\n"
        "Download from https://dumps.wikimedia.org/simplewiki/\n"
        "and rename it to simplewiki.xml"
    )

# ranker preference
RANKER = "bm25"  # options: "tfidf", "bm25"

#param for title weighting
TITLE_BOOST = 3.0

#embeddings path
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.json"


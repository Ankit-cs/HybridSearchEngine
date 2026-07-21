from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
CATALOG_DB_PATH = DATA_DIR / "catalog.db"
EPISODIC_MEMORY_DIR = DATA_DIR / "episodic_memory"
WORKING_MEMORY_DIR = DATA_DIR / "working_memory"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
EPISODIC_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
WORKING_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.json"
DOCUMENT_STORE_PATH = INDEX_DIR / "documents.parquet"
METADATA_PATH = INDEX_DIR / "metadata.json"
TITLE_INDEX_PATH = INDEX_DIR / "title_index.json"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

SIMPLEWIKI_XML = RAW_DATA_DIR / "simplewiki.xml"

RANKER = "bm25"
TITLE_BOOST = 3.0

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.index"
FAISS_ID_MAP_PATH = INDEX_DIR / "faiss_id_map.json"
CONTEXT_EMBEDDINGS_PATH = INDEX_DIR / "context_embeddings.index"
CONTEXT_ID_MAP_PATH = INDEX_DIR / "context_id_map.json"

QUANTIZED_INDEX_PATH = INDEX_DIR / "embeddings_quantized.index"
QUANTIZED_VECTORS_PATH = INDEX_DIR / "vectors_quantized.npy"
QUANTIZATION_CONFIG_PATH = INDEX_DIR / "quantization_config.json"

FTS_INDEX_DIR = INDEX_DIR / "fts"
FTS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRIC_STATS_PATH = INDEX_DIR / "geometric_stats.json"
VERSION_MANIFEST_PATH = INDEX_DIR / "version_manifest.json"

SEARCH_CONCURRENCY = 8
PRUNING_THRESHOLD = 0.95
COMPACTION_MIN_FILES = 4
COMPACTION_TARGET_SIZE_MB = 512
IDEMPOTENCY_TTL_HOURS = 72

MEMORY_DECAY_LAMBDA = 0.1
WORKING_MEMORY_MAX_CHUNKS = 500
EPISODIC_MEMORY_MAX_ENTRIES = 10000

CLOUD_STORAGE_BACKEND = "local"
CLOUD_BUCKET = ""
CLOUD_PREFIX = ""

LTR_MODEL_PATH = PROJECT_ROOT / "models" / "ltr_model.pkl"

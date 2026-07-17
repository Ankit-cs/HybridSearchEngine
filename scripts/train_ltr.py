"""
Learning-to-Rank Training Script
Trains a LightGBM Ranker (LambdaMART) on the indexed India Wikipedia dataset.

How it works:
  1. Loads the existing BM25 index + embeddings + doc store.
  2. Generates synthetic training data: for each query, top BM25 results are
     used as candidates. The best result (rank 1) is labeled "relevant" (label=1),
     the rest "non-relevant" (label=0). This simulates a click-through signal.
  3. Extracts the 6-dimensional feature vector for each (query, doc) pair.
  4. Trains a LightGBM Ranker using the LambdaMART objective.
  5. Saves the trained model to models/ltr_model.pkl.

Usage:
  python -m scripts.train_ltr
"""
import json
import os
import pickle
import numpy as np
import lightgbm as lgb

from src.query.query_parser import parse_query
from src.storage.index_reader import IndexReader
from src.storage.document_store import DocumentStore
from src.semantic.embedding_store import EmbeddingStore
from src.ranking.bm25 import BM25Ranker
from src.ranking.ltr_features import extract_features
from src.utils.config import (
    METADATA_PATH,
    TITLE_INDEX_PATH,
    EMBEDDINGS_PATH,
)

# ─── Config ────────────────────────────────────────────────────────────────────
INDEX_PATH      = "data/processed/inverted_index.json"
DOC_STORE_PATH  = "data/processed/documents.json"
MODEL_OUTPUT    = "models/ltr_model.pkl"

# Sample of diverse India-specific training queries
TRAINING_QUERIES = [
    "Prime Minister of India",
    "Indian independence movement",
    "Taj Mahal history",
    "Mumbai population",
    "Mahatma Gandhi non-violence",
    "Indian Space Research Organisation",
    "Himalaya mountains geography",
    "Reserve Bank of India",
    "Indus Valley Civilisation",
    "Battle of Panipat",
    "Mughal Empire Akbar",
    "Indian Constitution fundamental rights",
    "Bollywood film industry",
    "Indian National Congress formation",
    "Jawaharlal Nehru first prime minister",
    "Yoga origins India",
    "Karnataka Karnataka state capital",
    "Ganga river sacred",
    "Rajasthan desert culture",
    "Silicon Valley of India Bengaluru",
]

CANDIDATES_PER_QUERY = 20  # Number of BM25 candidates per query
# ───────────────────────────────────────────────────────────────────────────────


def main():
    print("[LTR Training] Loading indexes...")

    index_reader       = IndexReader(INDEX_PATH)
    title_index_reader = IndexReader(TITLE_INDEX_PATH)
    doc_store          = DocumentStore()
    doc_store.load(DOC_STORE_PATH)
    embedding_store    = EmbeddingStore()
    embedding_store.load(EMBEDDINGS_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    ranker = BM25Ranker(
        body_index=index_reader,
        title_index=title_index_reader,
        metadata=metadata,
    )

    avg_doc_length = metadata.get("avg_doc_length", 400)

    X_all    = []
    y_all    = []
    groups   = []  # Required by LightGBM LambdaMART: number of docs per query

    print(f"[LTR Training] Building features for {len(TRAINING_QUERIES)} queries...")

    for query in TRAINING_QUERIES:
        tokens = parse_query(query)
        if not tokens:
            continue

        scores = ranker.score(tokens)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = ranked[:CANDIDATES_PER_QUERY]

        if not candidates:
            continue

        max_bm25 = max(s for _, s in candidates) or 1.0
        group_size = 0

        for rank_idx, (doc_id, bm25_score) in enumerate(candidates):
            features = extract_features(
                query=query,
                query_tokens=tokens,
                doc_id=doc_id,
                bm25_score=bm25_score,
                max_bm25=max_bm25,
                embedding_store=embedding_store,
                doc_store=doc_store,
                avg_doc_length=avg_doc_length,
            )
            X_all.append(features)

            # Relevance label: 2 for top-1, 1 for top-5, 0 for the rest
            if rank_idx == 0:
                label = 2
            elif rank_idx < 5:
                label = 1
            else:
                label = 0

            y_all.append(label)
            group_size += 1

        groups.append(group_size)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    print(f"[LTR Training] Total training samples: {len(X)}")
    print(f"[LTR Training] Training LightGBM LambdaMART model...")

    train_dataset = lgb.Dataset(X, label=y, group=groups)

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5, 10],
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 5,
        "n_estimators":     200,
        "verbose":          -1,
    }

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=200,
    )

    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(model, f)

    print(f"[LTR Training] ✅ Model saved to {MODEL_OUTPUT}")
    print("[LTR Training] Run your search server now — LTR will be automatically enabled!")


if __name__ == "__main__":
    main()

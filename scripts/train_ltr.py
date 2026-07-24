"""
Learning-to-Rank Training Script
Trains a LightGBM Ranker (LambdaMART) on the indexed India Wikipedia dataset.

How it works:
  1. Loads the existing BM25 index + embeddings + doc store.
  2. Splits TRAINING_QUERIES into Train (80%) and Validation (20%) sets.
  3. Generates synthetic training data: for each query, top BM25 results are
     used as candidates. The best result (rank 1) is labeled "relevant" (label=2),
     ranks 2-4 (label=1), the rest (label=0).
  4. Extracts the 50-dimensional feature vector for each (query, doc) pair.
  5. Uses Optuna to find the optimal hyperparameters via the Validation set.
  6. Trains the final LightGBM Ranker using the LambdaMART objective.
  7. Saves the trained model to models/ltr_model.pkl.

Usage:
  python -m scripts.train_ltr
"""
import json
import os
import pickle
import random
import numpy as np
import lightgbm as lgb
import optuna

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


def build_dataset(queries, ranker, index_reader, doc_store, embedding_store, avg_doc_length, total_docs):
    X_all, y_all, groups = [], [], []

    for query in queries:
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
                index_reader=index_reader,
                total_docs=total_docs
            )
            X_all.append(features)

            if rank_idx == 0:
                label = 2
            elif rank_idx < 5:
                label = 1
            else:
                label = 0

            y_all.append(label)
            group_size += 1

        groups.append(group_size)

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int32), groups


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
    total_docs = metadata.get("total_docs", 0)

    # 1. Train / Validation Split
    print("[LTR Training] Splitting training queries...")
    random.shuffle(TRAINING_QUERIES)
    split_idx = int(len(TRAINING_QUERIES) * 0.8)
    train_queries = TRAINING_QUERIES[:split_idx]
    val_queries = TRAINING_QUERIES[split_idx:]

    print(f"[LTR Training] Building Train features ({len(train_queries)} queries)...")
    X_train, y_train, g_train = build_dataset(
        train_queries, ranker, index_reader, doc_store, embedding_store, avg_doc_length, total_docs
    )

    print(f"[LTR Training] Building Validation features ({len(val_queries)} queries)...")
    X_val, y_val, g_val = build_dataset(
        val_queries, ranker, index_reader, doc_store, embedding_store, avg_doc_length, total_docs
    )
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("[LTR Training] Not enough data to train/validate. Aborting.")
        return

    train_dataset = lgb.Dataset(X_train, label=y_train, group=g_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_dataset)

    # 2. Optuna Objective
    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "verbose": -1,
        }

        # Train with early stopping
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # We maximize the best validation ndcg@10 achieved
        return model.best_score["valid_0"]["ndcg@10"]

    print("[LTR Training] Running Optuna Hyperparameter Optimization...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # 10 trials for brevity

    best_params = study.best_params
    print(f"[LTR Training] Optuna Best NDCG@10: {study.best_value:.4f}")
    print(f"[LTR Training] Optuna Best Params: {best_params}")

    # 3. Train Final Model
    print("[LTR Training] Training final model with optimal parameters...")
    final_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "verbose": -1,
    }
    final_params.update(best_params)
    
    # We can train on the whole dataset or just train using early stopping on val
    final_model = lgb.train(
        final_params,
        train_dataset,
        valid_sets=[val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=20)
        ]
    )

    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(final_model, f)

    print(f"[LTR Training] ✅ Model saved to {MODEL_OUTPUT}")
    print("[LTR Training] Run your search server now — the upgraded LTR is automatically enabled!")


if __name__ == "__main__":
    main()

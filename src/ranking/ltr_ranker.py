"""
LightGBM Learning-to-Rank Ranker
Loads a pre-trained LightGBM model and re-ranks retrieved candidates
using rich feature vectors extracted per (query, document) pair.

Usage (inference only - training is done via scripts/train_ltr.py):
    ranker = LightGBMRanker(embedding_store, doc_store, metadata)
    results = ranker.rerank(query, query_tokens, candidates_with_bm25_scores)
"""
import os
import pickle
import numpy as np

from src.ranking.ltr_features import extract_features
from src.ranking.fixed_point import sort_by_fixed_point

# Default model path (set via config or fallback to this path)
DEFAULT_MODEL_PATH = "models/ltr_model.pkl"


class LightGBMRanker:
    def __init__(self, embedding_store, doc_store, metadata, index_reader=None, model_path=None):
        self.embedding_store = embedding_store
        self.doc_store = doc_store
        self.index_reader = index_reader
        self.total_docs = metadata.get("total_docs", 0)
        self.avg_doc_length = metadata.get("avg_doc_length", 400)

        # Load LightGBM model if it exists
        path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            print(f"[LTR] Loaded LightGBM ranker from {path}")
        else:
            print(f"[LTR] Warning: No model found at {path}. "
                  "Run scripts/train_ltr.py to train. "
                  "Falling back to RRF scoring.")

    def is_ready(self) -> bool:
        return self.model is not None

    def rerank(self, query, query_tokens, candidates):
        """
        candidates: [(doc_id, bm25_score), ...]  (pre-sorted by BM25)
        Returns:    [(doc_id, ltr_score), ...]   (sorted by predicted relevance)
        """
        if not candidates:
            return []

        max_bm25 = max(score for _, score in candidates) or 1.0

        # Build feature matrix
        feature_matrix = []
        doc_ids = []
        for doc_id, bm25_score in candidates:
            features = extract_features(
                query=query,
                query_tokens=query_tokens,
                doc_id=doc_id,
                bm25_score=bm25_score,
                max_bm25=max_bm25,
                embedding_store=self.embedding_store,
                doc_store=self.doc_store,
                avg_doc_length=self.avg_doc_length,
                index_reader=self.index_reader,
                total_docs=self.total_docs
            )
            feature_matrix.append(features)
            doc_ids.append(doc_id)

        X = np.array(feature_matrix, dtype=np.float32)

        if not self.is_ready():
            # Graceful fallback: RRF-style on norm_bm25 + semantic_score
            fallback_scores = (X[:, 0] + X[:, 1]).tolist()
            ranked = sort_by_fixed_point(list(zip(doc_ids, fallback_scores)))
            return ranked

        # LightGBM predict (returns relevance scores)
        scores = self.model.predict(X)
        ranked = sort_by_fixed_point(list(zip(doc_ids, scores.tolist())))
        return ranked

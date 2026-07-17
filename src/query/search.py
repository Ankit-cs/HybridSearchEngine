import json

from src.semantic.embedding_store import EmbeddingStore
from src.semantic.reranker import SemanticReranker

from src.query.query_parser import parse_query
from src.storage.index_reader import IndexReader
from src.storage.document_store import DocumentStore

from src.ranking.tfidf import TFIDFRanker
from src.ranking.bm25 import BM25Ranker
from src.ranking.ltr_ranker import LightGBMRanker

from src.utils.config import (
    METADATA_PATH,
    RANKER,
    TITLE_INDEX_PATH,
    EMBEDDINGS_PATH,
)

from src.semantic.query_expander import QueryExpander


class SearchEngine:
    def __init__(self, index_path, doc_store_path, total_docs=None):

        # ---- load indexes ----
        self.index_reader = IndexReader(index_path)
        self.title_index_reader = IndexReader(TITLE_INDEX_PATH)

        # ---- load document store ----
        self.doc_store = DocumentStore()
        self.doc_store.load(doc_store_path)

        # ---- load embeddings (semantic memory) ----
        self.embedding_store = EmbeddingStore()
        self.embedding_store.load(EMBEDDINGS_PATH)

        # ---- enable semantic reranking (RRF) ----
        self.semantic_enabled = True
        self.reranker = (
            SemanticReranker(self.embedding_store)
            if self.semantic_enabled
            else None
        )

        # ---- load metadata ----
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        # ---- choose ranking algorithm ----
        if RANKER == "bm25":
            self.ranker = BM25Ranker(
                body_index=self.index_reader,
                title_index=self.title_index_reader,
                metadata=metadata,
            )
        else:
            self.ranker = TFIDFRanker(
                self.index_reader,
                metadata["total_docs"]
            )

        # ---- Learning-to-Rank (LightGBM) ----
        # Automatically enabled when models/ltr_model.pkl exists.
        # Falls back to RRF gracefully if model is missing.
        self.ltr_ranker = LightGBMRanker(
            embedding_store=self.embedding_store,
            doc_store=self.doc_store,
            metadata=metadata,
        )

        # ---- query expander ----
        self.query_expander = QueryExpander(
            self.embedding_store,
            self.doc_store
        )

    # ======================================================
    # SEARCH
    # ======================================================

    def search(self, query, top_k=10):

        tokens = parse_query(query)

        # --- Tier 1: Initial BM25 pass ---
        scores = self.ranker.score(tokens)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # --- Tier 2: Semantic query expansion ---
        expanded_query = self.query_expander.expand(query, ranked[:20])
        tokens = parse_query(expanded_query)
        scores = self.ranker.score(tokens)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # --- Tier 3: RRF Semantic Fusion (top 50 candidates) ---
        if self.semantic_enabled and self.reranker:
            candidates = ranked[:50]
            try:
                ranked = self.reranker.rerank(query, candidates)
            except Exception as e:
                print("[RRF] Semantic reranking failed:", e)

        # --- Tier 4: Learning-to-Rank (LightGBM) final re-rank ---
        # Only re-ranks the top 20 post-RRF candidates for speed.
        # Falls back gracefully to RRF order if model is not trained yet.
        try:
            rrf_top = ranked[:20]
            # Pass original bm25 scores back by re-scoring the top candidates
            bm25_scores = self.ranker.score(tokens)
            rrf_with_bm25 = [
                (doc_id, bm25_scores.get(doc_id, 0.0))
                for doc_id, _ in rrf_top
            ]
            ranked = self.ltr_ranker.rerank(query, tokens, rrf_with_bm25)
        except Exception as e:
            print("[LTR] LightGBM re-rank failed, keeping RRF order:", e)

        # ---- Return final results ----
        return ranked[:top_k]

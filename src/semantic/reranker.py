from sentence_transformers import util
from src.semantic.embedding_model import EmbeddingModel


class SemanticReranker:

    def __init__(self, embedding_store, k=60):
        """
        k -> RRF constant (default: 60)
        """
        self.embedding_store = embedding_store
        self.k = k
        self.model = EmbeddingModel()

    def rerank(self, query, documents):
        """
        documents: [(doc_id, bm25_score)]
        returns: [(doc_id, final_score)] using Reciprocal Rank Fusion (RRF)
        """
        if not documents:
            return []

        # ---- Rank by BM25 (assume input is sorted or sort it to be safe) ----
        sorted_by_bm25 = sorted(documents, key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_by_bm25)}

        # ---- embed query once ----
        query_emb = self.model.encode(query)[0]

        # ---- Calculate Semantic Scores ----
        semantic_scores = []
        for doc_id, _ in sorted_by_bm25:
            doc_emb = self.embedding_store.get(doc_id)
            if doc_emb is None:
                semantic_score = -1.0
            else:
                semantic_score = float(util.cos_sim(query_emb, doc_emb)[0][0])
            semantic_scores.append((doc_id, semantic_score))

        # Sort by Semantic Score descending
        sorted_by_semantic = sorted(semantic_scores, key=lambda x: x[1], reverse=True)
        semantic_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_by_semantic)}

        # ---- Reciprocal Rank Fusion (RRF) ----
        fused_results = []
        for doc_id, _ in sorted_by_bm25:
            rank_bm25 = bm25_ranks[doc_id]
            rank_semantic = semantic_ranks[doc_id]

            # RRF Formula
            rrf_score = (1.0 / (self.k + rank_bm25)) + (1.0 / (self.k + rank_semantic))
            fused_results.append((doc_id, rrf_score))

        # Sort final results by RRF score descending
        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results


from sentence_transformers import CrossEncoder

class AgenticCrossEncoder:
    def __init__(self, doc_store):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        self.doc_store = doc_store

    def rerank(self, query, top_k_results):
        """
        Re-rank using a Cross-Encoder for maximum accuracy (slower but smarter).
        """
        if not top_k_results:
            return []

        pairs = []
        doc_ids = []
        for doc_id, _ in top_k_results:
            doc = self.doc_store.get(doc_id)
            if doc and doc.get("text"):
                pairs.append((query, doc["text"][:1000]))
                doc_ids.append(doc_id)
        
        if not pairs:
            return top_k_results

        scores = self.model.predict(pairs)
        
        reranked = [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked

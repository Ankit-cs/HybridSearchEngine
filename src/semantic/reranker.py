from sentence_transformers import util
from src.semantic.embedding_model import EmbeddingModel


class SemanticReranker:

    def __init__(self, embedding_store, alpha=0.6, beta=0.4):
        """
        alpha → BM25 importance
        beta  → semantic importance
        """
        self.embedding_store = embedding_store
        self.alpha = alpha
        self.beta = beta
        self.model = EmbeddingModel()

    def rerank(self, query, documents):
        """
        documents: [(doc_id, bm25_score)]
        returns: [(doc_id, final_score)]
        """

        if not documents:
            return []

        # ---- embed query once ----
        query_emb = self.model.encode(query)[0]

        # ---- normalize BM25 scores ----
        bm25_scores = [score for _, score in documents]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0

        fused_results = []

        for doc_id, bm25_score in documents:

            doc_emb = self.embedding_store.get(doc_id)
            if doc_emb is None:
                continue

            # semantic similarity
            semantic_score = util.cos_sim(query_emb, doc_emb)[0][0]
            semantic_score = float(semantic_score)

            # normalize bm25 → 0..1
            norm_bm25 = bm25_score / max_bm25

            # ⭐ fusion
            final_score = (
                self.alpha * norm_bm25 +
                self.beta * semantic_score
            )

            fused_results.append((doc_id, final_score))

        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results

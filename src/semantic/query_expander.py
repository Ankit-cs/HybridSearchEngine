from sentence_transformers import util
from src.semantic.embedding_model import EmbeddingModel


class QueryExpander:
    """
    Expands query using semantic similarity with document embeddings.
    """

    def __init__(self, embedding_store, doc_store, expansion_terms=3):
        self.embedding_store = embedding_store
        self.doc_store = doc_store
        self.model = EmbeddingModel()
        self.expansion_terms = expansion_terms

    def expand(self, query, candidates):
        """
        candidates: [(doc_id, bm25_score)]
        returns expanded query string
        """

        if not candidates:
            return query

        query_emb = self.model.encode(query)[0]

        similarities = []

        for doc_id, _ in candidates:
            doc_emb = self.embedding_store.get(doc_id)
            if doc_emb is None:
                continue

            sim = util.cos_sim(query_emb, doc_emb)[0][0]
            similarities.append((doc_id, float(sim)))

        # pick most semantically related docs
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_docs = similarities[:self.expansion_terms]

        extra_words = []

        for doc_id, _ in top_docs:
            doc = self.doc_store.get(doc_id)
            if not doc:
                continue

            title = doc.get("title", "")
            extra_words.extend(title.lower().split())

        expanded_query = query + " " + " ".join(extra_words)

        return expanded_query

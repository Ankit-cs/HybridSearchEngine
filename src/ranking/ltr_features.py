"""
Learning-to-Rank Feature Extractor
Extracts a feature vector for each (query, document) pair.
Features: bm25_score, semantic_score, title_match, doc_length_norm, query_term_coverage
"""
from sentence_transformers import util
from src.semantic.embedding_model import EmbeddingModel

# Shared embedding model (loaded once)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model


def extract_features(query, query_tokens, doc_id, bm25_score, max_bm25,
                     embedding_store, doc_store, avg_doc_length):
    """
    Returns a feature vector (list of floats) for a single (query, doc) pair.

    Feature layout (6 features):
      0: norm_bm25           - BM25 score normalized to [0, 1]
      1: semantic_score      - Cosine similarity between query and doc embeddings
      2: title_query_overlap - Fraction of query tokens present in the doc title
      3: body_query_overlap  - Fraction of query tokens present in the doc body snippet
      4: doc_length_norm     - Document length normalized by corpus avg length
      5: is_title_match      - Binary: 1 if any query token appears in title
    """
    doc = doc_store.get(doc_id) or {}
    title = doc.get("title", "").lower()
    text  = doc.get("text",  "").lower()
    doc_len = len(text.split()) if text else 0

    # Feature 0: Normalized BM25
    norm_bm25 = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0

    # Feature 1: Semantic cosine similarity
    semantic_score = 0.0
    doc_emb = embedding_store.get(doc_id)
    if doc_emb is not None:
        query_emb = _get_model().encode(query)[0]
        semantic_score = float(util.cos_sim(query_emb, doc_emb)[0][0])

    # Feature 2: Title query term coverage
    if query_tokens and title:
        title_hits = sum(1 for t in query_tokens if t in title)
        title_overlap = title_hits / len(query_tokens)
    else:
        title_overlap = 0.0

    # Feature 3: Body query term coverage
    if query_tokens and text:
        body_hits = sum(1 for t in query_tokens if t in text[:2000])
        body_overlap = body_hits / len(query_tokens)
    else:
        body_overlap = 0.0

    # Feature 4: Normalized doc length
    length_norm = (doc_len / avg_doc_length) if avg_doc_length > 0 else 1.0

    # Feature 5: Binary title match
    is_title_match = 1.0 if any(t in title for t in query_tokens) else 0.0

    return [
        norm_bm25,
        semantic_score,
        title_overlap,
        body_overlap,
        length_norm,
        is_title_match,
    ]

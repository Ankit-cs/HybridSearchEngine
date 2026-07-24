"""
Learning-to-Rank Feature Extractor
Extracts a 50-dimensional feature vector for each (query, document) pair.
Features include positional, proximity, coverage, and lexical (TF/IDF) aggregations.
"""
import math
import numpy as np
from collections import Counter
from sentence_transformers import util
from src.semantic.embedding_model import EmbeddingModel

# Shared embedding model (loaded once)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model

def compute_idf(term, index_reader, total_docs):
    if not index_reader or not total_docs:
        return 1.0
    postings = index_reader.get_postings(term)
    df = len(postings) if postings else 0
    return math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

def extract_features(query, query_tokens, doc_id, bm25_score, max_bm25,
                     embedding_store, doc_store, avg_doc_length,
                     index_reader=None, total_docs=0):
    
    doc = doc_store.get(doc_id) or {}
    text = doc.get("text", "").lower()
    doc_words = text.split()
    doc_len = len(doc_words)
    q_len = len(query_tokens)
    
    # 1. Base scores
    norm_bm25 = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
    semantic_score = 0.0
    doc_emb = embedding_store.get(doc_id)
    if doc_emb is not None:
        query_emb = _get_model().encode(query)[0]
        semantic_score = float(util.cos_sim(query_emb, doc_emb)[0][0])
        
    # 2. Coverage
    d_counter = Counter(doc_words)
    matched_terms = [w for w in query_tokens if w in d_counter]
    coverage_count = len(set(matched_terms))
    coverage_ratio = coverage_count / (q_len + 1e-6)
    
    # 3. TF Aggregations
    tf_list = [d_counter.get(w, 0) for w in query_tokens]
    sum_tf = sum(tf_list)
    min_tf = min(tf_list) if tf_list else 0
    max_tf = max(tf_list) if tf_list else 0
    mean_tf = np.mean(tf_list) if tf_list else 0
    var_tf = np.var(tf_list) if tf_list else 0
    
    # 4. IDF Aggregations
    idf_list = [compute_idf(w, index_reader, total_docs) for w in query_tokens]
    sum_idf = sum(idf_list)
    min_idf = min(idf_list) if idf_list else 0
    max_idf = max(idf_list) if idf_list else 0
    mean_idf = np.mean(idf_list) if idf_list else 0
    
    query_specificity = sum_idf / (q_len + 1e-6)
    
    # 5. TF-IDF Aggregations
    tfidf_list = [tf * idf for tf, idf in zip(tf_list, idf_list)]
    sum_tfidf = sum(tfidf_list)
    min_tfidf = min(tfidf_list) if tfidf_list else 0
    max_tfidf = max(tfidf_list) if tfidf_list else 0
    mean_tfidf = np.mean(tfidf_list) if tfidf_list else 0
    var_tfidf = np.var(tfidf_list) if tfidf_list else 0
    
    # 6. Positional
    exact_phrase = 1 if query.lower() in text else 0
    
    window_match = 0
    if len(query_tokens) >= 2:
        for i in range(len(doc_words) - 1):
            if doc_words[i] == query_tokens[0] and doc_words[i+1] == query_tokens[1]:
                window_match = 1
                break
                
    bigrams = [" ".join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)]
    bigram_match = sum(1 for bg in bigrams if bg in text)
    
    positions = [doc_words.index(w) for w in query_tokens if w in doc_words]
    first_pos = min(positions) / (doc_len + 1e-6) if positions else 1
    last_pos = max(positions) / (doc_len + 1e-6) if positions else 1
    
    density = 0
    if len(positions) > 1:
        density = 1 / (max(positions) - min(positions) + 1)
        
    bool_match = 1 if coverage_count > 0 else 0
    
    # Final feature vector (padded to 50 dimensions)
    features = [
        q_len, doc_len,
        coverage_count, coverage_ratio,
        sum_tf, min_tf, max_tf, mean_tf, var_tf,
        sum_idf, min_idf, max_idf, mean_idf,
        query_specificity,
        sum_tfidf, min_tfidf, max_tfidf, mean_tfidf, var_tfidf,
        norm_bm25, semantic_score,
        exact_phrase, window_match, bigram_match,
        density, bool_match, first_pos, last_pos
    ]
    
    # Pad to 50
    while len(features) < 50:
        features.append(0.0)
        
    return [float(f) for f in features]

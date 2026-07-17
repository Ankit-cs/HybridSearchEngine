import math


def precision_at_k(results, relevant_docs, k):
    hits = 0
    for i, (doc_id, _, _) in enumerate(results[:k]):
        if str(doc_id) in relevant_docs:
            hits += 1
    return hits / k


def average_precision(results, relevant_docs):
    score = 0.0
    hits = 0

    for i, (doc_id, _, _) in enumerate(results, start=1):
        if str(doc_id) in relevant_docs:
            hits += 1
            score += hits / i

    return score / max(len(relevant_docs), 1)


def ndcg_at_k(results, relevant_docs, k):
    def dcg():
        s = 0
        for i, (doc_id, _, _) in enumerate(results[:k], start=1):
            rel = relevant_docs.get(str(doc_id), 0)
            s += (2 ** rel - 1) / math.log2(i + 1)
        return s

    def idcg():
        rels = sorted(relevant_docs.values(), reverse=True)
        s = 0
        for i, rel in enumerate(rels[:k], start=1):
            s += (2 ** rel - 1) / math.log2(i + 1)
        return s

    return dcg() / max(idcg(), 1)

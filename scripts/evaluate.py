import time
import math
from src.query.search import SearchEngine
from src.utils.config import INVERTED_INDEX_PATH, DOCUMENT_STORE_PATH, METADATA_PATH
import json

# Simulated Evaluation Ground Truth Dataset
# Format: { "query": ["relevant_doc_id_1", "relevant_doc_id_2"] }
GROUND_TRUTH = {
    "Alan Turing": ["8", "250", "1902"],
    "April month": ["1", "2", "5"],
    "Adobe Illustrator": ["10", "40", "80"],
    "Spain autonomous communities": ["7", "90", "120"]
}

def compute_dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def evaluate_engine():
    print("Loading Search Engine for Evaluation...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    engine = SearchEngine(
        INVERTED_INDEX_PATH,
        DOCUMENT_STORE_PATH,
        total_docs=metadata["total_docs"]
    )
    print(f"\nEngine Loaded. Total Indexed Docs: {metadata['total_docs']}")
    print("="*50)

    total_queries = len(GROUND_TRUTH)
    total_ap = 0.0
    total_ndcg = 0.0
    avg_latency = 0.0

    for query, true_docs in GROUND_TRUTH.items():
        start = time.time()
        results = engine.search(query, top_k=10)
        latency = time.time() - start
        avg_latency += latency

        retrieved_ids = [str(r[0]) for r in results]
        
        # Compute Average Precision (AP)
        hits = 0
        sum_precisions = 0
        for idx, doc_id in enumerate(retrieved_ids):
            if doc_id in true_docs:
                hits += 1
                sum_precisions += hits / (idx + 1)
        
        ap = sum_precisions / len(true_docs) if true_docs else 0
        total_ap += ap

        # Compute NDCG@10
        relevances = [1 if doc_id in true_docs else 0 for doc_id in retrieved_ids]
        ideal_relevances = sorted([1]*len(true_docs) + [0]*(10 - len(true_docs)), reverse=True)[:10]
        
        dcg = compute_dcg(relevances)
        idcg = compute_dcg(ideal_relevances)
        ndcg = (dcg / idcg) if idcg > 0 else 0
        total_ndcg += ndcg

        print(f"Query: '{query}'")
        print(f"  Latency: {latency*1000:.2f}ms")
        print(f"  AP: {ap:.4f} | NDCG@10: {ndcg:.4f}")
        print("-" * 50)

    map_score = total_ap / total_queries
    mean_ndcg = total_ndcg / total_queries
    mean_latency = (avg_latency / total_queries) * 1000

    print("\n🔥 FINAL EVALUATION METRICS 🔥")
    print("="*50)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"NDCG@10:                      {mean_ndcg:.4f}")
    print(f"Average Latency per query:    {mean_latency:.2f}ms")
    print("="*50)

if __name__ == "__main__":
    evaluate_engine()

import json
from src.query.search import SearchEngine
from src.evaluation.metrics import precision_at_k, average_precision, ndcg_at_k
from src.utils.config import INVERTED_INDEX_PATH, DOCUMENT_STORE_PATH


def main():
    engine = SearchEngine(INVERTED_INDEX_PATH, DOCUMENT_STORE_PATH)

    with open("data/eval/qrels.json") as f:
        qrels = json.load(f)

    map_scores = []
    ndcg_scores = []

    for query, relevant_docs in qrels.items():
        results = engine.search(query, top_k=10)

        ap = average_precision(results, relevant_docs)
        ndcg = ndcg_at_k(results, relevant_docs, 10)

        map_scores.append(ap)
        ndcg_scores.append(ndcg)

        print(f"{query:20} | AP={ap:.3f} | NDCG@10={ndcg:.3f}")

    print("\nFINAL RESULTS")
    print(f"MAP   : {sum(map_scores)/len(map_scores):.3f}")
    print(f"NDCG@10: {sum(ndcg_scores)/len(ndcg_scores):.3f}")


if __name__ == "__main__":
    main()

from src.semantic.reranker import SemanticReranker

reranker = SemanticReranker()

docs = [
    (1, "machine learning is about learning from data", 1.0),
    (2, "football is a popular sport", 1.0),
]

results = reranker.rerank("learning from experience", docs)

print(results)

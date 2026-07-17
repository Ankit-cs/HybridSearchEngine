from src.semantic.embedding_model import EmbeddingModel

model = EmbeddingModel()

emb = model.encode("machine learning")

print(len(emb[0]))

from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model ready.")

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)

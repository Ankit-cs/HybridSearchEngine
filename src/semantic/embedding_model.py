from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if self._initialized:
            return
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model ready.")
        self._initialized = True

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)

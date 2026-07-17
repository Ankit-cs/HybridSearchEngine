import numpy as np
import json
import torch

class EmbeddingStore:

    def __init__(self):
        self.embeddings = {}

    def add(self, doc_id, vector):
        self.embeddings[doc_id] = vector.tolist()

    def get(self, doc_id):
        vec = self.embeddings.get(doc_id)
        if vec is None:
            return None
        return torch.tensor(vec, dtype = torch.float32)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.embeddings, f)

    def load(self, path):
        with open(path, "r") as f:
            self.embeddings = json.load(f)

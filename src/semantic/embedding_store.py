import faiss
import numpy as np
import json
import torch
from src.utils.config import FAISS_ID_MAP_PATH

class EmbeddingStore:

    def __init__(self):
        self.index = None
        self.id_to_doc = {}
        self.doc_to_id = {}
        self.next_id = 0

    def init_index(self, dimension=384):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

    def add(self, doc_id, vector):
        self.init_index(vector.shape[0])
        vec_np = np.array([vector], dtype=np.float32)
        faiss_id = self.next_id
        self.next_id += 1
        self.id_to_doc[str(faiss_id)] = str(doc_id)
        self.doc_to_id[str(doc_id)] = faiss_id
        self.index.add(vec_np)

    def get(self, doc_id):
        if str(doc_id) not in self.doc_to_id:
            return None
        faiss_id = self.doc_to_id[str(doc_id)]
        try:
            vec = self.index.reconstruct(faiss_id)
            return torch.tensor(vec, dtype=torch.float32)
        except Exception:
            return None

    def save(self, path):
        faiss.write_index(self.index, str(path))
        with open(FAISS_ID_MAP_PATH, "w") as f:
            json.dump({
                "id_to_doc": self.id_to_doc,
                "doc_to_id": self.doc_to_id,
                "next_id": self.next_id
            }, f)

    def load(self, path):
        self.index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        with open(FAISS_ID_MAP_PATH, "r") as f:
            data = json.load(f)
            self.id_to_doc = data["id_to_doc"]
            self.doc_to_id = data["doc_to_id"]
            self.next_id = data["next_id"]

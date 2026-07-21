import numpy as np
import json
import faiss
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.semantic.embedding_model import EmbeddingModel


@dataclass
class DualEmbeddingResult:
    content_embedding: np.ndarray
    context_embedding: np.ndarray


class DualEmbeddingGenerator:
    def __init__(self, model: Optional[EmbeddingModel] = None):
        self.model = model or EmbeddingModel()

    def generate(self, text: str, title: str = "", section: str = "",
                 summary: str = "") -> DualEmbeddingResult:
        content_emb = self.model.encode(text[:512])[0]
        context_parts = []
        if title:
            context_parts.append(title)
        if section:
            context_parts.append(section)
        if summary:
            context_parts.append(summary)
        context_parts.append(text[:300])
        context_text = " ".join(context_parts)
        context_emb = self.model.encode(context_text)[0]
        return DualEmbeddingResult(
            content_embedding=content_emb,
            context_embedding=context_emb,
        )


class DualEmbeddingStore:
    def __init__(self, content_dim: int = 384, context_dim: int = 384):
        self.content_index = None
        self.context_index = None
        self.content_id_to_doc = {}
        self.context_id_to_doc = {}
        self.doc_to_content_id = {}
        self.doc_to_context_id = {}
        self.content_next_id = 0
        self.context_next_id = 0
        self.content_dim = content_dim
        self.context_dim = context_dim

    def init_indexes(self, content_dim: int = None, context_dim: int = None):
        if content_dim:
            self.content_dim = content_dim
        if context_dim:
            self.context_dim = context_dim
        if self.content_index is None:
            self.content_index = faiss.IndexFlatIP(self.content_dim)
        if self.context_index is None:
            self.context_index = faiss.IndexFlatIP(self.context_dim)

    def add(self, doc_id: str, content_emb: np.ndarray, context_emb: np.ndarray):
        self.init_indexes(content_dim=content_emb.shape[0], context_dim=context_emb.shape[0])
        c_id = self.content_next_id
        self.content_next_id += 1
        self.content_id_to_doc[str(c_id)] = str(doc_id)
        self.doc_to_content_id[str(doc_id)] = c_id
        self.content_index.add(np.array([content_emb], dtype=np.float32))

        ct_id = self.context_next_id
        self.context_next_id += 1
        self.context_id_to_doc[str(ct_id)] = str(doc_id)
        self.doc_to_context_id[str(doc_id)] = ct_id
        self.context_index.add(np.array([context_emb], dtype=np.float32))

    def search_content(self, query_emb: np.ndarray, top_k: int = 10) -> list:
        if self.content_index is None or self.content_index.ntotal == 0:
            return []
        k = min(top_k, self.content_index.ntotal)
        query = query_emb.reshape(1, -1).astype(np.float32)
        distances, indices = self.content_index.search(query, k)
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            doc_id = self.content_id_to_doc.get(str(idx), "")
            results.append({"doc_id": doc_id, "score": float(dist), "rank": i + 1, "source": "content"})
        return results

    def search_context(self, query_emb: np.ndarray, top_k: int = 10) -> list:
        if self.context_index is None or self.context_index.ntotal == 0:
            return []
        k = min(top_k, self.context_index.ntotal)
        query = query_emb.reshape(1, -1).astype(np.float32)
        distances, indices = self.context_index.search(query, k)
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            doc_id = self.context_id_to_doc.get(str(idx), "")
            results.append({"doc_id": doc_id, "score": float(dist), "rank": i + 1, "source": "context"})
        return results

    def hybrid_search(self, query_emb: np.ndarray, top_k: int = 10,
                      content_weight: float = 0.5,
                      context_weight: float = 0.5) -> list:
        content_results = self.search_content(query_emb, top_k=top_k * 2)
        context_results = self.search_context(query_emb, top_k=top_k * 2)

        scores = {}
        for r in content_results:
            doc_id = r["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + content_weight * r["score"]
        for r in context_results:
            doc_id = r["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + context_weight * r["score"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": d, "score": s} for d, s in ranked[:top_k]]

    def save(self, content_path: str, context_path: str,
             content_map_path: str, context_map_path: str):
        if self.content_index:
            faiss.write_index(self.content_index, content_path)
        if self.context_index:
            faiss.write_index(self.context_index, context_path)
        with open(content_map_path, "w") as f:
            json.dump({
                "id_to_doc": self.content_id_to_doc,
                "doc_to_id": self.doc_to_content_id,
                "next_id": self.content_next_id,
                "dim": self.content_dim,
            }, f)
        with open(context_map_path, "w") as f:
            json.dump({
                "id_to_doc": self.context_id_to_doc,
                "doc_to_id": self.doc_to_context_id,
                "next_id": self.context_next_id,
                "dim": self.context_dim,
            }, f)

    def load(self, content_path: str, context_path: str,
             content_map_path: str, context_map_path: str):
        if os.path.exists(content_path):
            self.content_index = faiss.read_index(content_path, faiss.IO_FLAG_MMAP)
        if os.path.exists(context_path):
            self.context_index = faiss.read_index(context_path, faiss.IO_FLAG_MMAP)
        if os.path.exists(content_map_path):
            with open(content_map_path, "r") as f:
                data = json.load(f)
                self.content_id_to_doc = data["id_to_doc"]
                self.doc_to_content_id = data["doc_to_id"]
                self.content_next_id = data["next_id"]
                self.content_dim = data.get("dim", 384)
        if os.path.exists(context_map_path):
            with open(context_map_path, "r") as f:
                data = json.load(f)
                self.context_id_to_doc = data["id_to_doc"]
                self.doc_to_context_id = data["doc_to_id"]
                self.context_next_id = data["next_id"]
                self.context_dim = data.get("dim", 384)


import os

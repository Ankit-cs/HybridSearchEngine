import faiss
import numpy as np
import json
import torch
import os
from pathlib import Path


class EmbeddingStore:

    def __init__(self):
        self.index = None
        self.id_to_doc = {}
        self.doc_to_id = {}
        self.next_id = 0
        self._dimension = 384
        self._index_type = "flat"
        self._hnsw_m = 32
        self._hnsw_ef_search = 64

    def init_index(self, dimension=384, index_type="flat", hnsw_m=32,
                   hnsw_ef_search=64, nlist=100, m_pq=32, nbits=8):
        self._dimension = dimension
        self._index_type = index_type

        if self.index is not None:
            return

        if index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, hnsw_m)
            self.index.hnsw.efSearch = hnsw_ef_search
        elif index_type == "ivf_pq":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m_pq, nbits)
            self.index.nprobe = min(10, nlist)
        elif index_type == "gpu_ivf_pq":
            try:
                if faiss.get_num_gpus() > 0:
                    quantizer = faiss.IndexFlatIP(dimension)
                    cpu_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m_pq, nbits)
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self.index.nprobe = min(10, nlist)
                else:
                    quantizer = faiss.IndexFlatIP(dimension)
                    self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m_pq, nbits)
                    self.index.nprobe = min(10, nlist)
            except Exception:
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m_pq, nbits)
                self.index.nprobe = min(10, nlist)
        else:
            self.index = faiss.IndexFlatIP(dimension)

    def init_index_auto(self, dimension=384, total_vectors=1000):
        if total_vectors > 100_000:
            try:
                has_gpu = faiss.get_num_gpus() > 0
            except Exception:
                has_gpu = False
            if has_gpu:
                self.init_index(dimension, "gpu_ivf_pq")
            else:
                self.init_index(dimension, "ivf_pq")
        elif total_vectors > 10_000:
            self.init_index(dimension, "hnsw")
        else:
            self.init_index(dimension, "flat")

    def train_if_needed(self, vectors: np.ndarray):
        if isinstance(self.index, faiss.IndexIVFPQ) and not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))

    def add(self, doc_id, vector):
        if hasattr(vector, 'shape'):
            dim = vector.shape[0]
        else:
            dim = len(vector)

        if self.index is None:
            self.init_index(dim, self._index_type)

        if hasattr(vector, 'numpy'):
            vec_np = vector.numpy().astype(np.float32)
        else:
            vec_np = np.array(vector, dtype=np.float32)
        vec_np = vec_np.reshape(1, -1)

        faiss_id = self.next_id
        self.next_id += 1
        self.id_to_doc[str(faiss_id)] = str(doc_id)
        self.doc_to_id[str(doc_id)] = faiss_id

        if isinstance(self.index, faiss.IndexIVFPQ) and not self.index.is_trained:
            pass
        else:
            self.index.add(vec_np)

    def add_trained(self, doc_id, vector):
        if hasattr(vector, 'numpy'):
            vec_np = vector.numpy().astype(np.float32)
        else:
            vec_np = np.array(vector, dtype=np.float32)
        vec_np = vec_np.reshape(1, -1)

        faiss_id = self.next_id
        self.next_id += 1
        self.id_to_doc[str(faiss_id)] = str(doc_id)
        self.doc_to_id[str(doc_id)] = faiss_id
        self.index.add(vec_np)

    def add_batch(self, doc_ids: list, vectors: np.ndarray):
        if len(doc_ids) == 0:
            return
        self.init_index(vectors.shape[1])
        vectors = vectors.astype(np.float32)

        if isinstance(self.index, faiss.IndexIVFPQ) and not self.index.is_trained:
            self.index.train(vectors)
            self.index.nprobe = min(10, self.index.nlist)

        start_id = self.next_id
        ids = np.arange(start_id, start_id + len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            self.id_to_doc[str(ids[i])] = str(doc_id)
            self.doc_to_id[str(doc_id)] = int(ids[i])
        self.index.add_with_ids(vectors, ids)
        self.next_id += len(doc_ids)

    def get(self, doc_id):
        if str(doc_id) not in self.doc_to_id:
            return None
        faiss_id = self.doc_to_id[str(doc_id)]
        try:
            vec = self.index.reconstruct(faiss_id)
            return torch.tensor(vec, dtype=torch.float32)
        except Exception:
            return None

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        if hasattr(query_vector, 'numpy'):
            query_np = query_vector.numpy().astype(np.float32)
        else:
            query_np = np.array(query_vector, dtype=np.float32)
        query_np = query_np.reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.id_to_doc.get(str(idx), "")
            results.append({"doc_id": doc_id, "score": float(dist)})
        return results

    def prune_by_distance(self, query_vector: np.ndarray,
                          threshold: float = 0.95) -> list:
        results = self.search(query_vector, top_k=self.index.ntotal if self.index else 100)
        return [r["doc_id"] for r in results if r["score"] >= threshold]

    def create_hnsw_index(self, dimension: int = None, m: int = 32,
                          ef_search: int = 64):
        dim = dimension or self._dimension
        self._hnsw_m = m
        self._hnsw_ef_search = ef_search
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efSearch = ef_search
        self._index_type = "hnsw"
        return self.index

    def set_hnsw_params(self, ef_search: int = None, ef_construction: int = None):
        if isinstance(self.index, faiss.IndexHNSW):
            if ef_search is not None:
                self.index.hnsw.efSearch = ef_search
                self._hnsw_ef_search = ef_search
            if ef_construction is not None:
                self.index.hnsw.efConstruction = ef_construction

    def quantize_vectors(self, vectors: np.ndarray, precision: str = "f16") -> np.ndarray:
        if precision == "f16":
            return vectors.astype(np.float16)
        elif precision == "i8":
            vmin, vmax = vectors.min(), vectors.max()
            if vmax - vmin == 0:
                return np.zeros_like(vectors, dtype=np.int8)
            normalized = (vectors - vmin) / (vmax - vmin)
            return (normalized * 255 - 128).astype(np.int8)
        return vectors

    def create_ivf_pq_index(self, dimension: int, nlist: int = 100,
                            m: int = 32, nbits: int = 8):
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        self._index_type = "ivf_pq"
        return self.index

    def train_ivf_pq(self, vectors: np.ndarray, nlist: int = None):
        if not isinstance(self.index, faiss.IndexIVFPQ):
            return
        if nlist:
            self.index.nlist = min(nlist, len(vectors) // 10 + 1)
        self.index.train(vectors.astype(np.float32))
        self.index.nprobe = min(10, self.index.nlist)

    def save(self, path):
        if self.index:
            if isinstance(self.index, faiss.IndexHNSW):
                faiss.write_index(self.index, str(path))
            else:
                faiss.write_index(self.index, str(path))
        map_path = Path(path).parent / "faiss_id_map.json"
        with open(map_path, "w") as f:
            json.dump({
                "id_to_doc": self.id_to_doc,
                "doc_to_id": self.doc_to_id,
                "next_id": self.next_id,
                "dimension": self._dimension,
                "index_type": self._index_type,
                "hnsw_m": self._hnsw_m,
                "hnsw_ef_search": self._hnsw_ef_search,
            }, f)

    def load(self, path):
        if not os.path.exists(path):
            return
        self.index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        map_path = Path(path).parent / "faiss_id_map.json"
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                data = json.load(f)
                self.id_to_doc = data["id_to_doc"]
                self.doc_to_id = data["doc_to_id"]
                self.next_id = data["next_id"]
                self._dimension = data.get("dimension", 384)
                self._index_type = data.get("index_type", "flat")
                self._hnsw_m = data.get("hnsw_m", 32)
                self._hnsw_ef_search = data.get("hnsw_ef_search", 64)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    @property
    def index_type(self) -> str:
        return self._index_type

    def get_index_info(self) -> dict:
        info = {
            "type": self._index_type,
            "dimension": self._dimension,
            "total_vectors": self.total_vectors,
        }
        if isinstance(self.index, faiss.IndexHNSW):
            info["hnsw_m"] = self._hnsw_m
            info["hnsw_ef_search"] = self._hnsw_ef_search
            info["hnsw_ef_construction"] = self.index.hnsw.efConstruction
        elif isinstance(self.index, faiss.IndexIVFPQ):
            info["nlist"] = self.index.nlist
            info["nprobe"] = self.index.nprobe
            info["m"] = self.index.pq.M
        return info

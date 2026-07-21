import numpy as np
import faiss
import json
import os
from pathlib import Path
from typing import Optional


class VectorQuantizer:
    def __init__(self, precision: str = "f16"):
        self.precision = precision
        self.quantizer = None
        self.dequantizer = None
        self._setup()

    def _setup(self):
        if self.precision == "f16":
            self.quantizer = self._quantize_f16
            self.dequantizer = self._dequantize_f16
        elif self.precision == "i8":
            self.quantizer = self._quantize_i8
            self.dequantizer = self._dequantize_i8
        else:
            self.quantizer = lambda x: x
            self.dequantizer = lambda x: x

    def _quantize_f16(self, vectors: np.ndarray) -> np.ndarray:
        return vectors.astype(np.float16)

    def _dequantize_f16(self, vectors: np.ndarray) -> np.ndarray:
        return vectors.astype(np.float32)

    def _quantize_i8(self, vectors: np.ndarray) -> np.ndarray:
        self._i8_min = vectors.min()
        self._i8_max = vectors.max()
        if self._i8_max - self._i8_min == 0:
            return np.zeros_like(vectors, dtype=np.int8)
        normalized = (vectors - self._i8_min) / (self._i8_max - self._i8_min)
        return (normalized * 255 - 128).astype(np.int8)

    def _dequantize_i8(self, vectors: np.ndarray) -> np.ndarray:
        return (vectors.astype(np.float32) + 128) / 255 * (
            self._i8_max - self._i8_min
        ) + self._i8_min

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        return self.quantizer(vectors)

    def dequantize(self, vectors: np.ndarray) -> np.ndarray:
        return self.dequantizer(vectors)

    def save_quantized(self, vectors: np.ndarray, path: str):
        quantized = self.quantize(vectors)
        np.save(path, quantized)

    def load_quantized(self, path: str) -> np.ndarray:
        quantized = np.load(path)
        return self.dequantize(quantized)


class IVFPQIndex:
    def __init__(self, dimension: int, nlist: int = 100, m: int = 32,
                 nbits: int = 8, nprobe: int = 10):
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.index = None
        self.quantizer = None
        self._trained = False

    def train(self, vectors: np.ndarray):
        if len(vectors) < self.nlist:
            self.nlist = max(2, len(vectors) // 2)
            self.nprobe = min(self.nprobe, self.nlist)

        self.quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFPQ(
            self.quantizer, self.dimension,
            self.nlist, self.m, self.nbits,
        )
        self.index.nprobe = self.nprobe
        self.index.train(vectors)
        self._trained = True

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        if not self._trained:
            self.train(vectors)
        if ids is not None:
            self.index.add_with_ids(vectors, ids)
        else:
            self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int = 10) -> tuple:
        if not self._trained or self.index.ntotal == 0:
            return np.array([]), np.array([])
        query = query.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        self._trained = True

    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index else 0


class AdaptiveIndexSelector:
    def __init__(self, dimension: int, total_vectors: int):
        self.dimension = dimension
        self.total_vectors = total_vectors
        self._has_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        try:
            return faiss.get_num_gpus() > 0
        except Exception:
            return False

    def select(self) -> str:
        if self._has_gpu and self.total_vectors > 100_000:
            return "gpu_ivf_pq"
        elif self.total_vectors > 500_000:
            return "ivf_pq"
        elif self.total_vectors > 10_000:
            return "hnsw"
        else:
            return "flat"

    def create_index(self) -> faiss.Index:
        strategy = self.select()
        if strategy == "flat":
            return faiss.IndexFlatIP(self.dimension)
        elif strategy == "hnsw":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        elif strategy in ("ivf_pq", "gpu_ivf_pq"):
            nlist = min(int(np.sqrt(self.total_vectors)), 256)
            m = min(32, self.dimension)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(
                quantizer, self.dimension, nlist, m, 8,
            )
            return index
        return faiss.IndexFlatIP(self.dimension)


class GeometricPruner:
    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path
        self.file_stats = {}
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.file_stats = json.load(f)

    def compute_centroid(self, vectors: np.ndarray) -> tuple:
        centroid = vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        radius = float(distances.max())
        return centroid.tolist(), radius

    def register_file(self, file_path: str, centroid: list, radius: float,
                      doc_count: int):
        self.file_stats[file_path] = {
            "centroid": centroid,
            "radius": radius,
            "doc_count": doc_count,
        }
        self._save()

    def _save(self):
        if self.stats_path:
            Path(self.stats_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w") as f:
                json.dump(self.file_stats, f)

    def prune(self, query_vector: np.ndarray,
              threshold: float = 0.95) -> list[str]:
        surviving = []
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        for file_path, stats in self.file_stats.items():
            centroid = np.array(stats["centroid"])
            radius = stats["radius"]
            distance = float(np.linalg.norm(query_norm - centroid))
            if distance - radius <= threshold:
                surviving.append(file_path)

        return surviving

    def get_stats(self) -> dict:
        return dict(self.file_stats)

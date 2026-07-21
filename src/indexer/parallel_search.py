import os
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable
import numpy as np


class ConcurrentSearcher:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers

    def search_files(self, file_tasks: list[dict],
                     search_fn: Callable) -> list[dict]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(search_fn, task): task
                for task in file_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_results = future.result()
                    if task_results:
                        results.extend(task_results)
                except Exception as e:
                    print(f"[ConcurrentSearch] Error searching {task.get('file_path', '?')}: {e}")
        return results

    def merge_results(self, all_results: list[dict],
                      top_k: int = 100) -> list[dict]:
        seen = {}
        for result in all_results:
            doc_id = str(result.get("doc_id", ""))
            score = result.get("score", 0.0)
            if doc_id not in seen or score > seen[doc_id]["score"]:
                seen[doc_id] = result
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]


class RangeGETLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    @property
    def file_size(self) -> int:
        return self._file_size

    def read_footer(self, footer_size: int = 4096) -> bytes:
        if self._file_size == 0:
            return b""
        start = max(0, self._file_size - footer_size)
        with open(self.file_path, "rb") as f:
            f.seek(start)
            return f.read(footer_size)

    def read_range(self, offset: int, length: int) -> bytes:
        with open(self.file_path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def read_header(self, header_offset: int = 0, header_size: int = 256) -> bytes:
        return self.read_range(header_offset, header_size)

    def find_faiss_offset(self) -> Optional[int]:
        try:
            footer = self.read_footer(8)
            if len(footer) < 8:
                return None
            import struct
            offset_bytes = footer[:4]
            length_bytes = footer[4:8]
            offset_val = struct.unpack("<I", offset_bytes)[0]
            length_val = struct.unpack("<I", length_bytes)[0]
            if 0 < offset_val < self._file_size and 0 < length_val < self._file_size:
                return offset_val
        except Exception:
            pass
        return None

    def load_index_lazy(self, index_path: str) -> Optional[object]:
        try:
            import faiss
            return faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        except Exception:
            return None


class LazyIndexLoader:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = {}

    def load(self, key: str, file_path: str) -> Optional[object]:
        if key in self._loaded:
            return self._loaded[key]

        try:
            import faiss
            index = faiss.read_index(file_path, faiss.IO_FLAG_MMAP)
            self._loaded[key] = index
            return index
        except Exception:
            return None

    def unload(self, key: str):
        if key in self._loaded:
            del self._loaded[key]

    def cache_exists(self, key: str) -> bool:
        return key in self._loaded

    def clear_cache(self):
        self._loaded.clear()

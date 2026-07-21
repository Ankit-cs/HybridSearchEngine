import math
import time
import json
import uuid
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class EpisodicMemoryEntry:
    entry_id: str = ""
    agent_id: str = ""
    session_id: str = ""
    text: str = ""
    embedding: Optional[list] = None
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed_at: float = 0.0
    created_at: float = 0.0
    decay_lambda: float = 0.1
    step_index: int = 0
    mem_type: str = "observation"

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed_at:
            self.last_accessed_at = time.time()

    @property
    def recency_weight(self) -> float:
        days_old = (time.time() - self.last_accessed_at) / 86400
        return math.exp(-self.decay_lambda * days_old)

    def touch(self):
        self.access_count += 1
        self.last_accessed_at = time.time()

    def score(self, query_distance: float = 0.0) -> float:
        return query_distance * self.recency_weight * self.importance_score

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemoryEntry":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class EpisodicMemoryStore:
    def __init__(self, store_dir: str, agent_id: str = "default",
                 max_entries: int = 10000, decay_lambda: float = 0.1):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.max_entries = max_entries
        self.decay_lambda = decay_lambda
        self.entries: list[EpisodicMemoryEntry] = []
        self._load()

    def _file_path(self) -> Path:
        return self.store_dir / f"agent_{self.agent_id}_episodic.json"

    def _load(self):
        path = self._file_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = [EpisodicMemoryEntry.from_dict(e) for e in data]

    def _save(self):
        with open(self._file_path(), "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def add(self, text: str, embedding: Optional[list] = None,
            importance: float = 1.0, mem_type: str = "observation",
            session_id: str = "", step_index: int = 0) -> EpisodicMemoryEntry:
        entry = EpisodicMemoryEntry(
            agent_id=self.agent_id,
            session_id=session_id,
            text=text,
            embedding=embedding,
            importance_score=importance,
            decay_lambda=self.decay_lambda,
            mem_type=mem_type,
            step_index=step_index,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self._evict()
        self._save()
        return entry

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        if not self.entries:
            return []

        scored = []
        for entry in self.entries:
            if entry.embedding is not None:
                entry_emb = np.array(entry.embedding, dtype=np.float32)
                dist = float(np.dot(query_embedding, entry_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_emb) + 1e-10
                ))
            else:
                dist = 0.0
            final_score = entry.score(dist)
            entry.touch()
            scored.append((entry, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        self._save()
        return [entry for entry, _ in scored[:top_k]]

    def decay_all(self) -> int:
        updated = 0
        for entry in self.entries:
            old_weight = entry.recency_weight
            entry.last_accessed_at = entry.last_accessed_at
            if entry.recency_weight != old_weight:
                updated += 1
        self._save()
        return updated

    def _evict(self):
        self.entries.sort(key=lambda e: e.recency_weight * e.importance_score)
        excess = len(self.entries) - self.max_entries
        if excess > 0:
            self.entries = self.entries[excess:]

    def get_recent(self, n: int = 10) -> list:
        return self.entries[-n:]

    def count(self) -> int:
        return len(self.entries)

    def clear(self):
        self.entries = []
        self._save()

    def get_stats(self) -> dict:
        if not self.entries:
            return {"count": 0, "avg_age_days": 0, "avg_importance": 0}
        now = time.time()
        ages = [(now - e.created_at) / 86400 for e in self.entries]
        importances = [e.importance_score for e in self.entries]
        return {
            "count": len(self.entries),
            "avg_age_days": sum(ages) / len(ages),
            "avg_importance": sum(importances) / len(importances),
            "avg_recency_weight": sum(e.recency_weight for e in self.entries) / len(self.entries),
        }


class WorkingMemoryBuffer:
    def __init__(self, max_chunks: int = 500):
        self.max_chunks = max_chunks
        self.buffer = deque(maxlen=max_chunks)

    def push(self, text: str, embedding: Optional[np.ndarray] = None,
             importance: float = 1.0, metadata: Optional[dict] = None):
        entry = {
            "text": text,
            "embedding": embedding.tolist() if embedding is not None else None,
            "importance": importance,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self.buffer.append(entry)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        if not self.buffer:
            return []
        scored = []
        for entry in self.buffer:
            if entry["embedding"] is not None:
                emb = np.array(entry["embedding"], dtype=np.float32)
                dist = float(np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10
                ))
            else:
                dist = 0.0
            score = dist * entry["importance"]
            scored.append((entry, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_chunks

    def drain(self) -> list:
        items = list(self.buffer)
        self.buffer.clear()
        return items

    def size(self) -> int:
        return len(self.buffer)

    def to_list(self) -> list:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


class AgentPartitionManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._partitions: dict[str, EpisodicMemoryStore] = {}

    def get_partition(self, agent_id: str) -> EpisodicMemoryStore:
        if agent_id not in self._partitions:
            part_dir = self.base_dir / f"agent_{agent_id}"
            self._partitions[agent_id] = EpisodicMemoryStore(
                store_dir=str(part_dir),
                agent_id=agent_id,
            )
        return self._partitions[agent_id]

    def list_agents(self) -> list[str]:
        return [
            d.name.replace("agent_", "")
            for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("agent_")
        ]

    def delete_agent(self, agent_id: str):
        part_dir = self.base_dir / f"agent_{agent_id}"
        if part_dir.exists():
            import shutil
            shutil.rmtree(str(part_dir))
        self._partitions.pop(agent_id, None)

    def get_agent_stats(self, agent_id: str) -> dict:
        store = self.get_partition(agent_id)
        return store.get_stats()

import json
import os
import time
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.storage.catalog import Catalog, FileManifest, Snapshot


@dataclass
class WriteTransaction:
    tx_id: str
    started_at: float
    files_to_add: list
    batch_id: Optional[str] = None
    status: str = "pending"


class TransactionManager:
    def __init__(self, catalog: Catalog, index_dir: Path):
        self.catalog = catalog
        self.index_dir = index_dir
        self.journal_path = index_dir / "transaction_journal.json"
        self._load_journal()

    def _load_journal(self):
        if self.journal_path.exists():
            with open(self.journal_path, "r") as f:
                self.journal = json.load(f)
        else:
            self.journal = {"pending": [], "committed": []}

    def _save_journal(self):
        with open(self.journal_path, "w") as f:
            json.dump(self.journal, f)

    def begin_transaction(self, batch_id: Optional[str] = None) -> WriteTransaction:
        tx_id = hashlib.sha256(
            f"{time.time()}-{os.getpid()}-{id(self)}".encode()
        ).hexdigest()[:16]

        tx = WriteTransaction(
            tx_id=tx_id,
            started_at=time.time(),
            files_to_add=[],
            batch_id=batch_id,
        )
        self.journal["pending"].append({
            "tx_id": tx_id,
            "started_at": tx.started_at,
            "batch_id": batch_id,
        })
        self._save_journal()
        return tx

    def commit_transaction(self, tx: WriteTransaction, snapshot: Snapshot):
        if tx.batch_id:
            self.catalog.log_batch(tx.batch_id, len(tx.files_to_add))

        for file_info in tx.files_to_add:
            manifest = FileManifest(
                file_path=file_info["path"],
                doc_count=file_info.get("doc_count", 0),
                centroid=file_info.get("centroid"),
                radius=file_info.get("radius", 0.0),
                created_at=time.time(),
                batch_id=tx.batch_id,
                index_type=file_info.get("index_type", "hnsw"),
            )
            self.catalog.add_file(snapshot.snapshot_id, manifest)

        self.journal["pending"] = [
            j for j in self.journal["pending"]
            if j["tx_id"] != tx.tx_id
        ]
        self.journal["committed"].append({
            "tx_id": tx.tx_id,
            "batch_id": tx.batch_id,
            "committed_at": time.time(),
            "file_count": len(tx.files_to_add),
        })
        if len(self.journal["committed"]) > 100:
            self.journal["committed"] = self.journal["committed"][-50:]
        self._save_journal()

    def rollback_transaction(self, tx: WriteTransaction):
        self.journal["pending"] = [
            j for j in self.journal["pending"]
            if j["tx_id"] != tx.tx_id
        ]
        self._save_journal()

    def recover_incomplete(self) -> list[str]:
        pending = self.journal.get("pending", [])
        recovered = []
        for entry in pending:
            tx_id = entry["tx_id"]
            batch_id = entry.get("batch_id")
            if batch_id and self.catalog.batch_exists(batch_id):
                recovered.append(batch_id)
            self.journal["pending"] = [
                j for j in self.journal["pending"]
                if j["tx_id"] != tx_id
            ]
        self._save_journal()
        return recovered


class IdempotentWriter:
    def __init__(self, transaction_manager: TransactionManager, ttl_hours: int = 72):
        self.tm = transaction_manager
        self.ttl_seconds = ttl_hours * 3600

    def should_write(self, batch_id: str) -> bool:
        if not batch_id:
            return True
        if self.tm.catalog.batch_exists(batch_id):
            return False
        return True

    def mark_written(self, batch_id: str, doc_count: int):
        if batch_id:
            self.tm.catalog.log_batch(batch_id, doc_count, "committed")


class CompactionPlanner:
    def __init__(self, catalog: Catalog, index_dir: Path,
                 min_files: int = 4, target_size_mb: int = 512):
        self.catalog = catalog
        self.index_dir = index_dir
        self.min_files = min_files
        self.target_size_bytes = target_size_mb * 1024 * 1024

    def plan(self) -> Optional[dict]:
        files = self.catalog.get_active_files()
        if len(files) < self.min_files:
            return None

        total_docs = sum(f.doc_count for f in files)
        avg_docs = total_docs / len(files)

        compactable = []
        current_size = 0
        for f in sorted(files, key=lambda x: x.created_at):
            file_size = f.doc_count * 4096
            if current_size + file_size > self.target_size_bytes and compactable:
                break
            compactable.append(f)
            current_size += file_size

        if len(compactable) < 2:
            return None

        return {
            "files_to_compact": [f.file_path for f in compactable],
            "total_docs": sum(f.doc_count for f in compactable),
            "estimated_output_docs": sum(f.doc_count for f in compactable),
            "input_files": len(compactable),
        }

    def get_compacted_file_paths(self, plan: dict) -> list[str]:
        return plan.get("files_to_compact", [])

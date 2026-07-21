import json
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from src.storage.catalog import Catalog, Snapshot


class TimeTravelManager:
    def __init__(self, catalog: Catalog, index_dir: Path, snapshot_dir: Path):
        self.catalog = catalog
        self.index_dir = index_dir
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, description: str = "") -> Snapshot:
        snapshot = self.catalog.create_snapshot(description)
        snap_dir = self.snapshot_dir / f"v{snapshot.version}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        snapshot_files = [
            "inverted_index.json",
            "title_index.json",
            "documents.parquet",
            "metadata.json",
            "embeddings.index",
            "faiss_id_map.json",
        ]
        for fname in snapshot_files:
            src = self.index_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(snap_dir / fname))

        manifest_path = snap_dir / "snapshot_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)

        return snapshot

    def query_version(self, version: int) -> Optional[Path]:
        snap_dir = self.snapshot_dir / f"v{version}"
        if snap_dir.exists():
            return snap_dir
        return None

    def get_latest_version(self) -> Optional[int]:
        history = self.catalog.get_snapshot_history()
        return history[0].version if history else None

    def restore_version(self, version: int) -> bool:
        snap_dir = self.query_version(version)
        if not snap_dir:
            return False

        restore_files = [
            "inverted_index.json",
            "title_index.json",
            "documents.parquet",
            "metadata.json",
            "embeddings.index",
            "faiss_id_map.json",
        ]
        for fname in restore_files:
            src = snap_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(self.index_dir / fname))

        return True

    def list_versions(self) -> list[dict]:
        history = self.catalog.get_snapshot_history()
        versions = []
        for snap in history:
            snap_dir = self.snapshot_dir / f"v{snap.version}"
            versions.append({
                "version": snap.version,
                "snapshot_id": snap.snapshot_id,
                "description": snap.description,
                "created_at": snap.created_at,
                "has_files": snap_dir.exists(),
                "files_added": snap.files_added,
                "files_removed": snap.files_removed,
            })
        return versions

    def diff_versions(self, v1: int, v2: int) -> dict:
        files_v1 = set()
        files_v2 = set()

        dir1 = self.snapshot_dir / f"v{v1}"
        dir2 = self.snapshot_dir / f"v{v2}"

        if dir1.exists():
            files_v1 = {f.name for f in dir1.iterdir() if f.is_file()}
        if dir2.exists():
            files_v2 = {f.name for f in dir2.iterdir() if f.is_file()}

        return {
            "added": list(files_v2 - files_v1),
            "removed": list(files_v1 - files_v2),
            "unchanged": list(files_v1 & files_v2),
        }

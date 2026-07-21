import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class FileManifest:
    file_path: str
    doc_count: int
    centroid: Optional[list] = None
    radius: float = 0.0
    created_at: float = 0.0
    batch_id: Optional[str] = None
    column_stats: Optional[dict] = None
    index_type: str = "hnsw"
    is_active: bool = True


@dataclass
class Snapshot:
    snapshot_id: int
    version: int
    description: str
    created_at: float
    parent_snapshot_id: Optional[int] = None
    files_added: int = 0
    files_removed: int = 0


class Catalog:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                description TEXT DEFAULT '',
                created_at REAL NOT NULL,
                parent_snapshot_id INTEGER,
                files_added INTEGER DEFAULT 0,
                files_removed INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS file_manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                doc_count INTEGER DEFAULT 0,
                centroid TEXT,
                radius REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                batch_id TEXT,
                column_stats TEXT,
                index_type TEXT DEFAULT 'hnsw',
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                schema_def TEXT NOT NULL,
                created_at REAL NOT NULL,
                change_description TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS batch_log (
                batch_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                doc_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'committed'
            );
            CREATE INDEX IF NOT EXISTS idx_manifests_active
                ON file_manifests(is_active, snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_manifests_batch
                ON file_manifests(batch_id);
        """)
        self.conn.commit()

    def create_snapshot(self, description: str = "") -> Snapshot:
        cursor = self.conn.execute(
            "SELECT MAX(snapshot_id) FROM snapshots"
        )
        row = cursor.fetchone()
        last_id = row[0] if row and row[0] else None

        cursor = self.conn.execute(
            "SELECT MAX(version) FROM snapshots"
        )
        row = cursor.fetchone()
        last_version = row[0] if row and row[0] else 0

        now = time.time()
        cursor = self.conn.execute(
            "INSERT INTO snapshots (version, description, created_at, parent_snapshot_id) "
            "VALUES (?, ?, ?, ?)",
            (last_version + 1, description, now, last_id)
        )
        self.conn.commit()

        return Snapshot(
            snapshot_id=cursor.lastrowid,
            version=last_version + 1,
            description=description,
            created_at=now,
            parent_snapshot_id=last_id,
        )

    def add_file(self, snapshot_id: int, manifest: FileManifest):
        centroid_json = json.dumps(manifest.centroid) if manifest.centroid else None
        stats_json = json.dumps(manifest.column_stats) if manifest.column_stats else None
        self.conn.execute(
            "INSERT INTO file_manifests "
            "(snapshot_id, file_path, doc_count, centroid, radius, created_at, "
            "batch_id, column_stats, index_type, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id, manifest.file_path, manifest.doc_count,
                centroid_json, manifest.radius, manifest.created_at,
                manifest.batch_id, stats_json, manifest.index_type,
                int(manifest.is_active),
            )
        )
        self.conn.commit()

    def get_active_files(self, snapshot_id: Optional[int] = None) -> list[FileManifest]:
        if snapshot_id is None:
            snapshot_id = self._latest_snapshot_id()

        cursor = self.conn.execute(
            "SELECT file_path, doc_count, centroid, radius, created_at, "
            "batch_id, column_stats, index_type, is_active "
            "FROM file_manifests WHERE snapshot_id = ? AND is_active = 1",
            (snapshot_id,)
        )
        results = []
        for row in cursor.fetchall():
            results.append(FileManifest(
                file_path=row[0],
                doc_count=row[1],
                centroid=json.loads(row[2]) if row[2] else None,
                radius=row[3],
                created_at=row[4],
                batch_id=row[5],
                column_stats=json.loads(row[6]) if row[6] else None,
                index_type=row[7],
                is_active=bool(row[8]),
            ))
        return results

    def deactivate_file(self, snapshot_id: int, file_path: str):
        self.conn.execute(
            "UPDATE file_manifests SET is_active = 0 "
            "WHERE snapshot_id = ? AND file_path = ?",
            (snapshot_id, file_path)
        )
        self.conn.commit()

    def log_batch(self, batch_id: str, doc_count: int, status: str = "committed"):
        self.conn.execute(
            "INSERT OR REPLACE INTO batch_log (batch_id, created_at, doc_count, status) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, time.time(), doc_count, status)
        )
        self.conn.commit()

    def batch_exists(self, batch_id: str) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM batch_log WHERE batch_id = ? AND status = 'committed'",
            (batch_id,)
        )
        return cursor.fetchone() is not None

    def save_schema_version(self, version: int, schema_def: dict, description: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_versions (version, schema_def, created_at, change_description) "
            "VALUES (?, ?, ?, ?)",
            (version, json.dumps(schema_def), time.time(), description)
        )
        self.conn.commit()

    def get_schema_version(self, version: int) -> Optional[dict]:
        cursor = self.conn.execute(
            "SELECT schema_def FROM schema_versions WHERE version = ?", (version,)
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def get_latest_schema(self) -> Optional[dict]:
        cursor = self.conn.execute(
            "SELECT schema_def FROM schema_versions ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def get_snapshot_history(self) -> list[Snapshot]:
        cursor = self.conn.execute(
            "SELECT snapshot_id, version, description, created_at, "
            "parent_snapshot_id, files_added, files_removed "
            "FROM snapshots ORDER BY snapshot_id DESC"
        )
        return [
            Snapshot(
                snapshot_id=r[0], version=r[1], description=r[2],
                created_at=r[3], parent_snapshot_id=r[4],
                files_added=r[5], files_removed=r[6],
            )
            for r in cursor.fetchall()
        ]

    def _latest_snapshot_id(self) -> Optional[int]:
        cursor = self.conn.execute(
            "SELECT MAX(snapshot_id) FROM snapshots"
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else None

    def get_file_count(self, snapshot_id: Optional[int] = None) -> int:
        if snapshot_id is None:
            snapshot_id = self._latest_snapshot_id()
        if snapshot_id is None:
            return 0
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM file_manifests "
            "WHERE snapshot_id = ? AND is_active = 1",
            (snapshot_id,)
        )
        return cursor.fetchone()[0]

    def get_total_docs(self, snapshot_id: Optional[int] = None) -> int:
        if snapshot_id is None:
            snapshot_id = self._latest_snapshot_id()
        if snapshot_id is None:
            return 0
        cursor = self.conn.execute(
            "SELECT COALESCE(SUM(doc_count), 0) FROM file_manifests "
            "WHERE snapshot_id = ? AND is_active = 1",
            (snapshot_id,)
        )
        return cursor.fetchone()[0]

    def close(self):
        self.conn.close()

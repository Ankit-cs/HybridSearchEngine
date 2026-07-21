import os
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional


class StorageBackend(ABC):
    @abstractmethod
    def put(self, local_path: str, remote_path: str):
        ...

    @abstractmethod
    def get(self, remote_path: str, local_path: str):
        ...

    @abstractmethod
    def get_range(self, remote_path: str, offset: int, length: int) -> bytes:
        ...

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        ...

    @abstractmethod
    def delete(self, remote_path: str):
        ...

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        ...


class LocalStorage(StorageBackend):
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        return self.root / path

    def put(self, local_path: str, remote_path: str):
        dest = self._resolve(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, str(dest))

    def get(self, remote_path: str, local_path: str):
        src = self._resolve(remote_path)
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))

    def get_range(self, remote_path: str, offset: int, length: int) -> bytes:
        path = self._resolve(remote_path)
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()

    def delete(self, remote_path: str):
        path = self._resolve(remote_path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(str(path))
            else:
                path.unlink()

    def list_files(self, prefix: str = "") -> list[str]:
        base = self._resolve(prefix) if prefix else self.root
        results = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self.root))
                results.append(rel)
        return results


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        import boto3
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", region_name=region)

    def _key(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def put(self, local_path: str, remote_path: str):
        self.s3.upload_file(local_path, self.bucket, self._key(remote_path))

    def get(self, remote_path: str, local_path: str):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self._key(remote_path), local_path)

    def get_range(self, remote_path: str, offset: int, length: int) -> bytes:
        range_header = f"bytes={offset}-{offset + length - 1}"
        resp = self.s3.get_object(
            Bucket=self.bucket, Key=self._key(remote_path),
            Range=range_header,
        )
        return resp["Body"].read()

    def exists(self, remote_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(remote_path))
            return True
        except Exception:
            return False

    def delete(self, remote_path: str):
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(remote_path))

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix) if prefix else self.prefix
        results = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1:]
                results.append(key)
        return results


class GCSStorage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _blob_name(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def put(self, local_path: str, remote_path: str):
        blob = self.bucket.blob(self._blob_name(remote_path))
        blob.upload_from_filename(local_path)

    def get(self, remote_path: str, local_path: str):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.blob(self._blob_name(remote_path))
        blob.download_to_filename(local_path)

    def get_range(self, remote_path: str, offset: int, length: int) -> bytes:
        blob = self.bucket.blob(self._blob_name(remote_path))
        return blob.download_as_bytes(start=offset, end=offset + length - 1)

    def exists(self, remote_path: str) -> bool:
        blob = self.bucket.blob(self._blob_name(remote_path))
        return blob.exists()

    def delete(self, remote_path: str):
        blob = self.bucket.blob(self._blob_name(remote_path))
        blob.delete()

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._blob_name(prefix) if prefix else self.prefix
        return [
            b.name[len(self.prefix) + 1:] if self.prefix else b.name
            for b in self.bucket.list_blobs(prefix=full_prefix)
        ]


def get_storage_backend(backend: str = "local", **kwargs) -> StorageBackend:
    if backend == "s3":
        return S3Storage(
            bucket=kwargs.get("bucket", os.getenv("AWS_S3_BUCKET", "")),
            prefix=kwargs.get("prefix", ""),
            region=kwargs.get("region", "us-east-1"),
        )
    elif backend == "gcs":
        return GCSStorage(
            bucket=kwargs.get("bucket", os.getenv("GCS_BUCKET", "")),
            prefix=kwargs.get("prefix", ""),
        )
    else:
        root = kwargs.get("root_dir", str(Path(__file__).resolve().parents[2] / "data"))
        return LocalStorage(root)

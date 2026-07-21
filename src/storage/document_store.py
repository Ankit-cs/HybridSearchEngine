import pandas as pd
import numpy as np
from typing import Optional


class DocumentStore:
    def __init__(self):
        self.docs = {}

    def add(self, doc_id, title, url, text, **kwargs):
        entry = {
            "title": title,
            "url": url,
            "text": text,
        }
        for key, value in kwargs.items():
            entry[key] = value
        self.docs[str(doc_id)] = entry

    def get(self, doc_id):
        return self.docs.get(str(doc_id))

    def get_context_metadata(self, doc_id) -> dict:
        doc = self.docs.get(str(doc_id), {})
        return {
            "chunk_id": doc.get("chunk_id", ""),
            "document_id": doc.get("document_id", ""),
            "chunk_index": doc.get("chunk_index", 0),
            "total_chunks": doc.get("total_chunks", 1),
            "document_title": doc.get("title", ""),
            "section_path": doc.get("section_path", ""),
            "preceding_context": doc.get("preceding_context", ""),
            "following_context": doc.get("following_context", ""),
            "document_summary": doc.get("document_summary", ""),
            "source_uri": doc.get("url", ""),
            "page_number": doc.get("page_number"),
            "document_date": doc.get("document_date"),
        }

    def delete(self, doc_id):
        self.docs.pop(str(doc_id), None)

    def save(self, path):
        if not self.docs:
            return
        df = pd.DataFrame.from_dict(self.docs, orient='index')
        df.index.name = 'doc_id'
        df.to_parquet(path)

    def load(self, path):
        import os
        if not os.path.exists(path):
            return
        df = pd.read_parquet(path)
        if df.index.name == 'doc_id' or 'doc_id' not in df.columns:
            self.docs = df.to_dict(orient='index')
        else:
            self.docs = df.set_index('doc_id').to_dict(orient='index')
        self.docs = {str(k): v for k, v in self.docs.items()}

    def count(self) -> int:
        return len(self.docs)

    def batch_get(self, doc_ids: list) -> list:
        return [self.docs.get(str(did)) for did in doc_ids]

    def add_batch(self, entries: list[dict]):
        for entry in entries:
            doc_id = str(entry.get("doc_id", ""))
            if doc_id:
                self.docs[doc_id] = {k: v for k, v in entry.items() if k != "doc_id"}

    def filter_by(self, column: str, value: str) -> list:
        results = []
        for doc_id, doc in self.docs.items():
            if doc.get(column) == value:
                results.append(doc_id)
        return results

    def get_all_ids(self) -> list:
        return list(self.docs.keys())

    def get_column_values(self, column: str) -> dict:
        return {
            doc_id: doc.get(column)
            for doc_id, doc in self.docs.items()
            if column in doc
        }

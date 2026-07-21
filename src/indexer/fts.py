import os
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass


@dataclass
class FTSResult:
    doc_id: str
    score: float
    snippet: str = ""


class PersistentFTSIndex:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = {}
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.index_file = self.index_dir / "fts_inverted.json"
        self.meta_file = self.index_dir / "fts_meta.json"
        self._load()

    def _load(self):
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.inverted_index = defaultdict(dict, data)
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.doc_lengths = meta.get("doc_lengths", {})
                self.total_docs = meta.get("total_docs", 0)
                self.avg_doc_length = meta.get("avg_doc_length", 0.0)

    def _save(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(dict(self.inverted_index), f)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_lengths": self.doc_lengths,
                "total_docs": self.total_docs,
                "avg_doc_length": self.avg_doc_length,
            }, f)

    def index_document(self, doc_id: str, tokens: list[str]):
        token_freq = defaultdict(int)
        for token in tokens:
            token_freq[token] += 1
        for token, freq in token_freq.items():
            self.inverted_index[token][str(doc_id)] = freq
        doc_len = len(tokens)
        self.doc_lengths[str(doc_id)] = doc_len
        self.total_docs += 1
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs if self.total_docs else 0

    def commit(self):
        self._save()

    def search(self, query_tokens: list[str], top_k: int = 50,
               k1: float = 1.5, b: float = 0.75) -> list[FTSResult]:
        scores = defaultdict(float)
        for term in query_tokens:
            postings = self.inverted_index.get(term, {})
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(str(doc_id), 0)
                denom = tf + k1 * (1 - b + b * (dl / max(self.avg_doc_length, 1)))
                score = idf * (tf * (k1 + 1)) / denom
                scores[doc_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            FTSResult(doc_id=doc_id, score=score)
            for doc_id, score in ranked[:top_k]
        ]

    def get_postings(self, term: str) -> dict:
        return dict(self.inverted_index.get(term, {}))

    def delete_document(self, doc_id: str):
        doc_id_str = str(doc_id)
        for term in list(self.inverted_index.keys()):
            if doc_id_str in self.inverted_index[term]:
                del self.inverted_index[term][doc_id_str]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
        if doc_id_str in self.doc_lengths:
            del self.doc_lengths[doc_id_str]
            self.total_docs = max(0, self.total_docs - 1)
            if self.total_docs > 0:
                total_length = sum(self.doc_lengths.values())
                self.avg_doc_length = total_length / self.total_docs

    def merge_from(self, other: "PersistentFTSIndex"):
        for term, postings in other.inverted_index.items():
            for doc_id, freq in postings.items():
                if doc_id in self.inverted_index.get(term, {}):
                    self.inverted_index[term][doc_id] = max(
                        self.inverted_index[term][doc_id], freq
                    )
                else:
                    self.inverted_index[term][doc_id] = freq
        for doc_id, length in other.doc_lengths.items():
            if doc_id not in self.doc_lengths:
                self.total_docs += 1
            self.doc_lengths[doc_id] = length
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs if self.total_docs else 0

    @property
    def term_count(self) -> int:
        return len(self.inverted_index)

    @property
    def doc_count(self) -> int:
        return self.total_docs




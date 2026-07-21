import uuid
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ChunkMetadata:
    chunk_id: str = ""
    document_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    document_title: str = ""
    section_path: str = ""
    preceding_context: str = ""
    following_context: str = ""
    document_summary: str = ""
    chunk_summary: str = ""
    source_uri: str = ""
    page_number: Optional[int] = None
    created_at: float = 0.0
    document_date: Optional[str] = None

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())
        if not self.document_id:
            self.document_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LLMContextSchema:
    BASE_FIELDS = {
        "chunk_id": "str",
        "document_id": "str",
        "chunk_index": "int",
        "total_chunks": "int",
        "chunk_text": "str",
        "document_title": "str",
        "section_path": "str",
        "preceding_context": "str",
        "following_context": "str",
        "document_summary": "str",
        "chunk_summary": "str",
        "source_uri": "str",
        "page_number": "int?",
        "created_at": "float",
        "document_date": "str?",
    }

    @classmethod
    def get_schema(cls) -> dict:
        return dict(cls.BASE_FIELDS)

    @classmethod
    def build_chunk(cls, text: str, title: str = "", section: str = "",
                    prev_context: str = "", next_context: str = "",
                    doc_id: str = "", chunk_idx: int = 0,
                    total_chunks: int = 1, source_uri: str = "") -> ChunkMetadata:
        meta = ChunkMetadata(
            document_id=doc_id or str(uuid.uuid4()),
            document_title=title,
            section_path=section,
            preceding_context=prev_context,
            following_context=next_context,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            source_uri=source_uri,
        )
        return meta


class ContextAssembler:
    def __init__(self, max_tokens: int = 4000, dedup_threshold: float = 0.85,
                 group_by_document: bool = True, max_chunks_per_document: int = 5,
                 include_adjacent_context: bool = True):
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self.group_by_document = group_by_document
        self.max_chunks_per_document = max_chunks_per_document
        self.include_adjacent_context = include_adjacent_context

    def assemble(self, chunks: list[dict], scores: list[float] = None) -> str:
        if not chunks:
            return ""

        if scores is None:
            scores = [1.0] * len(chunks)

        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        deduped = self._dedup(scored_chunks)

        if self.group_by_document:
            grouped = self._group_by_document(deduped)
            assembled = []
            for doc_id, doc_chunks in grouped.items():
                doc_chunks.sort(key=lambda x: x[0].get("chunk_index", 0))
                for chunk, score in doc_chunks[:self.max_chunks_per_document]:
                    assembled.append(self._format_chunk(chunk))
            deduped = [(c, s) for c, s in deduped]
        else:
            assembled = [self._format_chunk(c) for c, _ in deduped]

        return self._fit_to_budget(assembled)

    def _dedup(self, scored_chunks: list) -> list:
        if self.dedup_threshold >= 1.0:
            return scored_chunks

        kept = []
        seen_texts = []
        for chunk, score in scored_chunks:
            text = chunk.get("text", "")[:500].lower()
            is_dup = False
            for seen in seen_texts:
                overlap = self._text_overlap(text, seen)
                if overlap >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((chunk, score))
                seen_texts.append(text)
        return kept

    def _text_overlap(self, a: str, b: str) -> float:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))

    def _group_by_document(self, chunks: list) -> dict:
        groups = {}
        for chunk, score in chunks:
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in groups:
                groups[doc_id] = []
            groups[doc_id].append((chunk, score))
        return groups

    def _format_chunk(self, chunk: dict) -> str:
        parts = []
        if chunk.get("section_path"):
            parts.append(f"[Section: {chunk['section_path']}]")
        if self.include_adjacent_context and chunk.get("preceding_context"):
            parts.append(f"(Context before: {chunk['preceding_context'][:200]})")
        parts.append(chunk.get("text", ""))
        if self.include_adjacent_context and chunk.get("following_context"):
            parts.append(f"(Context after: {chunk['following_context'][:200]})")
        return "\n".join(parts)

    def _fit_to_budget(self, chunks: list[str]) -> str:
        estimated_tokens = 0
        result = []
        for chunk in chunks:
            chunk_tokens = len(chunk.split()) * 4 // 3
            if estimated_tokens + chunk_tokens > self.max_tokens:
                remaining = self.max_tokens - estimated_tokens
                if remaining > 100:
                    words = chunk.split()
                    max_words = int(remaining * 3 / 4)
                    chunk = " ".join(words[:max_words]) + "..."
                    result.append(chunk)
                break
            result.append(chunk)
            estimated_tokens += chunk_tokens
        return "\n\n---\n\n".join(result)


class SchemaEvolver:
    def __init__(self, catalog):
        self.catalog = catalog

    def add_column(self, column_name: str, column_type: str,
                   description: str = "") -> dict:
        current = self.catalog.get_latest_schema() or {}
        version = max(current.get("version", 0) + 1, 1)

        new_schema = dict(current)
        new_schema["version"] = version
        if "columns" not in new_schema:
            new_schema["columns"] = {}
        new_schema["columns"][column_name] = {
            "type": column_type,
            "description": description,
            "added_at": time.time(),
        }
        self.catalog.save_schema_version(
            version, new_schema,
            f"Added column '{column_name}' ({column_type})"
        )
        return new_schema

    def rename_column(self, old_name: str, new_name: str) -> dict:
        current = self.catalog.get_latest_schema()
        if not current or "columns" not in current:
            raise ValueError("No schema with columns found")

        version = current["version"] + 1
        new_schema = dict(current)
        new_schema["version"] = version
        if old_name in new_schema["columns"]:
            new_schema["columns"][new_name] = new_schema["columns"].pop(old_name)
            new_schema["columns"][new_name]["renamed_from"] = old_name
            new_schema["columns"][new_name]["renamed_at"] = time.time()
        self.catalog.save_schema_version(
            version, new_schema,
            f"Renamed column '{old_name}' -> '{new_name}'"
        )
        return new_schema

    def get_current_columns(self) -> dict:
        schema = self.catalog.get_latest_schema()
        return schema.get("columns", {}) if schema else {}

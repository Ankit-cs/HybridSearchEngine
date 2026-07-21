import json
import time
from src.semantic.embedding_store import EmbeddingStore
from src.semantic.reranker import SemanticReranker

from src.query.query_parser import parse_query
from src.storage.index_reader import IndexReader
from src.storage.document_store import DocumentStore
from src.storage.catalog import Catalog
from src.storage.acid import TransactionManager, CompactionPlanner
from src.storage.schema import ContextAssembler, SchemaEvolver
from src.storage.time_travel import TimeTravelManager

from src.ranking.tfidf import TFIDFRanker
from src.ranking.bm25 import BM25Ranker
from src.ranking.ltr_ranker import LightGBMRanker

from src.utils.config import (
    METADATA_PATH,
    RANKER,
    TITLE_INDEX_PATH,
    EMBEDDINGS_PATH,
    CATALOG_DB_PATH,
    INDEX_DIR,
    SNAPSHOT_DIR,
    SEARCH_CONCURRENCY,
    GEOMETRIC_STATS_PATH,
    COMPACTION_MIN_FILES,
    COMPACTION_TARGET_SIZE_MB,
    CONTEXT_EMBEDDINGS_PATH,
    CONTEXT_ID_MAP_PATH,
)

from src.semantic.query_expander import QueryExpander
from src.semantic.dual_embeddings import DualEmbeddingStore, DualEmbeddingGenerator
from src.indexer.compression import GeometricPruner, AdaptiveIndexSelector
from src.indexer.parallel_search import ConcurrentSearcher, LazyIndexLoader
from src.indexer.fts import PersistentFTSIndex
from src.agent.memory import (
    WorkingMemoryBuffer,
    AgentPartitionManager,
    EpisodicMemoryStore,
)

from src.utils.config import (
    FTS_INDEX_DIR,
    WORKING_MEMORY_DIR,
    EPISODIC_MEMORY_DIR,
    WORKING_MEMORY_MAX_CHUNKS,
    MEMORY_DECAY_LAMBDA,
)


class SearchEngine:
    def __init__(self, index_path, doc_store_path, total_docs=None):

        self.index_reader = IndexReader(index_path)
        self.title_index_reader = IndexReader(TITLE_INDEX_PATH)

        self.doc_store = DocumentStore()
        self.doc_store.load(doc_store_path)

        self.embedding_store = EmbeddingStore()
        self.embedding_store.load(EMBEDDINGS_PATH)

        self.semantic_enabled = True
        self.reranker = (
            SemanticReranker(self.embedding_store)
            if self.semantic_enabled
            else None
        )

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        if RANKER == "bm25":
            self.ranker = BM25Ranker(
                body_index=self.index_reader,
                title_index=self.title_index_reader,
                metadata=metadata,
            )
        else:
            self.ranker = TFIDFRanker(
                self.index_reader,
                metadata["total_docs"]
            )

        self.ltr_ranker = LightGBMRanker(
            embedding_store=self.embedding_store,
            doc_store=self.doc_store,
            metadata=metadata,
        )

        self.query_expander = QueryExpander(
            self.embedding_store,
            self.doc_store
        )

        self.catalog = Catalog(str(CATALOG_DB_PATH))
        self.transaction_manager = TransactionManager(self.catalog, INDEX_DIR)
        self.compaction_planner = CompactionPlanner(
            self.catalog, INDEX_DIR,
            min_files=COMPACTION_MIN_FILES,
            target_size_mb=COMPACTION_TARGET_SIZE_MB,
        )
        self.schema_evolver = SchemaEvolver(self.catalog)
        self.context_assembler = ContextAssembler(
            max_tokens=4000, dedup_threshold=0.85
        )
        self.time_travel = TimeTravelManager(
            self.catalog, INDEX_DIR, SNAPSHOT_DIR
        )

        self.dual_store = DualEmbeddingStore()
        dual_content = str(CONTEXT_EMBEDDINGS_PATH)
        dual_context = str(CONTEXT_EMBEDDINGS_PATH).replace(
            "context_embeddings", "context_context_embeddings"
        )
        content_map = str(CONTEXT_ID_MAP_PATH)
        context_map = str(CONTEXT_ID_MAP_PATH).replace(
            "context_id_map", "context_context_id_map"
        )
        if Path(dual_content).exists():
            self.dual_store.load(dual_content, dual_context, content_map, context_map)

        self.geometric_pruner = GeometricPruner(str(GEOMETRIC_STATS_PATH))
        self.concurrent_searcher = ConcurrentSearcher(max_workers=SEARCH_CONCURRENCY)
        self.lazy_loader = LazyIndexLoader(str(INDEX_DIR / "cache"))

        self.fts_index = PersistentFTSIndex(str(FTS_INDEX_DIR))

        self.working_memory = WorkingMemoryBuffer(max_chunks=WORKING_MEMORY_MAX_CHUNKS)
        self.agent_partitions = AgentPartitionManager(str(EPISODIC_MEMORY_DIR))

        self.adaptive_selector = AdaptiveIndexSelector(
            dimension=self.embedding_store.dimension,
            total_vectors=self.embedding_store.total_vectors,
        )

    def search(self, query, top_k=10, agent_id: str = None,
               column_filter: dict = None, use_fts: bool = False,
               use_dual: bool = False, max_tokens: int = 4000):

        tokens = parse_query(query)

        if use_fts and self.fts_index.term_count > 0:
            fts_results = self.fts_index.search(tokens, top_k=top_k * 3)
            fts_scored = [(r.doc_id, r.score) for r in fts_results]
            if fts_scored:
                scores = self.ranker.score(tokens)
                for doc_id, fts_score in fts_results:
                    existing = scores.get(doc_id, 0.0)
                    scores[doc_id] = existing + fts_score * 0.5
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            else:
                scores = self.ranker.score(tokens)
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            scores = self.ranker.score(tokens)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if column_filter:
            ranked = self._apply_column_filter(ranked, column_filter)

        expanded_query = self.query_expander.expand(query, ranked[:20])
        tokens = parse_query(expanded_query)
        scores = self.ranker.score(tokens)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if use_dual and self.dual_store.content_index is not None:
            from src.semantic.embedding_model import EmbeddingModel
            model = EmbeddingModel()
            query_emb = model.encode(query)[0]
            dual_results = self.dual_store.hybrid_search(query_emb, top_k=top_k * 2)
            dual_scored = {}
            for r in dual_results:
                dual_scored[r["doc_id"]] = r["score"]
            for i, (doc_id, score) in enumerate(ranked):
                dual_boost = dual_scored.get(doc_id, 0.0)
                ranked[i] = (doc_id, score + dual_boost * 0.3)

        if self.semantic_enabled and self.reranker:
            candidates = ranked[:50]
            try:
                ranked = self.reranker.rerank(query, candidates)
            except Exception as e:
                print("[RRF] Semantic reranking failed:", e)

        try:
            rrf_top = ranked[:20]
            bm25_scores = self.ranker.score(tokens)
            rrf_with_bm25 = [
                (doc_id, bm25_scores.get(doc_id, 0.0))
                for doc_id, _ in rrf_top
            ]
            ranked = self.ltr_ranker.rerank(query, tokens, rrf_with_bm25)
        except Exception as e:
            print("[LTR] LightGBM re-rank failed, keeping RRF order:", e)

        if agent_id:
            agent_store = self.agent_partitions.get_partition(agent_id)
            if agent_store.count() > 0:
                from src.semantic.embedding_model import EmbeddingModel
                model = EmbeddingModel()
                query_emb = model.encode(query)[0]
                memory_results = agent_store.search(query_emb, top_k=5)
                memory_doc_ids = set()
                for entry in memory_results:
                    for doc_id, _ in ranked:
                        doc = self.doc_store.get(doc_id)
                        if doc and entry.text[:50] in doc.get("text", ""):
                            memory_doc_ids.add(doc_id)
                if memory_doc_ids:
                    boosted = []
                    for doc_id, score in ranked:
                        if doc_id in memory_doc_ids:
                            boosted.append((doc_id, score * 1.2))
                        else:
                            boosted.append((doc_id, score))
                    boosted.sort(key=lambda x: x[1], reverse=True)
                    ranked = boosted

        if self.working_memory.size() > 0:
            from src.semantic.embedding_model import EmbeddingModel
            model = EmbeddingModel()
            query_emb = model.encode(query)[0]
            wm_results = self.working_memory.search(query_emb, top_k=3)
            wm_texts = {r["text"][:50] for r in wm_results}
            for doc_id, _ in ranked:
                doc = self.doc_store.get(doc_id)
                if doc:
                    for wm_text in wm_texts:
                        if wm_text in doc.get("text", ""):
                            ranked = [(d, s * 1.1 if d == doc_id else s) for d, s in ranked]
                            break

        return ranked[:top_k]

    def search_as_context(self, query: str, top_k: int = 10,
                          max_tokens: int = 4000) -> str:
        results = self.search(query, top_k=top_k)
        chunks = []
        for doc_id, score in results:
            doc = self.doc_store.get(doc_id)
            if doc:
                ctx = self.doc_store.get_context_metadata(doc_id)
                chunk = {
                    "text": doc.get("text", ""),
                    "document_id": doc_id,
                    "section_path": ctx.get("section_path", ""),
                    "preceding_context": ctx.get("preceding_context", ""),
                    "following_context": ctx.get("following_context", ""),
                    "chunk_index": ctx.get("chunk_index", 0),
                    "document_title": ctx.get("document_title", doc.get("title", "")),
                }
                chunks.append(chunk)
        return self.context_assembler.assemble(chunks, max_tokens=max_tokens)

    def _apply_column_filter(self, ranked: list, column_filter: dict) -> list:
        column = column_filter.get("column", "")
        value = column_filter.get("value")
        op = column_filter.get("op", "eq")
        if not column or value is None:
            return ranked
        filtered = []
        for doc_id, score in ranked:
            doc = self.doc_store.get(doc_id)
            if not doc:
                continue
            doc_val = doc.get(column)
            if doc_val is None:
                continue
            match = False
            if op == "eq":
                match = str(doc_val) == str(value)
            elif op == "ne":
                match = str(doc_val) != str(value)
            elif op == "gt":
                match = float(doc_val) > float(value)
            elif op == "gte":
                match = float(doc_val) >= float(value)
            elif op == "lt":
                match = float(doc_val) < float(value)
            elif op == "lte":
                match = float(doc_val) <= float(value)
            elif op == "contains":
                match = str(value).lower() in str(doc_val).lower()
            if match:
                filtered.append((doc_id, score))
        return filtered

    def get_time_travel_versions(self) -> list:
        return self.time_travel.list_versions()

    def restore_version(self, version: int) -> bool:
        return self.time_travel.restore_version(version)

    def create_snapshot(self, description: str = "") -> dict:
        snapshot = self.time_travel.create_snapshot(description)
        return {
            "snapshot_id": snapshot.snapshot_id,
            "version": snapshot.version,
            "description": snapshot.description,
        }

    def get_catalog_stats(self) -> dict:
        return {
            "total_snapshots": len(self.catalog.get_snapshot_history()),
            "total_files": self.catalog.get_file_count(),
            "total_docs": self.catalog.get_total_docs(),
            "schema_version": self.catalog.get_latest_schema(),
        }

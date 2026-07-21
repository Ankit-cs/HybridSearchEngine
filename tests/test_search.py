import pytest
import os
import sys
import json
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.storage.catalog import Catalog, FileManifest, Snapshot
from src.storage.acid import TransactionManager, IdempotentWriter, CompactionPlanner
from src.storage.schema import (
    ChunkMetadata, LLMContextSchema, ContextAssembler,
    SchemaEvolver,
)
from src.storage.cloud_store import LocalStorage
from src.storage.time_travel import TimeTravelManager
from src.storage.document_store import DocumentStore
from src.indexer.compression import (
    VectorQuantizer, GeometricPruner, AdaptiveIndexSelector, IVFPQIndex,
)
from src.indexer.parallel_search import ConcurrentSearcher, RangeGETLoader, LazyIndexLoader
from src.indexer.fts import PersistentFTSIndex
from src.semantic.dual_embeddings import DualEmbeddingGenerator, DualEmbeddingStore
from src.agent.memory import (
    EpisodicMemoryStore, WorkingMemoryBuffer,
    AgentPartitionManager, EpisodicMemoryEntry,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestCatalog:
    def test_create_snapshot(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        snap = catalog.create_snapshot("test build")
        assert snap.version == 1
        assert snap.snapshot_id > 0
        catalog.close()

    def test_add_and_retrieve_files(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        snap = catalog.create_snapshot("build 1")
        manifest = FileManifest(
            file_path="data/test.parquet",
            doc_count=100,
            centroid=[0.1, 0.2, 0.3],
            radius=0.5,
            created_at=1.0,
            batch_id="batch-001",
        )
        catalog.add_file(snap.snapshot_id, manifest)
        files = catalog.get_active_files(snap.snapshot_id)
        assert len(files) == 1
        assert files[0].doc_count == 100
        assert files[0].centroid == [0.1, 0.2, 0.3]
        catalog.close()

    def test_batch_idempotency(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        catalog.log_batch("batch-123", 50)
        assert catalog.batch_exists("batch-123") is True
        assert catalog.batch_exists("batch-456") is False
        catalog.close()

    def test_schema_versioning(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        schema = {"version": 1, "columns": {"id": {"type": "int"}}}
        catalog.save_schema_version(1, schema, "initial")
        loaded = catalog.get_schema_version(1)
        assert loaded is not None
        assert loaded["version"] == 1
        latest = catalog.get_latest_schema()
        assert latest["version"] == 1
        catalog.close()


class TestSchema:
    def test_chunk_metadata(self):
        meta = ChunkMetadata(
            document_id="doc-1",
            document_title="Test Doc",
            section_path="intro",
            chunk_index=0,
            total_chunks=3,
        )
        assert meta.chunk_id != ""
        d = meta.to_dict()
        assert d["document_title"] == "Test Doc"

    def test_context_assembler(self):
        assembler = ContextAssembler(max_tokens=500)
        chunks = [
            {"text": "Chunk one about AI", "document_id": "d1", "chunk_index": 0},
            {"text": "Chunk two about ML", "document_id": "d1", "chunk_index": 1},
            {"text": "Chunk three about DL", "document_id": "d2", "chunk_index": 0},
        ]
        result = assembler.assemble(chunks)
        assert "AI" in result or "ML" in result or "DL" in result

    def test_schema_evolver(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        evolver = SchemaEvolver(catalog)
        schema = evolver.add_column("rating", "float", "Document rating")
        assert "rating" in schema.get("columns", {})
        evolver.rename_column("rating", "score")
        latest = evolver.get_current_columns()
        assert "score" in latest
        assert "rating" not in latest
        catalog.close()


class TestCompression:
    def test_f16_quantization(self):
        q = VectorQuantizer("f16")
        vectors = np.random.randn(10, 384).astype(np.float32)
        quantized = q.quantize(vectors)
        assert quantized.dtype == np.float16
        dequantized = q.dequantize(quantized)
        assert dequantized.dtype == np.float32
        assert dequantized.shape == vectors.shape

    def test_i8_quantization(self):
        q = VectorQuantizer("i8")
        vectors = np.random.randn(10, 384).astype(np.float32)
        quantized = q.quantize(vectors)
        assert quantized.dtype == np.int8

    def test_geometric_pruner(self, tmp_dir):
        pruner = GeometricPruner(os.path.join(tmp_dir, "geo.json"))
        vectors = np.random.randn(100, 384).astype(np.float32)
        centroid, radius = pruner.compute_centroid(vectors)
        assert len(centroid) == 384
        assert radius > 0
        pruner.register_file("test.parquet", centroid, radius, 100)
        query = np.random.randn(384).astype(np.float32)
        surviving = pruner.prune(query, threshold=2.0)
        assert len(surviving) >= 0

    def test_adaptive_selector(self):
        selector = AdaptiveIndexSelector(384, 100000)
        strategy = selector.select()
        assert strategy in ("flat", "hnsw", "ivf_pq", "gpu_ivf_pq")

    def test_ivf_pq_index(self):
        idx = IVFPQIndex(384, nlist=4, m=32, nbits=8)
        vectors = np.random.randn(200, 384).astype(np.float32)
        idx.train(vectors)
        idx.add(vectors)
        assert idx.ntotal == 200
        query = np.random.randn(384).astype(np.float32)
        dists, ids = idx.search(query, top_k=5)
        assert len(dists) == 5


class TestParallelSearch:
    def test_concurrent_searcher(self):
        searcher = ConcurrentSearcher(max_workers=2)
        tasks = [{"file_path": f"file_{i}.parquet"} for i in range(5)]

        def mock_search(task):
            return [{"doc_id": task["file_path"], "score": 0.5}]

        results = searcher.search_files(tasks, mock_search)
        assert len(results) == 5

    def test_range_get_loader(self, tmp_dir):
        test_file = os.path.join(tmp_dir, "test.bin")
        data = os.urandom(1024)
        with open(test_file, "wb") as f:
            f.write(data)
        loader = RangeGETLoader(test_file)
        assert loader.file_size == 1024
        chunk = loader.read_range(0, 10)
        assert len(chunk) == 10
        footer = loader.read_footer(64)
        assert len(footer) == 64


class TestFTS:
    def test_fts_index_and_search(self, tmp_dir):
        fts = PersistentFTSIndex(tmp_dir)
        fts.index_document("doc1", ["hello", "world", "search"])
        fts.index_document("doc2", ["hello", "python", "code"])
        fts.commit()
        results = fts.search(["hello"], top_k=10)
        assert len(results) == 2
        assert results[0].doc_id in ("doc1", "doc2")

    def test_fts_delete(self, tmp_dir):
        fts = PersistentFTSIndex(tmp_dir)
        fts.index_document("doc1", ["test", "delete"])
        fts.commit()
        assert fts.doc_count == 1
        fts.delete_document("doc1")
        assert fts.doc_count == 0
        fts.commit()


class TestDualEmbeddings:
    def test_dual_embedding_store(self):
        store = DualEmbeddingStore(content_dim=64, context_dim=64)
        c_emb = np.random.randn(64).astype(np.float32)
        ct_emb = np.random.randn(64).astype(np.float32)
        store.add("doc1", c_emb, ct_emb)
        query = np.random.randn(64).astype(np.float32)
        c_results = store.search_content(query, top_k=1)
        assert len(c_results) == 1
        assert c_results[0]["doc_id"] == "doc1"
        hybrid = store.hybrid_search(query, top_k=1)
        assert len(hybrid) == 1


class TestAgentMemory:
    def test_episodic_memory(self, tmp_dir):
        store = EpisodicMemoryStore(tmp_dir, agent_id="test_agent")
        entry = store.add("Remember this fact", importance=1.5)
        assert entry.text == "Remember this fact"
        assert entry.importance_score == 1.5
        assert store.count() == 1
        stats = store.get_stats()
        assert stats["count"] == 1

    def test_working_memory_buffer(self):
        buf = WorkingMemoryBuffer(max_chunks=5)
        buf.push("Memory 1", importance=1.0)
        buf.push("Memory 2", importance=2.0)
        assert buf.size() == 2
        assert not buf.is_full()
        items = buf.drain()
        assert len(items) == 2
        assert buf.size() == 0

    def test_agent_partition_manager(self, tmp_dir):
        mgr = AgentPartitionManager(tmp_dir)
        store_a = mgr.get_partition("agent_a")
        store_a.add("Agent A memory")
        store_b = mgr.get_partition("agent_b")
        store_b.add("Agent B memory")
        agents = mgr.list_agents()
        assert len(agents) == 2
        stats_a = mgr.get_agent_stats("agent_a")
        assert stats_a["count"] == 1


class TestTransactionManager:
    def test_transaction_commit(self, tmp_dir):
        catalog = Catalog(os.path.join(tmp_dir, "test.db"))
        tm = TransactionManager(catalog, tmp_dir)
        tx = tm.begin_transaction(batch_id="tx-001")
        snap = catalog.create_snapshot("tx test")
        tx.files_to_add = [{"path": "test.parquet", "doc_count": 10}]
        tm.commit_transaction(tx, snap)
        files = catalog.get_active_files(snap.snapshot_id)
        assert len(files) == 1
        catalog.close()


class TestDocumentStore:
    def test_add_and_get(self):
        store = DocumentStore()
        store.add("1", "Title", "url", "Text body")
        doc = store.get("1")
        assert doc["title"] == "Title"

    def test_context_metadata(self):
        store = DocumentStore()
        store.add("1", "Title", "url", "Text", section_path="intro")
        meta = store.get_context_metadata("1")
        assert meta["section_path"] == "intro"

    def test_batch_get(self):
        store = DocumentStore()
        store.add("1", "A", "u", "T1")
        store.add("2", "B", "u", "T2")
        docs = store.batch_get(["1", "2"])
        assert len(docs) == 2
        assert docs[0]["title"] == "A"

    def test_filter_by(self):
        store = DocumentStore()
        store.add("1", "A", "u", "T", category="news")
        store.add("2", "B", "u", "T", category="blog")
        ids = store.filter_by("category", "news")
        assert ids == ["1"]

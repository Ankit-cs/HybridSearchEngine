import json
import tempfile

from src.indexer.inverted_index import InvertedIndex
from src.indexer.index_writer import write_index
from src.storage.document_store import DocumentStore
from src.query.search import SearchEngine


def test_search_returns_results():
    # --- build small index ---
    index = InvertedIndex()
    index.add_document(1, ["india", "country"])
    index.add_document(2, ["india", "economy"])

    with tempfile.TemporaryDirectory() as tmp:
        index_path = f"{tmp}/index.json"
        docs_path = f"{tmp}/docs.json"
        meta_path = f"{tmp}/meta.json"

        write_index(index, index_path)

        store = DocumentStore()
        store.add(1, "India")
        store.add(2, "Indian Economy")
        store.save(docs_path)

        with open(meta_path, "w") as f:
            json.dump({"total_docs": 2}, f)

        engine = SearchEngine(index_path, docs_path, total_docs=2)

        results = engine.search("india")

        assert len(results) == 2
        assert results[0][1] in ["India", "Indian Economy"]

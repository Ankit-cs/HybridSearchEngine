from src.indexer.inverted_index import InvertedIndex


def test_inverted_index_basic():
    index = InvertedIndex()

    index.add_document(1, ["india", "country"])
    index.add_document(2, ["india", "economy"])

    postings = index.get_postings("india")

    assert postings[1] == 1
    assert postings[2] == 1
    assert "economy" in index.index

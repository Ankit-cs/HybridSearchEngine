import json

"""
reads inverted index and gives access to postings
"""
class IndexReader:
    def __init__(self, index_path: str):
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

    def get_postings(self, term: str) -> dict:
        return self.index.get(term, {})

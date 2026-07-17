import math
from collections import defaultdict

class TFIDFRanker:

    def __init__(self, index_reader, total_docs: int):
        self.index_reader = index_reader
        self.total_docs = total_docs

    def score(self, query_tokens: list[str]) -> dict:
        """
        Returns doc_id -> tf-idf score
        """
        scores = defaultdict(float)

        for term in query_tokens:
            
            postings = self.index_reader.get_postings(term)
            if not postings:
                continue

            df = len(postings)

            idf = math.log((self.total_docs + 1) / (df + 1)) + 1

            for doc_id, tf in postings.items():
                scores[doc_id] += tf * idf

        return scores

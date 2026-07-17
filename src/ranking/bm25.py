import math
from collections import defaultdict
from src.utils.config import TITLE_BOOST

class BM25Ranker:
    def __init__(self, body_index, title_index, metadata, k1=1.5, b=0.75):
        #self.index_reader = index_reader
        self.total_docs = metadata["total_docs"]
        self.avg_doc_length = metadata["avg_doc_length"]
        self.doc_lengths = metadata["doc_lengths"]
        self.body_index = body_index
        self.title_index = title_index

        # k1: term saturation (1.2–2.0)
        # b: length normalization (0.75 standard)
        self.k1 = k1
        self.b = b

    def score(self, query_tokens):

        #body score
        body_scores = defaultdict(float)
        
        for term in query_tokens:
            postings = self.body_index.get_postings(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(str(doc_id), 0)

                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_length))
                score = idf * (tf * (self.k1 + 1)) / denom

                body_scores[doc_id] += score


        #title score    
        title_scores = defaultdict(float)

        for term in query_tokens:
            postings = self.title_index.get_postings(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in postings.items():
                dl = 1  # titles are short → treat as 1

                denom = tf + self.k1
                score = idf * (tf * (self.k1 + 1)) / denom

                title_scores[doc_id] += score    


        #combine calculation (body_score + (title_score)*TITLE_BOOST)
        final_scores = defaultdict(float)

        for doc_id, s in body_scores.items():
            final_scores[doc_id] += s

        for doc_id, s in title_scores.items():
            final_scores[doc_id] += s * TITLE_BOOST

        return final_scores

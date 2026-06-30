from collections import defaultdict

class InvertedIndex:

    def __init__(self):
        #term -> doc_id : frequency
        #index[term][doc_id] = frequency

        self.index = defaultdict(dict)

    def add_document(self, doc_id: int, tokens: list[str]):
        
        """"
        add document(page) to inverted index
        """

        for token in tokens:

            if doc_id not in self.index[token]:
                self.index[token][doc_id] = 1

            else:
                self.index[token][doc_id] += 1        
    
    def get_postings(self, term: str) -> dict[int, int]:
        """
        Return posting list for a term.
        """
        return self.index.get(term, {})

    def __len__(self):
        """
        Number of unique terms in the index.
        """
        return len(self.index)
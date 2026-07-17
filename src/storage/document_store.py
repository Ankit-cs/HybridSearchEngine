import pandas as pd

class DocumentStore:
    def __init__(self):
        self.docs = {}

    def add(self, doc_id, title, url, text):
        self.docs[str(doc_id)] = {
            "title": title,
            "url": url,
            "text": text
        }
    
    def get(self, doc_id):
        return self.docs.get(str(doc_id))

    def save(self, path):
        # Convert dict to DataFrame and save as Parquet
        df = pd.DataFrame.from_dict(self.docs, orient='index')
        df.index.name = 'doc_id'
        df.to_parquet(path)

    def load(self, path):
        # Load Parquet file back to dict
        df = pd.read_parquet(path)
        # Reset index if doc_id was saved as index
        if df.index.name == 'doc_id' or 'doc_id' not in df.columns:
            self.docs = df.to_dict(orient='index')
        else:
            self.docs = df.set_index('doc_id').to_dict(orient='index')
        # ensure keys are strings
        self.docs = {str(k): v for k, v in self.docs.items()}

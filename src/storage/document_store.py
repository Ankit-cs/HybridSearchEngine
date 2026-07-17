import json


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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

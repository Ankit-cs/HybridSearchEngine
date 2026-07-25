import csv
from src.parser.base_parser import BaseParser


class MsmarcoParser(BaseParser):

    def parse(self, tsv_path):
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = parts[1]
                    title = f"Document {doc_id}"
                    url = f"doc://{doc_id}"
                    
                    if not text:
                        continue
                        
                    # Return integer doc_id if possible, otherwise string
                    try:
                        doc_id = int(doc_id)
                    except ValueError:
                        pass
                        
                    yield doc_id, title, text, url

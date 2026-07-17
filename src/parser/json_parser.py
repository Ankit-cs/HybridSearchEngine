import json
from src.parser.base_parser import BaseParser


class JsonParser(BaseParser):

    def parse(self, json_path):

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        doc_id = 1

        for item in data:

            title = item.get("title", f"Document {doc_id}")
            text = item.get("text", "")
            url = item.get("url", f"doc://{doc_id}")

            if not text:
                continue

            yield doc_id, title, text, url

            doc_id += 1

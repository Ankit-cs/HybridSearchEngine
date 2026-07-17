import csv
from src.parser.base_parser import BaseParser


class CsvParser(BaseParser):

    def parse(self, csv_path):

        doc_id = 1

        with open(csv_path, encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                title = row.get("title", f"Document {doc_id}")
                text = row.get("text", "")
                url = row.get("url", f"doc://{doc_id}")

                if not text:
                    continue

                yield doc_id, title, text, url

                doc_id += 1

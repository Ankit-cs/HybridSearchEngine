import os
from src.parser.base_parser import BaseParser


class TextParser(BaseParser):

    def parse(self, folder_path):

        doc_id = 1

        for filename in os.listdir(folder_path):

            path = os.path.join(folder_path, filename)

            if not filename.endswith(".txt"):
                continue

            with open(path, encoding="utf-8") as f:
                text = f.read()

            title = filename
            url = f"file://{path}"

            yield doc_id, title, text, url

            doc_id += 1

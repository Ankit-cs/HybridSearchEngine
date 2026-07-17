from src.query.search import SearchEngine
from src.utils.config import (
    INVERTED_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH
)
import json


def main():
    # load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    total_docs = metadata["total_docs"]

    engine = SearchEngine(
        index_path=INVERTED_INDEX_PATH,
        doc_store_path=DOCUMENT_STORE_PATH,
        total_docs=total_docs
    )

    while True:
        query = input("search> ")
        if query.strip().lower() in ("exit", "quit"):
            break

        results = engine.search(query)

        for doc_id, title, score in results:
            print(f"{doc_id} | {title} | {score}")



if __name__ == "__main__":
    main()

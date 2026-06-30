import argparse
import json


from src.semantic.embedding_model import EmbeddingModel
from src.semantic.embedding_store import EmbeddingStore
from src.utils.config import EMBEDDINGS_PATH

from src.parser.factory import get_parser, detect_parser
from src.parser.factory import get_parser
from src.preprocessing.cleaner import clean_wiki_text
from src.preprocessing.tokenizer import tokenize
from src.indexer.inverted_index import InvertedIndex
from src.indexer.index_writer import write_index
from src.storage.document_store import DocumentStore
from src.utils.config import (
    INVERTED_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH,
    TITLE_INDEX_PATH
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main():

    parser = argparse.ArgumentParser(description="Build AstraSearch Index")

    parser.add_argument(
        "--parser",
        type=str,
        required=False,
        help="Parser type (optional: auto-detected if not provided)"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to dataset (file or folder)"
    )


    args = parser.parse_args()

    source_path = args.source

    if args.parser:
        parser_type = args.parser
    else:
        parser_type = detect_parser(source_path)

    logger.info(f"Using parser type: {parser_type}")


    doc_lengths = {}
    total_length = 0
    total_docs = 0

    index = InvertedIndex()
    title_index = InvertedIndex()
    doc_store = DocumentStore()

    # --- Semantic embedding components ---
    embedding_model = EmbeddingModel()
    embedding_store = EmbeddingStore()


    # Get appropriate parser dynamically
    data_parser = get_parser(parser_type)

    for doc_id, title, text, url in data_parser.parse(source_path):

        tokens = tokenize(text)

        if not tokens:
            continue

        index.add_document(doc_id, tokens)
        doc_store.add(doc_id, title, url, text)

        # --- Generate semantic embedding ---
        short_text = text[:512]   # first 512 characters only
        embedding = embedding_model.encode(short_text)[0]
        embedding_store.add(doc_id, embedding)

        total_docs += 1

        if total_docs % 100 == 0:
            logger.info(f"Generated embeddings for {total_docs} docs")

        if total_docs % 1000 == 0:
            logger.info(f"Indexed {total_docs} documents")

        doc_len = len(tokens)
        doc_lengths[doc_id] = doc_len
        total_length += doc_len

        # Title weighting index
        title_tokens = tokenize(title)

        if title_tokens:
            title_index.add_document(doc_id, title_tokens)

    # Write indexes to disk
    write_index(index, INVERTED_INDEX_PATH)
    write_index(title_index, TITLE_INDEX_PATH)

    doc_store.save(DOCUMENT_STORE_PATH)

    avg_doc_length = total_length / total_docs if total_docs else 0

    metadata = {
        "total_docs": total_docs,
        "avg_doc_length": avg_doc_length,
        "doc_lengths": doc_lengths
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    # Save semantic embeddings
    embedding_store.save(EMBEDDINGS_PATH)

    logger.info("Indexing completed successfully")
    logger.info(f"Total documents indexed: {total_docs}")


if __name__ == "__main__":
    main()

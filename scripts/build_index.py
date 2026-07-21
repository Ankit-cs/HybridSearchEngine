import argparse
import json
import uuid
import time
import numpy as np

from src.semantic.embedding_model import EmbeddingModel
from src.semantic.embedding_store import EmbeddingStore
from src.semantic.dual_embeddings import DualEmbeddingGenerator, DualEmbeddingStore
from src.utils.config import (
    EMBEDDINGS_PATH,
    CONTEXT_EMBEDDINGS_PATH,
    CONTEXT_ID_MAP_PATH,
    GEOMETRIC_STATS_PATH,
    CATALOG_DB_PATH,
    FTS_INDEX_DIR,
)

from src.parser.factory import get_parser, detect_parser
from src.preprocessing.cleaner import clean_wiki_text
from src.preprocessing.tokenizer import tokenize
from src.indexer.inverted_index import InvertedIndex
from src.indexer.index_writer import write_index
from src.indexer.compression import GeometricPruner, VectorQuantizer
from src.indexer.fts import PersistentFTSIndex
from src.storage.document_store import DocumentStore
from src.storage.catalog import Catalog, FileManifest
from src.storage.schema import LLMContextSchema, ChunkMetadata
from src.utils.config import (
    INVERTED_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH,
    TITLE_INDEX_PATH,
    INDEX_DIR,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build AstraSearch Index")
    parser.add_argument("--parser", type=str, required=False)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--precision", type=str, default="f32",
                        choices=["f32", "f16", "i8"],
                        help="Vector quantization precision")
    parser.add_argument("--use-dual-embeddings", action="store_true",
                        help="Generate dual embeddings (content + context)")
    parser.add_argument("--use-fts", action="store_true",
                        help="Build persistent FTS index")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Batch ID for idempotent writes")
    parser.add_argument("--agent-id", type=str, default=None,
                        help="Agent ID for partition isolation")
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

    embedding_model = EmbeddingModel()
    embedding_store = EmbeddingStore()
    dual_generator = DualEmbeddingGenerator(embedding_model) if args.use_dual_embeddings else None
    dual_store = DualEmbeddingStore() if args.use_dual_embeddings else None

    fts_index = PersistentFTSIndex(str(FTS_INDEX_DIR)) if args.use_fts else None
    geometric_pruner = GeometricPruner(str(GEOMETRIC_STATS_PATH))
    quantizer = VectorQuantizer(precision=args.precision) if args.precision != "f32" else None

    catalog = Catalog(str(CATALOG_DB_PATH))
    snapshot = catalog.create_snapshot(f"Build from {source_path}")

    if args.batch_id:
        from src.storage.acid import TransactionManager, IdempotentWriter
        from src.utils.config import INDEX_DIR as idx_dir
        tm = TransactionManager(catalog, idx_dir)
        idempotent = IdempotentWriter(tm)
        if not idempotent.should_write(args.batch_id):
            logger.warning(f"Batch {args.batch_id} already written. Skipping.")
            return
        tx = tm.begin_transaction(args.batch_id)
    else:
        tx = None
        tm = None
        idempotent = None

    all_embeddings = []
    all_doc_ids = []

    data_parser = get_parser(parser_type)
    vectors_for_centroids = []

    for doc_id, title, text, url in data_parser.parse(source_path):
        text_lower = text.lower()
        title_lower = title.lower()

        india_keywords = [
            "india", "indian", "bharat", "hindustan", "modi", "gandhi", "nehru",
            "kalam", "vajpayee", "ashoka", "maurya", "gupta empire", "mughal",
            "chola", "maratha", "delhi", "mumbai", "bombay", "kolkata", "calcutta",
            "chennai", "madras", "bangalore", "bengaluru", "isro", "bollywood",
            "hinduism", "buddhism", "jainism", "sikhism", "vedic", "harappa",
            "mohenjo-daro", "indus valley", "east india company", "british raj",
            "mahabharata", "ramayana", "sanskrit", "hindi", "tamil", "telugu"
        ]

        is_indian = any(kw in title_lower or kw in text_lower for kw in india_keywords)
        if not is_indian:
            continue

        tokens = tokenize(text)
        if not tokens:
            continue

        index.add_document(doc_id, tokens)
        doc_store.add(doc_id, title, url, text)

        chunk_meta = ChunkMetadata(
            document_id=str(doc_id),
            document_title=title,
            source_uri=url,
            chunk_index=0,
            total_chunks=1,
        )

        content_emb = embedding_model.encode(text[:512])[0]
        embedding_store.add(doc_id, content_emb)
        all_embeddings.append(content_emb)
        all_doc_ids.append(doc_id)

        if dual_generator and dual_store:
            dual_result = dual_generator.generate(
                text[:512], title=title, summary=title
            )
            dual_store.add(
                doc_id,
                dual_result.content_embedding,
                dual_result.context_embedding,
            )

        if fts_index:
            fts_index.index_document(str(doc_id), tokens)

        total_docs += 1
        if total_docs % 100 == 0:
            logger.info(f"Indexed {total_docs} docs")

        doc_len = len(tokens)
        doc_lengths[doc_id] = doc_len
        total_length += doc_len

        title_tokens = tokenize(title)
        if title_tokens:
            title_index.add_document(doc_id, title_tokens)

        vectors_for_centroids.append(content_emb)

    if all_embeddings:
        all_embs_np = np.array(all_embeddings, dtype=np.float32)
        centroid, radius = geometric_pruner.compute_centroid(all_embs_np)
        geometric_pruner.register_file(
            str(DOCUMENT_STORE_PATH),
            centroid, radius, total_docs
        )

    write_index(index, INVERTED_INDEX_PATH)
    write_index(title_index, TITLE_INDEX_PATH)
    doc_store.save(DOCUMENT_STORE_PATH)

    avg_doc_length = total_length / total_docs if total_docs else 0
    metadata = {
        "total_docs": total_docs,
        "avg_doc_length": avg_doc_length,
        "doc_lengths": doc_lengths,
        "precision": args.precision,
        "use_dual_embeddings": args.use_dual_embeddings,
        "use_fts": args.use_fts,
        "created_at": time.time(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    if quantizer and all_embeddings:
        all_embs_np = np.array(all_embeddings, dtype=np.float32)
        quantized = quantizer.quantize(all_embs_np)
        quant_path = str(INDEX_DIR / f"vectors_quantized_{args.precision}.npy")
        np.save(quant_path, quantized)
        logger.info(f"Saved quantized vectors ({args.precision}): {quant_path}")

    embedding_store.save(EMBEDDINGS_PATH)

    if dual_store:
        dual_store.save(
            str(CONTEXT_EMBEDDINGS_PATH),
            str(CONTEXT_EMBEDDINGS_PATH).replace("context_embeddings", "context_context_embeddings"),
            str(CONTEXT_ID_MAP_PATH),
            str(CONTEXT_ID_MAP_PATH).replace("context_id_map", "context_context_id_map"),
        )
        logger.info("Saved dual embeddings (content + context)")

    if fts_index:
        fts_index.commit()
        logger.info(f"Saved FTS index: {fts_index.term_count} terms, {fts_index.doc_count} docs")

    file_manifest = FileManifest(
        file_path=str(DOCUMENT_STORE_PATH),
        doc_count=total_docs,
        centroid=centroid if all_embeddings else None,
        radius=radius if all_embeddings else 0.0,
        created_at=time.time(),
        batch_id=args.batch_id,
        index_type="hnsw",
    )
    catalog.add_file(snapshot.snapshot_id, file_manifest)

    if tx and tm:
        tx.files_to_add = [{
            "path": str(DOCUMENT_STORE_PATH),
            "doc_count": total_docs,
        }]
        tm.commit_transaction(tx, snapshot)

    if idempotent and args.batch_id:
        idempotent.mark_written(args.batch_id, total_docs)

    catalog.close()

    logger.info("Indexing completed successfully")
    logger.info(f"Total documents indexed: {total_docs}")
    logger.info(f"Vector precision: {args.precision}")
    logger.info(f"Dual embeddings: {args.use_dual_embeddings}")
    logger.info(f"Persistent FTS: {args.use_fts}")
    logger.info(f"Catalog snapshot: v{snapshot.version}")


if __name__ == "__main__":
    main()

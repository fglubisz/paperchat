"""Embedding and vector store layer using ChromaDB + sentence-transformers."""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from paperchat.chunk import Chunk

DEFAULT_DB_PATH = ".chroma"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, solid baseline


def _get_collection(db_path: str = DEFAULT_DB_PATH):
    """Open (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=db_path)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
    )


def add_chunks(chunks: list[Chunk], db_path: str = DEFAULT_DB_PATH) -> None:
    """Embed and store chunks. IDs are deterministic so re-indexing is idempotent."""
    if not chunks:
        return

    collection = _get_collection(db_path)

    ids = [f"{c.source}::p{c.page_number}::c{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"source": c.source, "page_number": c.page_number, "chunk_index": c.chunk_index}
        for c in chunks
    ]

    # upsert = insert or replace, so re-running `index` on the same folder is safe
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def reset_index(db_path: str = DEFAULT_DB_PATH) -> None:
    """Delete and recreate the collection. Useful for testing or full re-index."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist


def collection_size(db_path: str = DEFAULT_DB_PATH) -> int:
    """How many chunks are currently indexed."""
    return _get_collection(db_path).count()
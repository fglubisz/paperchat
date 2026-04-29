"""Embedding and vector store layer using ChromaDB + sentence-transformers."""
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from paperchat.chunk import Chunk
from dataclasses import dataclass

DEFAULT_DB_PATH = ".chroma"
COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, solid baseline

@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    source: str
    page_number: int
    chunk_index: int
    text: str
    distance: float  # lower = more similar

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

def search(query: str, k: int = 5, db_path: str = DEFAULT_DB_PATH) -> list[SearchResult]:
    """Find the k chunks most relevant to the query."""
    collection = _get_collection(db_path)

    if collection.count() == 0:
        return []

    raw = collection.query(query_texts=[query], n_results=k)

    results = []
    # Chroma returns parallel lists wrapped in an outer list (one per query).
    # We only sent one query, so we index [0] on each.
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(SearchResult(
            source=meta["source"],
            page_number=meta["page_number"],
            chunk_index=meta["chunk_index"],
            text=doc,
            distance=dist,
        ))
    return results
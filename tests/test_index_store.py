import pytest
from paperchat.chunk import Chunk
from paperchat.index_store import add_chunks, collection_size, reset_index

@pytest.fixture
def temp_db(tmp_path):
    """Use a throwaway DB path per test to avoid polluting the real one."""
    return str(tmp_path / "chroma_test")

def test_add_and_count(temp_db):
    chunks = [
        Chunk(source="a.pdf", page_number=1, chunk_index=0, text="hello world"),
        Chunk(source="a.pdf", page_number=1, chunk_index=1, text="goodbye world"),
    ]
    add_chunks(chunks, db_path=temp_db)
    assert collection_size(db_path=temp_db) == 2

def test_upsert_is_idempotent(temp_db):
    """Re-adding the same chunks should not create duplicates."""
    chunks = [Chunk(source="a.pdf", page_number=1, chunk_index=0, text="hello")]
    add_chunks(chunks, db_path=temp_db)
    add_chunks(chunks, db_path=temp_db)
    assert collection_size(db_path=temp_db) == 1

def test_empty_chunks_is_noop(temp_db):
    add_chunks([], db_path=temp_db)
    # No collection should have been created if we never added anything,
    # but if it was, it should be empty.
    # (Either way, this should not error.)

def test_reset(temp_db):
    chunks = [Chunk(source="a.pdf", page_number=1, chunk_index=0, text="hello")]
    add_chunks(chunks, db_path=temp_db)
    assert collection_size(db_path=temp_db) == 1
    reset_index(db_path=temp_db)
    assert collection_size(db_path=temp_db) == 0
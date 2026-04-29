import pytest

from paperchat.chunk import Chunk, chunk_page, chunk_pages
from paperchat.extract import Page


def test_short_page_yields_one_chunk():
    page = Page(source="a.pdf", page_number=1, text="short text")
    chunks = chunk_page(page, chunk_size=2000, overlap=400)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].chunk_index == 0


def test_long_page_yields_overlapping_chunks():
    long_text = "x" * 5000
    page = Page(source="a.pdf", page_number=1, text=long_text)
    chunks = chunk_page(page, chunk_size=2000, overlap=400)
    assert len(chunks) >= 2
    # Verify overlap: end of chunk[0] should overlap with start of chunk[1]
    assert chunks[0].text[-400:] == chunks[1].text[:400]


def test_invalid_overlap_raises():
    page = Page(source="a.pdf", page_number=1, text="hello")
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_page(page, chunk_size=100, overlap=100)


def test_chunk_pages_preserves_provenance():
    pages = [
        Page(source="a.pdf", page_number=1, text="page one"),
        Page(source="a.pdf", page_number=2, text="page two"),
    ]
    chunks = chunk_pages(pages)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.source == "a.pdf" for c in chunks)
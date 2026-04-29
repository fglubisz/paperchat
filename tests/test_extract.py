from pathlib import Path

from paperchat.extract import Page, extract_folder


def test_extract_folder_finds_pdfs(tmp_path):
    """extract_folder should raise on empty folder."""
    import pytest
    with pytest.raises(ValueError, match="No PDFs found"):
        extract_folder(tmp_path)


def test_page_dataclass():
    """Page should hold source, page_number, and text."""
    p = Page(source="test.pdf", page_number=1, text="hello")
    assert p.source == "test.pdf"
    assert p.page_number == 1
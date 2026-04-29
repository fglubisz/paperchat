"""Split extracted pages into overlapping chunks for retrieval."""
from dataclasses import dataclass
from paperchat.extract import Page


@dataclass
class Chunk:
    """A retrievable chunk of text with provenance back to its source."""
    source: str       # filename, e.g. "bmw_n55_combustion.pdf"
    page_number: int  # 1-indexed
    chunk_index: int  # 0-indexed within the page
    text: str

def chunk_page(page: Page, chunk_size: int = 2000, overlap: int = 400) -> list[Chunk]:
    """Split a single page's text into overlapping character windows.

    chunk_size and overlap are in characters. ~4 chars ≈ 1 token, so 2000 chars
    is roughly 500 tokens — a good size for embedding and retrieval.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = page.text
    if len(text) <= chunk_size:
        return [Chunk(
            source=page.source,
            page_number=page.page_number,
            chunk_index=0,
            text=text,
        )]

    chunks = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(text), step)):
        window = text[start:start + chunk_size]
        if not window.strip():
            continue
        chunks.append(Chunk(
            source=page.source,
            page_number=page.page_number,
            chunk_index=i,
            text=window,
        ))
        if start + chunk_size >= len(text):
            break
    return chunks

def chunk_pages(pages: list[Page], chunk_size: int = 2000, overlap: int = 400) -> list[Chunk]:
    """Chunk a whole list of pages."""
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_size, overlap))
    return chunks
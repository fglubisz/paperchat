"""PDF text extraction with page tracking."""
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Page:
    """A single page of extracted text from a PDF."""
    source: str       # filename, e.g. "bmw_n55_combustion.pdf"
    page_number: int  # 1-indexed
    text: str

def _clean_text(text: str) -> str:
    """Strip unicode replacement characters and normalise whitespace.

    pypdf produces \\uFFFD ('￿') for characters it can't decode — common in
    PDFs with embedded fonts (e.g. technical manuals). These add noise to
    embeddings without semantic value.
    """
    text = text.replace("\uFFFD", " ")
    # Collapse runs of whitespace (the substitution above creates double spaces)
    text = " ".join(text.split())
    return text

def extract_pdf(path: Path) -> list[Page]:
    """Extract text from every page of a PDF, preserving page numbers."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if text:  # skip empty pages (e.g. blank covers, image-only pages)
            pages.append(Page(source=path.name, page_number=i, text=text))
    return pages

def extract_folder(folder: Path) -> list[Page]:
    """Extract text from every PDF in a folder."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found in {folder}")
    all_pages = []
    for pdf in pdfs:
        all_pages.extend(extract_pdf(pdf))
    return all_pages
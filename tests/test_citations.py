from paperchat.citations import (
    find_cited_indices,
    render_inline_citations,
    used_sources,
)
from paperchat.index_store import SearchResult

def _make_result(source: str, page: int) -> SearchResult:
    return SearchResult(
        source=source, page_number=page, chunk_index=0,
        text="...", distance=0.5,
    )

def test_find_cited_indices_basic():
    text = "The engine uses direct injection [1] and dual turbos [2]."
    assert find_cited_indices(text) == [1, 2]

def test_find_cited_indices_dedupes_and_preserves_order():
    text = "First [3], then [1], then [3] again, then [2]."
    assert find_cited_indices(text) == [3, 1, 2]

def test_find_cited_indices_ignores_non_numeric_brackets():
    text = "See [Smith 2020] and figure [a]. Citation [1] is valid."
    assert find_cited_indices(text) == [1]

def test_render_inline_citations_replaces_numbers():
    sources = [
        _make_result("a.pdf", 7),
        _make_result("b.pdf", 12),
    ]
    text = "Claim one [1]. Claim two [2]."
    result = render_inline_citations(text, sources)
    assert "[1] (a.pdf p.7)" in result
    assert "[2] (b.pdf p.12)" in result

def test_render_handles_unknown_indices_gracefully():
    """If Claude cites [5] but we only have 2 sources, leave [5] untouched."""
    sources = [_make_result("a.pdf", 1), _make_result("b.pdf", 2)]
    text = "Real claim [1]. Bogus claim [5]."
    result = render_inline_citations(text, sources)
    assert "[1] (a.pdf p.1)" in result
    assert "[5]" in result  # left as-is
    assert "(p." not in result.split("[5]")[1][:20]  # nothing hallucinated after [5]

def test_used_sources_returns_only_cited():
    sources = [
        _make_result("a.pdf", 1),
        _make_result("b.pdf", 2),
        _make_result("c.pdf", 3),
    ]
    text = "Citing [1] and [3] only."
    used = used_sources(text, sources)
    assert len(used) == 2
    assert used[0].source == "a.pdf"
    assert used[1].source == "c.pdf"

def test_used_sources_empty_when_no_citations():
    sources = [_make_result("a.pdf", 1)]
    text = "I cannot answer that from the provided context."
    assert used_sources(text, sources) == []
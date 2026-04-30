"""Parse and render citations in answer text."""
import re
from paperchat.index_store import SearchResult

# Matches [1], [2], [10], etc. — but not [foo] or [1.2]
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def find_cited_indices(answer_text: str) -> list[int]:
    """Return all unique citation numbers used in the answer, in order of first appearance."""
    seen = []
    for match in CITATION_PATTERN.finditer(answer_text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen

def render_inline_citations(answer_text: str, sources: list[SearchResult]) -> str:
    """Replace [N] with [N] (file.pdf p.X) inline in the answer.

    If [N] points to an index outside `sources`, leave it untouched
    (defensive — Claude occasionally hallucinates citation numbers).
    """
    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(sources):
            src = sources[n - 1]
            return f"[{n}] ({src.source} p.{src.page_number})"
        return match.group(0)  # leave unknown refs untouched

    return CITATION_PATTERN.sub(replace, answer_text)

def used_sources(answer_text: str, sources: list[SearchResult]) -> list[SearchResult]:
    """Return only the sources that were actually cited in the answer.

    Useful for showing a focused 'sources used' list rather than all retrieved chunks.
    """
    cited_indices = find_cited_indices(answer_text)
    used = []
    for n in cited_indices:
        if 1 <= n <= len(sources):
            used.append(sources[n - 1])
    return used
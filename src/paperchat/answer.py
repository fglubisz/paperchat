"""Generate grounded answers from retrieved chunks using Claude."""
import os
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

from paperchat.index_store import SearchResult

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a research assistant answering questions about technical documents.

You will be given a user's question and a set of numbered excerpts retrieved from the documents.

Rules:
1. Answer ONLY using information in the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer the question, say so explicitly. Do not guess.
3. Cite excerpts using their bracketed numbers, e.g. [1], [2]. Cite every factual claim.
4. Be concise. Prefer 2-4 sentences unless the question genuinely needs more.
5. If excerpts conflict, note the conflict rather than picking one arbitrarily."""


@dataclass
class Answer:
    """A generated answer plus the chunks used to produce it."""
    text: str
    sources: list[SearchResult]


def _format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks as a numbered list for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        header = f"[{i}] ({r.source}, p.{r.page_number})"
        blocks.append(f"{header}\n{r.text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, results: list[SearchResult]) -> Answer:
    """Send the question and retrieved chunks to Claude, return the answer."""
    if not results:
        return Answer(
            text="No documents have been indexed yet. Run `paperchat index <folder>` first.",
            sources=[],
        )

    context = _format_context(results)
    user_message = (
        f"Question: {question}\n\n"
        f"Excerpts:\n{context}"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to a .env file in the project root."
        )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text
    return Answer(text=text, sources=results)
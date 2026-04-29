# paperchat

A CLI tool to chat with technical PDFs using RAG (retrieval-augmented generation). Point it at a folder of papers, ask questions in natural language, and get answers grounded in the source material with citations.

Built as an experiment in applied LLMs — originally tested against a corpus of automotive engineering papers (BMW engine design, combustion analysis, drivetrain research).

## Status
🚧 In progress

## Stack
- Python 3.10+
- `pypdf` for text extraction
- `sentence-transformers` for local embeddings
- `chromadb` as the vector store
- Anthropic Claude for answer generation
- `typer` + `rich` for the CLI
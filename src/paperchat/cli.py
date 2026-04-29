from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Chat with technical PDFs using RAG.")
console = Console()

@app.command()
def index(
    folder: str,
    reset: bool = typer.Option(False, "--reset", help="Wipe the index before adding."),
):
    """Index all PDFs in a folder."""
    from paperchat.extract import extract_folder
    from paperchat.chunk import chunk_pages
    from paperchat.index_store import add_chunks, reset_index, collection_size

    if reset:
        reset_index()
        console.print("[yellow]⚠[/yellow]  Index wiped.")

    with console.status("[cyan]Extracting PDFs..."):
        pages = extract_folder(Path(folder))

    with console.status("[cyan]Chunking..."):
        chunks = chunk_pages(pages)

    with console.status(f"[cyan]Embedding {len(chunks)} chunks (first run downloads the model)..."):
        add_chunks(chunks)

    sources = {p.source for p in pages}
    total = collection_size()
    console.print(
        f"[green]✓[/green] Indexed {len(chunks)} chunks "
        f"from {len(sources)} PDFs ({len(pages)} pages). "
        f"Collection now has {total} chunks total."
    )

@app.command()
def ask(question: str):
    """Ask a question against the indexed papers."""
    typer.echo(f"TODO: answer '{question}'")

if __name__ == "__main__":
    app()
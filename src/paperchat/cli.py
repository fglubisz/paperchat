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
def ask(
    question: str,
    k: int = typer.Option(5, "--k", help="Number of chunks to retrieve."),
):
    """Ask a question against the indexed papers."""
    from paperchat.index_store import search

    results = search(question, k=k)

    if not results:
        console.print("[red]No results — have you run `paperchat index` yet?[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Top {len(results)} results for:[/bold] {question}\n")
    for i, r in enumerate(results, start=1):
        console.print(
            f"[cyan]#{i}[/cyan] "
            f"[dim]{r.source} p.{r.page_number} chunk {r.chunk_index} "
            f"(distance {r.distance:.3f})[/dim]"
        )
        # Truncate long chunks for readability in the terminal
        preview = r.text[:400].replace("\n", " ")
        if len(r.text) > 400:
            preview += "…"
        console.print(f"  {preview}\n")

if __name__ == "__main__":
    app()
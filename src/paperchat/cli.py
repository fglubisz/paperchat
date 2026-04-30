from pathlib import Path
import typer
from rich.console import Console
import os
import warnings

# Quiet down the noisy first-time imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

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

    folder_path = Path(folder)
    if not folder_path.exists():
        console.print(f"[red]✗ Folder not found:[/red] {folder}")
        raise typer.Exit(code=1)
    if not folder_path.is_dir():
        console.print(f"[red]✗ Not a directory:[/red] {folder}")
        raise typer.Exit(code=1)

    if reset:
        reset_index()
        console.print("[yellow]⚠[/yellow]  Index wiped.")

    try:
        with console.status("[cyan]Extracting PDFs..."):
            pages = extract_folder(folder_path)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1)

    with console.status("[cyan]Chunking..."):
        chunks = chunk_pages(pages)

    with console.status(
        f"[cyan]Embedding {len(chunks)} chunks (first run downloads the model)..."
    ):
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
    show_sources: bool = typer.Option(
        False, "--show-sources", help="Print the retrieved chunks alongside the answer."
    ),
):
    """Ask a question against the indexed papers."""
    from paperchat.index_store import search
    from paperchat.answer import generate_answer
    from paperchat.citations import render_inline_citations, used_sources

    with console.status("[cyan]Searching..."):
        results = search(question, k=k)

    if not results:
        console.print(
            "[red]✗ Index is empty.[/red] Run `paperchat index <folder>` first."
        )
        raise typer.Exit(code=1)

    try:
        with console.status("[cyan]Asking Claude..."):
            answer = generate_answer(question, results)
    except RuntimeError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch API errors (auth, rate limits, billing, etc.) with a friendly message
        console.print(f"[red]✗ Claude API error:[/red] {e}")
        raise typer.Exit(code=1)

    # Render citations inline
    answer_with_citations = render_inline_citations(answer.text, answer.sources)
    cited = used_sources(answer.text, answer.sources)

    console.print(f"\n[bold]Question:[/bold] {question}\n")
    console.print(f"[bold green]Answer:[/bold green]\n{answer_with_citations}\n")

    if cited:
        console.print("[bold]Sources used:[/bold]")
        for i, r in enumerate(answer.sources, start=1):
            if r in cited:
                console.print(
                    f"  [cyan][{i}][/cyan] {r.source} p.{r.page_number} "
                    f"[dim](distance {r.distance:.3f})[/dim]"
                )
    else:
        console.print(
            "[dim]No citations in answer. Retrieved chunks were:[/dim]"
        )
        for i, r in enumerate(answer.sources, start=1):
            console.print(
                f"  [dim][{i}] {r.source} p.{r.page_number} "
                f"(distance {r.distance:.3f})[/dim]"
            )

    if show_sources:
        console.print("\n[bold]Retrieved excerpts:[/bold]")
        for i, r in enumerate(answer.sources, start=1):
            preview = r.text[:300].replace("\n", " ")
            if len(r.text) > 300:
                preview += "…"
            console.print(f"\n[cyan][{i}][/cyan] {preview}")

@app.command()
def status():
    """Show what's currently indexed."""
    from paperchat.index_store import collection_size, _get_collection

    count = collection_size()
    if count == 0:
        console.print("[yellow]Index is empty.[/yellow] Run `paperchat index <folder>` first.")
        return

    # Pull all metadata to see what files are in there
    collection = _get_collection()
    raw = collection.get(include=["metadatas"])
    sources = {}
    for meta in raw["metadatas"]:
        sources.setdefault(meta["source"], 0)
        sources[meta["source"]] += 1

    console.print(f"[green]✓[/green] {count} chunks indexed across {len(sources)} files:")
    for source, n in sorted(sources.items()):
        console.print(f"  [cyan]•[/cyan] {source} [dim]({n} chunks)[/dim]")

@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    ):
    """Wipe the entire index."""
    from paperchat.index_store import reset_index, collection_size

    count = collection_size()
    if count == 0:
        console.print("Index is already empty.")
        return

    if not yes:
        confirm = typer.confirm(f"Delete all {count} indexed chunks?")
        if not confirm:
            console.print("Cancelled.")
            return

    reset_index()
    console.print(f"[green]✓[/green] Deleted {count} chunks.")

if __name__ == "__main__":
    app()


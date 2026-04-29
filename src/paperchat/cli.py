from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Chat with technical PDFs using RAG.")
console = Console()

@app.command()
def index(folder: str):
    """Index all PDFs in a folder."""
    from paperchat.extract import extract_folder

    pages = extract_folder(Path(folder))
    sources = {p.source for p in pages}
    console.print(f"[green]✓[/green] Extracted {len(pages)} pages from {len(sources)} PDFs")

@app.command()
def ask(question: str):
    """Ask a question against the indexed papers."""
    typer.echo(f"TODO: answer '{question}'")

if __name__ == "__main__":
    app()
import typer

app = typer.Typer(help="Chat with academic PDFs using RAG.")

@app.command()
def index(folder: str):
    """Index all PDFs in a folder."""
    typer.echo(f"TODO: index {folder}")

@app.command()
def ask(question: str):
    """Ask a question against the indexed papers."""
    typer.echo(f"TODO: answer '{question}'")

if __name__ == "__main__":
    app()
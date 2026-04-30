# paperchat

A CLI tool for asking questions about technical PDFs, using local embeddings and the Claude API. Answers are grounded in retrieved excerpts and cite their source file and page number.


'''
$ paperchat ask "describe the oil supply system on the S58"
Answer:
Based on the provided excerpts, here is what can be 
determined about the S58 oil supply system:

**Components:**
The S58 uses a characteristic map-controlled external gear 
pump [4] (ST1926-S58-Engine.pdf p.44)[8] driven by an oil 
pump sprocket [4] (ST1926-S58-Engine.pdf p.44). The system 
includes an oil filter housing made from aluminum [5] 
(ST1926-S58-Engine.pdf p.64), an oil-level sensor [4] 
(ST1926-S58-Engine.pdf p.44), oil temperature sensor [4] 
(ST1926-S58-Engine.pdf p.44), and oil pressure sensor [4] 
(ST1926-S58-Engine.pdf p.44)[3] (ST1926-S58-Engine.pdf p.98).

**Oil Distribution:**
The system features main oil channels that supply oil to 
piston cooling components, including a piston cooling relay 
valve and piston cooling control valve [5] 
(ST1926-S58-Engine.pdf p.64). There are also channels to oil 
spray nozzles for piston cooling [5] (ST1926-S58-Engine.pdf 
p.64).

**Oil Cooling:**
A discrete engine oil cooler is mounted horizontally in front
of the radiator module [5] (ST1926-S58-Engine.pdf p.64). A 
thermostat at the oil filter housing controls oil flow to the
cooler based on engine oil temperature [5] 
(ST1926-S58-Engine.pdf p.64).

**Control:**
The system uses a map control valve [4] 
(ST1926-S58-Engine.pdf p.44) to regulate the characteristic 
map-controlled oil pump operation.

The excerpts show the physical layout and major components 
but do not provide complete details about operating 
pressures, flow rates, or the specific control algorithms 
used.

Sources used:
  [3] ST1926-S58-Engine.pdf p.98 (distance 0.606)
  [4] ST1926-S58-Engine.pdf p.44 (distance 0.607)
  [5] ST1926-S58-Engine.pdf p.64 (distance 0.632)
  '''

Built as an experiment in applied LLMs — a small, focused RAG (retrieval-augmented generation) pipeline that runs end-to-end on a laptop with no GPU, no managed services, and no embedding API costs.

## How it works
PDFs → text extraction → chunking → local embeddings → ChromaDB
                                                          │
                                                          ▼
                          Question → embedding → top-K retrieval
                                                          │
                                                          ▼
                              Claude (with grounding rules + cited excerpts)
                                                          │
                                                          ▼
                                           Answer with inline citations

Two stages:
Index time (offline, runs once per corpus): each PDF is parsed by pypdf with page numbers preserved, split into ~500-token overlapping chunks, embedded locally with sentence-transformers (all-MiniLM-L6-v2), and stored in a persistent ChromaDB collection. Chunks have deterministic IDs so re-indexing is idempotent.
Query time (per question): the question is embedded with the same model, the top-K nearest chunks are retrieved by vector similarity, and a system-prompted Claude call generates an answer constrained to the retrieved context. The prompt uses numbered excerpts so citations can be parsed back out and rendered inline.


## Quick start

'''
git clone https://github.com/fglubisz/paperchat.git
cd paperchat
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

paperchat index ./your-pdfs/
paperchat ask "your question here"
'''

First run downloads the embedding model (~80MB, cached after).

Commands
Command                             || What it does 
`paperchat index <folder>`          || Extract, chunk, embed, and store all PDFs in a folder
`paperchat index <folder> --reset`  || Wipe the index first, then re-
`paperchat ask "<question>"`        || Retrieve relevant chunks and answer with Claude
`paperchat ask "<q>" --show-sources`|| Also print the retrieved excerpts
`paperchat status`.                 || Show what's currently indexed
`paperchat reset`                   || Wipe the index


## Design decisions

**Local embeddings (`sentence-transformers`) over an embedding API.** Free, offline, no API key needed for indexing — the trade-off is slightly lower quality on technical text vs OpenAI's `text-embedding-3-small`. For a CLI that runs on a laptop, the trade is the right one.
**ChromaDB as the vector store.** Embedded, persistent, no server. The SQLite of vector databases. Swappable for FAISS / pgvector / Pinecone if scale ever justified it.
**Character-based chunking with overlap.** Pragmatic over precise: ~500 tokens is the right ballpark for retrieval, and chars-per-token is predictable enough in English. The 400-character overlap stops sentences being split across chunk boundaries.
**Page-number provenance carried end-to-end.** Every chunk knows its source PDF and page. Citations like `(file.pdf p.12)` are parsed from Claude's output and rendered inline, not inferred — so they're verifiable.
**Grounding via system prompt + numbered excerpts.** The model is told to answer only from context, refuse if context is insufficient, and cite excerpts by bracketed number. Tested explicitly with off-topic questions to confirm the refusal path.
**Mocked API in tests.** The Claude call is unit-tested with `unittest.mock.patch`, so the test suite runs deterministically, offline, with no API spend.

## Architecture
```
src/paperchat/
├── __init__.py        # package init, library quietening
├── cli.py             # Typer CLI entry point
├── extract.py         # PDF parsing with page tracking
├── chunk.py           # overlap-aware chunking
├── index_store.py     # ChromaDB + sentence-transformers
├── answer.py          # Claude prompting + grounded generation
└── citations.py       # parse and render [N] markers inline
tests/                 # unit tests for each module
```


## Stack
Python 3.10+ · `pypdf` · `sentence-transformers` · `chromadb` · `anthropic` · `typer` · `rich` · `python-dotenv` · `pytest`
## Licence
MIT.
# codebase-rag

Paste a GitHub repository URL, and ask natural-language questions about the code — with answers grounded in real excerpts and cited back to exact files and line numbers.

Built as a code intelligence assistant (in the spirit of GitHub Copilot Chat / Cursor / Sourcegraph Cody), not a generic document Q&A tool: answers are concise, structured, and avoid dumping raw code unless you actually ask for a specific file or function.

## How it works

```
GitHub URL
    │
    ▼
Clone (shallow, depth=1)  ──────────────► src/ingestion/github_loader.py
    │
    ▼
Walk repo, filter noise/binaries  ──────► src/ingestion/file_walker.py
    │
    ▼
AST-based chunking (tree-sitter)  ──────► src/chunking/ast_chunker.py
    │
    ▼
Embed each chunk  ──────────────────────► src/embedding/embedder.py
    │
    ▼
Store in ChromaDB (one collection/repo) ► src/storage/vector_store.py
    │
    ▼
   ... ask a question ...
    │
    ▼
Hybrid retrieval (vector + BM25, RRF)  ─► src/retrieval/hybrid_retriever.py, bm25_search.py
    │
    ▼
Grounded answer generation (Groq)  ─────► src/generation/answer_generator.py
    │
    ▼
Answer + inline citations
```

`pipeline.py` is the single orchestration point (`ingest_repository()` / `answer_question()`) — both the CLI and the web API call into it, so behavior is identical either way.

## Project structure

```
codebase-rag/
├── api/
│   └── main.py              # FastAPI backend — wraps pipeline.py for the web UI
├── static/
│   └── index.html           # Single-page frontend: paste a URL, ask questions
├── src/
│   ├── ingestion/
│   │   ├── github_loader.py # Clone a GitHub URL to a local shallow clone
│   │   └── file_walker.py   # Walk the clone, filter noise, tag file language
│   ├── chunking/
│   │   ├── ast_chunker.py   # AST-based chunking (tree-sitter) for py/js/ts/tsx
│   │   └── doc_chunker.py   # (planned) heading-aware chunking for Markdown
│   ├── embedding/
│   │   └── embedder.py      # Embeds chunks for vector search
│   ├── storage/
│   │   └── vector_store.py  # ChromaDB persistence, one collection per repo
│   ├── retrieval/
│   │   ├── bm25_search.py       # Keyword search half of hybrid retrieval
│   │   └── hybrid_retriever.py  # Vector + BM25 fusion via Reciprocal Rank Fusion
│   └── generation/
│       └── answer_generator.py  # Groq-backed, cited, structured answer generation
├── pipeline.py               # Orchestrates ingest_repository() / answer_question()
├── cli.py                    # Command-line interface
├── config.py                 # pydantic-settings: API keys, paths, model config
├── .env                      # API keys (not committed)
├── .env.example
└── requirements.txt
```

## Setup

**Requirements:** Python 3.14, Git, Windows/PowerShell (developed on; should work cross-platform).

```powershell
git clone <this-repo>
cd codebase-rag

python -m venv RAG_ENV
.\RAG_ENV\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
VOYAGE_API_KEY=your_voyage_key       # embeddings
GROQ_API_KEY=your_groq_key           # answer generation (llama-3.3-70b-versatile)
GITHUB_TOKEN=                        # optional — raises GitHub rate limits for private/large repos
```

> **Embedding provider note:** the project currently embeds via Voyage AI (`voyage-code-2`). Voyage's free tier is capped at **3 requests/minute** without a payment method on file, which is easy to hit on any repo with more than a handful of files. A migration to a locally-run, unlimited embedding model (`jina-embeddings-v2-base-code` via `sentence-transformers`, no API/rate limits) is planned but not yet wired in — see [Known Limitations](#known-limitations).

## Usage

### Command line

```powershell
# Ingest a repo (clone → chunk → embed → store)
python cli.py ingest https://github.com/owner/repo

# Ask a question about an ingested repo
python cli.py query owner/repo "How does authentication work in this repo?"

# List repos that have been ingested
python cli.py list
```

### Web UI

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` — paste a GitHub URL, click **Ingest**, then pick the repo from the dropdown and ask questions. Answers appear with source citations (`file:line-range`) underneath.

**API endpoints** (used by the frontend, callable directly too):

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/api/ingest` | `{"url": "...", "force_refresh": false}` | Clone, chunk, embed, store a repo |
| POST | `/api/query` | `{"repo": "owner/name", "question": "...", "n_results": 10}` | Ask a question about an ingested repo |
| GET | `/api/repos` | — | List repos already ingested |

## Design notes

- **Idempotent ingestion.** Re-ingesting a repo doesn't duplicate chunks — `vector_store.upsert_chunks()` upserts rather than inserts, and `force_refresh=True` forces a fresh re-clone when you want to pick up upstream changes.
- **Hybrid retrieval.** Pure vector search misses exact identifier/keyword matches (e.g. an exact function name); pure BM25 misses semantic paraphrases. Reciprocal Rank Fusion combines both result sets so retrieval isn't overly reliant on either.
- **Grounded, cited answers.** The generation prompt requires every claim to cite the excerpt it's based on (`[1]`, `[2]`, ...); only citations the model actually referenced in its answer are surfaced to the user, so the source list stays accurate rather than listing every chunk that happened to be retrieved.
- **Code-intelligence formatting, not document summarization.** The generation prompt explicitly avoids forcing markdown headers on short answers, avoids dumping full files, and detects broad/unscoped requests ("give me all the code") to respond with a project map instead of an unreadable wall of code.

## Known limitations

- **Embedding rate limits.** Voyage AI's free tier (3 RPM without billing) will slow or fail ingestion on repos with more than a few dozen chunks. Either add a payment method to your Voyage account, or wait for the planned local-embedding migration.
- **Cross-repo retrieval isolation is unverified.** Ingesting multiple different repos in the same session should keep each repo's chunks in its own ChromaDB collection and BM25 index, but this hasn't been explicitly tested/hardened yet — worth confirming before relying on multi-repo usage.
- **Single-file, single-language chunking scope.** AST chunking currently supports Python, JavaScript, TypeScript, and TSX; other text files (Markdown, JSON, YAML, etc.) fall back to line-based chunking rather than structure-aware chunking.
- **No code-editing capability yet.** The assistant currently only answers questions — it does not propose or apply code changes to the cloned repository. (On the roadmap; see below.)

## Roadmap

- [ ] Swap Voyage AI embeddings for a local, unlimited model (`jina-embeddings-v2-base-code` via `sentence-transformers`)
- [ ] Verify/harden per-repo isolation in vector storage and BM25 indexing
- [ ] Heading-aware Markdown chunking (`doc_chunker.py`)
- [ ] AI-proposed code edits: detect edit-intent questions, generate a reviewable diff against the cloned repo, and apply on explicit user confirmation (with VS Code integration to open/review the result)
- [ ] Incremental re-ingestion (only re-embed changed files on `force_refresh`, instead of a full re-clone/re-embed)

## License

_(Add your license here — e.g. MIT.)_

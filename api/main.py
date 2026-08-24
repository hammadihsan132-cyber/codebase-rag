"""
FastAPI backend for codebase-rag.

Endpoints:
  POST /api/ingest   {"url": "...", "force_refresh": false}
  POST /api/query    {"repo": "owner/name", "question": "...", "n_results": 10}
  GET  /api/repos

Serves the static frontend (static/index.html) at "/". This module is the
only thing that should ever be run as a server process; it talks to
pipeline.py exactly the way cli.py does, so behavior stays identical
between the CLI and the web UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import answer_question, ingest_repository
from src.ingestion.github_loader import InvalidGitHubUrlError, RepoCloneError
from src.storage.vector_store import list_ingested_repos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="codebase-rag",
    description="Paste a GitHub URL, then ask natural-language questions about the code.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class IngestRequest(BaseModel):
    url: str = Field(..., description="GitHub repository URL, e.g. https://github.com/psf/requests")
    force_refresh: bool = False


class IngestResponse(BaseModel):
    full_name: str
    files_processed: int
    chunks_created: int
    elapsed_seconds: float


class QueryRequest(BaseModel):
    repo: str = Field(..., description="'owner/name', e.g. psf/requests")
    question: str
    n_results: int = 10


class Citation(BaseModel):
    index: int
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class RepoSummary(BaseModel):
    owner: str
    repo: str
    collection: str


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Clone, chunk, embed, and store a GitHub repo. Safe to call again later."""
    try:
        result = ingest_repository(req.url, force_refresh=req.force_refresh)
    except InvalidGitHubUrlError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except RepoCloneError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    except Exception as e:
        logger.exception("Ingestion failed for %s", req.url)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from None

    return IngestResponse(
        full_name=result.repo.full_name,
        files_processed=result.files_processed,
        chunks_created=result.chunks_created,
        elapsed_seconds=result.elapsed_seconds,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Ask a question about an already-ingested repo."""
    if "/" not in req.repo:
        raise HTTPException(status_code=400, detail="repo must be in 'owner/repo' form")
    owner, repo_name = req.repo.split("/", 1)

    try:
        result = answer_question(owner, repo_name, req.question, n_results=req.n_results)
    except Exception as e:
        logger.exception("Query failed for %s/%s: %r", owner, repo_name, req.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from None

    return QueryResponse(
        answer=result.answer,
        citations=[
            Citation(
                index=c.index,
                file_path=str(c.file_path),
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
            )
            for c in result.citations
        ],
    )


@app.get("/api/repos", response_model=list[RepoSummary])
def repos() -> list[RepoSummary]:
    """List repos that have already been ingested, for the UI's dropdown/history."""
    return [RepoSummary(**r) for r in list_ingested_repos()]


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Frontend not found at {index_path}. Create static/index.html and restart the server.",
        )
    return FileResponse(index_path)
# Codebase RAG – AI-Powered Repository Intelligence

An AI-powered code intelligence assistant that allows developers to connect a GitHub repository, ask natural-language questions about its codebase, and receive repository-grounded answers with citations to exact files and line numbers.

Codebase RAG is designed in the spirit of GitHub Copilot Chat, Cursor, and Sourcegraph Cody, with a focus on understanding the actual structure and implementation of a software project rather than providing generic document-based answers.

## 1. Project Overview

Codebase RAG enables users to:

- Connect a GitHub repository using its URL.
- Clone and analyze the repository.
- Parse source code into meaningful chunks.
- Generate code embeddings using NVIDIA Nemotron.
- Store embeddings in ChromaDB.
- Retrieve relevant code using vector and keyword search.
- Ask questions about the repository.
- Receive AI-generated answers with source citations.
- Use advanced reasoning through Qwen4 Max.

**Future objective:** Extend the platform into an AI-powered development environment where coding agents can analyze, modify, test, and improve projects with user approval.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| AI API gateway | OpenRouter |
| Embedding model | NVIDIA Nemotron 3.5 Embed 8B |
| Default chat model | NVIDIA Nemotron 3.5 Lightning |
| Advanced reasoning model | Qwen4 Max |
| API client | OpenAI Python SDK |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Retrieval strategy | Reciprocal Rank Fusion (RRF) |
| Code parsing | Tree-sitter |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Repository management | GitPython |
| Configuration | Pydantic Settings |

## 3. AI Model Configuration

Codebase RAG uses OpenRouter to access its embedding and language models through an OpenAI-compatible API.

### 3.1 NVIDIA Nemotron 3.5 Embed 8B

Model ID:

`nvidia/nemotron-3.5-embed-8b:free`

Purpose:
- Generate embeddings for repository code chunks.
- Generate embeddings for user questions.
- Support semantic code retrieval.
- Produce 4096-dimensional vectors.

### 3.2 NVIDIA Nemotron 3.5 Lightning

Model ID:

`nvidia/nemotron-3.5-lightning:free`

Purpose:
- Default repository question answering.
- Explain functions and classes.
- Understand code relationships.
- Generate concise, repository-grounded responses.

### 3.3 Qwen4 Max

Model ID:

`qwen/qwen4-max`

Purpose:
- Advanced reasoning.
- Complex code analysis.
- Detailed architectural explanations.
- More demanding development and debugging questions.

Qwen4 Max is selected through the advanced generation option.

### OpenRouter Configuration

Base URL:

`https://openrouter.ai/api/v1`

API requests use the OpenAI Python SDK with OpenRouter as the API endpoint.

Model availability, pricing, and provider limits may change.

## 4. System Architecture

The current implemented workflow is:

```text
              GitHub Repository URL
                       |
                       v
                GitHub Loader
                       |
                       v
                 Clone Repository
                       |
                       v
                  File Walker
                       |
                       v
                 AST Chunking
                 (Tree-sitter)
                       |
                       v
               NVIDIA Nemotron
                 Embeddings
                       |
                       v
                   ChromaDB
                       |
                       v
                Repository Stored
                       |
                 User Question
                       |
                       v
               Query Embedding
                       |
             +---------+---------+
             |                   |
             v                   v
        Vector Search         BM25 Search
             |                   |
             +---------+---------+
                       |
                       v
             Reciprocal Rank Fusion
                       |
                       v
               Relevant Code Chunks
                       |
                       v
              OpenRouter LLM
                       |
                       v
             Answer + Citations
```

The central orchestration component is `pipeline.py`, which provides:

- `ingest_repository()`
- `answer_question()`

The CLI and web backend use this pipeline.

## 5. Project Structure

```text
codebase-rag/
│
├── api/
│   └── main.py
│
├── static/
│   └── index.html
│
├── src/
│   ├── ingestion/
│   │   ├── github_loader.py
│   │   └── file_walker.py
│   │
│   ├── chunking/
│   │   └── ast_chunker.py
│   │
│   ├── embedding/
│   │   └── embedder.py
│   │
│   ├── storage/
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   ├── bm25_search.py
│   │   └── hybrid_retriever.py
│   │
│   └── generation/
│       └── answer_generator.py
│
├── data/
│   ├── repos/
│   └── chroma/
│
├── pipeline.py
├── cli.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

## 6. Main Components

### 6.1 GitHub Repository Ingestion

File: `src/ingestion/github_loader.py`

Responsibilities:
- Accept GitHub repository URLs.
- Identify repository owner and name.
- Clone repositories locally.
- Support optional GitHub authentication.
- Reuse existing clones where applicable.

### 6.2 File Walker

File: `src/ingestion/file_walker.py`

Responsibilities:
- Traverse repository directories.
- Identify supported source files.
- Exclude unnecessary directories and binary files.
- Prepare source files for chunking.

### 6.3 AST-Based Code Chunking

File: `src/chunking/ast_chunker.py`

Uses Tree-sitter to identify meaningful code structures.

Supported structural parsing includes:
- Python
- JavaScript
- TypeScript
- TSX

Other supported text files use line-based chunking.

### 6.4 Embedding Engine

File: `src/embedding/embedder.py`

Uses NVIDIA Nemotron 3.5 Embed 8B through OpenRouter.

Implemented features:
- Batch embedding.
- Passage and query embedding modes.
- 4096-dimensional vector validation.
- Retry handling for transient API failures.
- Automatic subdivision of batches rejected with HTTP 422.
- Logging and identification of individually rejected chunks.

### 6.5 Vector Database

File: `src/storage/vector_store.py`

Uses ChromaDB to persist code embeddings.

Responsibilities:
- Create repository-specific collections.
- Store code chunks and metadata.
- Perform vector similarity searches.
- Retrieve relevant source code.
- Support deterministic chunk IDs and upserts.

The current Nemotron collection uses a versioned name, for example:

`hammadihsan132-cyber__codebase-rag__nemotron_4096`

### 6.6 Hybrid Retrieval

Files:
- `src/retrieval/bm25_search.py`
- `src/retrieval/hybrid_retriever.py`

Combines two retrieval approaches:

**Vector Search**
- Finds semantically related code.
- Supports natural-language questions.

**BM25 Search**
- Finds exact identifiers and keywords.
- Helps retrieve functions, classes, and variable names.

**Reciprocal Rank Fusion**
- Combines both result lists.
- Produces a unified ranking of relevant code chunks.

### 6.7 Answer Generation

File: `src/generation/answer_generator.py`

Uses OpenRouter to generate repository-grounded answers.

Implemented capabilities:
- Nemotron Lightning for normal questions.
- Qwen4 Max for advanced reasoning.
- Inline citation generation.
- Concise developer-oriented responses.
- Retry handling and empty-response diagnostics.
- Broad project questions handled through project maps instead of excessive code dumps.

Citation validation still requires further improvement.

## 7. Installation and Setup

### Requirements

- Python (tested in the current development environment with Python 3.14.6).
- Git.
- OpenRouter API key.
- Windows PowerShell or another supported terminal.

### Clone the repository

```powershell
git clone https://github.com/hammadihsan132-cyber/codebase-rag.git

cd codebase-rag
```

### Create a virtual environment

```powershell
python -m venv .venv
```

### Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key

OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

OPENROUTER_EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-8b:free

OPENROUTER_CHAT_MODEL=nvidia/nemotron-3.5-lightning:free

OPENROUTER_ADVANCED_MODEL=qwen/qwen4-max

GITHUB_TOKEN=your_optional_github_token
```

Never upload `.env` or API keys to GitHub.

## 8. Running the Application

### Repository ingestion

```powershell
python -m pipeline https://github.com/owner/repository.git
```

This performs repository cloning, file discovery, chunking, embedding, and ChromaDB storage.

### Command-line interface

Ingest:

```powershell
python cli.py ingest https://github.com/owner/repository
```

Ask questions:

```powershell
python cli.py query owner/repository "How does authentication work?"
```

List ingested repositories:

```powershell
python cli.py list
```

### Web application

Start the FastAPI server:

```powershell
uvicorn api.main:app --reload
```

Open:

`http://127.0.0.1:8000`

The current web interface supports repository ingestion and question answering.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/ingest` | Ingest a GitHub repository |
| POST | `/api/query` | Ask repository questions |
| GET | `/api/repos` | List ingested repositories |

## 9. Testing and Current Results

The following results were observed during development:

| Test | Result |
|---|---|
| OpenRouter connection | Successful |
| Nemotron embedding API | Successful |
| Embedding dimensions | 4096 |
| ChromaDB storage | Successful |
| Vector similarity retrieval | Successful |
| Nemotron Lightning generation | Successful |
| Qwen4 Max generation | Successfully tested |
| Repository ingestion | Successful |
| Repository question answering | Successful |
| Source citations | Returned; further validation needed |

### Repository ingestion test

Repository:

`hammadihsan132-cyber/codebase-rag`

Observed results:

- Files discovered: 16
- Chunks generated: 114
- Embeddings successfully generated: 113
- Chunks stored in ChromaDB: 113
- Provider-rejected chunks: 1
- Ingestion time: approximately 17.9 seconds

One chunk was rejected by the embedding provider's content-safety screening. The application isolated and reported it without stopping the entire ingestion process.

These figures describe one development test, not a general performance benchmark.

## 10. Current Known Limitations

1. **Embedding checkpointing:** Successful batches are not yet persistently checkpointed for resumption after interruption.

2. **Repository synchronization:** Existing clone reuse and upstream refresh behavior need further verification.

3. **Citation accuracy:** Citation references require stricter validation against retrieved excerpts.

4. **Outdated documentation:** Previous Voyage AI and Groq references must be removed from documentation and checked throughout the source code.

5. **Rejected chunks:** Provider-rejected chunks need appropriate investigation and handling.

6. **Retrieval evaluation:** A systematic test set for retrieval relevance and answer correctness has not yet been implemented.

7. **Incremental ingestion:** Changed files are not yet handled through a fully verified incremental embedding workflow.

8. **Code modification:** The current application does not yet implement AI-powered code editing.

## 11. Development Roadmap

### Phase 1 – Stabilize Existing RAG

- [ ] Update outdated documentation.
- [ ] Fix citation integrity.
- [ ] Improve repository synchronization.
- [ ] Implement persistent embedding checkpoints.
- [ ] Improve handling of rejected chunks.
- [ ] Test cross-repository isolation.
- [ ] Build a RAG evaluation framework.

### Phase 2 – AI Coding Agent System

- [ ] Introduce repository analysis agents.
- [ ] Develop a planning agent.
- [ ] Develop a code-generation agent.
- [ ] Develop a code-review agent.
- [ ] Develop a testing agent.
- [ ] Implement code-diff generation.
- [ ] Implement approval-based code modification.

### Phase 3 – VS Code Integration

- [ ] Develop a VS Code extension.
- [ ] Connect the extension with the existing backend.
- [ ] Allow agents to inspect the local workspace.
- [ ] Display proposed changes and diffs.
- [ ] Require approval before applying changes.
- [ ] Execute tests and display results.

### Phase 4 – Advanced Development Features

- [ ] GitHub branch management.
- [ ] Automated bug detection.
- [ ] Feature implementation workflows.
- [ ] Pull request generation.
- [ ] Multi-agent task coordination.
- [ ] Project-wide code quality analysis.

## 12. Project Development Philosophy

Codebase RAG should evolve from a repository question-answering system into an AI-assisted software development platform.

The intended workflow is:

**Connect Repository → Understand Codebase → Ask Questions → Plan Changes → Generate Code → Review Changes → Obtain Approval → Apply Changes → Run Tests**

The user must remain in control of code modifications.

The existing RAG system will serve as the knowledge and retrieval foundation for the future coding agents.

---

## Repository Information

**GitHub:** https://github.com/hammadihsan132-cyber/codebase-rag

**Primary branch:** `main`

**Latest confirmed pushed commit at the time of this documentation:** `e7fa9fe`

**Project status:** Working repository ingestion and RAG question answering; coding-agent and VS Code capabilities are planned.

**Documentation note:** This README describes the implementation and testing status reported during development. Features marked as planned or unfinished must not be considered implemented.

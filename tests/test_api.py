
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as api


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def fake_ingest_result():
    return SimpleNamespace(
        repo=SimpleNamespace(
            full_name="owner/test-repo"
        ),
        files_processed=10,
        chunks_created=25,
        elapsed_seconds=2.5,
    )


@pytest.fixture
def fake_answer():
    return SimpleNamespace(
        answer="The repository uses ChromaDB for vector storage.",
        citations=[
            SimpleNamespace(
                index=1,
                file_path=Path("src/storage/vector_store.py"),
                start_line=10,
                end_line=25,
                symbol_name="upsert_chunks",
            )
        ],
    )


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200


def test_ingest_success(
    client,
    monkeypatch,
    fake_ingest_result,
):
    def mock_ingest(url, force_refresh=False):
        return fake_ingest_result

    monkeypatch.setattr(
        api,
        "ingest_repository",
        mock_ingest,
    )

    response = client.post(
        "/api/ingest",
        json={
            "url": "https://github.com/owner/test-repo",
            "force_refresh": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["full_name"] == "owner/test-repo"
    assert data["files_processed"] == 10
    assert data["chunks_created"] == 25


def test_ingest_invalid_github_url(
    client,
    monkeypatch,
):
    def mock_ingest(*args, **kwargs):
        raise api.InvalidGitHubUrlError("Invalid GitHub URL")

    monkeypatch.setattr(
        api,
        "ingest_repository",
        mock_ingest,
    )

    response = client.post(
        "/api/ingest",
        json={
            "url": "invalid-url",
        },
    )

    assert response.status_code == 400


def test_ingest_clone_failure(
    client,
    monkeypatch,
):
    def mock_ingest(*args, **kwargs):
        raise api.RepoCloneError("Clone failed")

    monkeypatch.setattr(
        api,
        "ingest_repository",
        mock_ingest,
    )

    response = client.post(
        "/api/ingest",
        json={
            "url": "https://github.com/owner/test-repo",
        },
    )

    assert response.status_code == 502


def test_query_success(
    client,
    monkeypatch,
    fake_answer,
):
    def mock_answer(owner, repo, question, n_results=10):
        assert owner == "owner"
        assert repo == "test-repo"
        return fake_answer

    monkeypatch.setattr(
        api,
        "answer_question",
        mock_answer,
    )

    response = client.post(
        "/api/query",
        json={
            "repo": "owner/test-repo",
            "question": "Explain vector storage",
            "n_results": 10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "ChromaDB" in data["answer"]
    assert len(data["citations"]) == 1

    citation = data["citations"][0]

    assert citation["file_path"] == (
        "src/storage/vector_store.py"
    )
    assert citation["start_line"] == 10
    assert citation["end_line"] == 25


def test_query_invalid_repo_format(client):
    response = client.post(
        "/api/query",
        json={
            "repo": "invalid-repository",
            "question": "Explain the project",
        },
    )

    assert response.status_code == 400


def test_query_internal_error(
    client,
    monkeypatch,
):
    def mock_answer(*args, **kwargs):
        raise RuntimeError("Test internal failure")

    monkeypatch.setattr(
        api,
        "answer_question",
        mock_answer,
    )

    response = client.post(
        "/api/query",
        json={
            "repo": "owner/test-repo",
            "question": "Explain the project",
        },
    )

    assert response.status_code == 500


def test_list_repositories(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "list_ingested_repos",
        lambda: [
            {
                "owner": "owner",
                "repo": "test-repo",
                "collection": "owner__test-repo__nemotron_4096",
            }
        ],
    )

    response = client.get("/api/repos")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["owner"] == "owner"
    assert data[0]["repo"] == "test-repo"


def test_pydantic_rejects_missing_question(client):
    response = client.post(
        "/api/query",
        json={
            "repo": "owner/test-repo",
        },
    )

    assert response.status_code == 422


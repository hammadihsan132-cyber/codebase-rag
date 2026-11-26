
from pathlib import Path

import pytest

from src.chunking.ast_chunker import Chunk
from src.storage.embedding_checkpoint import EmbeddingCheckpoint


OWNER = "test-owner"
REPO = "test-repo"


@pytest.fixture
def checkpoint(tmp_path):
    """Create an isolated temporary SQLite database."""
    db_path = tmp_path / "test_embeddings.sqlite3"

    return EmbeddingCheckpoint(
        db_path=db_path,
        dimensions=3,
    )


@pytest.fixture
def sample_chunk():
    return Chunk(
        file_path=Path("src/example.py"),
        language="python",
        symbol_name="calculate",
        chunk_type="function",
        content="def calculate(a, b): return a + b",
        start_line=1,
        end_line=2,
        part=None,
        total_parts=None,
    )


@pytest.fixture
def sample_vector():
    return [0.1, 0.2, 0.3]


# ---------------------------------------------------------
# 1. DATABASE INITIALIZATION
# ---------------------------------------------------------

def test_database_initialization(checkpoint):
    assert checkpoint.db_path.exists()
    assert checkpoint.count() == 0
    assert checkpoint.count_rejected() == 0


# ---------------------------------------------------------
# 2. FINGERPRINT TESTS
# ---------------------------------------------------------

def test_fingerprint_is_deterministic(
    checkpoint,
    sample_chunk,
):
    first = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    second = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    assert first == second


def test_changed_content_produces_new_fingerprint(
    checkpoint,
    sample_chunk,
):
    first = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    changed_chunk = Chunk(
        file_path=sample_chunk.file_path,
        language=sample_chunk.language,
        symbol_name=sample_chunk.symbol_name,
        chunk_type=sample_chunk.chunk_type,
        content="def calculate(a, b): return a * b",
        start_line=1,
        end_line=2,
        part=None,
        total_parts=None,
    )

    second = checkpoint.fingerprint_chunk(
        OWNER, REPO, changed_chunk
    )

    assert first != second


def test_different_repository_produces_new_fingerprint(
    checkpoint,
    sample_chunk,
):
    first = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    second = checkpoint.fingerprint_chunk(
        OWNER, "another-repo", sample_chunk
    )

    assert first != second


# ---------------------------------------------------------
# 3. SAVE AND RETRIEVE EMBEDDINGS
# ---------------------------------------------------------

def test_save_and_retrieve_embedding(
    checkpoint,
    sample_chunk,
    sample_vector,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    checkpoint.save_many(
        OWNER,
        REPO,
        [(fingerprint, sample_vector)],
    )

    result = checkpoint.get_many([fingerprint])

    assert fingerprint in result
    assert result[fingerprint] == pytest.approx(
        sample_vector,
        abs=1e-6,
    )

    assert checkpoint.count(OWNER, REPO) == 1


def test_missing_fingerprint_returns_empty(
    checkpoint,
):
    result = checkpoint.get_many(["missing-fingerprint"])

    assert result == {}


def test_empty_fingerprint_list(checkpoint):
    assert checkpoint.get_many([]) == {}


def test_empty_save_does_not_change_database(
    checkpoint,
):
    checkpoint.save_many(OWNER, REPO, [])

    assert checkpoint.count() == 0


# ---------------------------------------------------------
# 4. VALIDATION TESTS
# ---------------------------------------------------------

def test_invalid_dimensions_are_rejected(
    checkpoint,
):
    with pytest.raises(ValueError):
        checkpoint.save_many(
            OWNER,
            REPO,
            [("invalid-fingerprint", [0.1, 0.2])],
        )

    assert checkpoint.count() == 0


def test_non_finite_values_are_rejected(
    checkpoint,
):
    invalid_vector = [0.1, float("nan"), 0.3]

    with pytest.raises(ValueError):
        checkpoint.save_many(
            OWNER,
            REPO,
            [("invalid-fingerprint", invalid_vector)],
        )

    assert checkpoint.count() == 0


# ---------------------------------------------------------
# 5. REJECTION TRACKING
# ---------------------------------------------------------

def test_rejected_chunk_is_recorded(
    checkpoint,
    sample_chunk,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    checkpoint.mark_rejected_many(
        OWNER,
        REPO,
        [(fingerprint, "Provider returned HTTP 422")],
    )

    rejected = checkpoint.get_rejected([fingerprint])

    assert fingerprint in rejected
    assert rejected[fingerprint]["error"] == (
        "Provider returned HTTP 422"
    )
    assert rejected[fingerprint]["attempts"] == 1

    assert checkpoint.count_rejected(OWNER, REPO) == 1


def test_repeated_rejection_increments_attempts(
    checkpoint,
    sample_chunk,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    for _ in range(3):
        checkpoint.mark_rejected_many(
            OWNER,
            REPO,
            [(fingerprint, "HTTP 422")],
        )

    rejected = checkpoint.get_rejected([fingerprint])

    assert rejected[fingerprint]["attempts"] == 3


def test_rejection_error_is_updated(
    checkpoint,
    sample_chunk,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    checkpoint.mark_rejected_many(
        OWNER, REPO,
        [(fingerprint, "First error")],
    )

    checkpoint.mark_rejected_many(
        OWNER, REPO,
        [(fingerprint, "Second error")],
    )

    rejected = checkpoint.get_rejected([fingerprint])

    assert rejected[fingerprint]["error"] == "Second error"
    assert rejected[fingerprint]["attempts"] == 2


# ---------------------------------------------------------
# 6. CLEAR REJECTION AFTER SUCCESS
# ---------------------------------------------------------

def test_successful_embedding_clears_rejection(
    checkpoint,
    sample_chunk,
    sample_vector,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    checkpoint.mark_rejected_many(
        OWNER,
        REPO,
        [(fingerprint, "HTTP 422")],
    )

    assert checkpoint.count_rejected(OWNER, REPO) == 1

    checkpoint.save_many(
        OWNER,
        REPO,
        [(fingerprint, sample_vector)],
    )

    assert checkpoint.count_rejected(OWNER, REPO) == 0
    assert checkpoint.count(OWNER, REPO) == 1

    assert checkpoint.get_rejected([fingerprint]) == {}

    assert fingerprint in checkpoint.get_many([fingerprint])


def test_manual_rejection_clearing(
    checkpoint,
    sample_chunk,
):
    fingerprint = checkpoint.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    checkpoint.mark_rejected_many(
        OWNER, REPO,
        [(fingerprint, "HTTP 422")],
    )

    checkpoint.clear_rejected_many([fingerprint])

    assert checkpoint.get_rejected([fingerprint]) == {}
    assert checkpoint.count_rejected(OWNER, REPO) == 0


# ---------------------------------------------------------
# 7. DATABASE PERSISTENCE
# ---------------------------------------------------------

def test_embeddings_survive_database_reinitialization(
    tmp_path,
    sample_chunk,
    sample_vector,
):
    db_path = tmp_path / "persistent_test.sqlite3"

    first = EmbeddingCheckpoint(
        db_path=db_path,
        dimensions=3,
    )

    fingerprint = first.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    first.save_many(
        OWNER,
        REPO,
        [(fingerprint, sample_vector)],
    )

    # Simulate restarting the application.
    second = EmbeddingCheckpoint(
        db_path=db_path,
        dimensions=3,
    )

    recovered = second.get_many([fingerprint])

    assert fingerprint in recovered
    assert recovered[fingerprint] == pytest.approx(
        sample_vector,
        abs=1e-6,
    )

    assert second.count(OWNER, REPO) == 1


# ---------------------------------------------------------
# 8. REJECTION PERSISTENCE
# ---------------------------------------------------------

def test_rejection_survives_database_reinitialization(
    tmp_path,
    sample_chunk,
):
    db_path = tmp_path / "rejection_test.sqlite3"

    first = EmbeddingCheckpoint(
        db_path=db_path,
        dimensions=3,
    )

    fingerprint = first.fingerprint_chunk(
        OWNER, REPO, sample_chunk
    )

    first.mark_rejected_many(
        OWNER,
        REPO,
        [(fingerprint, "Content safety rejection")],
    )

    second = EmbeddingCheckpoint(
        db_path=db_path,
        dimensions=3,
    )

    rejected = second.get_rejected([fingerprint])

    assert fingerprint in rejected
    assert rejected[fingerprint]["attempts"] == 1


from pathlib import Path
from types import SimpleNamespace

import pytest

import src.embedding.embedder as embedder_module

from src.embedding.embedder import Embedder
from src.chunking.ast_chunker import Chunk


DIMENSIONS = 4096


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            file_path=Path("src/example.py"),
            language="python",
            symbol_name="function_one",
            chunk_type="function",
            content="def function_one(): return 1",
            start_line=1,
            end_line=2,
        ),
        Chunk(
            file_path=Path("src/example.py"),
            language="python",
            symbol_name="function_two",
            chunk_type="function",
            content="def function_two(): return 2",
            start_line=4,
            end_line=5,
        ),
    ]


@pytest.fixture
def embedder():
    return Embedder(api_key="test-api-key")


def make_vector(value=0.1):
    return [value] * DIMENSIONS


def test_embed_single_chunk(embedder, sample_chunks, monkeypatch):

    monkeypatch.setattr(
        embedder,
        "_embed_with_retry",
        lambda texts, input_type: [make_vector()],
    )

    results = embedder.embed_chunks([sample_chunks[0]])

    assert len(results) == 1
    assert results[0].chunk == sample_chunks[0]
    assert len(results[0].embedding) == DIMENSIONS
    assert embedder.failed_chunks == []


def test_embed_multiple_chunks(embedder, sample_chunks, monkeypatch):

    monkeypatch.setattr(
        embedder,
        "_embed_with_retry",
        lambda texts, input_type: [
            make_vector(0.1),
            make_vector(0.2),
        ],
    )

    results = embedder.embed_chunks(sample_chunks)

    assert len(results) == 2
    assert results[0].chunk == sample_chunks[0]
    assert results[1].chunk == sample_chunks[1]


def test_empty_input_returns_empty_list(embedder):

    results = embedder.embed_chunks([])

    assert results == []
    assert embedder.failed_chunks == []
    assert embedder.failed_chunk_errors == {}
def test_format_for_embedding(sample_chunks):

    formatted = Embedder._format_for_embedding(
        sample_chunks[0]
    )

    # Normalize Windows and Unix path separators
    formatted = formatted.replace("\\", "/")

    assert "# File: src/example.py" in formatted
    assert "# function: function_one" in formatted
    assert "def function_one(): return 1" in formatted


# def test_format_for_embedding(sample_chunks):

#     formatted = Embedder._format_for_embedding(
#         sample_chunks[0]
#     )

#     assert "# File: src/example.py" in formatted
#     assert "# function: function_one" in formatted
#     assert "def function_one(): return 1" in formatted


def test_rejected_chunk_is_isolated(
    embedder,
    sample_chunks,
    monkeypatch,
):

    class RejectedError(Exception):
        pass

    monkeypatch.setattr(
        embedder_module,
        "UnprocessableEntityError",
        RejectedError,
    )

    def mock_embedding(texts, input_type):

        if any("function_one" in text for text in texts):
            raise RejectedError("Content safety rejection")

        return [make_vector(0.2) for _ in texts]

    monkeypatch.setattr(
        embedder,
        "_embed_with_retry",
        mock_embedding,
    )

    results = embedder.embed_chunks(sample_chunks)

    assert len(results) == 1
    assert results[0].chunk == sample_chunks[1]

    assert len(embedder.failed_chunks) == 1
    assert embedder.failed_chunks[0] == sample_chunks[0]

    assert "Content safety rejection" in (
        embedder.failed_chunk_errors[id(sample_chunks[0])]
    )


def test_api_response_order_is_preserved(embedder, monkeypatch):

    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=1,
                embedding=make_vector(0.2),
            ),
            SimpleNamespace(
                index=0,
                embedding=make_vector(0.1),
            ),
        ]
    )

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: response
        )
    )

    embedder._client = fake_client

    results = embedder._embed_with_retry(
        ["first", "second"],
        input_type="passage",
    )

    assert results[0][0] == 0.1
    assert results[1][0] == 0.2


def test_invalid_input_type_is_rejected(embedder):

    with pytest.raises(ValueError):

        embedder._embed_with_retry(
            ["test"],
            input_type="invalid",
        )


def test_empty_api_input_returns_empty(embedder):

    result = embedder._embed_with_retry(
        [],
        input_type="passage",
    )

    assert result == []


def test_rate_limit_retries_successfully(
    embedder,
    monkeypatch,
):

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(
        embedder_module,
        "RateLimitError",
        FakeRateLimitError,
    )

    calls = {"count": 0}

    def fake_create(**kwargs):

        calls["count"] += 1

        if calls["count"] == 1:
            raise FakeRateLimitError("Rate limited")

        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=make_vector(),
                )
            ]
        )

    embedder._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=fake_create
        )
    )

    monkeypatch.setattr(
        embedder_module.time,
        "sleep",
        lambda seconds: None,
    )

    result = embedder._embed_with_retry(
        ["test"],
        input_type="passage",
    )

    assert calls["count"] == 2
    assert len(result) == 1
    assert len(result[0]) == DIMENSIONS


def test_retry_exhaustion_raises_error(
    embedder,
    monkeypatch,
):

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(
        embedder_module,
        "RateLimitError",
        FakeRateLimitError,
    )

    def fake_create(**kwargs):
        raise FakeRateLimitError("Still rate limited")

    embedder._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=fake_create
        )
    )

    monkeypatch.setattr(
        embedder_module.time,
        "sleep",
        lambda seconds: None,
    )

    with pytest.raises(RuntimeError, match="5 attempts"):

        embedder._embed_with_retry(
            ["test"],
            input_type="passage",
        )


def test_invalid_vector_dimensions_raise_error(
    embedder,
):

    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=0,
                embedding=[0.1] * 100,
            )
        ]
    )

    embedder._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: response
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Expected 4096",
    ):

        embedder._embed_with_retry(
            ["test"],
            input_type="passage",
        )


def test_unexpected_vector_count_raises_error(
    embedder,
):

    response = SimpleNamespace(
        data=[]
    )

    embedder._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: response
        )
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected number of vectors",
    ):

        embedder._embed_with_retry(
            ["test"],
            input_type="passage",
        )


def test_failed_tracking_resets_between_runs(
    embedder,
    sample_chunks,
    monkeypatch,
):

    embedder.failed_chunks = [sample_chunks[0]]
    embedder.failed_chunk_errors = {
        id(sample_chunks[0]): "Old error"
    }

    monkeypatch.setattr(
        embedder,
        "_embed_with_retry",
        lambda texts, input_type: [
            make_vector() for _ in texts
        ],
    )

    results = embedder.embed_chunks([sample_chunks[1]])

    assert len(results) == 1
    assert embedder.failed_chunks == []
    assert embedder.failed_chunk_errors == {}

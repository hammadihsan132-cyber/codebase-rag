
from argparse import Namespace
from types import SimpleNamespace

import pytest

import cli


def test_positive_int_accepts_valid_value():
    assert cli.positive_int("10") == 10


def test_positive_int_rejects_zero():
    with pytest.raises(Exception):
        cli.positive_int("0")


def test_parser_registers_all_commands():

    parser = cli.build_parser()

    commands = parser._subparsers._group_actions[0].choices

    expected = {
        "ingest",
        "query",
        "list",
        "status",
        "sync",
        "retry-rejected",
    }

    assert expected.issubset(commands.keys())


def test_parse_repo_name():

    assert cli.parse_repo_name(
        "owner/project"
    ) == ("owner", "project")


def test_parse_repo_name_rejects_invalid():

    with pytest.raises(ValueError):
        cli.parse_repo_name("invalid")


def test_query_rejects_invalid_repository(
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        cli,
        "answer_question",
        lambda *args, **kwargs: pytest.fail(
            "Answer generator should not run."
        ),
    )

    with pytest.raises(SystemExit):
        cli._cmd_query(
            Namespace(
                repo="invalid",
                question="Test",
                n_results=10,
            )
        )


def test_status_displays_repository(
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        cli,
        "list_ingested_repos",
        lambda: [
            {
                "owner": "test-owner",
                "repo": "test-repo",
                "collection": "test-collection",
            }
        ],
    )

    class FakeCheckpoint:

        def count(self, owner, repo):
            return 100

        def count_rejected(self, owner, repo):
            return 2

    monkeypatch.setattr(
        cli,
        "EmbeddingCheckpoint",
        FakeCheckpoint,
    )

    monkeypatch.setattr(
        cli,
        "get_or_create_collection",
        lambda owner, repo: SimpleNamespace(
            name="test-collection",
            count=lambda: 98,
        ),
    )

    cli._cmd_status(
        Namespace(repo=None)
    )

    output = capsys.readouterr().out

    assert "test-owner/test-repo" in output
    assert "98" in output
    assert "100" in output
    assert "2" in output


def test_status_handles_empty_repository_list(
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        cli,
        "list_ingested_repos",
        lambda: [],
    )

    cli._cmd_status(
        Namespace(repo=None)
    )

    output = capsys.readouterr().out

    assert "No ingested repositories found" in output

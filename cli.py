
"""
Advanced command-line interface for codebase-rag.

Commands:
    python cli.py ingest <github_url>
    python cli.py query <owner/repo> "<question>"
    python cli.py list
    python cli.py status [owner/repo]
    python cli.py sync <github_url>
    python cli.py retry-rejected <github_url>
"""

from __future__ import annotations

import argparse
import logging
import sys

from git import Repo, GitCommandError, InvalidGitRepositoryError

from config import settings
from pipeline import answer_question, ingest_repository
from src.ingestion.github_loader import (
    InvalidGitHubUrlError,
    RepoCloneError,
    parse_github_url,
)
from src.storage.embedding_checkpoint import EmbeddingCheckpoint
from src.storage.vector_store import (
    get_or_create_collection,
    list_ingested_repos,
)

logger = logging.getLogger("codebase-rag.cli")


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Value must be an integer."
        )

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def parse_repo_name(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise ValueError(
            "Repository must use owner/repo format."
        )

    owner, repo = value.split("/", 1)

    if not owner or not repo or "/" in repo:
        raise ValueError(
            "Invalid repository format. Expected owner/repo."
        )

    return owner, repo


# ---------------------------------------------------------
# INGEST
# ---------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> None:

    result = ingest_repository(
        args.url,
        force_refresh=args.force_refresh,
        retry_rejected=args.retry_rejected,
    )

    print("\nINGESTION COMPLETED")
    print("-" * 55)

    print(f"Repository       : {result.repo.full_name}")
    print(f"Files processed : {result.files_processed}")
    print(f"Chunks created  : {result.chunks_created}")
    print(f"Elapsed time    : {result.elapsed_seconds:.2f}s")

    print("\nNext command:")
    print(
        f'python cli.py query {result.repo.full_name} '
        '"Explain this repository."'
    )


# ---------------------------------------------------------
# QUERY
# ---------------------------------------------------------

def _cmd_query(args: argparse.Namespace) -> None:

    try:
        owner, repo = parse_repo_name(args.repo)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

    result = answer_question(
        owner,
        repo,
        args.question,
        n_results=args.n_results,
    )

    print("\nANSWER")
    print("=" * 70)
    print(result.answer)

    if result.citations:

        print("\nSOURCES")
        print("-" * 70)

        for citation in result.citations:

            symbol = (
                f" ({citation.symbol_name})"
                if citation.symbol_name
                else ""
            )

            print(
                f"[{citation.index}] "
                f"{citation.file_path}:"
                f"{citation.start_line}-"
                f"{citation.end_line}"
                f"{symbol}"
            )


# ---------------------------------------------------------
# LIST REPOSITORIES
# ---------------------------------------------------------

def _cmd_list(_: argparse.Namespace) -> None:

    repos = list_ingested_repos()

    if not repos:
        print("No repositories have been ingested.")
        return

    print("\nINGESTED REPOSITORIES")
    print("=" * 70)

    for item in repos:

        print(
            f"\nRepository : "
            f"{item['owner']}/{item['repo']}"
        )

        print(
            f"Collection : {item['collection']}"
        )


# ---------------------------------------------------------
# REPOSITORY STATUS
# ---------------------------------------------------------

def _show_repository_status(
    owner: str,
    repo: str,
    checkpoint: EmbeddingCheckpoint,
) -> None:

    collection = get_or_create_collection(
        owner,
        repo,
    )

    vector_count = collection.count()

    successful = checkpoint.count(
        owner,
        repo,
    )

    rejected = checkpoint.count_rejected(
        owner,
        repo,
    )

    print("\n" + "=" * 65)
    print(f"REPOSITORY: {owner}/{repo}")
    print("=" * 65)

    print(f"ChromaDB collection : {collection.name}")
    print(f"Stored vectors      : {vector_count}")
    print(f"Successful cache    : {successful}")
    print(f"Rejected chunks     : {rejected}")

    print("-" * 65)

    if rejected:
        print("Status: Attention required - rejected chunks exist.")

    elif vector_count == 0:
        print("Status: No vectors stored.")

    else:
        print("Status: Repository available for querying.")


def _cmd_status(args: argparse.Namespace) -> None:

    repos = list_ingested_repos()

    checkpoint = EmbeddingCheckpoint()

    if args.repo:

        try:
            owner, repo = parse_repo_name(args.repo)
        except ValueError as error:
            print(f"Error: {error}", file=sys.stderr)
            raise SystemExit(1)

        exists = any(
            item["owner"].lower() == owner.lower()
            and item["repo"].lower() == repo.lower()
            for item in repos
        )

        if not exists:
            print(
                f"Repository {owner}/{repo} "
                "has not been ingested."
            )
            return

        _show_repository_status(
            owner,
            repo,
            checkpoint,
        )

        return

    if not repos:
        print("No ingested repositories found.")
        return

    print("\nCODEBASE-RAG SYSTEM STATUS")
    print("=" * 65)

    for item in repos:

        _show_repository_status(
            item["owner"],
            item["repo"],
            checkpoint,
        )


# ---------------------------------------------------------
# SAFE GIT SYNCHRONIZATION
# ---------------------------------------------------------

def _cmd_sync(args: argparse.Namespace) -> None:

    owner, repo_name, branch = parse_github_url(
        args.url
    )

    local_path = (
        settings.clone_dir / f"{owner}__{repo_name}"
    )

    # First-time ingestion
    if not local_path.exists():

        print("Repository not cloned. Starting initial ingestion.")

        result = ingest_repository(
            args.url,
            retry_rejected=args.retry_rejected,
        )

        print(
            f"\nInitial ingestion completed: "
            f"{result.repo.full_name}"
        )

        return

    try:

        repository = Repo(local_path)

    except InvalidGitRepositoryError as error:

        raise RuntimeError(
            f"Invalid local Git repository: {local_path}"
        ) from error

    # Never overwrite local modifications.
    if repository.is_dirty(untracked_files=True):

        raise RuntimeError(
            "Synchronization stopped: local repository "
            "contains uncommitted or untracked changes. "
            "Commit, stash, or remove them before syncing."
        )

    if repository.head.is_detached:

        raise RuntimeError(
            "Synchronization stopped: repository is in "
            "detached HEAD state. Check out a branch first."
        )

    active_branch = repository.active_branch.name

    if branch and branch != active_branch:

        raise RuntimeError(
            f"Requested branch '{branch}' does not match "
            f"the checked-out branch '{active_branch}'. "
            "Switch branches explicitly before syncing."
        )

    print(f"\nSynchronizing {owner}/{repo_name}")
    print(f"Branch: {active_branch}")

    try:

        repository.remotes.origin.fetch()

        repository.git.pull(
            "--ff-only",
            "origin",
            active_branch,
        )

    except GitCommandError as error:

        raise RuntimeError(
            f"Git synchronization failed: {error}"
        ) from error

    print("\nGit update completed.")
    print("Starting RAG re-ingestion...")

    result = ingest_repository(
        args.url,
        force_refresh=False,
        retry_rejected=args.retry_rejected,
    )

    print("\nSYNCHRONIZATION COMPLETED")
    print("-" * 55)

    print(f"Repository : {result.repo.full_name}")
    print(f"Files      : {result.files_processed}")
    print(f"Chunks     : {result.chunks_created}")
    print(f"Time       : {result.elapsed_seconds:.2f}s")


# ---------------------------------------------------------
# RETRY REJECTED EMBEDDINGS
# ---------------------------------------------------------

def _cmd_retry_rejected(
    args: argparse.Namespace,
) -> None:

    owner, repo, _ = parse_github_url(
        args.url
    )

    checkpoint = EmbeddingCheckpoint()

    rejected_before = checkpoint.count_rejected(
        owner,
        repo,
    )

    if rejected_before == 0:

        print(
            f"No rejected chunks found for {owner}/{repo}."
        )
        return

    print(
        f"\nPreviously rejected chunks: {rejected_before}"
    )

    print("Starting retry process...")

    result = ingest_repository(
        args.url,
        force_refresh=False,
        retry_rejected=True,
    )

    rejected_after = checkpoint.count_rejected(
        owner,
        repo,
    )

    print("\nRETRY SUMMARY")
    print("-" * 55)

    print(f"Repository          : {result.repo.full_name}")
    print(f"Rejected before     : {rejected_before}")
    print(f"Rejected remaining  : {rejected_after}")
    print(f"Successfully cleared: {rejected_before - rejected_after}")


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Manage, ingest and query GitHub repositories "
            "using Codebase-RAG."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # INGEST
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a GitHub repository.",
    )

    ingest_parser.add_argument("url")

    ingest_parser.add_argument(
        "--force-refresh",
        action="store_true",
    )

    ingest_parser.add_argument(
        "--retry-rejected",
        action="store_true",
    )

    ingest_parser.set_defaults(
        func=_cmd_ingest,
    )

    # QUERY
    query_parser = subparsers.add_parser(
        "query",
        help="Ask questions about a repository.",
    )

    query_parser.add_argument("repo")
    query_parser.add_argument("question")

    query_parser.add_argument(
        "-n",
        "--n-results",
        type=positive_int,
        default=10,
    )

    query_parser.set_defaults(
        func=_cmd_query,
    )

    # LIST
    list_parser = subparsers.add_parser(
        "list",
        help="List ingested repositories.",
    )

    list_parser.set_defaults(
        func=_cmd_list,
    )

    # STATUS
    status_parser = subparsers.add_parser(
        "status",
        help="Show embedding and vector-store status.",
    )

    status_parser.add_argument(
        "repo",
        nargs="?",
        help="Optional owner/repo filter.",
    )

    status_parser.set_defaults(
        func=_cmd_status,
    )

    # SYNC
    sync_parser = subparsers.add_parser(
        "sync",
        help="Pull Git updates and synchronize RAG storage.",
    )

    sync_parser.add_argument("url")

    sync_parser.add_argument(
        "--retry-rejected",
        action="store_true",
    )

    sync_parser.set_defaults(
        func=_cmd_sync,
    )

    # RETRY REJECTED
    retry_parser = subparsers.add_parser(
        "retry-rejected",
        help="Retry previously rejected embeddings.",
    )

    retry_parser.add_argument("url")

    retry_parser.set_defaults(
        func=_cmd_retry_rejected,
    )

    return parser


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)

    except (
        InvalidGitHubUrlError,
        RepoCloneError,
        GitCommandError,
        RuntimeError,
    ) as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )

    main()

"""
Command-line interface for codebase-rag.

Usage:
  python cli.py ingest <github_url> [--force-refresh]
  python cli.py query <owner/repo> "<question>" [-n 10]
  python cli.py list
"""

from __future__ import annotations

import argparse
import sys

from pipeline import answer_question, ingest_repository
from src.storage.vector_store import list_ingested_repos


def _cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest_repository(args.url, force_refresh=args.force_refresh)
    print(f"\nIngested {result.repo.full_name}")
    print(f"  Files processed : {result.files_processed}")
    print(f"  Chunks created  : {result.chunks_created}")
    print(f"  Time            : {result.elapsed_seconds:.1f}s")
    print(f"\nYou can now query it with:")
    print(f'  python cli.py query {result.repo.full_name} "your question here"')


def _cmd_query(args: argparse.Namespace) -> None:
    if "/" not in args.repo:
        print(f"Error: repo must be in 'owner/repo' form, got {args.repo!r}", file=sys.stderr)
        sys.exit(1)
    owner, repo = args.repo.split("/", 1)

    result = answer_question(owner, repo, args.question, n_results=args.n_results)

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            symbol = f" ({c.symbol_name})" if c.symbol_name else ""
            print(f"  [{c.index}] {c.file_path}:{c.start_line}-{c.end_line}{symbol}")


def _cmd_list(_: argparse.Namespace) -> None:
    repos = list_ingested_repos()
    if not repos:
        print("No repos ingested yet. Run: python cli.py ingest <github_url>")
        return
    print("Ingested repos:")
    for r in repos:
        print(f"  {r['owner']}/{r['repo']}  (collection: {r['collection']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="Query GitHub codebases in natural language.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Clone, chunk, embed, and store a GitHub repo")
    ingest_parser.add_argument("url", help="GitHub repository URL")
    ingest_parser.add_argument(
        "--force-refresh", action="store_true", help="Re-clone even if the repo was already ingested"
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question about an ingested repo")
    query_parser.add_argument("repo", help="Repo in 'owner/repo' form, e.g. psf/requests")
    query_parser.add_argument("question", help="Your question about the codebase")
    query_parser.add_argument("-n", "--n-results", type=int, default=10, help="Number of chunks to retrieve")
    query_parser.set_defaults(func=_cmd_query)

    list_parser = subparsers.add_parser("list", help="List repos that have been ingested")
    list_parser.set_defaults(func=_cmd_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
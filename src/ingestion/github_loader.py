"""
Handles turning a GitHub URL into a local, shallow-cloned repository
ready for ingestion.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

from config import settings

logger = logging.getLogger(__name__)

_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?"
    r"(?:/tree/(?P<branch>[A-Za-z0-9_./-]+))?"
    r"/?$"
)


class InvalidGitHubUrlError(ValueError):
    """Raised when a URL doesn't look like a valid GitHub repo URL."""


class RepoCloneError(RuntimeError):
    """Raised when git clone fails (network, auth, repo not found, etc.)."""


@dataclass
class RepoInfo:
    owner: str
    name: str
    branch: str | None
    source_url: str
    local_path: Path

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


def parse_github_url(url: str) -> tuple[str, str, str | None]:
    """
    Extract (owner, repo, branch) from a GitHub URL.

    Accepts forms like:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      github.com/owner/repo
      https://github.com/owner/repo/tree/branch-name
    """
    match = _GITHUB_URL_RE.match(url.strip())
    if not match:
        raise InvalidGitHubUrlError(f"'{url}' doesn't look like a valid GitHub repository URL")

    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    return owner, repo, branch


def _authenticated_clone_url(owner: str, repo: str) -> str:
    """Build the clone URL, injecting a token if one is configured (for private repos / rate limits)."""
    if settings.github_token:
        return f"https://{settings.github_token}@github.com/{owner}/{repo}.git"
    return f"https://github.com/{owner}/{repo}.git"


def clone_repo(url: str, force_refresh: bool = False) -> RepoInfo:
    """
    Shallow-clone a GitHub repository (depth=1) into settings.clone_dir.

    If the repo was already cloned previously, it's reused unless
    force_refresh=True, in which case it's deleted and re-cloned.
    """
    owner, repo, branch = parse_github_url(url)
    local_path = settings.clone_dir / f"{owner}__{repo}"

    if local_path.exists():
        if force_refresh:
            logger.info("Removing existing clone at %s for refresh", local_path)
            shutil.rmtree(local_path)
        else:
            logger.info("Repo already cloned at %s, reusing", local_path)
            return RepoInfo(owner=owner, name=repo, branch=branch, source_url=url, local_path=local_path)

    clone_url = _authenticated_clone_url(owner, repo)
    clone_kwargs = {"depth": 1}
    if branch:
        clone_kwargs["branch"] = branch

    logger.info("Cloning %s/%s (branch=%s) into %s", owner, repo, branch or "default", local_path)
    try:
        Repo.clone_from(clone_url, local_path, **clone_kwargs)
    except GitCommandError as e:
        # Clean up a partial clone so a retry doesn't hit the "already exists" branch
        if local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)
        raise RepoCloneError(
            f"Failed to clone {owner}/{repo}: {e.stderr.strip() if e.stderr else e}"
        ) from e

    return RepoInfo(owner=owner, name=repo, branch=branch, source_url=url, local_path=local_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/psf/requests"
    info = clone_repo(test_url)
    print(f"Cloned {info.full_name} -> {info.local_path}")
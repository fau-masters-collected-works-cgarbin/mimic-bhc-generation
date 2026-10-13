"""Provenance for run records.

The git commit is the provenance for preparation, generation, and scoring, so
each refuses to run on a dirty tree. Also file fingerprints for the cohort
build, and the launch settings the serve script recorded when it started the
model server.
"""

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file.

    Reads in chunks to keep memory usage constant for any file size.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str | None:
    """Return the full git commit hash of HEAD, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 (git on PATH is safe)
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_status_porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607 (git on PATH is safe)
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes, recorded beside the commit
    so a run made under BHC_SKIP_GIT_CHECK=1 does not pass as clean."""
    return bool(_git_status_porcelain())


def require_clean_tree() -> None:
    """Exit unless the working tree is clean, since the commit is the only provenance.

    Set BHC_SKIP_GIT_CHECK=1 to bypass this check during debugging.
    """
    if os.environ.get("BHC_SKIP_GIT_CHECK") == "1":
        log.info("Skipping git clean check (BHC_SKIP_GIT_CHECK=1)")
        return
    if git_is_dirty():
        sys.exit("Uncommitted changes detected. Commit or stash before running.")


def git_head(repo_dir: Path) -> tuple[str, str]:
    """Return (short_hash, subject) for HEAD of the repo at ``repo_dir``."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%h\t%s"],  # noqa: S607 (git on PATH is safe)
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    short_hash, subject = result.stdout.strip().split("\t", 1)
    return short_hash, subject


# serve_local_llm.sh writes one record per server start under this name. Keep the
# pattern in sync with that script.
LAUNCH_RECORD_TEMPLATE = "launch-record-{alias}.txt"


def read_launch_record(logs_dir: Path, alias: str) -> dict[str, str] | None:
    """Read the launch record serve_local_llm.sh wrote for ``alias``, or None.

    The launch script writes one KEY=VALUE record per server start so the run
    manifest can record exactly how llama-server was launched: flash attention,
    KV cache types, and batch sizes. The experiment runs in a separate process
    from the server, so the record is how it learns that configuration.

    Returns the parsed fields, or None when no record exists for the alias (for
    example, a server launched by hand outside the script).
    """
    path = logs_dir / LAUNCH_RECORD_TEMPLATE.format(alias=alias)
    if not path.is_file():
        return None
    record: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        record[key.strip()] = value.strip()
    return record

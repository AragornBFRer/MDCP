#!/usr/bin/env python3
"""Clone or update the upstream WILDS repository under external/wilds_upstream.

Run this helper via ``python -m wilds.setup_wilds_repo`` (preferred) or
``python wilds/setup_wilds_repo.py`` from the repository root.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_REPO_URL = "https://github.com/p-lambda/wilds.git"
DEFAULT_BRANCH = "main"


def run_git(args: Sequence[str], cwd: Path | None = None) -> None:
    cmd = ["git", *args]
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stderr.write("Command failed: {}\n".format(" ".join(cmd)))
        if completed.stdout:
            sys.stderr.write(completed.stdout + "\n")
        if completed.stderr:
            sys.stderr.write(completed.stderr + "\n")
        raise SystemExit(completed.returncode)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)


def ensure_repo(target_dir: Path, url: str, branch: str) -> None:
    if not target_dir.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url} into {target_dir}")
        run_git(["clone", "--branch", branch, url, str(target_dir)])
        return

    git_dir = target_dir / ".git"
    if not git_dir.exists():
        raise SystemExit(f"{target_dir} exists but is not a git repository. Remove it or choose --target.")

    print(f"Updating {target_dir} (branch {branch})")
    run_git(["fetch", "origin", branch], cwd=target_dir)
    run_git(["checkout", branch], cwd=target_dir)
    run_git(["pull", "--ff-only", "origin", branch], cwd=target_dir)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Clone or update the upstream WILDS repository.")
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="Path to the MDCP repository root (defaults to script parent)")
    parser.add_argument("--target", type=Path, default=None, help="Directory where WILDS should live (defaults to <repo-root>/external/wilds_upstream)")
    parser.add_argument("--url", type=str, default=DEFAULT_REPO_URL, help="Git URL of the WILDS repository")
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH, help="Branch to checkout (default: main)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    target_dir = (args.target or (repo_root / "external" / "wilds_upstream")).resolve()
    ensure_repo(target_dir, args.url, args.branch)
    print(f"WILDS repository ready at {target_dir}")


if __name__ == "__main__":
    main()

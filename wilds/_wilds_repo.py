from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

ENV_VAR_NAME = "WILDS_REPO"
_DEFAULT_SUBDIRS: Tuple[Tuple[str, ...], ...] = (
    ("external", "wilds_upstream"),
    ("external", "wilds"),
    ("get_data", "get_wilds_data"),
)


def _iter_candidates(search_root: Path, override: Path | None) -> Iterable[Path]:
    if override is not None:
        yield override.expanduser().resolve()
    env_override = os.environ.get(ENV_VAR_NAME)
    if env_override:
        yield Path(env_override).expanduser().resolve()
    for parts in _DEFAULT_SUBDIRS:
        yield search_root.joinpath(*parts).resolve()


def resolve_wilds_repo(
    search_root: Path,
    override: Path | None = None,
    *,
    must_exist: bool = True,
) -> Optional[Path]:
    """Return the first existing WILDS source tree based on the known layout."""

    search_root = search_root.resolve()
    for candidate in _iter_candidates(search_root, override):
        if candidate.exists():
            return candidate
    if must_exist:
        raise FileNotFoundError(
            "Could not locate the upstream WILDS repository. Clone it into "
            "'external/wilds_upstream' (see external/README.md or run python -m wilds.setup_wilds_repo) "
            "or set the WILDS_REPO environment variable."
        )
    return None


def ensure_wilds_repo_on_path(repo_path: Path | None) -> Optional[Path]:
    if repo_path is None:
        return None
    path_str = repo_path.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return repo_path


def prepare_wilds_repo(
    project_root: Path,
    override: Path | None = None,
    *,
    require_examples: bool = False,
    optional: bool = False,
) -> Optional[Path]:
    """Ensure the upstream WILDS repo is importable and optionally validate examples/ access."""

    repo = resolve_wilds_repo(project_root, override, must_exist=not optional)
    if repo is None:
        return None
    ensure_wilds_repo_on_path(repo)
    if require_examples:
        try:
            importlib.import_module("examples.models.resnet_multispectral")
        except ImportError as exc:
            raise ImportError(
                (
                    f"The upstream WILDS repository was located at {repo} but the 'examples' modules could not be "
                    "imported. Clone the official repo (git clone https://github.com/p-lambda/wilds.git external/wilds_upstream) "
                    "or set WILDS_REPO to a checkout that includes the examples directory."
                )
            ) from exc
    return repo

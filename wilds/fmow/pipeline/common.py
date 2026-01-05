from __future__ import annotations

import sys
from pathlib import Path


def import_wilds(repo_root: Path | None):
    if repo_root is not None:
        repo_root = repo_root.resolve()
        if repo_root.as_posix() not in sys.path:
            sys.path.insert(0, repo_root.as_posix())
        existing = sys.modules.get("wilds")
        if existing is not None:
            existing_path = getattr(existing, "__file__", None)
            should_reload = True
            if existing_path is not None:
                resolved = Path(existing_path).resolve()
                try:
                    resolved.relative_to(repo_root)
                    should_reload = False
                except ValueError:
                    should_reload = True
            if should_reload:
                prefixes = [name for name in list(sys.modules.keys()) if name == "wilds" or name.startswith("wilds.")]
                for name in prefixes:
                    sys.modules.pop(name, None)
    try:
        from wilds import get_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import the WILDS package. Provide --wilds-repo or install the package."
        ) from exc
    return get_dataset


def load_fmow_dataset(root_dir: Path, wilds_repo: Path | None = None, download: bool = False):
    get_dataset = import_wilds(wilds_repo)
    root_dir = root_dir.resolve()
    version_suffix = 'fmow_v1.1'
    candidate_dir = root_dir / version_suffix
    metadata_file = root_dir / 'rgb_metadata.csv'
    if not candidate_dir.exists() and metadata_file.exists():
        # root_dir already points at extracted fmow_v1.1 folder; use its parent per WILDS expectation
        root_dir = root_dir.parent
    return get_dataset(
        dataset="fmow",
        root_dir=root_dir.as_posix(),
        download=download,
        split_scheme="official",
    )

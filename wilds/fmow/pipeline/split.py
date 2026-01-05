import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .common import load_fmow_dataset


def _ensure_datetime(metadata: pd.DataFrame) -> pd.Series:
    if np.issubdtype(metadata["timestamp"].dtype, np.datetime64):
        return metadata["timestamp"].dt.tz_convert("UTC") if metadata["timestamp"].dt.tz is not None else metadata["timestamp"].dt.tz_localize("UTC")
    ts = pd.to_datetime(metadata["timestamp"], utc=True, format="mixed")
    return ts


def _stratified_split(idxs: np.ndarray, labels: np.ndarray, train_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if len(idxs) != len(labels):
        raise ValueError("idxs and labels must have the same length")
    train, holdout = [], []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_mask = labels == label
        label_idxs = idxs[label_mask]
        if len(label_idxs) == 0:
            continue
        perm = rng.permutation(label_idxs)
        cutoff = int(np.floor(train_frac * len(perm)))
        if len(label_idxs) == 1:
            cutoff = 0
        elif cutoff == len(label_idxs):
            cutoff -= 1
        train.extend(perm[:cutoff])
        holdout.extend(perm[cutoff:])
    return np.array(train, dtype=np.int64), np.array(holdout, dtype=np.int64)


def create_year_filtered_split(
    dataset,
    target_year: int = 2016,
    train_frac: float = 0.375,
    seed: int = 0,
    group_field: str = "region",
    output_dir: Path | None = None,
) -> Dict[str, np.ndarray]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)

    metadata = dataset.metadata.iloc[dataset.full_idxs].reset_index(drop=True).copy()
    metadata["dataset_index"] = np.arange(len(metadata), dtype=np.int64)
    timestamps = _ensure_datetime(metadata)
    metadata["timestamp_parsed"] = timestamps
    year_mask = metadata["timestamp_parsed"].dt.year == target_year
    filtered = metadata.loc[year_mask].copy()
    if filtered.empty:
        raise RuntimeError(f"No samples found for year {target_year}.")

    if group_field not in filtered.columns:
        raise KeyError(f"Group field '{group_field}' not present in metadata columns: {filtered.columns.tolist()}")

    train_indices: List[int] = []
    holdout_indices: List[int] = []
    per_group_counts: Dict[str, Dict[str, int]] = {}

    for group_value, group_df in filtered.groupby(group_field):
        idxs = group_df["dataset_index"].to_numpy(dtype=np.int64)
        labels = group_df["y"].to_numpy(dtype=np.int64)
        group_train, group_holdout = _stratified_split(idxs, labels, train_frac, rng)
        train_indices.extend(group_train.tolist())
        holdout_indices.extend(group_holdout.tolist())
        per_group_counts[str(group_value)] = {
            "total": int(len(group_df)),
            "train": int(len(group_train)),
            "holdout": int(len(group_holdout)),
        }

    train_idx = np.array(sorted(train_indices), dtype=np.int64)
    holdout_idx = np.array(sorted(holdout_indices), dtype=np.int64)

    summary = {
        "train_frac": train_frac,
        "seed": seed,
        "target_year": target_year,
        "n_train": int(train_idx.size),
        "n_holdout": int(holdout_idx.size),
        "group_field": group_field,
        "per_group_counts": per_group_counts,
    }

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "train_idx.npy", train_idx)
        np.save(output_dir / "holdout_idx.npy", holdout_idx)
        filtered.to_csv(output_dir / "filtered_metadata_2016.csv", index=False)
        with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {"train_idx": train_idx, "holdout_idx": holdout_idx, "summary": summary, "filtered_metadata": filtered}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 2016-only FMoW splits grouped by region.")
    parser.add_argument("--root", type=Path, required=True, help="Path to the fmow_v1.1 directory")
    parser.add_argument("--wilds-repo", type=Path, default=None, help="Path to get_data/get_wilds_data (optional)")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save index files")
    parser.add_argument("--train-frac", type=float, default=0.375, help="Fraction of data per group used for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
    parser.add_argument("--target-year", type=int, default=2016, help="Year to filter on")
    parser.add_argument("--group-field", type=str, default="region", help="Metadata column used for per-group splits")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Dict[str, np.ndarray]:
    args = parse_args(argv)
    dataset = load_fmow_dataset(args.root, args.wilds_repo)
    result = create_year_filtered_split(
        dataset=dataset,
        target_year=args.target_year,
        train_frac=args.train_frac,
        seed=args.seed,
        group_field=args.group_field,
        output_dir=args.output,
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    main()

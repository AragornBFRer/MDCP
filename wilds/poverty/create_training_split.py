#!/usr/bin/env python3
"""Create MDCP training split for the WILDS PovertyMap dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_TRAIN_FRAC = 0.375
DEFAULT_YEARS = (2014, 2015, 2016)


@dataclass
class SplitSummary:
    total_candidates: int
    train_count: int
    year_counts: dict[int, int]
    country_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_candidates": self.total_candidates,
            "train_count": self.train_count,
            "year_counts": self.year_counts,
            "country_counts": self.country_counts,
        }


def select_train_indices(
    candidate_indices: np.ndarray,
    train_frac: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < train_frac <= 1.0:
        raise ValueError("train_frac must be in (0, 1]")
    rng = np.random.default_rng(seed)
    n_candidates = candidate_indices.size
    train_count = max(1, int(np.floor(train_frac * n_candidates)))
    if train_count > n_candidates:
        train_count = n_candidates
    perm = rng.permutation(candidate_indices)
    chosen = np.sort(perm[:train_count])
    return chosen


def write_summary_markdown(
    path: Path,
    filtered: pd.DataFrame,
    train_indices: Sequence[int],
    year_counts: dict[int, int],
    country_counts: dict[str, int],
) -> None:
    train_df = filtered.loc[train_indices]
    lines = [
        "# MDCP PovertyMap Training Split",
        "",
        f"- Candidate samples (years 2014-2016): {filtered.shape[0]:,}",
        f"- Selected training samples ({len(train_indices)} entries, 37.5% of candidates)",
        "",
        "## Year Distribution",
        "| Year | Candidate Count | Train Count | Train Share (%) |",
        "|---|---|---|---|",
    ]
    total_train = max(1, len(train_indices))
    for year in sorted(year_counts):
        cand = year_counts[year]
        train = int(train_df[train_df["year"] == year].shape[0])
        share = 100.0 * train / total_train
        lines.append(f"| {year} | {cand:,} | {train:,} | {share:5.2f} |")

    lines.extend([
        "",
        "## Top Countries in Training Split",
        "| Country | Train Count | Share (%) |",
        "|---|---|---|",
    ])
    top_train = (
        train_df["country"].value_counts().head(10).sort_values(ascending=False)
    )
    for country, count in top_train.items():
        share = 100.0 * count / total_train
        lines.append(f"| {country} | {count:,} | {share:5.2f} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MDCP poverty training split")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the MDCP repository root",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help="Fraction of eligible samples to draw for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Survey years to include (inclusive)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data" / "poverty_v1.1"
    output_root = repo_root / "eval_out" / "poverty" / "splits"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = data_root / "dhs_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    year_set = set(args.years)
    filtered = metadata[metadata["year"].isin(year_set)].copy()
    if filtered.empty:
        raise RuntimeError("No samples found for requested years")

    candidate_indices = filtered.index.to_numpy(dtype=np.int64)
    train_indices = select_train_indices(candidate_indices, args.train_frac, args.seed)

    summary = SplitSummary(
        total_candidates=int(candidate_indices.size),
        train_count=int(train_indices.size),
        year_counts={int(year): int(count) for year, count in filtered["year"].value_counts().items()},
        country_counts={country: int(count) for country, count in filtered["country"].value_counts().items()},
    )

    # Save indices for downstream tasks
    split_dir = output_root
    (split_dir / "train_indices.json").write_text(
        json.dumps(train_indices.tolist(), indent=2),
        encoding="utf-8",
    )

    train_index_set = set(train_indices.tolist())
    split_csv = pd.DataFrame(
        {
            "index": candidate_indices,
            "split": ["train" if idx in train_index_set else "pending" for idx in candidate_indices],
        }
    )
    csv_path = data_root / "mdcp_split.csv"
    split_csv.to_csv(csv_path, index=False)

    (split_dir / "train_split.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )

    write_summary_markdown(
        split_dir / "split_report.md",
        filtered=filtered,
        train_indices=train_indices,
        year_counts=summary.year_counts,
        country_counts=summary.country_counts,
    )

    print(f"Training indices saved to {split_dir / 'train_indices.json'}")
    print(f"Split CSV saved to {csv_path}")
    print(f"Summary report saved to {split_dir / 'split_report.md'}")


if __name__ == "__main__":
    main()

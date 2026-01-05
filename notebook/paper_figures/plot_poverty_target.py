#!/usr/bin/env python3
"""Visualize PovertyMap response distributions overall and by urban/rural."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


warnings.filterwarnings(
    "ignore",
    message="use_inf_as_na option is deprecated and will be removed in a future version.",
    category=FutureWarning,
)


def load_metadata(repo_root: Path) -> pd.DataFrame:
    data_path = repo_root / "data" / "poverty_v1.1" / "dhs_metadata.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {data_path}")
    return pd.read_csv(data_path)


def plot_response(
    df: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
    year_start: int,
    year_end: int,
) -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        },
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    response = df["wealthpooled"].dropna()
    sns.histplot(response, bins=40, stat="density", color="#9ecae1", alpha=0.6, ax=ax, label="Overall")
    sns.kdeplot(response, color="#08519c", ax=ax)

    group_palette = {True: "#ef3b2c", False: "#31a354"}
    for is_urban in [True, False]:
        subset = df.loc[df["urban"] == is_urban, "wealthpooled"].dropna()
        label = "Urban" if is_urban else "Rural"
        if subset.empty:
            continue
        sns.kdeplot(subset, color=group_palette[is_urban], ax=ax, lw=4, label=f"{label} KDE")

    ax.set_title(None)
    # ax.set_xlabel("Wealth index (target)", fontsize=25, fontweight="bold")
    # ax.set_ylabel("Density", fontsize=25, fontweight="bold")
    # ax.tick_params(axis="both", labelsize=25, width=1.5)
    ax.set_xlabel("Wealth index (target)", fontsize=30, fontweight="bold")
    ax.set_ylabel("Density", fontsize=30, fontweight="bold")
    ax.tick_params(axis="both", labelsize=25, width=1.5)

    summary_rows = [
        {
            "group": "Overall",
            "count": len(response),
            "mean": response.mean(),
            "std": response.std(ddof=1),
            "median": response.median(),
            "year_start": year_start,
            "year_end": year_end,
        },
    ]
    for is_urban in [True, False]:
        subset = df.loc[df["urban"] == is_urban, "wealthpooled"].dropna()
        label = "Urban" if is_urban else "Rural"
        summary_rows.append(
            {
                "group": label,
                "count": len(subset),
                "mean": subset.mean() if not subset.empty else float("nan"),
                "std": subset.std(ddof=1) if len(subset) > 1 else float("nan"),
                "median": subset.median() if not subset.empty else float("nan"),
                "year_start": year_start,
                "year_end": year_end,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=len(labels),
            frameon=False,
            prop={"size": 25, "weight": "bold"},  # legend font size
        )

    fig.tight_layout(rect=(0, 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PovertyMap response distributions")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None, help="Path to save the plot (PDF)")
    parser.add_argument("--summary-output", type=Path, default=None, help="Path to save summary statistics (CSV)")
    parser.add_argument("--year-start", type=int, default=2014, help="Inclusive start year for filtering")
    parser.add_argument("--year-end", type=int, default=2016, help="Inclusive end year for filtering")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    df = load_metadata(repo_root)

    if "year" not in df.columns:
        raise KeyError("'year' column not found in poverty metadata.")

    year_start = args.year_start
    year_end = args.year_end
    if year_start > year_end:
        raise ValueError("year_start must be less than or equal to year_end")

    df_filtered = df[df["year"].between(year_start, year_end)].copy()
    if df_filtered.empty:
        raise ValueError(f"No records found between years {year_start} and {year_end}.")

    df = df_filtered

    default_output = repo_root / "eval_out" / "final_paper_ready" / "poverty_target.pdf"
    output_path = (args.output or default_output).with_suffix(".pdf").resolve()

    summary_default = output_path.with_suffix(".csv") if output_path != default_output else output_path.parent / "poverty_target_summary.csv"
    summary_path = (args.summary_output or summary_default).with_suffix(".csv").resolve()

    plot_response(df, output_path, summary_path, year_start, year_end)
    print(f"Saved response distribution plot to {output_path}")
    print(f"Saved summary statistics to {summary_path}")


if __name__ == "__main__":
    main()

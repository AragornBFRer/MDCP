"""Generate publication-ready MDCP FMoW plots.

This script loads the aggregated evaluation tables under
``eval_out/fmow*/mdcp_analysis`` and produces combined coverage/set-size
figures comparable to the poverty results. It outputs:

- subset-level coverage + set-size panels (nonpenalized MDCP only)
- subset-level coverage + set-size panels (including mimic-selected MDCP)
- aggregate coverage / worst-case coverage / set-size panels (nonpenalized)
- aggregate coverage / worst-case coverage / set-size panels (including
  mimic-selected MDCP)

All figures are saved to ``eval_out/final_paper_ready``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 12

plt.rcParams.update(
    {
        "font.size": TICK_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": LABEL_FONT_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

COVERAGE_KEY = "coverage"
OVERALL_COVERAGE_KEY = "overall_coverage"
WORST_CASE_COVERAGE_KEY = "worst_case_coverage"
WIDTH_KEY = "avg_set_size"

METHOD_BASELINE_AGG = "Baseline agg"
METHOD_MDCP_FIXED = "MDCP"
METHOD_MDCP_TUNED = "MDCP tuned"
METHOD_BASELINE_PREFIX = "Baseline "

REGION_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Other"]

PALETTE: Dict[str, str] = {
    METHOD_BASELINE_AGG: "#5E6674",
    METHOD_MDCP_FIXED: "#5C8FD4",
    METHOD_MDCP_TUNED: "#81ABDE",
    f"{METHOD_BASELINE_PREFIX}Africa": "#8F7A62",
    f"{METHOD_BASELINE_PREFIX}Americas": "#9B866C",
    f"{METHOD_BASELINE_PREFIX}Asia": "#A79276",
    f"{METHOD_BASELINE_PREFIX}Europe": "#B39E80",
    f"{METHOD_BASELINE_PREFIX}Oceania": "#BFAB8A",
    f"{METHOD_BASELINE_PREFIX}Other": "#CBB794",
}

SUBSET_LABEL_COLORS = {
    "Africa": PALETTE[f"{METHOD_BASELINE_PREFIX}Africa"],
    "Americas": PALETTE[f"{METHOD_BASELINE_PREFIX}Americas"],
    "Asia": PALETTE[f"{METHOD_BASELINE_PREFIX}Asia"],
    "Europe": PALETTE[f"{METHOD_BASELINE_PREFIX}Europe"],
    "Oceania": PALETTE[f"{METHOD_BASELINE_PREFIX}Oceania"],
    "Other": PALETTE[f"{METHOD_BASELINE_PREFIX}Other"],
    "Overall coverage": PALETTE[METHOD_MDCP_FIXED],
    "Worst-case coverage": PALETTE[METHOD_MDCP_FIXED],
}

DISPLAY_LABELS: Dict[str, str] = {
    METHOD_BASELINE_AGG: "Baseline agg",
    METHOD_MDCP_FIXED: "MDCP",
    METHOD_MDCP_TUNED: "MDCP tuned",
    f"{METHOD_BASELINE_PREFIX}Africa": "Africa",
    f"{METHOD_BASELINE_PREFIX}Americas": "Americas",
    f"{METHOD_BASELINE_PREFIX}Asia": "Asia",
    f"{METHOD_BASELINE_PREFIX}Europe": "Europe",
    f"{METHOD_BASELINE_PREFIX}Oceania": "Oceania",
    f"{METHOD_BASELINE_PREFIX}Other": "Other",
}

MIN_COVERAGE_Y = 0.8
JITTER_MAX = 0.18
COVERAGE_COMPRESS_FACTOR = 0.25
COVERAGE_PADDING = 0.01
COVERAGE_METRICS = {COVERAGE_KEY, OVERALL_COVERAGE_KEY, WORST_CASE_COVERAGE_KEY}

_WIDTH_LABEL = "Avg set size"


def _color_for_method(method: str) -> str:
    if method in PALETTE:
        return PALETTE[method]
    if method.startswith(METHOD_BASELINE_PREFIX):
        base_colors = ["#5e5e5e", "#777777", "#909090", "#a9a9a9", "#c2c2c2", "#d6d6d6"]
        idx = sum(ord(ch) for ch in method) % len(base_colors)
        PALETTE[method] = base_colors[idx]
        return PALETTE[method]
    fallback_palette = sns.color_palette("tab10")
    idx = len(PALETTE) % len(fallback_palette)
    color = matplotlib.colors.to_hex(fallback_palette[idx])
    PALETTE[method] = color
    return color


def _transform_metric_value(metric: str, value: float) -> float:
    if metric not in COVERAGE_METRICS or np.isnan(value):
        return value
    if value >= MIN_COVERAGE_Y:
        return value
    return MIN_COVERAGE_Y - (MIN_COVERAGE_Y - value) * COVERAGE_COMPRESS_FACTOR


def _coverage_axis_limits(values: np.ndarray, coverage_target: Optional[float], metric: str) -> Tuple[float, float, List[float], List[str]]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        baseline = MIN_COVERAGE_Y
        ticks = [baseline]
        return baseline - COVERAGE_PADDING, baseline + COVERAGE_PADDING, ticks, [f"{baseline:.2f}"]

    min_val = float(valid.min())
    max_val = float(valid.max())

    bottom = _transform_metric_value(metric, min_val) if min_val < MIN_COVERAGE_Y else MIN_COVERAGE_Y
    bottom = min(bottom, MIN_COVERAGE_Y - COVERAGE_PADDING)

    top_candidates = [MIN_COVERAGE_Y, _transform_metric_value(metric, max_val)]
    if coverage_target is not None and not np.isnan(coverage_target):
        top_candidates.append(_transform_metric_value(metric, float(coverage_target)))
    top = max(top_candidates)
    top = max(top, MIN_COVERAGE_Y + COVERAGE_PADDING)

    ordered_actuals: List[float] = []
    if min_val < MIN_COVERAGE_Y:
        ordered_actuals.append(min_val)
    ordered_actuals.append(MIN_COVERAGE_Y)
    if coverage_target is not None and not np.isnan(coverage_target):
        ordered_actuals.append(float(coverage_target))
    ordered_actuals.append(max_val)

    ticks: List[float] = []
    labels: List[str] = []
    seen: set[float] = set()
    for actual in ordered_actuals:
        if np.isnan(actual):
            continue
        key = round(actual, 6)
        if key in seen:
            continue
        seen.add(key)
        ticks.append(_transform_metric_value(metric, actual))
        labels.append(f"{actual:.2f}")

    order = np.argsort(ticks)
    ticks = [ticks[i] for i in order]
    labels = [labels[i] for i in order]

    bottom_limit = min(ticks) - COVERAGE_PADDING if ticks else MIN_COVERAGE_Y - COVERAGE_PADDING
    top_limit = max(ticks) + COVERAGE_PADDING if ticks else MIN_COVERAGE_Y + COVERAGE_PADDING
    return bottom_limit, top_limit, ticks, labels


def _draw_bars_with_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    method_order: Sequence[str],
    coverage_target: Optional[float],
    metric: str,
) -> None:
    means = df.groupby("method")["value"].mean()
    x_positions = range(len(method_order))
    for idx, method in enumerate(method_order):
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        mean_val = float(means.get(method, np.nan))
        color = _color_for_method(method)
        disp_mean = _transform_metric_value(metric, mean_val) if not np.isnan(mean_val) else np.nan
        ax.bar(
            idx,
            disp_mean,
            color=color,
            width=0.58,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        values = subset["value"].to_numpy(dtype=float)
        if values.size:
            disp_vals = np.array([_transform_metric_value(metric, val) for val in values])
            offsets = np.linspace(-JITTER_MAX, JITTER_MAX, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.shape, idx, dtype=float) + offsets,
                disp_vals,
                color=color,
                s=10,
                linewidths=0.0,
                zorder=3,
                clip_on=False,
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        [DISPLAY_LABELS.get(method, method) for method in method_order],
        rotation=30,
        ha="right",
        fontsize=TICK_FONT_SIZE,
    )
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    if method_order:
        ax.set_xlim(-0.5 - JITTER_MAX, len(method_order) - 0.5 + JITTER_MAX)

    if metric in COVERAGE_METRICS:
        bottom, top, ticks, labels = _coverage_axis_limits(df["value"].to_numpy(dtype=float), coverage_target, metric)
        ax.set_ylim(bottom, top)
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        if coverage_target is not None and not np.isnan(coverage_target):
            ax.axhline(
                _transform_metric_value(metric, float(coverage_target)),
                color="#666666",
                linestyle="--",
                linewidth=1.0,
            )
    else:
        if metric == WIDTH_KEY:
            ax.set_ylim(bottom=2.0)
        else:
            ax.set_ylim(bottom=0.0)

    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _rename_method(label: str) -> Optional[str]:
    if label == "Max-p aggregate":
        return METHOD_BASELINE_AGG
    if label == "MDCP (gamma=0 nonpenalized)":
        return METHOD_MDCP_FIXED
    if label == "MDCP (mimic-selected)":
        return METHOD_MDCP_TUNED
    if label.startswith("Single (") and label.endswith(")"):
        inner = label[len("Single (") : -1]
        return f"{METHOD_BASELINE_PREFIX}{inner}"
    return None


def _load_subset_metrics(input_dir: Path) -> pd.DataFrame:
    analysis_dir = input_dir.parent / "mdcp_analysis"
    table_path = analysis_dir / "tables" / "per_trial_region_metrics.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Subset metrics table not found: {table_path}. Run the MDCP analysis pipeline first."
        )
    df = pd.read_csv(table_path)
    df = df[df["metric"].isin([COVERAGE_KEY, WIDTH_KEY])].copy()
    df["subset"] = df["subset"].astype(str)
    df["method"] = df["method"].map(_rename_method)
    df = df[df["method"].notna()].copy()
    return df


def _load_overall_metrics(input_dir: Path) -> pd.DataFrame:
    analysis_dir = input_dir.parent / "mdcp_analysis"
    table_path = analysis_dir / "tables" / "per_trial_overall_metrics.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Overall metrics table not found: {table_path}. Run the MDCP analysis pipeline first."
        )
    df = pd.read_csv(table_path)
    df["method"] = df["method"].map(_rename_method)
    df = df[df["method"].notna()].copy()
    return df


def _prepare_overall_panel(
    df: pd.DataFrame,
    metrics: Sequence[str],
    alias: Dict[str, str],
    metric_key: str,
) -> pd.DataFrame:
    subset_df = df[df["metric"].isin(metrics)].copy()
    if subset_df.empty:
        return subset_df
    subset_df["subset"] = subset_df["metric"].map(alias)
    subset_df["metric"] = metric_key
    return subset_df


def _infer_coverage_target(input_dir: Path, default: float = 0.9) -> float:
    analysis_dir = input_dir.parent / "mdcp_analysis"
    manifest_path = analysis_dir / "analysis_summary.json"
    if manifest_path.exists():
        metadata = json.load(manifest_path.open()).get("metadata", {})
        target = metadata.get("target_coverage")
        if target is not None:
            return float(target)
    manifest_path = analysis_dir / "analysis_manifest.json"
    if manifest_path.exists():
        metadata = json.load(manifest_path.open()).get("metadata", {})
        target = metadata.get("target_coverage")
        if target is not None:
            return float(target)
    for trial_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        summary_path = trial_dir / "summary.json"
        if not summary_path.exists():
            continue
        config = json.load(summary_path.open()).get("config", {})
        alpha = config.get("alpha")
        if alpha is not None:
            return 1.0 - float(alpha)
    return default


def _plot_combined_panels(
    coverage_df: pd.DataFrame,
    width_df: pd.DataFrame,
    coverage_subsets: Sequence[str],
    width_subsets: Sequence[str],
    method_order: Sequence[str],
    coverage_target: Optional[float],
    width_ylabel: str,
    output_path: Path,
) -> None:
    n_cols = max(len(coverage_subsets), len(width_subsets))
    if n_cols == 0:
        raise ValueError("At least one subset must be provided to plot combined panels.")

    coverage_methods = set(coverage_df["method"].unique())
    width_methods = set(width_df["method"].unique())
    present_methods = [m for m in method_order if m in coverage_methods or m in width_methods]
    if not present_methods:
        raise RuntimeError("No methods available in the provided data to plot.")

    fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 6.4), squeeze=False)

    for col in range(n_cols):
        ax = axes[0, col]
        if col >= len(coverage_subsets):
            ax.axis("off")
            continue
        subset = coverage_subsets[col]
        subset_df = coverage_df[(coverage_df["subset"] == subset) & (coverage_df["metric"] == COVERAGE_KEY)]
        if subset_df.empty:
            ax.axis("off")
            continue
        _draw_bars_with_points(ax, subset_df, present_methods, coverage_target, COVERAGE_KEY)
        ylabel = "Coverage" if col == 0 else ""
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=9)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.text(
            0.02,
            0.94,
            subset,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="semibold",
            color=SUBSET_LABEL_COLORS.get(subset, "#333333"),
        )

    first_width_label = True
    for col in range(n_cols):
        ax = axes[1, col]
        if col >= len(width_subsets):
            ax.axis("off")
            continue
        subset = width_subsets[col]
        subset_df = width_df[(width_df["subset"] == subset) & (width_df["metric"] == WIDTH_KEY)]
        if subset_df.empty:
            ax.axis("off")
            continue
        _draw_bars_with_points(ax, subset_df, present_methods, None, WIDTH_KEY)
        ylabel = width_ylabel if first_width_label else ""
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=8)
        else:
            ax.set_ylabel("")
        if first_width_label:
            first_width_label = False
        ax.set_xlabel("Method", fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=10)
        ax.text(
            0.02,
            0.94,
            subset,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="semibold",
            color=SUBSET_LABEL_COLORS.get(subset, "#333333"),
        )

    handles = [
        mpatches.Patch(
            facecolor=_color_for_method(method),
            edgecolor="black",
            linewidth=0.6,
            label=DISPLAY_LABELS.get(method, method),
        )
        for method in present_methods
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=max(1, len(handles)),
        columnspacing=1.2,
        handlelength=1.1,
        prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
    )
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.03, right=0.99, wspace=0.3, hspace=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_horizontal_panels(
    panels: Sequence[Tuple[str, str, pd.DataFrame]],
    method_order: Sequence[str],
    coverage_target: Optional[float],
    width_ylabel: str,
    output_path: Path,
) -> None:
    n_cols = len(panels)
    if n_cols == 0:
        raise ValueError("At least one panel must be provided for horizontal plotting.")

    present_method_set: set[str] = set()
    for _, _, panel_df in panels:
        present_method_set.update(panel_df["method"].unique())
    present_methods = [m for m in method_order if m in present_method_set]
    if not present_methods:
        raise RuntimeError("No methods available in horizontal panels to plot.")

    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.3), squeeze=False)
    first_width_label = True

    for ax, (label, metric, panel_df) in zip(axes[0], panels):
        if panel_df.empty:
            ax.axis("off")
            continue
        coverage_ref = coverage_target if metric in COVERAGE_METRICS else None
        _draw_bars_with_points(ax, panel_df, present_methods, coverage_ref, metric)
        if metric in COVERAGE_METRICS:
            ax.set_ylabel(label, fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=10, color="#000000")
        else:
            ylabel = width_ylabel if first_width_label else ""
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=8)
            else:
                ax.set_ylabel("")
            if first_width_label:
                first_width_label = False
        ax.set_xlabel("Method", fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=10)

    handles = [
        mpatches.Patch(
            facecolor=_color_for_method(method),
            edgecolor="black",
            linewidth=0.6,
            label=DISPLAY_LABELS.get(method, method),
        )
        for method in present_methods
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=max(1, len(handles)),
        columnspacing=1.2,
        handlelength=1.1,
        prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
    )
    fig.subplots_adjust(top=0.87, bottom=0.3, left=0.065, right=0.99, wspace=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-ready FMoW MDCP plots")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval_out/fmow/mdcp"),
        help="Directory containing per-trial MDCP evaluation folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_out/final_paper_ready"),
        help="Directory where figures will be saved",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    subset_df = _load_subset_metrics(args.input_dir)
    if subset_df.empty:
        raise RuntimeError("No subset metrics found to plot.")

    coverage_target = _infer_coverage_target(args.input_dir)

    subset_order = [subset for subset in REGION_ORDER if subset in subset_df["subset"].unique()]
    subset_order.extend(sorted({subset for subset in subset_df["subset"].unique() if subset not in subset_order}))

    present_methods = list(subset_df["method"].unique())
    baseline_sources = [f"{METHOD_BASELINE_PREFIX}{subset}" for subset in subset_order]
    method_order_all: List[str] = [METHOD_BASELINE_AGG]
    method_order_all.extend([method for method in baseline_sources if method in present_methods])
    if METHOD_MDCP_FIXED in present_methods:
        method_order_all.append(METHOD_MDCP_FIXED)
    if METHOD_MDCP_TUNED in present_methods:
        method_order_all.append(METHOD_MDCP_TUNED)

    has_tuned = METHOD_MDCP_TUNED in present_methods
    method_order_fixed = [m for m in method_order_all if m != METHOD_MDCP_TUNED]
    subset_df_fixed = subset_df[subset_df["method"] != METHOD_MDCP_TUNED].copy() if has_tuned else subset_df
    if not method_order_fixed:
        raise RuntimeError("No methods available for nonpenalized comparison plot.")

    subset_coverage_fixed = subset_df_fixed[subset_df_fixed["metric"] == COVERAGE_KEY]
    subset_width_fixed = subset_df_fixed[subset_df_fixed["metric"] == WIDTH_KEY]
    if subset_coverage_fixed.empty or subset_width_fixed.empty:
        raise RuntimeError("Missing coverage or set-size metrics for subset plots.")

    subset_metrics_nonpen_path = args.output_dir / "fmow_subset_vanilla.pdf"

    _plot_combined_panels(
        subset_coverage_fixed,
        subset_width_fixed,
        subset_order,
        subset_order,
        method_order_fixed,
        coverage_target,
        _WIDTH_LABEL,
        subset_metrics_nonpen_path,
    )

    print(f"Saved subset metrics (vanilla): {subset_metrics_nonpen_path}")

    if has_tuned:
        subset_coverage_all = subset_df[subset_df["metric"] == COVERAGE_KEY]
        subset_width_all = subset_df[subset_df["metric"] == WIDTH_KEY]
        if subset_coverage_all.empty or subset_width_all.empty:
            raise RuntimeError("Missing coverage or set-size metrics for tuned subset plots.")

        subset_metrics_tuned_path = args.output_dir / "fmow_subset_tuned.pdf"

        _plot_combined_panels(
            subset_coverage_all,
            subset_width_all,
            subset_order,
            subset_order,
            method_order_all,
            coverage_target,
            _WIDTH_LABEL,
            subset_metrics_tuned_path,
        )

        print(f"Saved subset metrics (tuned): {subset_metrics_tuned_path}")

    overall_df = _load_overall_metrics(args.input_dir)
    if overall_df.empty:
        raise RuntimeError("No overall metrics found to plot.")

    overall_method_order_all = [m for m in method_order_all if m in overall_df["method"].unique()]
    overall_method_order_fixed = [m for m in method_order_fixed if m in overall_method_order_all]
    if not overall_method_order_fixed:
        raise RuntimeError("No methods available for nonpenalized overall comparison plot.")

    coverage_alias = {
        OVERALL_COVERAGE_KEY: "Overall coverage",
        WORST_CASE_COVERAGE_KEY: "Worst-case coverage",
    }
    overall_coverage_df = _prepare_overall_panel(overall_df, coverage_alias.keys(), coverage_alias, OVERALL_COVERAGE_KEY)
    if overall_coverage_df.empty:
        raise RuntimeError("Overall coverage metrics not found in overall metrics table.")

    width_alias = {WIDTH_KEY: "Average set size"}
    overall_width_df = _prepare_overall_panel(overall_df, width_alias.keys(), width_alias, WIDTH_KEY)
    if overall_width_df.empty:
        raise RuntimeError("Average set size metric not found in overall metrics table.")

    overall_coverage_fixed = overall_coverage_df[overall_coverage_df["method"] != METHOD_MDCP_TUNED]
    overall_width_fixed = overall_width_df[overall_width_df["method"] != METHOD_MDCP_TUNED]
    if overall_coverage_fixed.empty or overall_width_fixed.empty:
        raise RuntimeError("Missing coverage or set-size metrics for overall plots.")

    coverage_subsets_overall = list(coverage_alias.values())
    width_subsets_overall = list(width_alias.values())

    overall_metrics_nonpen_path = args.output_dir / "fmow_overall_vanilla.pdf"

    panels_nonpen: List[Tuple[str, str, pd.DataFrame]] = []
    for label in coverage_subsets_overall:
        panel_df = overall_coverage_fixed[overall_coverage_fixed["subset"] == label]
        panels_nonpen.append((label, OVERALL_COVERAGE_KEY, panel_df))
    for label in width_subsets_overall:
        panel_df = overall_width_fixed[overall_width_fixed["subset"] == label]
        panels_nonpen.append((label, WIDTH_KEY, panel_df))

    _plot_horizontal_panels(
        panels_nonpen,
        overall_method_order_fixed,
        coverage_target,
        _WIDTH_LABEL,
        overall_metrics_nonpen_path,
    )

    print(f"Saved overall metrics (vanilla): {overall_metrics_nonpen_path}")

    if has_tuned:
        overall_metrics_tuned_path = args.output_dir / "fmow_overall_tuned.pdf"

        panels_tuned: List[Tuple[str, str, pd.DataFrame]] = []
        for label in coverage_subsets_overall:
            panel_df = overall_coverage_df[overall_coverage_df["subset"] == label]
            panels_tuned.append((label, OVERALL_COVERAGE_KEY, panel_df))
        for label in width_subsets_overall:
            panel_df = overall_width_df[overall_width_df["subset"] == label]
            panels_tuned.append((label, WIDTH_KEY, panel_df))

        _plot_horizontal_panels(
            panels_tuned,
            overall_method_order_all,
            coverage_target,
            _WIDTH_LABEL,
            overall_metrics_tuned_path,
        )

        print(f"Saved overall metrics (tuned): {overall_metrics_tuned_path}")


if __name__ == "__main__":
    main()

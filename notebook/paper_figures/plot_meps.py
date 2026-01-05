"""Paper-ready MEPS MDCP comparison plots.

This script reconstructs MDCP and baseline metrics from the raw MEPS
evaluation payloads and renders concise bar+dot plots that are ready for
statistics-style publications. The workflow follows the repository's
MDCP mimic calibration logic: for each trial we select the penalty
parameter that performs best on the mimic test split (subject to the
coverage target) and evaluate that choice on the true test set. Both
non-penalized MDCP and mimic-selected MDCP are compared against the
single-source baselines (shown individually) and the max-aggregated
baseline. Outputs are saved under ``eval_out/final_paper_ready``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meps.plot import _extract_metrics, _load_payload


METHOD_BASELINE_MAX = "Baseline max-agg"
METHOD_BASELINE_SRC_TEMPLATE = "Baseline source {idx}"
METHOD_BASELINE_SRC_PREFIX = "Baseline source "
METHOD_MDCP_NONPEN = "MDCP nonpen"
METHOD_MDCP_MIMIC = "MDCP mimic sel"

DISPLAY_NAMES: Dict[str, str] = {
    METHOD_BASELINE_MAX: "Baseline agg",
    METHOD_BASELINE_SRC_TEMPLATE.format(idx=0): "Non-white",
    METHOD_BASELINE_SRC_TEMPLATE.format(idx=1): "White",
    METHOD_MDCP_NONPEN: "MDCP",
    METHOD_MDCP_MIMIC: "MDCP tuned",
}

PALETTE: Dict[str, str] = {
    METHOD_BASELINE_MAX: "#5E6674",
    METHOD_BASELINE_SRC_TEMPLATE.format(idx=0): "#9B866C",
    METHOD_BASELINE_SRC_TEMPLATE.format(idx=1): "#BFAB8A",
    METHOD_MDCP_NONPEN: "#5C8FD4",
    METHOD_MDCP_MIMIC: "#81ABDE",
}

BASELINE_SOURCE_COLORS = ["#8F7A62", "#9B866C", "#A79276", "#B39E80", "#BFAB8A", "#CBB794"]

METRIC_CONFIGS: Sequence[Tuple[str, str]] = (
    ("overall_coverage", "Overall coverage"),
    ("worst_case_coverage", "Worst-case coverage"),
    ("avg_width", "Avg interval width"),
)

METRIC_TYPES: Dict[str, str] = {
    "overall_coverage": "coverage",
    "worst_case_coverage": "coverage",
    "avg_width": "width",
}

COVERAGE_FLOOR = 0.8
COVERAGE_COMPRESS = 0.25
SCATTER_JITTER = 0.18
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 12
COLUMN_HEADER_PAD = 0.012
MIN_WIDTH_Y = 2.0


def _transform_coverage(value: float) -> float:
    if math.isnan(value) or value >= COVERAGE_FLOOR:
        return value
    return COVERAGE_FLOOR - (COVERAGE_FLOOR - value) * COVERAGE_COMPRESS


def _coverage_axis_limits(
    values: np.ndarray,
    coverage_target: Optional[float],
) -> Tuple[float, float, List[float], List[str]]:
    cleaned = values[~np.isnan(values)]
    if cleaned.size == 0:
        ticks = [_transform_coverage(COVERAGE_FLOOR)]
        labels = [f"{COVERAGE_FLOOR:.2f}"]
        bottom = COVERAGE_FLOOR - 0.02
        top = COVERAGE_FLOOR + 0.02
        return bottom, top, ticks, labels

    actual_values: List[float] = [float(cleaned.min()), float(cleaned.max()), COVERAGE_FLOOR]
    if coverage_target is not None and not math.isnan(coverage_target):
        actual_values.append(float(coverage_target))

    ticks: List[float] = []
    labels: List[str] = []
    seen: set[float] = set()
    for actual in sorted(actual_values):
        if math.isnan(actual):
            continue
        key = round(actual, 5)
        if key in seen:
            continue
        seen.add(key)
        ticks.append(_transform_coverage(actual))
        labels.append(f"{actual:.2f}")

    bottom = min(ticks) - 0.015 if ticks else COVERAGE_FLOOR - 0.02
    top = max(ticks) + 0.02 if ticks else COVERAGE_FLOOR + 0.05
    bottom = min(bottom, _transform_coverage(float(cleaned.min())) - 0.015)
    top = max(top, _transform_coverage(float(cleaned.max())) + 0.02)
    bottom = max(bottom, _transform_coverage(COVERAGE_FLOOR) - 0.005)
    top = max(top, 1.0)
    return bottom, top, ticks, labels


def _color_for_method(method: str) -> str:
    if method in PALETTE:
        return PALETTE[method]
    if method.startswith(METHOD_BASELINE_SRC_PREFIX):
        try:
            idx = int(method.replace(METHOD_BASELINE_SRC_PREFIX, ""))
        except ValueError:
            idx = 0
        color = BASELINE_SOURCE_COLORS[idx % len(BASELINE_SOURCE_COLORS)]
        PALETTE[method] = color
        return color
    palette = matplotlib.cm.get_cmap("tab10")
    idx = len(PALETTE) % palette.N
    color = matplotlib.colors.to_hex(palette(idx))
    PALETTE[method] = color
    return color


def _display_name(method: str) -> str:
    if method in DISPLAY_NAMES:
        return DISPLAY_NAMES[method]
    if method.startswith(METHOD_BASELINE_SRC_PREFIX):
        suffix = method.replace(METHOD_BASELINE_SRC_PREFIX, "").strip()
        if suffix.isdigit():
            return f"Baseline src {suffix}"
    return method


def _pretty_source_label(raw: object) -> str:
    if raw is None:
        return "Source"
    text = str(raw).strip()
    if not text:
        return "Source"
    lowered = text.lower()
    if lowered in {"non-white", "non white", "nonwhite"}:
        return "Non-white"
    if lowered in {"white"}:
        return "White"
    return text.title()


def _seed_from_path(path: Path) -> Optional[int]:
    stem = path.stem
    if "seed_" not in stem:
        return None
    try:
        return int(stem.split("seed_")[-1].split("_")[0])
    except ValueError:
        return None


def _add_metric_rows(
    container: List[Dict[str, object]],
    base_info: Dict[str, object],
    metrics: Dict[str, float],
) -> None:
    for metric_key, metric_value in metrics.items():
        if metric_key not in METRIC_TYPES:
            continue
        if metric_value is None or math.isnan(metric_value):
            continue
        entry = dict(base_info)
        entry["metric"] = metric_key
        entry["value"] = float(metric_value)
        container.append(entry)


def _select_mimic_entry(
    entries: Iterable[Dict[str, object]],
    coverage_target: float,
) -> Optional[Dict[str, object]]:
    chosen: Optional[Dict[str, object]] = None
    chosen_key: Optional[Tuple[float, float]] = None
    fallback: Optional[Dict[str, object]] = None
    fallback_key: Optional[Tuple[float, float]] = None

    for entry in entries:
        mimic_metrics = entry.get("mimic_metrics")
        if not isinstance(mimic_metrics, dict):
            continue
        extracted = _extract_metrics(mimic_metrics)
        coverage = extracted.get("overall_coverage")
        width = extracted.get("avg_width")
        if coverage is None or width is None:
            continue
        if math.isnan(coverage) or math.isnan(width):
            continue
        candidate_key = (float(width), -float(coverage))
        fallback_candidate = (-float(coverage), float(width))
        if coverage >= coverage_target:
            if chosen_key is None or candidate_key < chosen_key:
                chosen = entry
                chosen_key = candidate_key
        if fallback_key is None or fallback_candidate < fallback_key:
            fallback = entry
            fallback_key = fallback_candidate

    return chosen if chosen is not None else fallback


def _gather_records(eval_root: Path, coverage_target: float) -> pd.DataFrame:
    if not eval_root.exists():
        raise FileNotFoundError(f"Evaluation root not found: {eval_root}")

    records: List[Dict[str, object]] = []

    trial_dirs = sorted(path for path in eval_root.iterdir() if path.is_dir())
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories discovered under {eval_root}")

    for trial_dir in trial_dirs:
        payload_paths = sorted(trial_dir.glob("meps_panel_*_alpha_*.npz"))
        for payload_path in payload_paths:
            payload = _load_payload(payload_path)
            panel = str(payload.get("panel"))

            metadata = payload.get("metadata")
            seed = getattr(metadata, "random_seed", None)
            if seed is None:
                seed = _seed_from_path(payload_path)
            run_id = f"panel{panel}_seed{seed}" if seed is not None else f"panel{panel}_{trial_dir.name}"

            base_info = {"panel": panel, "run_id": run_id}

            source_label_map: Dict[str, str] = {}
            source_mapping = getattr(metadata, "source_mapping", None) if metadata is not None else None
            if isinstance(source_mapping, dict):
                for key, value in source_mapping.items():
                    try:
                        idx = int(key)
                    except (TypeError, ValueError):
                        continue
                    method_label = METHOD_BASELINE_SRC_TEMPLATE.format(idx=idx)
                    pretty = _pretty_source_label(value)
                    DISPLAY_NAMES[method_label] = pretty
                    source_label_map[str(idx)] = pretty
                    source_label_map[str(float(idx))] = pretty

            baseline = payload.get("baseline", {})
            if isinstance(baseline, dict):
                metrics = baseline.get("Max Aggregation")
                if isinstance(metrics, dict):
                    extracted = _extract_metrics(metrics)
                    info = dict(base_info)
                    info["method"] = METHOD_BASELINE_MAX
                    _add_metric_rows(records, info, extracted)

            comprehensive = payload.get("baseline_comprehensive", {})
            if isinstance(comprehensive, dict):
                for idx in range(4):
                    key = f"Source_{idx}"
                    if key not in comprehensive:
                        continue
                    method_label = METHOD_BASELINE_SRC_TEMPLATE.format(idx=idx)
                    overall_metrics = comprehensive[key].get("Overall")
                    if not isinstance(overall_metrics, dict):
                        continue
                    extracted = _extract_metrics(overall_metrics)
                    info = dict(base_info)
                    info["method"] = method_label
                    pretty = source_label_map.get(str(idx))
                    if pretty:
                        DISPLAY_NAMES[method_label] = pretty
                    _add_metric_rows(records, info, extracted)

            gamma_entries = payload.get("mdcp_gamma_results", [])
            if isinstance(gamma_entries, list) and gamma_entries:
                nonpen_entry = next(
                    (entry for entry in gamma_entries if float(entry.get("gamma", float("nan"))) == 0.0),
                    None,
                )
                if nonpen_entry is None:
                    for entry in gamma_entries:
                        name = entry.get("gamma_name", "")
                        if isinstance(name, str) and "g1_0" in name and "g2_0" in name:
                            nonpen_entry = entry
                            break
                if nonpen_entry is not None:
                    metrics = nonpen_entry.get("metrics")
                    if isinstance(metrics, dict):
                        extracted = _extract_metrics(metrics)
                        info = dict(base_info)
                        info["method"] = METHOD_MDCP_NONPEN
                        _add_metric_rows(records, info, extracted)

                mimic_entry = _select_mimic_entry(gamma_entries, coverage_target)
                if mimic_entry is not None:
                    metrics = mimic_entry.get("metrics")
                    if isinstance(metrics, dict):
                        extracted = _extract_metrics(metrics)
                        info = dict(base_info)
                        info["method"] = METHOD_MDCP_MIMIC
                        _add_metric_rows(records, info, extracted)

    if not records:
        raise RuntimeError("No metrics collected from the provided evaluation payloads.")

    df = pd.DataFrame(records)
    df["panel"] = df["panel"].astype(str)
    df["run_id"] = df["run_id"].astype(str)
    return df


def _plot_metric_panels(
    df: pd.DataFrame,
    methods: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    coverage_target: Optional[float],
    output_path: Path,
) -> None:
    panels = sorted(
        df["panel"].unique(),
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )
    if not panels:
        raise RuntimeError("No panels available for plotting.")

    n_rows = len(panels)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_rows,
        n_metrics,
        figsize=(3.4 * n_metrics, 2.4 * n_rows + 0.9),
        squeeze=False,
    )

    label_axes: List[Optional[Axes]] = [None] * n_rows
    row_anchor_axes: List[Optional[Axes]] = [None] * n_rows

    for row_idx, panel in enumerate(panels):
        panel_subset = df[df["panel"] == panel]
        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            subset = panel_subset[panel_subset["metric"] == metric_key]
            present_methods = [m for m in methods if m in subset["method"].unique()]
            if not present_methods:
                ax.axis("off")
                continue

            x_positions = np.arange(len(present_methods), dtype=float)
            for xpos, method in zip(x_positions, present_methods):
                method_subset = subset[subset["method"] == method]["value"].to_numpy(dtype=float)
                if method_subset.size == 0:
                    continue
                mean_val = float(np.nanmean(method_subset))
                color = _color_for_method(method)

                if METRIC_TYPES.get(metric_key) == "coverage":
                    disp_mean = _transform_coverage(mean_val)
                else:
                    disp_mean = mean_val

                ax.bar(
                    xpos,
                    disp_mean,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    width=0.58,
                    alpha=0.9,
                )

                if method_subset.size:
                    if METRIC_TYPES.get(metric_key) == "coverage":
                        disp_values = np.array([_transform_coverage(v) for v in method_subset], dtype=float)
                    else:
                        disp_values = method_subset
                    if disp_values.size > 1:
                        offsets = np.linspace(-SCATTER_JITTER, SCATTER_JITTER, disp_values.size)
                    else:
                        offsets = np.zeros_like(disp_values, dtype=float)
                    ax.scatter(
                        np.full_like(disp_values, xpos, dtype=float) + offsets,
                        disp_values,
                        color=color,
                        s=12,
                        linewidths=0.0,
                        alpha=0.9,
                        zorder=3,
                        clip_on=False,
                    )

            ax.set_xticks(x_positions)
            tick_labels = [_display_name(method) for method in present_methods]
            ax.set_xticklabels(tick_labels, rotation=18, ha="right", fontsize=TICK_FONT_SIZE)
            ax.set_xlim(-0.5 - SCATTER_JITTER, len(present_methods) - 0.5 + SCATTER_JITTER)

            if METRIC_TYPES.get(metric_key) == "coverage":
                values = subset[subset["method"].isin(present_methods)]["value"].to_numpy(dtype=float)
                bottom, top, ticks, tick_labels = _coverage_axis_limits(values, coverage_target)
                ax.set_ylim(bottom, top)
                if ticks:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels, fontsize=TICK_FONT_SIZE)
                if coverage_target is not None and not math.isnan(coverage_target):
                    ax.axhline(
                        y=_transform_coverage(float(coverage_target)),
                        linestyle="--",
                        color="#626262",
                        linewidth=0.9,
                    )
            else:
                values = subset[subset["method"].isin(present_methods)]["value"].to_numpy(dtype=float)
                finite = values[~np.isnan(values)]
                if finite.size:
                    top_val = float(finite.max())
                else:
                    top_val = MIN_WIDTH_Y + 0.5
                top_val = max(top_val * 1.05, MIN_WIDTH_Y + 0.1)
                ax.set_ylim(MIN_WIDTH_Y, top_val)

            ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

            panel_label = str(panels[row_idx])  # Panel 19, 20, 21
            ax.set_ylabel(f"Panel {panel_label}")
            ax.set_xlabel("")

            if label_axes[row_idx] is None or col_idx == n_metrics // 2:
                label_axes[row_idx] = ax

            if row_anchor_axes[row_idx] is None:
                row_anchor_axes[row_idx] = ax

    for row_idx, ax in enumerate(label_axes):
        if ax is not None and row_idx == 2:
            ax.set_xlabel("Method", fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=8)

    handles = [
        mpatches.Patch(
            facecolor=_color_for_method(method),
            edgecolor="black",
            linewidth=0.6,
            label=_display_name(method),
        )
        for method in methods
        if method in df["method"].unique()
    ]
    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            ncol=len(handles),
            handlelength=1.1,
            columnspacing=1.1,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )

    # fig.subplots_adjust(top=0.93, bottom=0.09, left=0.07, right=0.99, hspace=0.4, wspace=0.3)
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.07, right=0.99, hspace=0.4, wspace=0.3)
    fig.canvas.draw()

    for col_idx, (_, metric_label) in enumerate(metrics):
        ax = axes[0, col_idx]
        if not ax.has_data():
            continue
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y1 + COLUMN_HEADER_PAD,
            metric_label,
            va="bottom",
            ha="center",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _summarize_metrics(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    subset = df[df["method"].isin(methods)].copy()
    summary = (
        subset.groupby(["method", "metric"])
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
        .reset_index()
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready MEPS MDCP comparison plots.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_out/meps"),
        help="Directory containing MEPS raw evaluation payloads (NPZ files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_out/final_paper_ready"),
        help="Destination directory for generated figures.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.9,
        help="Target coverage used when selecting gamma on the mimic set.",
    )
    args = parser.parse_args(argv)

    records = _gather_records(args.input, args.coverage_target)

    present_methods = set(records["method"].unique())
    source_methods = sorted(
        (m for m in present_methods if m.startswith(METHOD_BASELINE_SRC_PREFIX)),
        key=lambda name: int(name.replace(METHOD_BASELINE_SRC_PREFIX, ""))
        if name.replace(METHOD_BASELINE_SRC_PREFIX, "").isdigit()
        else 0,
    )

    nonpen_methods = [METHOD_BASELINE_MAX]
    nonpen_methods.extend(source_methods)
    if METHOD_MDCP_NONPEN in present_methods:
        nonpen_methods.append(METHOD_MDCP_NONPEN)

    combined_methods = list(nonpen_methods)
    if METHOD_MDCP_MIMIC in present_methods:
        combined_methods.append(METHOD_MDCP_MIMIC)

    nonpen_path = args.output / "meps_overall_vanilla.pdf"
    _plot_metric_panels(records, nonpen_methods, METRIC_CONFIGS, args.coverage_target, nonpen_path)
    print(f"Saved overall metrics (vanilla): {nonpen_path}")

    combined_path = args.output / "meps_overall_tuned.pdf"
    _plot_metric_panels(records, combined_methods, METRIC_CONFIGS, args.coverage_target, combined_path)
    print(f"Saved overall metrics (tuned): {combined_path}")

    summary_path = args.output / "meps_overall_summary.csv"
    summary = _summarize_metrics(records, combined_methods)
    summary["method"] = summary["method"].apply(_display_name)
    summary.to_csv(summary_path, index=False)
    print(f"Saved overall summary: {summary_path}")


if __name__ == "__main__":
    main()

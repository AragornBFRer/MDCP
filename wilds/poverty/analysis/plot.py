#!/usr/bin/env python3
"""Aggregate and visualise PovertyMap MDCP evaluation trials with mimic-selected penalties."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
POVERTY_ROOT = SCRIPT_DIR.parent
WILDS_ROOT = POVERTY_ROOT.parent
REPO_ROOT = WILDS_ROOT.parent
for candidate in (REPO_ROOT, REPO_ROOT / "model", REPO_ROOT / "notebook"):
    candidate = candidate.resolve()
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

 

DEFAULT_GAMMA_GRID: Tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MDCP_NONPENALIZED_GAMMA: float = 0.0
BASELINE_LABEL = "Max-p aggregate"
MDCP_NONPENALIZED_LABEL = "MDCP (gamma=0 nonpenalized)"
MDCP_SELECTED_LABEL = "MDCP (selected gamma)"
TRUE_SPLIT = "true"
SOURCE_LABEL_OVERRIDES = {"rural": "Rural", "urban": "Urban"}
BASE_SINGLE_COLORS = ["#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#a6cee3", "#f781bf"]
GAMMA_COLOR_CYCLE = sns.color_palette("colorblind", 10)

PALETTE: Dict[str, str] = {
    BASELINE_LABEL: "#e41a1c",
    MDCP_NONPENALIZED_LABEL: "#377eb8",
    MDCP_SELECTED_LABEL: "#ff8c00",
}

OVERALL_METRICS = ["overall_coverage", "worst_case_coverage", "avg_width"]
METRIC_LABELS = {
    "overall_coverage": "Coverage",
    "worst_case_coverage": "Worst-case coverage",
    "avg_width": "Average width",
    "coverage": "Coverage",
}


@dataclass(frozen=True)
class TrialMetadata:
    alpha: float
    train_frac: float
    cal_frac: float
    test_frac: float
    gamma_values: Tuple[float, ...]
    y_grid_size: int
    y_margin: float


def _existing_path_with_fallback(path: Path) -> Path:
    candidates: List[Path] = [path]
    if not path.is_absolute():
        candidates.append((REPO_ROOT / path).resolve())
    if "hpc3_trials" in path.parts:
        try:
            mdcp_index = path.parts.index("mdcp")
            fallback = Path(*path.parts[: mdcp_index + 1])
            candidates.append(fallback if fallback.is_absolute() else (REPO_ROOT / fallback).resolve())
        except ValueError:
            candidates.append((REPO_ROOT / "eval_out" / "poverty" / "mdcp").resolve())
    else:
        candidates.append((REPO_ROOT / "eval_out" / "poverty" / "mdcp").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise PovertyMap MDCP evaluation results and generate plots.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval_out/poverty/mdcp"),
        help="Directory containing MDCP evaluation .npz artifacts (default: eval_out/poverty/mdcp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_out/poverty/mdcp_analysis"),
        help="Directory to store aggregated tables and figures (default: eval_out/poverty/mdcp_analysis).",
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=300,
        help="Resolution for saved figures (default: 300).",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=DEFAULT_GAMMA_GRID,
        help="Gamma values to highlight in gamma trend plots (default: MDCP evaluation grid).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _compute_overall_metrics(payload: Mapping[str, object]) -> Dict[str, float]:
    coverage = _to_float(payload.get("coverage"))
    avg_width = _to_float(payload.get("avg_width"))
    individual_cov = np.asarray(payload.get("individual_coverage", []), dtype=float)
    individual_widths = np.asarray(payload.get("individual_widths", []), dtype=float)
    worst_cov = float(np.nanmin(individual_cov)) if individual_cov.size else math.nan
    worst_width = float(np.nanmax(individual_widths)) if individual_widths.size else math.nan
    return {
        "overall_coverage": coverage,
        "worst_case_coverage": worst_cov,
        "avg_width": avg_width,
        "worst_case_width": worst_width,
    }


def _iter_subset_records(
    metrics: Mapping[str, object],
    source_id_to_label: Mapping[int, str],
) -> Iterable[Tuple[str, float, float]]:
    coverage = np.asarray(metrics.get("individual_coverage", []), dtype=float)
    widths = np.asarray(metrics.get("individual_widths", []), dtype=float)
    unique_sources = metrics.get("unique_sources")
    if unique_sources is None:
        source_ids = np.arange(coverage.size, dtype=int)
    else:
        source_ids = np.asarray(unique_sources, dtype=int)
    for idx, source_id in enumerate(source_ids):
        label = source_id_to_label.get(int(source_id), f"Source {int(source_id)}")
        cov_val = coverage[idx] if idx < coverage.size else math.nan
        width_val = widths[idx] if idx < widths.size else math.nan
        yield label, float(cov_val), float(width_val)


def _append_overall_rows(
    rows: List[Dict[str, object]],
    run_id: str,
    trial_index: int,
    method: str,
    metrics: Mapping[str, object],
    split: str,
    gamma: float | None,
) -> None:
    aggregate = _compute_overall_metrics(metrics)
    for metric_name, value in aggregate.items():
        if metric_name == "worst_case_width":
            continue
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "method": method,
                "metric": metric_name,
                "value": value,
                "split": split,
                "gamma": gamma if gamma is not None else math.nan,
            }
        )


def _append_subset_rows(
    rows: List[Dict[str, object]],
    run_id: str,
    trial_index: int,
    method: str,
    metrics: Mapping[str, object],
    split: str,
    source_id_to_label: Mapping[int, str],
    gamma: float | None,
) -> None:
    for subset_label, cov_val, width_val in _iter_subset_records(metrics, source_id_to_label):
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "method": method,
                "subset": subset_label,
                "metric": "coverage",
                "value": cov_val,
                "split": split,
                "gamma": gamma if gamma is not None else math.nan,
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "method": method,
                "subset": subset_label,
                "metric": "avg_width",
                "value": width_val,
                "split": split,
                "gamma": gamma if gamma is not None else math.nan,
            }
        )


def _append_gamma_rows(
    rows: List[Dict[str, object]],
    run_id: str,
    trial_index: int,
    gamma_value: float,
    metrics: Mapping[str, object],
    split: str,
) -> None:
    aggregate = _compute_overall_metrics(metrics)
    for metric_name, value in aggregate.items():
        if metric_name == "worst_case_width":
            continue
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "gamma": gamma_value,
                "metric": metric_name,
                "value": value,
                "split": split,
            }
        )


def _find_metrics_for_gamma(
    metrics_map: Mapping[str, Mapping[str, object]],
    target_gamma: float,
    tol: float = 1e-9,
) -> Optional[Mapping[str, object]]:
    for key, metrics in metrics_map.items():
        gamma_candidate = _to_float(key)
        if math.isnan(gamma_candidate):
            gamma_candidate = _to_float(metrics.get("gamma"))
        if math.isnan(gamma_candidate):
            continue
        if abs(gamma_candidate - target_gamma) <= tol:
            return metrics
    return None


def _select_best_gamma(
    mimic_metrics: Mapping[str, Mapping[str, object]],
    target_cov: float,
) -> Optional[Tuple[float, Mapping[str, object]]]:
    feasible: List[Tuple[float, float, float, Mapping[str, object]]] = []
    fallback: List[Tuple[float, float, float, Mapping[str, object]]] = []
    for gamma_key, metrics in mimic_metrics.items():
        gamma_value = _to_float(gamma_key)
        if math.isnan(gamma_value):
            gamma_value = _to_float(metrics.get("gamma"))
        coverage = _to_float(metrics.get("coverage"))
        avg_width = _to_float(metrics.get("avg_width"))
        if math.isnan(gamma_value) or math.isnan(coverage) or math.isnan(avg_width):
            continue
        record = (gamma_value, coverage, avg_width, metrics)
        if coverage >= target_cov:
            feasible.append(record)
        else:
            fallback.append(record)
    if feasible:
        feasible.sort(key=lambda item: (item[2], -item[1], abs(item[0])))
        best = feasible[0]
        return best[0], best[3]
    if fallback:
        fallback.sort(key=lambda item: (-item[1], item[2], abs(item[0])))
        best = fallback[0]
        return best[0], best[3]
    return None


def _select_entry_by_mimic(
    entries: Sequence[Mapping[str, object]],
    target_cov: float,
) -> Optional[Mapping[str, object]]:
    feasible: List[Tuple[float, float, float, Mapping[str, object]]] = []
    fallback: List[Tuple[float, float, float, Mapping[str, object]]] = []
    for entry in entries:
        gamma_value = _to_float(entry.get("gamma"))
        mimic_metrics = entry.get("mimic_metrics")
        if mimic_metrics is None or not isinstance(mimic_metrics, Mapping):
            continue
        mimic_overall = _compute_overall_metrics(mimic_metrics)
        coverage = float(mimic_overall.get("overall_coverage", math.nan))
        avg_width = float(mimic_overall.get("avg_width", math.nan))
        if math.isnan(coverage) or math.isnan(avg_width) or math.isnan(gamma_value):
            continue
        if coverage >= target_cov:
            feasible.append((avg_width, -coverage, abs(gamma_value), entry))
        else:
            fallback.append((-coverage, avg_width, abs(gamma_value), entry))
    if feasible:
        feasible.sort(key=lambda item: (item[0], item[1], item[2]))
        return feasible[0][3]
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1], item[2]))
        return fallback[0][3]
    return None


def _metadata_from_npz(meta: Mapping[str, object]) -> TrialMetadata:
    alpha = _to_float(meta.get("alpha"))
    train_frac = _to_float(meta.get("train_frac"))
    cal_frac = _to_float(meta.get("cal_frac"))
    test_frac = _to_float(meta.get("test_frac"))
    gamma_values_raw = meta.get("gamma_grid")
    gamma_values = tuple(_to_float(g) for g in (gamma_values_raw or DEFAULT_GAMMA_GRID))
    y_grid_size = int(meta.get("y_grid_size", 512))
    y_margin = _to_float(meta.get("y_margin", 0.05))
    if math.isnan(alpha) or math.isnan(train_frac) or math.isnan(cal_frac) or math.isnan(test_frac):
        raise ValueError("Missing required metadata fields in artifact")
    return TrialMetadata(
        alpha=float(alpha),
        train_frac=float(train_frac),
        cal_frac=float(cal_frac),
        test_frac=float(test_frac),
        gamma_values=tuple(gamma_values),
        y_grid_size=int(y_grid_size),
        y_margin=float(y_margin),
    )


def _source_name_from_raw_id(raw_id: int) -> str:
    if raw_id == 0:
        return "rural"
    if raw_id == 1:
        return "urban"
    return f"source_{raw_id}"


def _baseline_entry_candidates(baseline_results: Mapping[str, object]) -> List[str]:
    priority = [
        "Max_Aggregated",
    ]
    ordered: List[str] = []
    for name in priority:
        if name in baseline_results:
            ordered.append(name)
    for name in baseline_results.keys():
        if name not in ordered:
            ordered.append(str(name))
    return ordered


def _extract_overall_metrics(payload: object) -> Optional[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        if "Overall" in payload and isinstance(payload.get("Overall"), Mapping):
            return payload["Overall"]  # type: ignore[return-value]
        return payload
    return None


def _append_subset_rows_from_entry(
    rows: List[Dict[str, object]],
    run_id: str,
    trial_index: int,
    method: str,
    entry: Mapping[str, object],
    split: str,
    source_id_to_label: Mapping[int, str],
    gamma: float | None,
) -> None:
    overall = entry.get("Overall")
    if isinstance(overall, Mapping) and ("individual_coverage" in overall or "unique_sources" in overall):
        _append_subset_rows(rows, run_id, trial_index, method, overall, split, source_id_to_label, gamma)
        return

    # Fallback: use per-source subgroup metrics if available.
    for key, value in entry.items():
        if not isinstance(key, str) or not key.startswith("Source_"):
            continue
        try:
            src_idx = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if not isinstance(value, Mapping):
            continue
        label = source_id_to_label.get(src_idx, f"Source {src_idx}")
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "method": method,
                "subset": label,
                "metric": "coverage",
                "value": _to_float(value.get("coverage")),
                "split": split,
                "gamma": gamma if gamma is not None else math.nan,
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "trial_index": trial_index,
                "method": method,
                "subset": label,
                "metric": "avg_width",
                "value": _to_float(value.get("avg_width")),
                "split": split,
                "gamma": gamma if gamma is not None else math.nan,
            }
        )


def collect_results(
    input_dir: Path,
    source_id_to_label: Mapping[int, str],
    single_method_labels: Mapping[str, str],
    target_cov: float,
    gamma_highlight: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    artifact_paths = sorted(input_dir.glob("*.npz"))
    if not artifact_paths:
        raise FileNotFoundError(f"No .npz artifacts found under {input_dir}")

    overall_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []
    gamma_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    selection_notes: List[str] = []
    gamma_targets = {float(g) for g in gamma_highlight}

    for artifact_path in artifact_paths:
        with np.load(artifact_path, allow_pickle=True) as bundle:
            meta = dict(bundle["metadata"][0])
            trials_payload = bundle["trials"][0]

        trials: List[Mapping[str, object]]
        if isinstance(trials_payload, list):
            trials = [dict(item) for item in trials_payload]
        elif isinstance(trials_payload, Mapping):
            trials = [dict(trials_payload)]
        else:
            selection_notes.append(f"Unexpected trials payload in {artifact_path.name}")
            continue

        for trial in trials:
            trial_index = int(trial.get("trial_index", 0))
            random_seed = int(trial.get("random_seed", 0))
            run_id = f"trial{trial_index:03d}_seed{random_seed}"

            baseline_results = trial.get("baseline_results")
            if isinstance(baseline_results, Mapping):
                candidates = _baseline_entry_candidates(baseline_results)
                if candidates:
                    baseline_entry = baseline_results.get(candidates[0])
                    baseline_entry_map = baseline_entry if isinstance(baseline_entry, Mapping) else {}
                    baseline_overall = _extract_overall_metrics(baseline_entry_map)
                    if baseline_overall is not None:
                        _append_overall_rows(
                            overall_rows,
                            run_id,
                            trial_index,
                            BASELINE_LABEL,
                            baseline_overall,
                            TRUE_SPLIT,
                            gamma=None,
                        )
                    if isinstance(baseline_entry_map, Mapping):
                        _append_subset_rows_from_entry(
                            subset_rows,
                            run_id,
                            trial_index,
                            BASELINE_LABEL,
                            baseline_entry_map,
                            TRUE_SPLIT,
                            source_id_to_label,
                            gamma=None,
                        )
                else:
                    selection_notes.append(f"Missing baseline results for {run_id}")

                # Optional: include any explicit single-source baselines that appear (Source_0, Source_1, ...)
                for method_key, entry in baseline_results.items():
                    if not isinstance(method_key, str) or not method_key.startswith("Source_"):
                        continue
                    if not isinstance(entry, Mapping):
                        continue
                    # Derive the method label from the source index using provided labels.
                    try:
                        src_idx = int(method_key.split("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    src_label = source_id_to_label.get(src_idx, f"Source {src_idx}")
                    method_label = f"Single ({src_label})"
                    overall_metrics = _extract_overall_metrics(entry)
                    if overall_metrics is not None:
                        _append_overall_rows(overall_rows, run_id, trial_index, method_label, overall_metrics, TRUE_SPLIT, gamma=None)
                    # Append per-subset rows (if present)
                    _append_subset_rows_from_entry(
                        subset_rows,
                        run_id,
                        trial_index,
                        method_label,
                        entry,
                        TRUE_SPLIT,
                        source_id_to_label,
                        gamma=None,
                    )

            mdcp_results_obj = trial.get("mdcp_results")
            mdcp_results: List[Mapping[str, object]] = []
            if isinstance(mdcp_results_obj, list):
                mdcp_results = [dict(item) for item in mdcp_results_obj if isinstance(item, Mapping)]

            nonpen_metrics: Optional[Mapping[str, object]] = None
            for entry in mdcp_results:
                gamma_value = _to_float(entry.get("gamma"))
                metrics = entry.get("metrics")
                if not isinstance(metrics, Mapping) or math.isnan(gamma_value):
                    continue
                if abs(gamma_value - MDCP_NONPENALIZED_GAMMA) <= 1e-12:
                    nonpen_metrics = metrics
                if gamma_value in gamma_targets:
                    _append_gamma_rows(gamma_rows, run_id, trial_index, float(gamma_value), metrics, TRUE_SPLIT)

            if nonpen_metrics is not None:
                _append_overall_rows(
                    overall_rows,
                    run_id,
                    trial_index,
                    MDCP_NONPENALIZED_LABEL,
                    nonpen_metrics,
                    TRUE_SPLIT,
                    gamma=MDCP_NONPENALIZED_GAMMA,
                )
                _append_subset_rows(
                    subset_rows,
                    run_id,
                    trial_index,
                    MDCP_NONPENALIZED_LABEL,
                    nonpen_metrics,
                    TRUE_SPLIT,
                    source_id_to_label,
                    gamma=MDCP_NONPENALIZED_GAMMA,
                )

            selected_entry = _select_entry_by_mimic(mdcp_results, target_cov)
            if selected_entry is not None:
                selected_gamma = _to_float(selected_entry.get("gamma"))
                true_metrics = selected_entry.get("metrics")
                if isinstance(true_metrics, Mapping) and not math.isnan(selected_gamma):
                    _append_overall_rows(
                        overall_rows,
                        run_id,
                        trial_index,
                        MDCP_SELECTED_LABEL,
                        true_metrics,
                        TRUE_SPLIT,
                        gamma=float(selected_gamma),
                    )
                    _append_subset_rows(
                        subset_rows,
                        run_id,
                        trial_index,
                        MDCP_SELECTED_LABEL,
                        true_metrics,
                        TRUE_SPLIT,
                        source_id_to_label,
                        gamma=float(selected_gamma),
                    )
                    selection_rows.append(
                        {
                            "run_id": run_id,
                            "trial_index": trial_index,
                            "gamma": float(selected_gamma),
                            "true_coverage": _to_float(true_metrics.get("coverage")),
                            "true_avg_width": _to_float(true_metrics.get("avg_width")),
                        }
                    )
                else:
                    selection_notes.append(f"Selected entry missing true metrics for {run_id}")
            else:
                selection_notes.append(f"Selection unavailable for {run_id}")

    overall_df = pd.DataFrame.from_records(overall_rows)
    subset_df = pd.DataFrame.from_records(subset_rows)
    gamma_df = pd.DataFrame.from_records(gamma_rows)
    selection_df = pd.DataFrame.from_records(selection_rows)
    return overall_df, subset_df, gamma_df, selection_df, selection_notes


def summarize_results(
    overall_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    gamma_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    summaries: Dict[str, pd.DataFrame] = {}
    if not overall_df.empty:
        summaries["overall_summary"] = (
            overall_df.groupby(["split", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
        )
    if not subset_df.empty:
        summaries["subset_summary"] = (
            subset_df.groupby(["split", "subset", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
        )
    if not gamma_df.empty:
        summaries["gamma_summary"] = (
            gamma_df.groupby(["split", "gamma", "metric"])["value"].agg(["mean", "std", "count"]).reset_index().sort_values(["split", "gamma", "metric"])
        )
    if not selection_df.empty:
        summaries["selection_counts"] = selection_df.groupby("gamma").size().reset_index(name="count").sort_values("gamma")
        summaries["selection_metrics"] = selection_df.describe()
    return summaries


def _method_color(method: str) -> str:
    return PALETTE.get(method, "#4c72b0")


def _short_method_label(method: str) -> str:
    if method == MDCP_NONPENALIZED_LABEL:
        return "MDCP gamma=0"
    if method == MDCP_SELECTED_LABEL:
        return "MDCP selected"
    if method == BASELINE_LABEL:
        return "Max-p"
    return method


def _draw_points_with_mean(
    ax: plt.Axes,
    data: pd.DataFrame,
    method_order: Sequence[str],
    coverage_target: float | None,
    metric_name: str,
) -> None:
    filtered = data[data["method"].isin(method_order)]
    for idx, method in enumerate(method_order):
        method_slice = filtered[filtered["method"] == method]
        if method_slice.empty:
            continue
        values = method_slice["value"].astype(float).to_numpy()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        color = _method_color(method)
        offsets = np.linspace(-0.15, 0.15, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(
            np.full(values.shape, idx, dtype=float) + offsets,
            values,
            color=color,
            s=10,
            alpha=0.75,
            linewidth=0,
            zorder=3,
        )
        mean_val = float(np.mean(values))
        ax.plot([idx - 0.2, idx + 0.2], [mean_val, mean_val], color=color, linewidth=2.0, zorder=4)
        ax.scatter(
            idx,
            mean_val,
            color=color,
            marker="D",
            s=50,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels([_short_method_label(m) for m in method_order], rotation=25, ha="right")
    if "coverage" in metric_name and coverage_target is not None and not math.isnan(coverage_target):
        ax.axhline(coverage_target, color="#808080", linestyle="--", linewidth=1.2, label="Target")
        ax.set_ylim(0.6, 1.02)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_xlim(-0.5, len(method_order) - 0.5)


def _annotate_penalty(fig: plt.Figure, annotation: str) -> None:
    fig.text(0.5, 0.01, annotation, ha="center", va="bottom", fontsize=10)


def plot_overall_panel(
    overall_df: pd.DataFrame,
    method_order: Sequence[str],
    split: str,
    coverage_target: float,
    output_path: Path,
    title: str,
    subtitle: str,
    penalty_note: str,
    dpi: int,
) -> Optional[Path]:
    subset = overall_df[(overall_df["split"] == split) & (overall_df["metric"].isin(OVERALL_METRICS))]
    if subset.empty:
        return None
    available_methods = subset["method"].unique().tolist()
    ordered_methods = [method for method in method_order if method in available_methods]
    if not ordered_methods:
        return None
    fig, axes = plt.subplots(1, len(OVERALL_METRICS), figsize=(4.2 * len(OVERALL_METRICS), 3.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for axis, metric_name in zip(axes_arr, OVERALL_METRICS):
        metric_slice = subset[subset["metric"] == metric_name]
        _draw_points_with_mean(axis, metric_slice, ordered_methods, coverage_target, metric_name)
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name))
    fig.suptitle(f"{title}\n{subtitle}")
    _annotate_penalty(fig, penalty_note)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_subset_grid(
    subset_df: pd.DataFrame,
    method_order: Sequence[str],
    split: str,
    metric_name: str,
    coverage_target: float,
    output_path: Path,
    title: str,
    subtitle: str,
    penalty_note: str,
    dpi: int,
) -> Optional[Path]:
    slice_df = subset_df[(subset_df["split"] == split) & (subset_df["metric"] == metric_name)]
    if slice_df.empty:
        return None
    subsets = sorted(slice_df["subset"].unique())
    if not subsets:
        return None
    n_cols = min(2, len(subsets))
    n_rows = int(math.ceil(len(subsets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, subset in enumerate(subsets):
        ax = axes_flat[idx]
        data_slice = slice_df[slice_df["subset"] == subset]
        available_methods = data_slice["method"].unique().tolist()
        ordered_methods = [method for method in method_order if method in available_methods]
        _draw_points_with_mean(
            ax,
            data_slice,
            ordered_methods,
            coverage_target if "coverage" in metric_name else None,
            metric_name,
        )
        ax.set_title(subset)
    for idx in range(len(subsets), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.suptitle(f"{title}\n{subtitle}")
    _annotate_penalty(fig, penalty_note)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_gamma_comparison(
    gamma_df: pd.DataFrame,
    split: str,
    coverage_target: float,
    output_path: Path,
    title: str,
    subtitle: str,
    penalty_note: str,
    dpi: int,
) -> Optional[Path]:
    slice_df = gamma_df[gamma_df["split"] == split]
    if slice_df.empty:
        return None
    gamma_values = sorted(float(g) for g in slice_df["gamma"].dropna().unique())
    if not gamma_values:
        return None
    palette = {
        gamma: mcolors.to_hex(GAMMA_COLOR_CYCLE[idx % len(GAMMA_COLOR_CYCLE)])
        for idx, gamma in enumerate(gamma_values)
    }
    positions = {gamma: idx for idx, gamma in enumerate(gamma_values)}
    fig, axes = plt.subplots(1, len(OVERALL_METRICS), figsize=(4.4 * len(OVERALL_METRICS), 3.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for axis, metric_name in zip(axes_arr, OVERALL_METRICS):
        metric_slice = slice_df[slice_df["metric"] == metric_name]
        if metric_slice.empty:
            axis.set_visible(False)
            continue
        for gamma in gamma_values:
            gamma_slice = metric_slice[metric_slice["gamma"] == gamma]
            values = gamma_slice["value"].astype(float).to_numpy()
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            offsets = np.linspace(-0.15, 0.15, values.size) if values.size > 1 else np.array([0.0])
            axis.scatter(
                np.full(values.shape, positions[gamma], dtype=float) + offsets,
                values,
                color=palette[gamma],
                s=10,
                alpha=0.75,
                linewidth=0,
            )
            mean_val = float(np.mean(values))
            axis.plot(
                [positions[gamma] - 0.22, positions[gamma] + 0.22],
                [mean_val, mean_val],
                color=palette[gamma],
                linewidth=2.0,
            )
            axis.scatter(
                positions[gamma],
                mean_val,
                color=palette[gamma],
                marker="D",
                s=50,
                edgecolor="white",
                linewidth=0.8,
            )
        if "coverage" in metric_name and not math.isnan(coverage_target):
            axis.axhline(coverage_target, color="#808080", linestyle="--", linewidth=1.2)
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name))
        axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        axis.set_xticks(range(len(gamma_values)))
        axis.set_xticklabels([f"{gamma:g}" for gamma in gamma_values], rotation=25, ha="right")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[gamma],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            label=f"gamma={gamma:g}",
        )
        for gamma in gamma_values
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(gamma_values), 6),
    )
    fig.suptitle(f"{title}\n{subtitle}")
    _annotate_penalty(fig, penalty_note)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_selection_hist(selection_df: pd.DataFrame, output_path: Path, dpi: int) -> Optional[Path]:
    if selection_df.empty:
        return None
    counts = selection_df.groupby("gamma").size().reset_index(name="count").sort_values("gamma")
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(counts["gamma"].astype(str), counts["count"], color="#2ca02c")
    ax.set_xlabel("Selected gamma (mimic)")
    ax.set_ylabel("Trial count")
    ax.set_title("Mimic-selected MDCP penalties")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def dataframe_to_json_ready(df: pd.DataFrame) -> List[Dict[str, object]]:
    return df.to_dict(orient="records") if not df.empty else []


def main() -> None:
    args = parse_args()

    args.input_dir = _existing_path_with_fallback(args.input_dir)

    artifact_paths = sorted(args.input_dir.glob("*.npz")) if args.input_dir.exists() else []
    if not artifact_paths:
        raise FileNotFoundError(f"No .npz artifacts found under {args.input_dir}")

    with np.load(artifact_paths[0], allow_pickle=True) as bundle:
        meta = dict(bundle["metadata"][0])

    metadata = _metadata_from_npz(meta)
    contig_to_raw = meta.get("contig_to_raw_source_id")
    if not isinstance(contig_to_raw, list):
        raise ValueError("Artifact metadata missing contig_to_raw_source_id")

    source_order = tuple(_source_name_from_raw_id(int(raw)) for raw in contig_to_raw)
    source_id_to_label = {
        idx: SOURCE_LABEL_OVERRIDES.get(name, name.title()) for idx, name in enumerate(source_order)
    }
    single_method_labels = {
        name: f"Single ({SOURCE_LABEL_OVERRIDES.get(name, name.title())})" for name in source_order
    }
    for idx, name in enumerate(source_order):
        label = single_method_labels[name]
        if label not in PALETTE:
            PALETTE[label] = BASE_SINGLE_COLORS[idx % len(BASE_SINGLE_COLORS)]

    target_cov = 1.0 - metadata.alpha
    overall_df, subset_df, gamma_df, selection_df, selection_notes = collect_results(
        args.input_dir,
        source_id_to_label,
        single_method_labels,
        target_cov,
        args.gamma_grid,
    )

    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(output_dir / "figures")
    table_dir = ensure_dir(output_dir / "tables")

    overall_path = table_dir / "per_trial_overall_metrics.csv"
    subset_path = table_dir / "per_trial_subset_metrics.csv"
    gamma_path = table_dir / "per_trial_gamma_metrics.csv"
    selection_path = table_dir / "selection_metrics.csv"

    overall_df.to_csv(overall_path, index=False)
    subset_df.to_csv(subset_path, index=False)
    gamma_df.to_csv(gamma_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    summaries = summarize_results(overall_df, subset_df, gamma_df, selection_df)
    summary_json_path = output_dir / "summary_statistics.json"
    summary_json_path.write_text(
        json.dumps({name: dataframe_to_json_ready(df) for name, df in summaries.items()}, indent=2),
        encoding="utf-8",
    )

    method_order_original = [BASELINE_LABEL, MDCP_NONPENALIZED_LABEL] + [single_method_labels[name] for name in source_order]
    method_order_selected = [BASELINE_LABEL, MDCP_NONPENALIZED_LABEL, MDCP_SELECTED_LABEL] + [single_method_labels[name] for name in source_order]

    figures: List[Path] = []

    fig = plot_overall_panel(
        overall_df,
        method_order_original,
        TRUE_SPLIT,
        target_cov,
        figure_dir / "overall_true_original.png",
        "True test metrics",
        "Baseline vs nonpenalized MDCP",
        "MDCP penalties: gamma=0 nonpenalized",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_overall_panel(
        overall_df,
        method_order_selected,
        TRUE_SPLIT,
        target_cov,
        figure_dir / "overall_true_selected.png",
        "True test metrics",
        "Baselines vs nonpenalized and selected MDCP",
        "MDCP penalties: gamma=0 nonpenalized and mimic-selected gamma",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_subset_grid(
        subset_df,
        method_order_original,
        TRUE_SPLIT,
        "coverage",
        target_cov,
        figure_dir / "subset_true_original_coverage.png",
        "Per-source coverage",
        "True split",
        "MDCP penalties: gamma=0 nonpenalized",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_subset_grid(
        subset_df,
        method_order_original,
        TRUE_SPLIT,
        "avg_width",
        target_cov,
        figure_dir / "subset_true_original_width.png",
        "Per-source interval width",
        "True split",
        "MDCP penalties: gamma=0 nonpenalized",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_subset_grid(
        subset_df,
        method_order_selected,
        TRUE_SPLIT,
        "coverage",
        target_cov,
        figure_dir / "subset_true_selected_coverage.png",
        "Per-source coverage",
        "True split",
        "MDCP penalties: gamma=0 nonpenalized and mimic-selected gamma",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_subset_grid(
        subset_df,
        method_order_selected,
        TRUE_SPLIT,
        "avg_width",
        target_cov,
        figure_dir / "subset_true_selected_width.png",
        "Per-source interval width",
        "True split",
        "MDCP penalties: gamma=0 nonpenalized and mimic-selected gamma",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_gamma_comparison(
        gamma_df,
        TRUE_SPLIT,
        target_cov,
        figure_dir / "gamma_trends_true.png",
        "MDCP gamma sweep",
        "True test split",
        "MDCP penalties: varying gamma grid",
        args.fig_dpi,
    )
    if fig is not None:
        figures.append(fig)

    fig = plot_selection_hist(selection_df, figure_dir / "gamma_selection_hist.png", args.fig_dpi)
    if fig is not None:
        figures.append(fig)

    manifest = {
        "metadata": {
            "alpha": metadata.alpha,
            "target_coverage": target_cov,
            "train_fraction": metadata.train_frac,
            "calibration_fraction": metadata.cal_frac,
            "test_fraction": metadata.test_frac,
            "gamma_grid": list(metadata.gamma_values),
            "notes": selection_notes,
        },
        "tables": {
            "overall": str(overall_path),
            "subset": str(subset_path),
            "gamma": str(gamma_path),
            "selection": str(selection_path),
        },
        "summary_json": str(summary_json_path),
        "figures": [str(path) for path in figures],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved tables to {table_dir}")
    print(f"Saved figures to {figure_dir}")
    print(f"Summary JSON: {summary_json_path}")
    if selection_notes:
        print("Selection notes:")
        for note in selection_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
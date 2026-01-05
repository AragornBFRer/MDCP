#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]

sns.set_theme(style="whitegrid")
GAMMA_COLOR_CYCLE = sns.color_palette("colorblind", 10)

MDCP_NONPEN_METHOD = "MDCP (gamma=0 nonpenalized)"
MDCP_MIMIC_METHOD = "MDCP (mimic-selected)"
BASELINE_METHOD = "Max-p aggregate"

MDCP_NONPEN_NOTE = "MDCP results use gamma=0 (non-penalized)."
MDCP_MIMIC_NOTE = (
    "MDCP results select gamma via mimic split, applied to true test metrics. "
    "Trials without mimic metrics are reported as missing and omitted from the mimic-selected method."
)

REGION_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Other"]
SINGLE_SOURCE_METHODS = [f"Single ({name})" for name in REGION_ORDER]

PALETTE = {
    MDCP_NONPEN_METHOD: "#377eb8",  # blue
    MDCP_MIMIC_METHOD: "#ff8c00",  # orange
    BASELINE_METHOD: "#e41a1c",  # red
    "Single (Africa)": "#4daf4a",
    "Single (Americas)": "#984ea3",
    "Single (Asia)": "#ff7f00",
    "Single (Europe)": "#a65628",
    "Single (Oceania)": "#f781bf",
    "Single (Other)": "#999999",
}

OVERALL_METRICS = [
    "overall_coverage",
    "worst_case_coverage",
    "avg_set_size",
]

METRIC_LABELS = {
    "overall_coverage": "Coverage",
    "worst_case_coverage": "Worst-case Coverage",
    "avg_set_size": "Avg Set Size",
    "coverage": "Coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MDCP FMoW evaluation aggregates with baseline comparison.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval_out/fmow/mdcp"),
        help="Directory containing MDCP evaluation .npz artifacts (default: eval_out/fmow/mdcp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_out/fmow/mdcp_analysis"),
        help="Directory to store aggregated summaries and figures (default: eval_out/fmow/mdcp_analysis).",
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=300,
        help="Resolution for saved figures (default: 300).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
            candidates.append((REPO_ROOT / "eval_out" / "fmow" / "mdcp").resolve())
    else:
        candidates.append((REPO_ROOT / "eval_out" / "fmow" / "mdcp").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    priority = [MDCP_NONPEN_METHOD, MDCP_MIMIC_METHOD, BASELINE_METHOD] + SINGLE_SOURCE_METHODS
    method_list = list(dict.fromkeys(methods))
    method_list.sort(key=lambda name: priority.index(name) if name in priority else len(priority) + method_list.index(name))
    return method_list


def _method_color(method: str) -> str:
    return PALETTE.get(method, "#4c72b0")


def _compute_overall_metrics(payload: Dict[str, object]) -> Dict[str, float]:
    coverage = float(payload.get("coverage", np.nan))
    avg_set_size = float(payload.get("avg_set_size", np.nan))
    individual_cov = np.asarray(payload.get("individual_coverage", []), dtype=float)
    individual_widths = np.asarray(payload.get("individual_widths", []), dtype=float)
    worst_cov = float(np.nanmin(individual_cov)) if individual_cov.size else np.nan
    worst_size = float(np.nanmax(individual_widths)) if individual_widths.size else np.nan
    return {
        "overall_coverage": coverage,
        "worst_case_coverage": worst_cov,
        "avg_set_size": avg_set_size,
        "worst_case_set_size": worst_size,
    }


def _infer_region_lookup(meta: Dict[str, object]) -> Dict[int, str]:
    """Infer a mapping from contiguous region indices -> human-readable names.

    Newer FMoW evaluation artifacts store region ids as:
      - contig_to_raw_region_id: list[int]
      - raw_region_id_to_contig: dict[int, int]

    Region *names* must be inferred using a WILDS-provided id->name mapping when
    available. Recent prediction artifacts store this mapping under:
      meta['embeddings_meta']['region_id_to_name']  (list[str], indexed by raw region id)

    When the mapping is missing (legacy artifacts), we fall back to a best-effort
    guess that assumes raw ids 0..5 match REGION_ORDER.
    """
    # Preferred path: use exported WILDS mapping from prediction artifacts.
    region_id_to_name = None
    embeddings_meta = meta.get("embeddings_meta")
    if isinstance(embeddings_meta, dict):
        candidate = embeddings_meta.get("region_id_to_name")
        if isinstance(candidate, list) and candidate:
            region_id_to_name = [str(name) for name in candidate]

    contig_to_raw = meta.get("contig_to_raw_region_id")
    if not isinstance(contig_to_raw, list) or not contig_to_raw:
        # Fall back to assuming standard ordering.
        n_sources = int(meta.get("n_sources", len(REGION_ORDER)))
        if n_sources == len(REGION_ORDER):
            return {idx: REGION_ORDER[idx] for idx in range(n_sources)}
        return {idx: f"Region_{idx}" for idx in range(n_sources)}

    lookup: Dict[int, str] = {}
    for contig_idx, raw_id in enumerate(contig_to_raw):
        try:
            raw_int = int(raw_id)
        except (TypeError, ValueError):
            raw_int = -1
        if region_id_to_name is not None and 0 <= raw_int < len(region_id_to_name):
            lookup[int(contig_idx)] = region_id_to_name[raw_int]
        elif 0 <= raw_int < len(REGION_ORDER):
            # Legacy fallback: only correct if raw ids were 0..5 in REGION_ORDER.
            lookup[int(contig_idx)] = REGION_ORDER[raw_int]
        elif raw_int >= 0:
            lookup[int(contig_idx)] = f"Region_{raw_int}"
        else:
            lookup[int(contig_idx)] = f"Region_{contig_idx}"
    return lookup


def _collect_subset_metrics(
    metrics: Dict[str, object],
    region_lookup: Dict[int, str],
) -> List[Tuple[str, float, float]]:
    unique_sources = metrics.get("unique_sources")
    individual_cov = metrics.get("individual_coverage")
    individual_widths = metrics.get("individual_widths")
    if unique_sources is None or individual_cov is None or individual_widths is None:
        return []
    records: List[Tuple[str, float, float]] = []
    for src, cov_val, width_val in zip(unique_sources, individual_cov, individual_widths):
        src_idx = int(src)
        region_name = region_lookup.get(src_idx, f"Source_{src_idx}")
        records.append((region_name, float(cov_val), float(width_val)))
    return records


def _select_best_entry(entries: List[Dict[str, object]], target_cov: float) -> Dict[str, object] | None:
    feasible: List[Tuple[float, float, Dict[str, object]]] = []
    fallback: List[Tuple[float, float, Dict[str, object]]] = []
    for entry in entries:
        summary = entry.get("summary", {})
        coverage = float(summary.get("coverage", np.nan))
        avg_set_size = float(summary.get("avg_set_size", np.nan))
        if np.isnan(coverage) or np.isnan(avg_set_size):
            continue
        if coverage >= target_cov:
            feasible.append((avg_set_size, -coverage, entry))
        else:
            fallback.append((-coverage, avg_set_size, entry))
    if feasible:
        feasible.sort(key=lambda item: (item[0], item[1]))
        return feasible[0][2]
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1]))
        return fallback[0][2]
    return None


def _select_entry_by_mimic(entries: List[Dict[str, object]], target_cov: float) -> Dict[str, object] | None:
    feasible: List[Tuple[float, float, float, Dict[str, object]]] = []
    fallback: List[Tuple[float, float, float, Dict[str, object]]] = []
    for entry in entries:
        gamma_value = float(entry.get("gamma", np.nan))
        mimic_metrics = entry.get("mimic_metrics")
        if mimic_metrics is None:
            continue
        mimic_overall = _compute_overall_metrics(mimic_metrics)
        coverage = float(mimic_overall.get("overall_coverage", np.nan))
        avg_size = float(mimic_overall.get("avg_set_size", np.nan))
        if np.isnan(coverage) or np.isnan(avg_size) or np.isnan(gamma_value):
            continue
        if coverage >= target_cov:
            feasible.append((avg_size, -coverage, abs(gamma_value), entry))
        else:
            fallback.append((-coverage, avg_size, abs(gamma_value), entry))
    if feasible:
        feasible.sort(key=lambda item: (item[0], item[1], item[2]))
        return feasible[0][3]
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1], item[2]))
        return fallback[0][3]
    return None


def collect_results(
    input_dir: Path,
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    artifact_paths = sorted(input_dir.glob("*.npz"))
    if not artifact_paths:
        raise ValueError(f"No .npz artifacts found under {input_dir}")

    metadata: Dict[str, object] | None = None
    region_lookup: Dict[int, str] = {}
    contig_to_raw_reference: List[int] | None = None

    overall_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []
    gamma_rows: List[Dict[str, object]] = []
    gamma_subset_rows: List[Dict[str, object]] = []
    selection_mimic_rows: List[Dict[str, object]] = []

    for artifact_path in artifact_paths:
        with np.load(artifact_path, allow_pickle=True) as bundle:
            meta = dict(bundle["metadata"][0])
            trial = dict(bundle["trials"][0])

        if metadata is None:
            metadata = meta
            region_lookup = _infer_region_lookup(meta)
            if isinstance(meta.get("contig_to_raw_region_id"), list):
                contig_to_raw_reference = [int(v) for v in meta["contig_to_raw_region_id"]]
        else:
            if isinstance(meta.get("contig_to_raw_region_id"), list) and contig_to_raw_reference is not None:
                current = [int(v) for v in meta["contig_to_raw_region_id"]]
                if current != contig_to_raw_reference:
                    raise ValueError("Region mappings are inconsistent across artifacts (contig_to_raw_region_id mismatch).")

        trial_index = int(trial["trial_index"])
        random_seed = int(trial["random_seed"])
        run_id = f"trial{trial_index:03d}_seed{random_seed}"
        target_cov = 1.0 - float(meta["alpha"])

        mdcp_results: List[Dict[str, object]] = list(trial["mdcp_results"])
        baseline_results: Dict[str, Dict[str, object]] = trial["baseline_results"]

        nonpen_entry: Dict[str, object] | None = None

        for entry in mdcp_results:
            gamma_value = float(entry["gamma"])
            metrics = entry["metrics"]
            overall_metrics = _compute_overall_metrics(metrics)
            for metric_name, value in overall_metrics.items():
                gamma_rows.append(
                    {
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": gamma_value,
                        "metric": metric_name,
                        "value": value,
                    }
                )
            for region_name, cov_val, width_val in _collect_subset_metrics(metrics, region_lookup):
                gamma_subset_rows.append(
                    {
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": gamma_value,
                        "subset": region_name,
                        "metric": "coverage",
                        "value": cov_val,
                    }
                )
                gamma_subset_rows.append(
                    {
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": gamma_value,
                        "subset": region_name,
                        "metric": "avg_set_size",
                        "value": width_val,
                    }
                )
            if nonpen_entry is None and abs(gamma_value) <= 1e-12:
                nonpen_entry = entry

        if nonpen_entry is None:
            raise RuntimeError("Missing gamma=0 entry in MDCP results")

        nonpen_gamma = float(nonpen_entry.get("gamma", 0.0))
        nonpen_metrics = _compute_overall_metrics(nonpen_entry["metrics"])
        for metric_name, value in nonpen_metrics.items():
            overall_rows.append(
                {
                    "method": MDCP_NONPEN_METHOD,
                    "metric": metric_name,
                    "value": value,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": nonpen_gamma,
                }
            )
        for region_name, cov_val, width_val in _collect_subset_metrics(nonpen_entry["metrics"], region_lookup):
            subset_rows.append(
                {
                    "method": MDCP_NONPEN_METHOD,
                    "subset": region_name,
                    "metric": "coverage",
                    "value": cov_val,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": nonpen_gamma,
                }
            )
            subset_rows.append(
                {
                    "method": MDCP_NONPEN_METHOD,
                    "subset": region_name,
                    "metric": "avg_set_size",
                    "value": width_val,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": nonpen_gamma,
                }
            )

        mimic_entry = _select_entry_by_mimic(mdcp_results, target_cov)
        mimic_metrics_payload = mimic_entry.get("mimic_metrics") if mimic_entry is not None else None

        if mimic_entry is None or mimic_metrics_payload is None:
            selection_mimic_rows.append(
                {
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": np.nan,
                    "strategy": "missing",
                    "note": "no_mimic_metrics",
                    "mimic_coverage": np.nan,
                    "mimic_avg_set_size": np.nan,
                    "true_coverage": np.nan,
                    "true_avg_set_size": np.nan,
                }
            )
        else:
            mimic_gamma = float(mimic_entry.get("gamma", np.nan))
            mimic_true_metrics = _compute_overall_metrics(mimic_entry["metrics"])
            mimic_split_metrics = _compute_overall_metrics(mimic_metrics_payload)

            selection_mimic_rows.append(
                {
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": mimic_gamma,
                    "strategy": "mimic",
                    "note": "",
                    "mimic_coverage": float(mimic_split_metrics.get("overall_coverage", np.nan)),
                    "mimic_avg_set_size": float(mimic_split_metrics.get("avg_set_size", np.nan)),
                    "true_coverage": float(mimic_true_metrics.get("overall_coverage", np.nan)),
                    "true_avg_set_size": float(mimic_true_metrics.get("avg_set_size", np.nan)),
                }
            )

            for metric_name, value in mimic_true_metrics.items():
                overall_rows.append(
                    {
                        "method": MDCP_MIMIC_METHOD,
                        "metric": metric_name,
                        "value": value,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": mimic_gamma,
                    }
                )
            for region_name, cov_val, width_val in _collect_subset_metrics(mimic_entry["metrics"], region_lookup):
                subset_rows.append(
                    {
                        "method": MDCP_MIMIC_METHOD,
                        "subset": region_name,
                        "metric": "coverage",
                        "value": cov_val,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": mimic_gamma,
                    }
                )
                subset_rows.append(
                    {
                        "method": MDCP_MIMIC_METHOD,
                        "subset": region_name,
                        "metric": "avg_set_size",
                        "value": width_val,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": mimic_gamma,
                    }
                )

        for key, payload in baseline_results.items():
            if not key.startswith("Source_"):
                continue
            try:
                source_idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                source_idx = -1
            region_name = region_lookup.get(source_idx, key)
            method_label = f"Single ({region_name})"
            overall_payload = payload.get("Overall")
            if overall_payload is None:
                continue
            single_metrics = _compute_overall_metrics(overall_payload)
            for metric_name, value in single_metrics.items():
                overall_rows.append(
                    {
                        "method": method_label,
                        "metric": metric_name,
                        "value": value,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": np.nan,
                    }
                )
            for subset_key, subset_payload in payload.items():
                if subset_key == "Overall":
                    continue
                if not subset_key.startswith("Source_"):
                    continue
                try:
                    target_idx = int(subset_key.split("_")[1])
                except (IndexError, ValueError):
                    target_idx = -1
                target_region = region_lookup.get(target_idx, subset_key)
                coverage = float(subset_payload.get("coverage", np.nan))
                avg_set_size = float(subset_payload.get("avg_set_size", np.nan))
                subset_rows.append(
                    {
                        "method": method_label,
                        "subset": target_region,
                        "metric": "coverage",
                        "value": coverage,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": np.nan,
                    }
                )
                subset_rows.append(
                    {
                        "method": method_label,
                        "subset": target_region,
                        "metric": "avg_set_size",
                        "value": avg_set_size,
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "seed": random_seed,
                        "gamma": np.nan,
                    }
                )

        max_payload = baseline_results.get("Max_Aggregated", {})
        overall_baseline = max_payload.get("Overall")
        if overall_baseline is None:
            raise RuntimeError("Baseline Max_Aggregated missing Overall metrics")
        baseline_metrics = _compute_overall_metrics(overall_baseline)
        for metric_name, value in baseline_metrics.items():
            overall_rows.append(
                {
                    "method": BASELINE_METHOD,
                    "metric": metric_name,
                    "value": value,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": np.nan,
                }
            )

        for key, subset_payload in max_payload.items():
            if key == "Overall":
                continue
            if not key.startswith("Source_"):
                continue
            try:
                source_idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                source_idx = -1
            region_name = region_lookup.get(source_idx, key)
            coverage = float(subset_payload.get("coverage", np.nan))
            avg_set_size = float(subset_payload.get("avg_set_size", np.nan))
            subset_rows.append(
                {
                    "method": BASELINE_METHOD,
                    "subset": region_name,
                    "metric": "coverage",
                    "value": coverage,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": np.nan,
                }
            )
            subset_rows.append(
                {
                    "method": BASELINE_METHOD,
                    "subset": region_name,
                    "metric": "avg_set_size",
                    "value": avg_set_size,
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "seed": random_seed,
                    "gamma": np.nan,
                }
            )

    assert metadata is not None

    overall_df = pd.DataFrame.from_records(overall_rows)
    subset_df = pd.DataFrame.from_records(subset_rows)
    gamma_df = pd.DataFrame.from_records(gamma_rows)
    gamma_subset_df = pd.DataFrame.from_records(gamma_subset_rows)
    selection_mimic_df = pd.DataFrame.from_records(selection_mimic_rows)

    return (
        metadata,
        overall_df,
        subset_df,
        gamma_df,
        gamma_subset_df,
        selection_mimic_df,
    )


def summarize_results(
    overall_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    gamma_df: pd.DataFrame,
    gamma_subset_df: pd.DataFrame,
    selection_mimic_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    summaries: Dict[str, pd.DataFrame] = {}
    if not overall_df.empty:
        summaries["overall_summary"] = (
            overall_df.groupby(["method", "metric"]) ["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    if not subset_df.empty:
        summaries["subset_summary"] = (
            subset_df.groupby(["subset", "method", "metric"]) ["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    if not gamma_df.empty:
        summaries["gamma_summary"] = (
            gamma_df.groupby(["gamma", "metric"]) ["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("gamma")
        )
    if not gamma_subset_df.empty:
        summaries["gamma_subset_summary"] = (
            gamma_subset_df.groupby(["gamma", "subset", "metric"]) ["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["subset", "gamma"])
        )
    if not selection_mimic_df.empty:
        summaries["selection_counts_mimic"] = (
            selection_mimic_df.groupby("gamma")
            .size()
            .reset_index(name="count")
            .sort_values("gamma")
        )
    return summaries


def _short_method_label(method: str) -> str:
    if method == MDCP_NONPEN_METHOD:
        return "MDCP\nnonpen"
    if method == MDCP_MIMIC_METHOD:
        return "MDCP\nmimic"
    if method == BASELINE_METHOD:
        return "Max-p"
    if method.startswith("Single (") and method.endswith(")"):
        region = method[len("Single ("):-1]
        return f"Single\n{region}"
    return method


def _draw_points_with_mean(
    ax: plt.Axes,
    data: pd.DataFrame,
    method_order: List[str],
    coverage_target: float | None,
    metric_name: str,
) -> None:
    filtered = data[data["method"].isin(method_order)]
    all_values = filtered["value"].to_numpy(dtype=float)

    for idx, method in enumerate(method_order):
        method_slice = data[data["method"] == method]
        if method_slice.empty:
            continue
        values = method_slice["value"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        color = _method_color(method)
        if values.size == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.15, 0.15, values.size)
        ax.scatter(
            np.full(values.shape, idx, dtype=float) + offsets,
            values,
            color=color,
            s=9,
            alpha=0.7,
            linewidth=0,
            zorder=3,
        )
        mean_val = float(np.mean(values))
        ax.plot([idx - 0.2, idx + 0.2], [mean_val, mean_val], color=color, linewidth=2.0, zorder=4)
        ax.scatter(
            idx,
            mean_val,
            color=color,
            s=45,
            marker="D",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels([_short_method_label(m) for m in method_order], rotation=25, ha="right")
    if "coverage" in metric_name:
        lower_bound = 0.6
        finite_values = all_values[np.isfinite(all_values)]
        if finite_values.size:
            min_val = float(np.min(finite_values))
            if min_val < lower_bound:
                lower_bound = max(0.0, min_val - 0.02)
        ax.set_ylim(lower_bound, 1.0)
        if coverage_target is not None:
            ax.axhline(coverage_target, linestyle="--", color="#9a9a9a", linewidth=1.0, zorder=1)
        ax.set_ylabel("Coverage")
    elif metric_name == "avg_set_size":
        ax.set_ylabel("Avg set size")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_xlim(-0.5, len(method_order) - 0.5)


def plot_overall_panels(
    overall_df: pd.DataFrame,
    coverage_target: float,
    method_priority: List[str],
    note: str,
    output_path: Path,
    dpi: int,
) -> Path:
    if overall_df.empty:
        raise ValueError("No overall records to plot.")
    available_methods = overall_df["method"].unique()
    method_order = [m for m in method_priority if m in available_methods]
    if not method_order:
        raise ValueError("No overlapping methods for plotting.")
    fig, axes = plt.subplots(1, len(OVERALL_METRICS), figsize=(4.2 * len(OVERALL_METRICS), 3.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for idx, metric_name in enumerate(OVERALL_METRICS):
        ax = axes_arr[idx]
        metric_slice = overall_df[overall_df["metric"] == metric_name]
        if metric_slice.empty:
            ax.axis("off")
            continue
        _draw_points_with_mean(ax, metric_slice, method_order, coverage_target, metric_name)
        ax.set_title(METRIC_LABELS.get(metric_name, metric_name))
        if idx > 0:
            ax.set_ylabel("")
    fig.suptitle("Overall evaluation targets\n" + note)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_subset_metric_grid(
    subset_df: pd.DataFrame,
    metric_name: str,
    coverage_target: float,
    method_priority: List[str],
    note: str,
    output_path: Path,
    dpi: int,
) -> Path | None:
    metric_slice = subset_df[subset_df["metric"] == metric_name]
    if metric_slice.empty:
        return None
    regions = [region for region in REGION_ORDER if region in metric_slice["subset"].unique()]
    if not regions:
        return None
    n_cols = 3
    n_rows = int(np.ceil(len(regions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, region in enumerate(regions):
        ax = axes_flat[idx]
        region_slice = metric_slice[metric_slice["subset"] == region]
        available_methods = region_slice["method"].unique()
        method_order = [m for m in method_priority if m in available_methods]
        if not method_order:
            ax.axis("off")
            continue
        _draw_points_with_mean(ax, region_slice, method_order, coverage_target, metric_name)
        ax.set_title(region)
        ax.set_xlabel("")

    for idx, ax in enumerate(axes_flat):
        if idx >= len(regions):
            ax.axis("off")
        else:
            if idx % n_cols != 0:
                ax.set_ylabel("")

    title = "Per-source Coverage" if "coverage" in metric_name else "Per-source Average Set Size"
    fig.suptitle(f"{title}\n{note}")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_gamma_comparison(gamma_df: pd.DataFrame, coverage_target: float, output_path: Path, dpi: int) -> Path | None:
    if gamma_df.empty:
        return None
    gammas = sorted(gamma_df["gamma"].dropna().unique())
    if not gammas:
        return None
    gamma_labels = [f"{gamma:g}" for gamma in gammas]
    positions = np.arange(len(gammas), dtype=float)
    palette = {
        gamma: mcolors.to_hex(GAMMA_COLOR_CYCLE[idx % len(GAMMA_COLOR_CYCLE)])
        for idx, gamma in enumerate(gammas)
    }

    metrics_to_plot = ["overall_coverage", "worst_case_coverage", "avg_set_size"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.2 * len(metrics_to_plot), 3.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for axis, metric_name in zip(axes_arr, metrics_to_plot):
        metric_slice = gamma_df[gamma_df["metric"] == metric_name]
        if metric_slice.empty:
            axis.axis("off")
            continue
        metric_values = metric_slice["value"].to_numpy(dtype=float)
        for pos, gamma in enumerate(gammas):
            values = metric_slice[metric_slice["gamma"] == gamma]["value"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color = palette[gamma]
            if values.size == 1:
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(-0.12, 0.12, values.size)
            axis.scatter(
                np.full(values.shape, pos, dtype=float) + offsets,
                values,
                color=color,
                s=9,
                alpha=0.75,
                linewidth=0,
                zorder=3,
            )
            mean_val = float(np.mean(values))
            axis.plot([pos - 0.18, pos + 0.18], [mean_val, mean_val], color=color, linewidth=2.0, zorder=4)
            axis.scatter(pos, mean_val, color=color, s=40, marker="D", edgecolor="white", linewidth=0.8, zorder=5)

        axis.set_xticks(positions)
        axis.set_xticklabels(gamma_labels)
        if "coverage" in metric_name:
            lower_bound = 0.6
            finite_vals = metric_values[np.isfinite(metric_values)]
            if finite_vals.size:
                min_val = float(np.min(finite_vals))
                if min_val < lower_bound:
                    lower_bound = max(0.0, min_val - 0.02)
            axis.set_ylim(lower_bound, 1.0)
            axis.axhline(coverage_target, linestyle="--", color="#9a9a9a", linewidth=1.0, zorder=1)
            axis.set_ylabel("Coverage")
        else:
            axis.set_ylabel("Avg set size")
        axis.set_xlabel("gamma")
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name))
        axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

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
        for gamma in gammas
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=min(len(gammas), 6))
    fig.suptitle("MDCP gamma comparison (per-trial)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_gamma_selection(selection_df: pd.DataFrame, title: str, output_path: Path, dpi: int) -> Path | None:
    if selection_df.empty:
        return None
    counts = selection_df.groupby("gamma").size().reset_index(name="count").sort_values("gamma")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(counts["gamma"].astype(str), counts["count"], color="#2ca02c")
    ax.set_xlabel("Selected gamma")
    ax.set_ylabel("Trial count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    input_dir = _existing_path_with_fallback(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    ensure_dir(output_dir)
    figure_dir = ensure_dir(output_dir / "figures")
    table_dir = ensure_dir(output_dir / "tables")

    (
        metadata,
        overall_df,
        subset_df,
        gamma_df,
        gamma_subset_df,
        selection_mimic_df,
    ) = collect_results(input_dir)
    summaries = summarize_results(
        overall_df,
        subset_df,
        gamma_df,
        gamma_subset_df,
        selection_mimic_df,
    )

    overall_path = table_dir / "per_trial_overall_metrics.csv"
    subset_path = table_dir / "per_trial_region_metrics.csv"
    gamma_path = table_dir / "per_trial_gamma_metrics.csv"
    gamma_subset_path = table_dir / "per_trial_gamma_region_metrics.csv"
    selection_mimic_path = table_dir / "selected_gamma_mimic_per_trial.csv"

    overall_df.to_csv(overall_path, index=False)
    subset_df.to_csv(subset_path, index=False)
    gamma_df.to_csv(gamma_path, index=False)
    gamma_subset_df.to_csv(gamma_subset_path, index=False)
    selection_mimic_df.to_csv(selection_mimic_path, index=False)

    for name, df in summaries.items():
        df.to_csv(table_dir / f"{name}.csv", index=False)

    target_cov = 1.0 - float(metadata["alpha"])

    original_priority = [MDCP_NONPEN_METHOD, BASELINE_METHOD] + SINGLE_SOURCE_METHODS
    mimic_priority = [MDCP_NONPEN_METHOD, MDCP_MIMIC_METHOD, BASELINE_METHOD] + SINGLE_SOURCE_METHODS

    original_overall_df = overall_df[overall_df["method"].isin(original_priority)]
    mimic_overall_df = overall_df[overall_df["method"].isin(mimic_priority)]
    original_subset_df = subset_df[subset_df["method"].isin(original_priority)]
    mimic_subset_df = subset_df[subset_df["method"].isin(mimic_priority)]

    figures: List[Path] = []
    figures.append(
        plot_overall_panels(
            original_overall_df,
            target_cov,
            original_priority,
            MDCP_NONPEN_NOTE,
            figure_dir / "overall_panels.png",
            args.fig_dpi,
        )
    )

    figures.append(
        plot_overall_panels(
            mimic_overall_df,
            target_cov,
            mimic_priority,
            MDCP_MIMIC_NOTE,
            figure_dir / "overall_panels_mimic.png",
            args.fig_dpi,
        )
    )

    coverage_subset_fig = plot_subset_metric_grid(
        original_subset_df,
        "coverage",
        target_cov,
        original_priority,
        MDCP_NONPEN_NOTE,
        figure_dir / "per_region_coverage.png",
        args.fig_dpi,
    )
    if coverage_subset_fig is not None:
        figures.append(coverage_subset_fig)

    coverage_subset_fig_mimic = plot_subset_metric_grid(
        mimic_subset_df,
        "coverage",
        target_cov,
        mimic_priority,
        MDCP_MIMIC_NOTE,
        figure_dir / "per_region_coverage_mimic.png",
        args.fig_dpi,
    )
    if coverage_subset_fig_mimic is not None:
        figures.append(coverage_subset_fig_mimic)

    size_subset_fig = plot_subset_metric_grid(
        original_subset_df,
        "avg_set_size",
        target_cov,
        original_priority,
        MDCP_NONPEN_NOTE,
        figure_dir / "per_region_set_size.png",
        args.fig_dpi,
    )
    if size_subset_fig is not None:
        figures.append(size_subset_fig)

    size_subset_fig_mimic = plot_subset_metric_grid(
        mimic_subset_df,
        "avg_set_size",
        target_cov,
        mimic_priority,
        MDCP_MIMIC_NOTE,
        figure_dir / "per_region_set_size_mimic.png",
        args.fig_dpi,
    )
    if size_subset_fig_mimic is not None:
        figures.append(size_subset_fig_mimic)

    gamma_fig = plot_gamma_comparison(gamma_df, target_cov, figure_dir / "gamma_trends.png", args.fig_dpi)
    if gamma_fig is not None:
        figures.append(gamma_fig)

    selection_fig_mimic = plot_gamma_selection(
        selection_mimic_df,
        "Gamma selection frequency (mimic)",
        figure_dir / "gamma_selection.png",
        args.fig_dpi,
    )
    if selection_fig_mimic is not None:
        figures.append(selection_fig_mimic)

    tables_payload: Dict[str, str] = {
        "per_trial_overall": str(overall_path.resolve()),
        "per_trial_region": str(subset_path.resolve()),
        "per_trial_gamma": str(gamma_path.resolve()),
        "per_trial_gamma_region": str(gamma_subset_path.resolve()),
        "selected_gamma_mimic": str(selection_mimic_path.resolve()),
    }
    for name in summaries:
        tables_payload[name] = str((table_dir / f"{name}.csv").resolve())

    summary_payload = {
        "metadata": metadata,
        "target_coverage": target_cov,
        "tables": tables_payload,
        "figures": [str(path.resolve()) for path in figures],
    }

    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print("Saved tables:")
    print(f"  per_trial_overall: {overall_path}")
    print(f"  per_trial_region: {subset_path}")
    print(f"  per_trial_gamma: {gamma_path}")
    print(f"  per_trial_gamma_region: {gamma_subset_path}")
    print(f"  selected_gamma_mimic: {selection_mimic_path}")
    for name in summaries:
        print(f"  {name}: {table_dir / f'{name}.csv'}")
    print("Saved figures:")
    for path in figures:
        print(f"  {path}")
    print(f"Analysis summary: {summary_path}")


if __name__ == "__main__":
    main()

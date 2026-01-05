"""Visualize MDCP MEPS regression evaluation results.

This script aggregates the outputs produced by ``meps/eval.py``
and generates summary figures that compare MDCP (with gamma penalties)
against baseline conformal predictors. The figures highlight overall and
worst-case coverage as well as average interval width, both for per-trial
MDCP selections (mimic- and true-tuned) and baseline aggregations. Gamma
selection behaviour is also summarized to mirror the iterative evaluation
plots used for synthetic experiments.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure non-interactive backend for script execution
plt.switch_backend("Agg")

sns.set_theme(style="whitegrid")

MDCP_PER_TRIAL_MIMIC = "MDCP per-trial sel (mimic)"
MDCP_PER_TRIAL_TRUE = "MDCP per-trial sel (true)"
MDCP_AVG_MIMIC = "MDCP avg sel (mimic)"
MDCP_AVG_TRUE = "MDCP avg sel (true)"
BASELINE_MAX = "Baseline max-agg"
BASELINE_SINGLE = "Baseline single-src"
BASELINE_SOURCE_PREFIX = "Baseline source "
BASELINE_SOURCE_COLORS = ["#c44e52", "#dd8452", "#937860", "#8c564b"]

CQR_MAX = "CQR max-agg"
CQR_SOURCE_PREFIX = "CQR source "
CQR_SOURCE_COLORS = ["#4c72b0", "#6a8caf", "#8aa2c2", "#a9b9d5"]

REGRESSION_METRIC_KEY = "avg_width"
REGRESSION_WORST_METRIC_KEY = "worst_case_interval_width"

METRIC_LABELS = {
    "overall_coverage": "Overall Coverage",
    "worst_case_coverage": "Worst-case Coverage",
    REGRESSION_METRIC_KEY: "Avg Interval Width",
    REGRESSION_WORST_METRIC_KEY: "Worst-case Interval Width",
}

METHOD_LABELS = {
    MDCP_PER_TRIAL_MIMIC: "MDCP per-trial sel (mimic)",
    MDCP_PER_TRIAL_TRUE: "MDCP per-trial sel (true)",
    MDCP_AVG_MIMIC: "MDCP avg sel (mimic)",
    MDCP_AVG_TRUE: "MDCP avg sel (true)",
    BASELINE_MAX: "Baseline max-agg",
    BASELINE_SINGLE: "Baseline single-src",
    CQR_MAX: "CQR max-agg",
}

PALETTE = {
    MDCP_PER_TRIAL_MIMIC: "#1f77b4",
    MDCP_PER_TRIAL_TRUE: "#2ca02c",
    MDCP_AVG_MIMIC: "#4c72b0",
    MDCP_AVG_TRUE: "#3c8d2f",
    BASELINE_MAX: "#55a868",
    BASELINE_SINGLE: "#c44e52",
    CQR_MAX: "#6c8ebf",
}

EVALUATION_STYLES = {
    "mimic": {"label": "mimic calib->test", "color": "#1f77b4"},
    "true": {"label": "true calib->test", "color": "#2ca02c"},
}

_P_LOADED = False


def _format_gamma_label(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "gamma=?"
    return f"gamma={value:g}"


def _register_pickled_classes() -> None:
    """Register dataclasses used when saving MEPS payloads for unpickling."""

    global _P_LOADED
    if _P_LOADED:
        return

    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import meps.evaluate_meps_rand_avg as eval_meps
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Unable to import meps.evaluate_meps_rand_avg; ensure the repo root is on PYTHONPATH."
        ) from exc

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "PanelMetadata", eval_meps.PanelMetadata)
        setattr(main_mod, "SplitPayload", eval_meps.SplitPayload)
    _P_LOADED = True


@dataclass
class RecordBundle:
    records: pd.DataFrame
    comprehensive: pd.DataFrame
    gamma: pd.DataFrame
    gamma_comprehensive: pd.DataFrame
    selection: pd.DataFrame
    panels: List[str]


def _safe_array(values: object) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=float)
    return np.asarray([float(values)], dtype=float)


def _extract_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    coverage = float(metrics.get("coverage", float("nan")))
    width = float(metrics.get(REGRESSION_METRIC_KEY, float("nan")))

    worst_cov = float("nan")
    worst_width = float("nan")

    individual_cov = metrics.get("individual_coverage")
    if individual_cov is not None:
        arr = _safe_array(individual_cov)
        if arr.size:
            worst_cov = float(np.nanmin(arr))

    individual_width = metrics.get("individual_widths")
    if individual_width is not None:
        arr = _safe_array(individual_width)
        if arr.size:
            worst_width = float(np.nanmax(arr))

    return {
        "overall_coverage": coverage,
        "worst_case_coverage": worst_cov,
        REGRESSION_METRIC_KEY: width,
        REGRESSION_WORST_METRIC_KEY: worst_width,
    }


def _add_records(
    container: List[Dict[str, object]],
    base_info: Dict[str, object],
    metric_values: Dict[str, float],
) -> None:
    for metric_name, value in metric_values.items():
        entry = dict(base_info)
        entry["metric"] = metric_name
        entry["metric_label"] = METRIC_LABELS.get(metric_name, metric_name)
        entry["value"] = float(value)
        container.append(entry)


def _subset_metrics_for_mdcp(metrics: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    subset_map: Dict[str, Dict[str, float]] = {}

    overall = {
        "coverage": float(metrics.get("coverage", float("nan"))),
        REGRESSION_METRIC_KEY: float(metrics.get(REGRESSION_METRIC_KEY, float("nan"))),
    }
    subset_map["Overall"] = overall

    unique_sources = metrics.get("unique_sources")
    ind_cov = metrics.get("individual_coverage")
    ind_width = metrics.get("individual_widths")

    sources = _safe_array(unique_sources).astype(int, copy=False)
    cov_arr = _safe_array(ind_cov)
    width_arr = _safe_array(ind_width)
    if sources.size and cov_arr.size and width_arr.size:
        for idx, src in enumerate(sources):
            label = f"Source_{int(src)}"
            subset_map[label] = {
                "coverage": float(cov_arr[idx] if idx < cov_arr.size else float("nan")),
                REGRESSION_METRIC_KEY: float(
                    width_arr[idx] if idx < width_arr.size else float("nan")
                ),
            }

    return subset_map


def _select_entry(
    entries: Sequence[Dict[str, object]],
    metrics_key: str,
    coverage_target: float,
) -> Optional[Dict[str, object]]:
    candidates: List[Tuple[float, float, Dict[str, object]]] = []
    fallback: List[Tuple[float, float, Dict[str, object]]] = []

    for entry in entries:
        metrics = entry.get(metrics_key)
        if not isinstance(metrics, dict):
            continue
        extracted = _extract_metrics(metrics)
        coverage = extracted["overall_coverage"]
        efficiency = extracted[REGRESSION_METRIC_KEY]
        if math.isnan(coverage) or math.isnan(efficiency):
            continue
        if coverage >= coverage_target:
            candidates.append((efficiency, -coverage, entry))
        fallback.append((efficiency, -coverage, entry))

    if candidates:
        best = min(candidates, key=lambda x: (x[0], x[1]))
        return best[2]
    if fallback:
        best = max(fallback, key=lambda x: (-x[1], -x[0]))
        return best[2]
    return None


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    order: Dict[str, Tuple[int, int]] = {}
    for method in methods:
        if method == MDCP_PER_TRIAL_MIMIC:
            order[method] = (0, 0)
        elif method == MDCP_PER_TRIAL_TRUE:
            order[method] = (0, 1)
        elif method == MDCP_AVG_MIMIC:
            order[method] = (1, 0)
        elif method == MDCP_AVG_TRUE:
            order[method] = (1, 1)
        elif method == BASELINE_MAX:
            order[method] = (2, 0)
        elif method.startswith(BASELINE_SOURCE_PREFIX):
            try:
                idx = int(method.split()[-1])
            except ValueError:
                idx = 99
            order[method] = (3, idx)
        elif method == BASELINE_SINGLE:
            order[method] = (4, 0)
        elif method == CQR_MAX:
            order[method] = (2, 1)
        elif method.startswith(CQR_SOURCE_PREFIX):
            try:
                idx = int(method.split()[-1])
            except ValueError:
                idx = 99
            order[method] = (3, idx + 100)
        else:
            order[method] = (5, 0)
    return sorted(set(methods), key=lambda m: order.get(m, (5, 0)))


def _ordered_subsets(subsets: Iterable[str]) -> List[str]:
    def sort_key(name: str) -> Tuple[int, int]:
        if name == "Overall":
            return (0, 0)
        if name.startswith("Source_"):
            try:
                return (1, int(name.split("_")[1]))
            except (IndexError, ValueError):
                return (1, 99)
        return (2, 0)

    return sorted(set(subsets), key=sort_key)


def _ordered_panels(panels: Iterable[str]) -> List[str]:
    try:
        return sorted(set(panels), key=lambda p: int(p))
    except ValueError:
        return sorted(set(panels))


def _load_payload(npz_path: Path) -> Dict[str, object]:
    _register_pickled_classes()
    with np.load(npz_path, allow_pickle=True) as data:
        payload = data["payload"].item()
    return payload


def _collect_records(
    eval_root: Path,
    coverage_target: float,
    panels: Optional[Sequence[str]] = None,
) -> RecordBundle:
    records: List[Dict[str, object]] = []
    comprehensive_records: List[Dict[str, object]] = []
    gamma_records: List[Dict[str, object]] = []
    gamma_comprehensive_records: List[Dict[str, object]] = []
    selection_records: List[Dict[str, object]] = []

    desired_panels = set(panels) if panels else None
    discovered_panels: set[str] = set()

    trial_dirs = sorted([path for path in eval_root.iterdir() if path.is_dir()])
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found under {eval_root}")

    for trial_dir in trial_dirs:
        panel_paths = sorted(trial_dir.glob("meps_panel_*_alpha_*.npz"))
        if not panel_paths:
            continue
        for panel_path in panel_paths:
            payload = _load_payload(panel_path)
            panel = str(payload.get("panel"))
            if desired_panels and panel not in desired_panels:
                continue
            metadata = payload.get("metadata")
            source_label_map: Dict[str, str] = {}
            seed = None
            if metadata is not None:
                seed = getattr(metadata, "random_seed", None)
                raw_mapping = getattr(metadata, "source_mapping", None)
                if isinstance(raw_mapping, dict):
                    source_label_map = {
                        str(key): str(value) for key, value in raw_mapping.items()
                    }
            if seed is None:
                # attempt to parse from path name
                try:
                    seed = int(trial_dir.name.rsplit("_", 1)[-1])
                except ValueError:
                    seed = -1
            run_id = f"panel{panel}_seed{seed}"
            discovered_panels.add(panel)

            baseline_results = payload.get("baseline", {})
            if isinstance(baseline_results, dict):
                for method_key, metrics in baseline_results.items():
                    if not isinstance(metrics, dict):
                        continue
                    if method_key == "Max Aggregation":
                        label = BASELINE_MAX
                    elif method_key == "Single Source":
                        label = BASELINE_SINGLE
                    elif method_key == "CQR Max Aggregation":
                        label = CQR_MAX
                    elif method_key.startswith("CQR Source "):
                        suffix = method_key.split(" ")[-1]
                        label = f"{CQR_SOURCE_PREFIX}{suffix}"
                        color_idx = int(suffix) if suffix.isdigit() else 0
                        color = CQR_SOURCE_COLORS[color_idx % len(CQR_SOURCE_COLORS)]
                        METHOD_LABELS.setdefault(label, f"CQR source {suffix}")
                        PALETTE.setdefault(label, color)
                    else:
                        label = method_key
                    METHOD_LABELS.setdefault(label, label)
                    PALETTE.setdefault(label, "#888888")
                    base_info = {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "true",
                        "method": label,
                    }
                    _add_records(records, base_info, _extract_metrics(metrics))

            baseline_comp = payload.get("baseline_comprehensive")
            if isinstance(baseline_comp, dict):
                for agg_key, subset_map in baseline_comp.items():
                    if agg_key == "Max_Aggregated":
                        method_label = BASELINE_MAX
                    elif agg_key == "Single_Source":
                        method_label = BASELINE_SINGLE
                    elif agg_key == "CQR_Max_Aggregated":
                        method_label = CQR_MAX
                    elif agg_key.startswith("Source_"):
                        suffix = agg_key.split("_")[-1]
                        method_label = f"{BASELINE_SOURCE_PREFIX}{suffix}"
                        display = source_label_map.get(suffix)
                        if display is None and suffix.isdigit():
                            display = source_label_map.get(str(int(suffix)))
                        if display is None:
                            display = f"src {suffix}"
                        pretty_label = f"Baseline single-src ({display})"
                        METHOD_LABELS.setdefault(method_label, pretty_label)
                        color_idx = int(suffix) if suffix.isdigit() else 0
                        color = BASELINE_SOURCE_COLORS[color_idx % len(BASELINE_SOURCE_COLORS)]
                        PALETTE.setdefault(method_label, color)
                    elif agg_key.startswith("CQR_Source_"):
                        suffix = agg_key.split("_")[-1]
                        method_label = f"{CQR_SOURCE_PREFIX}{suffix}"
                        display = source_label_map.get(suffix)
                        if display is None and suffix.isdigit():
                            display = source_label_map.get(str(int(suffix)))
                        if display is None:
                            display = f"src {suffix}"
                        pretty_label = f"CQR single-src ({display})"
                        METHOD_LABELS.setdefault(method_label, pretty_label)
                        color_idx = int(suffix) if suffix.isdigit() else 0
                        color = CQR_SOURCE_COLORS[color_idx % len(CQR_SOURCE_COLORS)]
                        PALETTE.setdefault(method_label, color)
                    else:
                        method_label = f"Baseline {agg_key.replace('_', ' ')}"
                        METHOD_LABELS.setdefault(method_label, method_label)
                        PALETTE.setdefault(method_label, "#777777")
                    if not isinstance(subset_map, dict):
                        continue
                    for subset_name, metrics in subset_map.items():
                        if not isinstance(metrics, dict):
                            continue
                        metric_values = {
                            "coverage": float(metrics.get("coverage", float("nan"))),
                            REGRESSION_METRIC_KEY: float(
                                metrics.get(REGRESSION_METRIC_KEY, float("nan"))
                            ),
                        }
                        base_info = {
                            "panel": panel,
                            "run_id": run_id,
                            "evaluation": "true",
                            "method": method_label,
                            "subset": subset_name,
                        }
                        _add_records(comprehensive_records, base_info, metric_values)

            gamma_entries = payload.get("mdcp_gamma_results", [])
            if not isinstance(gamma_entries, list) or not gamma_entries:
                continue

            best_mimic = _select_entry(gamma_entries, "mimic_metrics", coverage_target)
            best_true = _select_entry(gamma_entries, "metrics", coverage_target)

            if best_mimic is not None:
                mimic_gamma_name = best_mimic.get("gamma_name")
                mimic_gamma_value = float(best_mimic.get("gamma", float("nan")))
                mimic_gamma_label = _format_gamma_label(mimic_gamma_value)
                selection_records.append(
                    {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "mimic",
                        "gamma": mimic_gamma_name,
                        "gamma_label": mimic_gamma_label,
                        "gamma_value": mimic_gamma_value,
                    }
                )

                mimic_metrics = best_mimic.get("mimic_metrics")
                if isinstance(mimic_metrics, dict):
                    base_mimic = {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "mimic",
                        "method": MDCP_PER_TRIAL_MIMIC,
                        "gamma": mimic_gamma_name,
                        "gamma_label": mimic_gamma_label,
                        "gamma_value": mimic_gamma_value,
                    }
                    _add_records(records, base_mimic, _extract_metrics(mimic_metrics))

                true_metrics_from_mimic = best_mimic.get("metrics")
                if isinstance(true_metrics_from_mimic, dict):
                    base_true_for_mimic = {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "true",
                        "method": MDCP_PER_TRIAL_MIMIC,
                        "gamma": mimic_gamma_name,
                        "gamma_label": mimic_gamma_label,
                        "gamma_value": mimic_gamma_value,
                    }
                    _add_records(records, base_true_for_mimic, _extract_metrics(true_metrics_from_mimic))

                    for subset_name, metrics in _subset_metrics_for_mdcp(true_metrics_from_mimic).items():
                        base_info = {
                            "panel": panel,
                            "run_id": run_id,
                            "evaluation": "true",
                            "method": MDCP_PER_TRIAL_MIMIC,
                            "subset": subset_name,
                        }
                        _add_records(comprehensive_records, base_info, metrics)

            if best_true is not None:
                true_gamma_name = best_true.get("gamma_name")
                true_gamma_value = float(best_true.get("gamma", float("nan")))
                true_gamma_label = _format_gamma_label(true_gamma_value)
                selection_records.append(
                    {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "true",
                        "gamma": true_gamma_name,
                        "gamma_label": true_gamma_label,
                        "gamma_value": true_gamma_value,
                    }
                )
                true_metrics = best_true.get("metrics")
                if isinstance(true_metrics, dict):
                    base_true = {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": "true",
                        "method": MDCP_PER_TRIAL_TRUE,
                        "gamma": true_gamma_name,
                        "gamma_label": true_gamma_label,
                        "gamma_value": true_gamma_value,
                    }
                    _add_records(records, base_true, _extract_metrics(true_metrics))

                    for subset_name, metrics in _subset_metrics_for_mdcp(true_metrics).items():
                        base_info = {
                            "panel": panel,
                            "run_id": run_id,
                            "evaluation": "true",
                            "method": MDCP_PER_TRIAL_TRUE,
                            "subset": subset_name,
                        }
                        _add_records(comprehensive_records, base_info, metrics)

            for entry in gamma_entries:
                gamma_name = entry.get("gamma_name")
                gamma_value = float(entry.get("gamma", float("nan")))
                gamma_label = _format_gamma_label(gamma_value)
                for eval_label, key in (("true", "metrics"), ("mimic", "mimic_metrics")):
                    metrics = entry.get(key)
                    if not isinstance(metrics, dict):
                        continue
                    extracted = _extract_metrics(metrics)
                    base_info = {
                        "panel": panel,
                        "run_id": run_id,
                        "evaluation": eval_label,
                        "gamma": gamma_name,
                        "gamma_label": gamma_label,
                        "gamma_value": gamma_value,
                    }
                    _add_records(gamma_records, base_info, extracted)

                    for subset_name, values in _subset_metrics_for_mdcp(metrics).items():
                        comp_info = {
                            "panel": panel,
                            "run_id": run_id,
                            "evaluation": eval_label,
                            "subset": subset_name,
                            "gamma": gamma_name,
                            "gamma_label": gamma_label,
                            "gamma_value": gamma_value,
                        }
                        _add_records(gamma_comprehensive_records, comp_info, values)

    record_df = pd.DataFrame.from_records(records)
    comp_df = pd.DataFrame.from_records(comprehensive_records)
    gamma_df = pd.DataFrame.from_records(gamma_records)
    gamma_comp_df = pd.DataFrame.from_records(gamma_comprehensive_records)
    selection_df = pd.DataFrame.from_records(selection_records)

    return RecordBundle(
        records=record_df,
        comprehensive=comp_df,
        gamma=gamma_df,
        gamma_comprehensive=gamma_comp_df,
        selection=selection_df,
        panels=sorted(discovered_panels, key=lambda x: int(x) if str(x).isdigit() else x),
    )


def _determine_best_gamma(
    gamma_df: pd.DataFrame,
    panel: str,
    evaluation: str,
    coverage_target: float,
) -> Tuple[Optional[str], Optional[float]]:
    if gamma_df.empty:
        return (None, None)
    mask = (gamma_df["panel"] == panel) & (gamma_df["evaluation"] == evaluation)
    subset = gamma_df[mask]
    if subset.empty:
        return (None, None)

    coverage_mean = (
        subset[subset["metric"] == "overall_coverage"].groupby("gamma_label")["value"].mean()
    )
    width_mean = (
        subset[subset["metric"] == REGRESSION_METRIC_KEY].groupby("gamma_label")["value"].mean()
    )
    if width_mean.empty:
        return (None, None)

    candidate_labels = width_mean.index.tolist()
    best_label: Optional[str] = None
    best_eff = float("inf")
    best_cov = float("nan")

    for label in candidate_labels:
        eff = float(width_mean.get(label, float("nan")))
        cov = float(coverage_mean.get(label, float("nan")))
        if math.isnan(eff) or math.isnan(cov):
            continue
        if cov >= coverage_target:
            if best_label is None or eff < best_eff or (math.isclose(eff, best_eff) and cov > best_cov):
                best_label = label
                best_eff = eff
                best_cov = cov

    if best_label is None:
        best_cov = -float("inf")
        best_eff = float("inf")
        for label in candidate_labels:
            eff = float(width_mean.get(label, float("nan")))
            cov = float(coverage_mean.get(label, float("nan")))
            if math.isnan(eff) or math.isnan(cov):
                continue
            if cov > best_cov or (math.isclose(cov, best_cov) and eff < best_eff):
                best_label = label
                best_cov = cov
                best_eff = eff

    if best_label is None:
        return (None, None)

    value_series = subset[(subset["gamma_label"] == best_label)]["gamma_value"].dropna()
    best_value = value_series.iloc[0] if not value_series.empty else None
    return (best_label, best_value)


def _draw_bars_with_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    method_order: List[str],
    coverage_target: float,
    show_target_line: bool,
) -> None:
    means = data.groupby("method")["value"].mean()
    positions = np.arange(len(method_order))
    line_half_width = 0.25
    jitter_span = 0.2
    rng = np.random.default_rng(42)

    for idx, method in enumerate(method_order):
        subset = data[data["method"] == method]
        if subset.empty:
            continue
        color = PALETTE.get(method, "#555555")
        mean_val = means.get(method, np.nan)

        jitter = (rng.random(len(subset)) - 0.5) * jitter_span
        ax.scatter(
            np.full(len(subset), idx, dtype=float) + jitter,
            subset["value"],
            color=color,
            s=8,
            alpha=0.8,
            edgecolors="none",
        )
        if not math.isnan(mean_val):
            ax.hlines(mean_val, idx - line_half_width, idx + line_half_width, colors=color, linewidth=2.0, alpha=0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in method_order], rotation=18, ha="right", fontsize=9)
    if show_target_line:
        ax.axhline(coverage_target, color="gray", linestyle="--", linewidth=1.0)
        ymin, ymax = ax.get_ylim()
        ymax = max(ymax, coverage_target * 1.05 if coverage_target > 0 else ymax)
        ax.set_ylim(bottom=0.7, top=ymax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def _plot_summary_panels(
    df: pd.DataFrame,
    evaluation: str,
    coverage_target: float,
    output_path: Path,
    title: str,
    method_priority: Optional[List[str]] = None,
) -> None:
    metric_order = ["overall_coverage", "worst_case_coverage", REGRESSION_METRIC_KEY, REGRESSION_WORST_METRIC_KEY]
    panels = _ordered_panels(df["panel"].unique())
    if not panels:
        return

    n_rows = len(panels)
    n_cols = len(metric_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.2 * n_rows), squeeze=False)

    for row_idx, panel in enumerate(panels):
        panel_slice = df[(df["evaluation"] == evaluation) & (df["panel"] == panel)]
        for col_idx, metric in enumerate(metric_order):
            ax = axes[row_idx, col_idx]
            metric_slice = panel_slice[panel_slice["metric"] == metric]
            if metric_slice.empty:
                ax.axis("off")
                continue
            method_order = _ordered_methods(metric_slice["method"]) if method_priority is None else [m for m in method_priority if m in metric_slice["method"].unique()]
            if not method_order:
                ax.axis("off")
                continue
            show_target = "coverage" in metric
            _draw_bars_with_points(ax, metric_slice, method_order, coverage_target, show_target)
            if row_idx == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric))
            if col_idx == 0:
                ax.set_ylabel(f"Panel {panel}")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_comprehensive_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    coverage_target: Optional[float] = None,
) -> None:
    if df.empty:
        return
    panels = _ordered_panels(df["panel"].unique())
    subset_order = _ordered_subsets(df["subset"].unique())
    method_order = _ordered_methods(df["method"].unique())
    if not subset_order or not method_order:
        return

    n_rows = len(subset_order)
    n_cols = len(panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.1 * n_rows), squeeze=False)

    for row_idx, subset in enumerate(subset_order):
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            mask = (
                (df["subset"] == subset)
                & (df["panel"] == panel)
                & (df["metric"] == metric)
            )
            subset_slice = df[mask]
            if subset_slice.empty:
                ax.axis("off")
                continue
            means = subset_slice.groupby("method")["value"].mean()
            positions = np.arange(len(method_order), dtype=float)
            rng = np.random.default_rng(42)
            line_half_width = 0.25
            jitter_span = 0.2
            for idx, method in enumerate(method_order):
                method_mask = subset_slice["method"] == method
                if not method_mask.any():
                    continue
                values = subset_slice.loc[method_mask, "value"]
                color = PALETTE.get(method, "#555555")
                mean_val = means.get(method, np.nan)
                jitter = (rng.random(len(values)) - 0.5) * jitter_span
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    color=color,
                    s=10,
                    alpha=0.8,
                    edgecolors="none",
                )
                if not math.isnan(mean_val):
                    ax.hlines(mean_val, idx - line_half_width, idx + line_half_width, colors=color, linewidth=2.0, alpha=0.9)
            ax.set_xticks(positions)
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in method_order], rotation=18, ha="right", fontsize=8)
            if coverage_target is not None and "coverage" in metric:
                ax.axhline(coverage_target, color="gray", linestyle="--", linewidth=1.0)
                ymin, ymax = ax.get_ylim()
                ymax = max(ymax, coverage_target * 1.05 if coverage_target > 0 else ymax)
                ax.set_ylim(bottom=0.7, top=ymax)
            if col_idx == 0:
                ax.set_ylabel(subset)
            if row_idx == 0:
                ax.set_title(f"Panel {panel}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gamma_comparison(
    gamma_df: pd.DataFrame,
    evaluation: str,
    coverage_target: float,
    output_path: Path,
    title: str,
) -> None:
    subset = gamma_df[gamma_df["evaluation"] == evaluation]
    if subset.empty:
        return

    panels = _ordered_panels(subset["panel"].unique())
    metrics = ["overall_coverage", REGRESSION_METRIC_KEY]

    fig, axes = plt.subplots(len(panels), len(metrics), figsize=(4.0 * len(metrics), 3.0 * len(panels)), squeeze=False)

    for row_idx, panel in enumerate(panels):
        panel_slice = subset[subset["panel"] == panel]
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            metric_slice = panel_slice[panel_slice["metric"] == metric]
            if metric_slice.empty:
                ax.axis("off")
                continue
            summary = (
                metric_slice.groupby(["gamma_label", "gamma_value"])["value"]
                .agg(["mean", "std"])
                .reset_index()
            )
            summary = summary.sort_values(by="gamma_value")
            positions = np.arange(len(summary), dtype=float)
            ax.errorbar(
                positions,
                summary["mean"],
                yerr=summary["std"],
                fmt="-o",
                color=EVALUATION_STYLES.get(evaluation, {}).get("color", "#555555"),
                ecolor="#333333",
                capsize=3,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(summary["gamma_label"], rotation=18, ha="right")
            if "coverage" in metric:
                ax.axhline(coverage_target, color="gray", linestyle="--", linewidth=1.0)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(f"Panel {panel}")
            ax.set_title(METRIC_LABELS.get(metric, metric))

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gamma_selection(
    selection_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if selection_df.empty:
        return

    panels = _ordered_panels(selection_df["panel"].unique())
    evaluations = [eval_label for eval_label in ("mimic", "true") if eval_label in selection_df["evaluation"].unique()]
    if not evaluations:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(4.8, 2.8 * len(panels)), squeeze=False)

    for row_idx, panel in enumerate(panels):
        ax = axes[row_idx, 0]
        panel_slice = selection_df[selection_df["panel"] == panel]
        if panel_slice.empty:
            ax.axis("off")
            continue
        ordering = (
            panel_slice[["gamma_label", "gamma_value"]]
            .drop_duplicates()
            .sort_values(by="gamma_value")
        )
        gamma_labels = ordering["gamma_label"].tolist()
        positions = np.arange(len(gamma_labels), dtype=float)
        bar_width = 0.8 / max(1, len(evaluations))
        ymax = 0
        for idx, eval_label in enumerate(evaluations):
            offset = (idx - (len(evaluations) - 1) / 2.0) * bar_width
            style = EVALUATION_STYLES.get(eval_label, {"label": eval_label, "color": "#555555"})
            counts = panel_slice[panel_slice["evaluation"] == eval_label].groupby("gamma_label")["run_id"].nunique()
            heights = [int(counts.get(label, 0)) for label in gamma_labels]
            ymax = max(ymax, max(heights, default=0))
            ax.bar(
                positions + offset,
                heights,
                width=bar_width,
                color=style.get("color", "#555555"),
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label=style.get("label", eval_label),
            )
            for pos, height in zip(positions + offset, heights):
                if height > 0:
                    ax.text(pos, height + 0.05 * max(1, ymax), str(height), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(gamma_labels, rotation=18, ha="right")
        ax.set_ylabel(f"Panel {panel}")
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.25 + 0.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def collect_meps_records(
    eval_root: Path,
    coverage_target: float,
    panels: Optional[Sequence[str]] = None,
) -> RecordBundle:
    return _collect_records(eval_root, coverage_target, panels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MEPS MDCP regression evaluation metrics.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval_out" / "meps1018" / "meps_raw",
        help="Path to the directory containing per-trial MEPS evaluation payloads.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.9,
        help="Target coverage used when selecting gamma (default: 0.9).",
    )
    parser.add_argument(
        "--panels",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of panel identifiers to include (e.g. 19 20 21).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval_out" / "meps1018" / "fig",
        help="Directory where figures will be written.",
    )
    args = parser.parse_args()

    eval_root = args.eval_root
    if not eval_root.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_root}")

    bundle = collect_meps_records(eval_root, args.coverage_target, args.panels)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = bundle.records
    comp_df = bundle.comprehensive
    gamma_df = bundle.gamma
    gamma_comp_df = bundle.gamma_comprehensive
    selection_df = bundle.selection

    panels = bundle.panels
    if not panels:
        raise RuntimeError("No panels found in evaluation payloads.")

    extra_rows: List[pd.DataFrame] = []
    extra_comp_rows: List[pd.DataFrame] = []

    for panel in panels:
        mimic_label, _ = _determine_best_gamma(gamma_df, panel, "mimic", args.coverage_target)
        if mimic_label:
            mimic_rows_true = gamma_df[
                (gamma_df["panel"] == panel)
                & (gamma_df["evaluation"] == "true")
                & (gamma_df["gamma_label"] == mimic_label)
            ].copy()
            mimic_rows_true["method"] = MDCP_AVG_MIMIC
            extra_rows.append(mimic_rows_true)

            mimic_rows_mimic = gamma_df[
                (gamma_df["panel"] == panel)
                & (gamma_df["evaluation"] == "mimic")
                & (gamma_df["gamma_label"] == mimic_label)
            ].copy()
            mimic_rows_mimic["method"] = MDCP_AVG_MIMIC
            extra_rows.append(mimic_rows_mimic)

            mimic_comp = gamma_comp_df[
                (gamma_comp_df["panel"] == panel)
                & (gamma_comp_df["evaluation"] == "true")
                & (gamma_comp_df["gamma_label"] == mimic_label)
            ].copy()
            mimic_comp["method"] = MDCP_AVG_MIMIC
            extra_comp_rows.append(mimic_comp)

        true_label, _ = _determine_best_gamma(gamma_df, panel, "true", args.coverage_target)
        if true_label:
            true_rows_true = gamma_df[
                (gamma_df["panel"] == panel)
                & (gamma_df["evaluation"] == "true")
                & (gamma_df["gamma_label"] == true_label)
            ].copy()
            true_rows_true["method"] = MDCP_AVG_TRUE
            extra_rows.append(true_rows_true)

            true_comp = gamma_comp_df[
                (gamma_comp_df["panel"] == panel)
                & (gamma_comp_df["evaluation"] == "true")
                & (gamma_comp_df["gamma_label"] == true_label)
            ].copy()
            true_comp["method"] = MDCP_AVG_TRUE
            extra_comp_rows.append(true_comp)

    if extra_rows:
        df = pd.concat([df] + extra_rows, ignore_index=True)
    if extra_comp_rows:
        comp_df = pd.concat([comp_df] + extra_comp_rows, ignore_index=True)

    dfs_summary = (
        df.groupby(["evaluation", "panel", "method", "metric"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_path = out_dir / "meps_rand_metrics_summary.csv"
    dfs_summary.to_csv(summary_path, index=False)
    print(f"Saved metric summary: {summary_path}")

    method_priority = [
        MDCP_PER_TRIAL_MIMIC,
        MDCP_PER_TRIAL_TRUE,
        MDCP_AVG_MIMIC,
        MDCP_AVG_TRUE,
        BASELINE_MAX,
        BASELINE_SINGLE,
    ]

    true_summary_path = out_dir / "meps_rand_eval_summary_true.png"
    _plot_summary_panels(df, "true", args.coverage_target, true_summary_path, "MEPS MDCP Evaluation (True)", method_priority)
    if true_summary_path.exists():
        print(f"Saved summary figure (true eval): {true_summary_path}")

    mimic_summary_path = out_dir / "meps_rand_eval_summary_mimic.png"
    _plot_summary_panels(df, "mimic", args.coverage_target, mimic_summary_path, "MEPS MDCP Evaluation (Mimic)", [MDCP_PER_TRIAL_MIMIC, MDCP_AVG_MIMIC])
    if mimic_summary_path.exists():
        print(f"Saved summary figure (mimic eval): {mimic_summary_path}")

    comp_slice = comp_df[comp_df["evaluation"] == "true"]
    comp_cov_path = out_dir / "meps_rand_eval_comprehensive_true.png"
    _plot_comprehensive_metric(
        comp_slice,
        "coverage",
        comp_cov_path,
        "MEPS MDCP Comprehensive (True Eval – Coverage)",
        coverage_target=args.coverage_target,
    )
    if comp_cov_path.exists():
        print(f"Saved comprehensive coverage figure: {comp_cov_path}")

    comp_width_path = out_dir / "meps_rand_eval_comprehensive_true_width.png"
    _plot_comprehensive_metric(
        comp_slice,
        REGRESSION_METRIC_KEY,
        comp_width_path,
        "MEPS MDCP Comprehensive (True Eval – Interval Width)",
    )
    if comp_width_path.exists():
        print(f"Saved comprehensive width figure: {comp_width_path}")

    gamma_true_path = out_dir / "meps_rand_gamma_comparison_true.png"
    _plot_gamma_comparison(gamma_df, "true", args.coverage_target, gamma_true_path, "MEPS Gamma Comparison (True)")
    if gamma_true_path.exists():
        print(f"Saved gamma comparison (true): {gamma_true_path}")

    gamma_mimic_path = out_dir / "meps_rand_gamma_comparison_mimic.png"
    _plot_gamma_comparison(gamma_df, "mimic", args.coverage_target, gamma_mimic_path, "MEPS Gamma Comparison (Mimic)")
    if gamma_mimic_path.exists():
        print(f"Saved gamma comparison (mimic): {gamma_mimic_path}")

    selection_path = out_dir / "meps_rand_gamma_selection.png"
    _plot_gamma_selection(selection_df, selection_path, "MEPS Gamma Selection Distribution")
    if selection_path.exists():
        print(f"Saved gamma selection figure: {selection_path}")


if __name__ == "__main__":
    main()

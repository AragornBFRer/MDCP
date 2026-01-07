from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

CLASSIFICATION = "classification"
REGRESSION = "regression"

COVERAGE_KEY = "coverage"
WORST_CASE_KEY = "worst_case_coverage"
CLASS_EFF_KEY = "avg_set_size"
REG_EFF_KEY = "avg_width"

METHOD_BASELINE_AGG = "Max-agg baseline"
METHOD_BASELINE_SINGLE = "Single-source avg"
METHOD_BASELINE_SOURCE_PREFIX = "Source "
METHOD_MDCP_NONPEN = "MDCP"
METHOD_MDCP_MIMIC = "MDCP tuned"

BASELINE_AGG_COLOR = "#5A6270"
BASELINE_SINGLE_COLOR = "#747D8A"
BASELINE_SOURCE_COLORS = [
    "#8B94A1",
    "#9CA5B1",
    "#ADB5BF",
    "#BEC6CE",
    "#CFD7DD",
]
MDCP_NONPEN_COLOR = "#5C8FD4"
MDCP_MIMIC_COLOR = "#81ABDE"

PALETTE: Dict[str, str] = {
    METHOD_BASELINE_AGG: BASELINE_AGG_COLOR,
    METHOD_BASELINE_SINGLE: BASELINE_SINGLE_COLOR,
    METHOD_MDCP_NONPEN: MDCP_NONPEN_COLOR,
    METHOD_MDCP_MIMIC: MDCP_MIMIC_COLOR,
}

_COLOR_CACHE: Dict[str, str] = {}

SOURCE_MARKERS = ["P", "s", "^", "v", "X", "D", "o", "*"]
SOURCE_LINESTYLES: List[object] = [
    ":",
    "--",
    "-.",
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (2, 1, 2, 1)),
    (0, (5, 2)),
    (0, (3, 2, 1, 2)),
]
SOURCE_MARKER_SIZES = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]  # for single source

MDCP_EDGE_COLOR = "#1D3B5B"

MIN_COVERAGE_Y = 0.7
COVERAGE_COMPRESS_FACTOR = 0.35
COVERAGE_PADDING = 0.015
JITTER_MAX = 0.16
SUMMARY_POINT_JITTER = 0.12
SUMMARY_POINT_SIZE = 14
SUMMARY_LINE_WIDTH = 1.9

METHOD_RENAMES: Dict[str, str] = {
    "Max Aggregation": METHOD_BASELINE_AGG,
    "Max_Aggregation": METHOD_BASELINE_AGG,
    "Max_Aggregated": METHOD_BASELINE_AGG,
    "baseline_max": METHOD_BASELINE_AGG,
    "baseline_max_agg": METHOD_BASELINE_AGG,
    "Single_Source": METHOD_BASELINE_SINGLE,
    "Single Source": METHOD_BASELINE_SINGLE,
    "baseline_single": METHOD_BASELINE_SINGLE,
    "MDCP": METHOD_MDCP_NONPEN,
    "MDCP_nonpen": METHOD_MDCP_NONPEN,
    "MDCP_tuned": METHOD_MDCP_MIMIC,
    "mdcp_nonpen": METHOD_MDCP_NONPEN,
    "mdcp_mimic": METHOD_MDCP_MIMIC,
    "MDCP (non-penalized)": METHOD_MDCP_NONPEN,
    "MDCP (mimic-selected)": METHOD_MDCP_MIMIC,
}


def _efficiency_key(task: str) -> str:
    return CLASS_EFF_KEY if task == CLASSIFICATION else REG_EFF_KEY


def _normalized_source_label(label: str) -> str:
    candidates = [
        ("baseline_source_", label),
        ("Source_", label),
        ("source_", label),
        (METHOD_BASELINE_SOURCE_PREFIX, label),
    ]
    suffix = label
    for prefix, value in candidates:
        if value.startswith(prefix):
            suffix = value[len(prefix) :]
            break
    suffix = suffix.replace("_", " ").strip()
    if suffix.isdigit():
        suffix = str(int(suffix))
    return f"{METHOD_BASELINE_SOURCE_PREFIX}{suffix}"


def _method_label(method: str) -> str:
    if method is None:
        return "Unknown"
    label = METHOD_RENAMES.get(method, method)
    if label.startswith("Source_"):
        return _normalized_source_label(label)
    if label.startswith("baseline_source_"):
        return _normalized_source_label(label)
    if label.startswith(METHOD_BASELINE_SOURCE_PREFIX):
        remainder = label[len(METHOD_BASELINE_SOURCE_PREFIX) :].strip()
        if remainder.isdigit():
            remainder = str(int(remainder))
        return f"{METHOD_BASELINE_SOURCE_PREFIX}{remainder}"
    return label


def _with_alpha(color: str, alpha: float) -> Tuple[float, float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r, g, b, max(0.0, min(1.0, alpha)))


def _point_offsets(count: int, jitter: float = SUMMARY_POINT_JITTER) -> np.ndarray:
    if count <= 1:
        return np.array([0.0])
    if count == 2:
        return np.array([-jitter, jitter])
    return np.linspace(-jitter, jitter, count)


def _color_for_method(method: str) -> str:
    if method in PALETTE:
        return PALETTE[method]
    if method in _COLOR_CACHE:
        return _COLOR_CACHE[method]
    if method.startswith(METHOD_BASELINE_SOURCE_PREFIX):
        try:
            idx = int(method.replace(METHOD_BASELINE_SOURCE_PREFIX, ""))
        except ValueError:
            idx = 0
        color = BASELINE_SOURCE_COLORS[idx % len(BASELINE_SOURCE_COLORS)]
    else:
        palette = sns.color_palette("tab10")
        idx = len(_COLOR_CACHE) % len(palette)
        color = mcolors.to_hex(palette[idx])
    _COLOR_CACHE[method] = color
    return color


def _style_for_method(method: str) -> Dict[str, object]:
    color = _color_for_method(method)
    base_style: Dict[str, object] = {
        "color": color,
        "linewidth": 1.7,
        "linestyle": "-",
        "alpha": 0.9,
        "marker": "o",
        "markersize": 4.6,
        "marker_facecolor": color,
        "marker_edgecolor": "black",
        "marker_edgewidth": 0.5,
        "scatter_alpha": 0.35,
    }

    if method == METHOD_BASELINE_AGG:
        base_style.update({
            "linewidth": 2.1,
            "linestyle": "-",
            "alpha": 0.92,
            "marker": "o",
            "markersize": 6.2,
            "marker_edgecolor": "#222832",
            "marker_edgewidth": 0.6,
            "scatter_alpha": 0.30,
        })
    elif method == METHOD_BASELINE_SINGLE:
        base_style.update({
            "linewidth": 1.6,
            "linestyle": "-.",
            "alpha": 0.85,
            "marker": "D",
            "markersize": 5.4,
            "scatter_alpha": 0.28,
        })
    elif method == METHOD_MDCP_NONPEN:
        base_style.update({
            "linewidth": 2.4,
            "linestyle": "-",
            "alpha": 0.95,
            "marker": "o",
            "markersize": 6.8,
            "marker_edgecolor": MDCP_EDGE_COLOR,
            "marker_edgewidth": 0.65,
            "scatter_alpha": 0.45,
        })
    elif method == METHOD_MDCP_MIMIC:
        base_style.update({
            "linewidth": 2.4,
            "linestyle": "-",
            "alpha": 0.95,
            "marker": "D",
            "markersize": 5.8,
            "marker_edgecolor": MDCP_EDGE_COLOR,
            "marker_edgewidth": 0.65,
            "scatter_alpha": 0.45,
        })
    elif method.startswith(METHOD_BASELINE_SOURCE_PREFIX):
        suffix = method[len(METHOD_BASELINE_SOURCE_PREFIX) :].strip()
        idx_val = 0
        if suffix:
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                idx_val = int(digits)
            else:
                idx_val = sum(ord(ch) for ch in suffix)
        marker = SOURCE_MARKERS[idx_val % len(SOURCE_MARKERS)]
        linestyle = SOURCE_LINESTYLES[idx_val % len(SOURCE_LINESTYLES)]
        markersize = SOURCE_MARKER_SIZES[idx_val % len(SOURCE_MARKER_SIZES)]
        base_style.update({
            "linewidth": 1.25,
            "linestyle": linestyle,
            "alpha": 0.78,
            "marker": marker,
            "markersize": markersize,
            "marker_facecolor": color,
            "marker_edgecolor": "#1A1A1A",
            "marker_edgewidth": 0.45,
            "scatter_alpha": 0.24,
        })
    return base_style


def _ordered_terms(terms: Iterable[object]) -> List[str]:
    unique_terms: List[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        if term not in unique_terms:
            unique_terms.append(term)

    def sort_key(term: str) -> Tuple[int, str]:
        lower = term.lower()
        return (0 if lower == "linear" else 1, lower)

    return sorted(unique_terms, key=sort_key)


def _transform_metric_value(metric: str, value: float) -> float:
    if metric != COVERAGE_KEY and metric != WORST_CASE_KEY:
        return value
    if np.isnan(value) or value >= MIN_COVERAGE_Y:
        return value
    return MIN_COVERAGE_Y - (MIN_COVERAGE_Y - value) * COVERAGE_COMPRESS_FACTOR


def _coverage_axis_limits(values: np.ndarray, coverage_target: Optional[float]) -> Tuple[float, float, List[float], List[str]]:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        base = MIN_COVERAGE_Y
        return base - COVERAGE_PADDING, base + COVERAGE_PADDING, [base], [f"{base:.2f}"]

    min_val = float(clean.min())
    max_val = float(clean.max())

    bottom = _transform_metric_value(COVERAGE_KEY, min_val) if min_val < MIN_COVERAGE_Y else MIN_COVERAGE_Y
    bottom = min(bottom, MIN_COVERAGE_Y - COVERAGE_PADDING)

    top_candidates = [MIN_COVERAGE_Y, _transform_metric_value(COVERAGE_KEY, max_val)]
    if coverage_target is not None and not np.isnan(coverage_target):
        top_candidates.append(_transform_metric_value(COVERAGE_KEY, coverage_target))
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
        ticks.append(_transform_metric_value(COVERAGE_KEY, actual))
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
    positions = range(len(method_order))
    for idx, method in enumerate(method_order):
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        mean_val = float(means.get(method, np.nan))
        color = _color_for_method(method)
        display_mean = _transform_metric_value(metric, mean_val) if not np.isnan(mean_val) else np.nan
        ax.bar(
            idx,
            display_mean,
            color=color,
            width=0.58,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        values = subset["value"].to_numpy(dtype=float)
        if values.size:
            display_vals = np.array([_transform_metric_value(metric, val) for val in values])
            offsets = np.linspace(-JITTER_MAX, JITTER_MAX, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.shape, idx, dtype=float) + offsets,
                display_vals,
                color=color,
                s=10,
                linewidths=0.0,
                zorder=3,
                clip_on=False,
            )

    ax.set_xticks(list(positions))
    labels = [method for method in method_order]
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    if method_order:
        ax.set_xlim(-0.5 - JITTER_MAX, len(method_order) - 0.5 + JITTER_MAX)

    if metric in {COVERAGE_KEY, WORST_CASE_KEY}:
        bottom, top, ticks, tick_labels = _coverage_axis_limits(df["value"].to_numpy(dtype=float), coverage_target)
        ax.set_ylim(bottom, top)
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
        if coverage_target is not None and not np.isnan(coverage_target):
            ax.axhline(
                _transform_metric_value(metric, float(coverage_target)),
                color="#666666",
                linestyle="--",
                linewidth=1.0,
            )
    else:
        ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = set(methods)
    ordered: List[str] = []
    if METHOD_BASELINE_AGG in methods:
        ordered.append(METHOD_BASELINE_AGG)
    source_methods = sorted(
        (m for m in methods if m.startswith(METHOD_BASELINE_SOURCE_PREFIX)),
        key=lambda name: int(name.replace(METHOD_BASELINE_SOURCE_PREFIX, "")) if name.replace(METHOD_BASELINE_SOURCE_PREFIX, "").isdigit() else 0,
    )
    ordered.extend(source_methods)
    if METHOD_MDCP_NONPEN in methods:
        ordered.append(METHOD_MDCP_NONPEN)
    if METHOD_MDCP_MIMIC in methods:
        ordered.append(METHOD_MDCP_MIMIC)
    return ordered


def _compute_worst_case(metrics: Dict[str, object], subset_map: Optional[Dict[str, Dict[str, object]]] = None) -> float:
    values: List[float] = []
    indiv = metrics.get("individual_coverage") if isinstance(metrics, dict) else None
    if isinstance(indiv, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(indiv, dtype=float)
        except Exception:
            arr = np.array([], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            values.append(float(arr.min()))
    if subset_map:
        for subset_name, subset_metrics in subset_map.items():
            if subset_name == "Overall" or not isinstance(subset_metrics, dict):
                continue
            try:
                cov = float(subset_metrics.get(COVERAGE_KEY, np.nan))
            except (TypeError, ValueError):
                cov = np.nan
            if not np.isnan(cov):
                values.append(cov)
    if values:
        return float(np.nanmin(values))
    return float("nan")


def _bundle_metrics(
    metrics: Optional[Dict[str, object]],
    task: str,
    *,
    subset_map: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    eff_key = _efficiency_key(task)
    bundle: Dict[str, float] = {}
    try:
        bundle[COVERAGE_KEY] = float(metrics.get(COVERAGE_KEY, np.nan))
    except (TypeError, ValueError):
        bundle[COVERAGE_KEY] = float("nan")
    try:
        bundle[eff_key] = float(metrics.get(eff_key, np.nan))
    except (TypeError, ValueError):
        bundle[eff_key] = float("nan")
    worst = _compute_worst_case(metrics, subset_map=subset_map)
    bundle[WORST_CASE_KEY] = worst
    return bundle


def _record_metrics(
    records: List[Dict[str, object]],
    bundle: Dict[str, float],
    *,
    task: str,
    term: str,
    method: str,
    run_id: str,
    coverage_target: float,
) -> None:
    if not bundle:
        return
    for metric_name, value in bundle.items():
        if np.isnan(value):
            continue
        records.append(
            {
                "task": task,
                "term": term,
                "method": _method_label(method),
                "metric": COVERAGE_KEY if metric_name == COVERAGE_KEY else (WORST_CASE_KEY if metric_name == WORST_CASE_KEY else metric_name),
                "value": float(value),
                "run_id": run_id,
                "coverage_target": coverage_target,
            }
        )

def _baseline_label_from_key(key: str) -> str:
    if key == "Max_Aggregated":
        return METHOD_BASELINE_AGG
    if key == "Single_Source":
        return METHOD_BASELINE_SINGLE
    if key.startswith("Source_"):
        suffix = key.split("_", 1)[1]
        if suffix.isdigit():
            return f"{METHOD_BASELINE_SOURCE_PREFIX}{int(suffix)}"
        return f"{METHOD_BASELINE_SOURCE_PREFIX}{suffix}"
    return key.replace("_", " ")


def _is_zero_gamma(entry: Dict[str, object], tol: float = 1e-10) -> bool:
    try:
        return abs(float(entry.get("gamma1", 0.0))) <= tol and abs(float(entry.get("gamma2", 0.0))) <= tol and abs(float(entry.get("gamma3", 0.0))) <= tol
    except (TypeError, ValueError):
        return False


def _select_best_gamma(
    entries: Iterable[Dict[str, object]],
    metric_key: str,
    coverage_target: float,
) -> Optional[Dict[str, object]]:
    best_entry: Optional[Dict[str, object]] = None
    best_eff = float("inf")
    best_cov = -float("inf")

    for entry in entries:
        metrics = entry.get("mimic_metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        cov = float(metrics.get(COVERAGE_KEY, np.nan))
        eff = float(metrics.get(metric_key, np.nan))
        if np.isnan(cov) or np.isnan(eff):
            continue
        if cov >= coverage_target:
            if best_entry is None or eff < best_eff - 1e-12 or (abs(eff - best_eff) <= 1e-12 and cov > best_cov):
                best_entry = entry
                best_eff = eff
                best_cov = cov

    if best_entry is not None:
        return best_entry

    for entry in entries:
        metrics = entry.get("mimic_metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        cov = float(metrics.get(COVERAGE_KEY, np.nan))
        eff = float(metrics.get(metric_key, np.nan))
        if np.isnan(cov) or np.isnan(eff):
            continue
        if cov > best_cov + 1e-12 or (abs(cov - best_cov) <= 1e-12 and eff < best_eff - 1e-12):
            best_entry = entry
            best_cov = cov
            best_eff = eff

    return best_entry


def _iter_npz(eval_dir: Path) -> Iterable[Tuple[str, Path]]:
    for task in (CLASSIFICATION, REGRESSION):
        task_dir = eval_dir / task
        if not task_dir.exists():
            continue
        for path in sorted(task_dir.glob("*.npz")):
            yield task, path


def _collect_records(eval_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records: List[Dict[str, object]] = []
    coverage_targets: Dict[str, List[float]] = {CLASSIFICATION: [], REGRESSION: []}

    for task, npz_path in _iter_npz(eval_dir):
        data = np.load(npz_path, allow_pickle=True)
        metadata = data["metadata"].item()
        term = metadata.get("term_name", npz_path.stem)
        alpha = float(metadata.get("alpha", 0.1))
        coverage_target = 1.0 - alpha
        coverage_targets.setdefault(task, []).append(coverage_target)

        trial_index = int(metadata.get("trial_index", 0))
        seed = metadata.get("trial_seed", "na")
        run_id = f"{term}_trial{trial_index:02d}_seed{seed}"

        eval_results = data["evaluation_results"].item()
        baseline_comp = eval_results.get("baseline_comprehensive")
        if isinstance(baseline_comp, np.ndarray) and baseline_comp.size == 1:
            baseline_comp = baseline_comp.item()
        if isinstance(baseline_comp, dict):
            for key, subset_map in baseline_comp.items():
                if not isinstance(subset_map, dict):
                    continue
                overall = subset_map.get("Overall")
                bundle = _bundle_metrics(overall, task, subset_map=subset_map)
                method_label = _baseline_label_from_key(key)
                _record_metrics(records, bundle, task=task, term=term, method=method_label, run_id=run_id, coverage_target=coverage_target)
        else:
            for key in ("Max_Aggregation", "Single_Source"):
                metrics = eval_results.get(key)
                bundle = _bundle_metrics(metrics, task)
                method_label = _baseline_label_from_key(key)
                _record_metrics(records, bundle, task=task, term=term, method=method_label, run_id=run_id, coverage_target=coverage_target)

        gamma_entries = list(data.get("gamma_results", []))
        if not gamma_entries:
            continue

        zero_entry = next((entry for entry in gamma_entries if _is_zero_gamma(entry)), None)
        if zero_entry is not None:
            true_metrics = zero_entry.get("true_metrics")
            bundle = _bundle_metrics(true_metrics, task)
            _record_metrics(
                records,
                bundle,
                task=task,
                term=term,
                method=METHOD_MDCP_NONPEN,
                run_id=run_id,
                coverage_target=coverage_target,
            )

        metric_key = _efficiency_key(task)
        chosen_entry = _select_best_gamma(gamma_entries, metric_key, coverage_target)
        if chosen_entry is not None:
            true_metrics = chosen_entry.get("true_metrics")
            bundle = _bundle_metrics(true_metrics, task)
            _record_metrics(
                records,
                bundle,
                task=task,
                term=term,
                method=METHOD_MDCP_MIMIC,
                run_id=run_id,
                coverage_target=coverage_target,
            )

    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame.from_records(records)
    avg_targets = {task: float(np.mean(vals)) for task, vals in coverage_targets.items() if vals}
    return df, avg_targets


def _plot_term_panels(
    df: pd.DataFrame,
    task: str,
    methods: Sequence[str],
    coverage_target: Optional[float],
    output_path: Path,
) -> None:
    if not methods:
        return
    metric_order = [COVERAGE_KEY, WORST_CASE_KEY, _efficiency_key(task)]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(4.1 * len(metric_order), 3.4), squeeze=False)

    for col_idx, metric in enumerate(metric_order):
        ax = axes[0, col_idx]
        subset = df[(df["metric"] == metric) & df["method"].isin(methods)]
        if subset.empty:
            ax.axis("off")
            continue
        method_order = [m for m in methods if m in subset["method"].unique()]
        _draw_bars_with_points(ax, subset, method_order, coverage_target, metric)
        if metric == COVERAGE_KEY:
            ax.set_ylabel("Coverage", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        elif metric == WORST_CASE_KEY:
            ax.set_ylabel("Worst-case coverage", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        else:
            ax.set_ylabel(
                "Avg width" if task == REGRESSION else "Avg set size",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )
        ax.set_xlabel("Method", fontsize=LABEL_FONT_SIZE, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=_color_for_method(method), edgecolor="black", linewidth=0.6, label=method)
        for method in methods
        if method in df["method"].unique()
    ]
    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            ncol=max(1, len(handles)),
            columnspacing=1.2,
            handlelength=1.2,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_task_summary_lines(
    ax: plt.Axes,
    subset: pd.DataFrame,
    method_order: Sequence[str],
    coverage_target: Optional[float],
    metric: str,
) -> None:
    terms_series = subset.get("term")
    if terms_series is None or terms_series.dropna().empty:
        ax.axis("off")
        return
    term_order = _ordered_terms(terms_series.dropna().unique())
    if not term_order:
        ax.axis("off")
        return
    x_positions = np.arange(len(term_order), dtype=float)

    values = subset["value"].to_numpy(dtype=float)
    if metric in {COVERAGE_KEY, WORST_CASE_KEY}:
        bottom, top, ticks, labels = _coverage_axis_limits(values, coverage_target)
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
                zorder=2,
            )
    else:
        clean = values[~np.isnan(values)]
        if clean.size:
            y_min = float(clean.min())
            y_max = float(clean.max())
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                y_min, y_max = 0.0, 1.0
            if abs(y_max - y_min) < 1e-6:
                span = max(abs(y_min), 1.0) * 0.05
                y_min -= span
                y_max += span
            margin = max((y_max - y_min) * 0.08, 1e-3)
            bottom = max(0.0, y_min - margin)
            top = y_max + margin
        else:
            bottom, top = 0.0, 1.0
        ax.set_ylim(bottom, top)

    for method in method_order:
        method_subset = subset[subset["method"] == method]
        if method_subset.empty:
            continue

        means = method_subset.groupby("term", dropna=False)["value"].mean()
        line_vals: List[float] = []
        for term in term_order:
            raw_val = float(means.get(term, np.nan))
            if np.isnan(raw_val):
                line_vals.append(np.nan)
            else:
                line_vals.append(_transform_metric_value(metric, raw_val))

        style = _style_for_method(method)
        color = style["color"]
        marker_shape = str(style.get("marker", "o"))
        marker_size = float(style.get("markersize", 4.6))
        ax.plot(
            x_positions,
            np.array(line_vals, dtype=float),
            color=color,
            linewidth=float(style.get("linewidth", SUMMARY_LINE_WIDTH)),
            linestyle=str(style.get("linestyle", "-")),
            alpha=float(style.get("alpha", 0.9)),
            marker=marker_shape,
            markersize=marker_size,
            markerfacecolor=style.get("marker_facecolor", color),
            markeredgecolor=style.get("marker_edgecolor", "black"),
            markeredgewidth=float(style.get("marker_edgewidth", 0.5)),
            zorder=3,
        )

        rgba = _with_alpha(color, float(style.get("scatter_alpha", 0.35)))
        for term_idx, term in enumerate(term_order):
            run_values = method_subset[method_subset["term"] == term]["value"].to_numpy(dtype=float)
            run_values = run_values[~np.isnan(run_values)]
            if run_values.size == 0:
                continue
            offsets = _point_offsets(run_values.size)
            transformed = np.array([_transform_metric_value(metric, float(val)) for val in run_values], dtype=float)
            ax.scatter(
                np.full(run_values.shape, x_positions[term_idx]) + offsets,
                transformed,
                s=SUMMARY_POINT_SIZE,
                color=[rgba],
                edgecolors="none",
                marker=marker_shape,
                zorder=2,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(term_order, rotation=25, ha="right", fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    if term_order:
        ax.set_xlim(-0.5, len(term_order) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _plot_summary(
    df: pd.DataFrame,
    task: str,
    methods: Sequence[str],
    coverage_target: Optional[float],
    output_path: Path,
) -> None:
    metric_order = [COVERAGE_KEY, WORST_CASE_KEY, CLASS_EFF_KEY] if task == CLASSIFICATION else [COVERAGE_KEY, WORST_CASE_KEY, REG_EFF_KEY]
    if df.empty:
        raise RuntimeError(f"No records available for task '{task}'.")

    fig, axes = plt.subplots(1, len(metric_order), figsize=(4.0 * len(metric_order), 3.4), squeeze=False)
    ax_row = axes[0]

    for col_idx, metric in enumerate(metric_order):
        ax = ax_row[col_idx]
        subset = df[(df["metric"] == metric) & df["method"].isin(methods)]
        if subset.empty:
            ax.axis("off")
            continue
        method_order = [m for m in methods if m in subset["method"].unique()]
        _plot_task_summary_lines(ax, subset, method_order, coverage_target, metric)
        if metric == COVERAGE_KEY:
            ax.set_title("Overall coverage", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        elif metric == WORST_CASE_KEY:
            ax.set_title("Worst-case coverage", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        else:
            ax.set_title("Avg set size" if task == CLASSIFICATION else "Avg interval width", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        # # y-label, no need since classification/regression is separated
        # if col_idx == 0:
        #     ylabel = "Classification" if task == CLASSIFICATION else "Regression"
        #     ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold")
        # else:
        #     ax.set_ylabel("")
        ax.set_xlabel("Nonlinear term", fontsize=LABEL_FONT_SIZE, fontweight="bold")

    handles: List[Line2D] = []
    task_methods = df["method"].unique()
    for method in methods:
        if method not in task_methods:
            continue
        style = _style_for_method(method)
        marker_shape = str(style.get("marker", "o"))
        marker_size = float(style.get("markersize", 5.2))
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=float(style.get("linewidth", SUMMARY_LINE_WIDTH)),
                linestyle=str(style.get("linestyle", "-")),
                alpha=float(style.get("alpha", 0.9)),
                marker=marker_shape,
                markersize=marker_size,
                markerfacecolor=style.get("marker_facecolor", style["color"]),
                markeredgecolor=style.get("marker_edgecolor", "black"),
                markeredgewidth=float(style.get("marker_edgewidth", 0.5)),
                label=method,
            )
        )

    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            ncol=max(1, len(handles)),
            columnspacing=1.1,
            handlelength=1.35,
            borderaxespad=0.2,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )

    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.09, right=0.995, hspace=0.28, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready MDCP nonlinear evaluation plots")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "nonlinear",
        help="Directory containing nonlinear evaluation NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "final_paper_ready",
        help="Directory where figures will be saved",
    )
    args = parser.parse_args()

    if not args.eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {args.eval_dir}")

    df, coverage_targets = _collect_records(args.eval_dir)
    if df.empty:
        raise RuntimeError("No evaluation records found for nonlinear plots.")

    df["method"] = df["method"].apply(_method_label)
    term_lower = df["term"].astype(str).str.lower()
    df = df[~((df["task"] == REGRESSION) & (term_lower == "quadratic"))].copy()

    nonlinear_root = args.output_dir / "nonlinear"
    nonlinear_root.mkdir(parents=True, exist_ok=True)

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        task_dir = nonlinear_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        term_list = _ordered_terms(task_df["term"].dropna().unique())
        for term in term_list:
            term_df = task_df[task_df["term"] == term]
            coverage_target = float(term_df["coverage_target"].dropna().mean()) if not term_df["coverage_target"].dropna().empty else None

            present_methods = set(term_df["method"].unique())
            baseline_sources = sorted(
                (m for m in present_methods if m.startswith(METHOD_BASELINE_SOURCE_PREFIX)),
                key=lambda name: int(name.replace(METHOD_BASELINE_SOURCE_PREFIX, "")) if name.replace(METHOD_BASELINE_SOURCE_PREFIX, "").isdigit() else 0,
            )

            vanilla_methods: List[str] = []
            if METHOD_BASELINE_AGG in present_methods:
                vanilla_methods.append(METHOD_BASELINE_AGG)
            vanilla_methods.extend(baseline_sources)
            if METHOD_BASELINE_SINGLE in present_methods:
                vanilla_methods.append(METHOD_BASELINE_SINGLE)
            if METHOD_MDCP_NONPEN in present_methods:
                vanilla_methods.append(METHOD_MDCP_NONPEN)
            vanilla_methods = _ordered_methods(vanilla_methods)

            vanilla_path = task_dir / f"{term}_overall_vanilla.pdf"
            _plot_term_panels(term_df, task, vanilla_methods, coverage_target, vanilla_path)
            print(f"Saved {task} {term} vanilla figure: {vanilla_path}")

            tuned_methods = list(vanilla_methods)
            tuned_path: Optional[Path] = None
            if METHOD_MDCP_MIMIC in present_methods:
                if METHOD_MDCP_MIMIC not in tuned_methods:
                    tuned_methods.append(METHOD_MDCP_MIMIC)
                tuned_methods = _ordered_methods(tuned_methods)
                tuned_path = task_dir / f"{term}_overall_tuned.pdf"
                _plot_term_panels(term_df, task, tuned_methods, coverage_target, tuned_path)
                print(f"Saved {task} {term} tuned figure: {tuned_path}")
            else:
                print(f"No mimic-selected MDCP available for {task} {term}; tuned figure skipped.")

            term_summary = (
                term_df.groupby(["task", "term", "method", "metric"], dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
            )
            term_summary_path = task_dir / f"{term}_overall_summary.csv"
            term_summary.to_csv(term_summary_path, index=False)
            print(f"Saved {task} {term} summary: {term_summary_path}")

    present_methods_all = set(df["method"].unique())
    baseline_sources_all = sorted(
        (m for m in present_methods_all if m.startswith(METHOD_BASELINE_SOURCE_PREFIX)),
        key=lambda name: int(name.replace(METHOD_BASELINE_SOURCE_PREFIX, "")) if name.replace(METHOD_BASELINE_SOURCE_PREFIX, "").isdigit() else 0,
    )

    vanilla_all: List[str] = []
    if METHOD_BASELINE_AGG in present_methods_all:
        vanilla_all.append(METHOD_BASELINE_AGG)
    vanilla_all.extend(baseline_sources_all)
    if METHOD_BASELINE_SINGLE in present_methods_all:
        vanilla_all.append(METHOD_BASELINE_SINGLE)
    if METHOD_MDCP_NONPEN in present_methods_all:
        vanilla_all.append(METHOD_MDCP_NONPEN)
    vanilla_all = _ordered_methods(vanilla_all)

    tuned_all = list(vanilla_all)
    if METHOD_MDCP_MIMIC in present_methods_all and METHOD_MDCP_MIMIC not in tuned_all:
        tuned_all.append(METHOD_MDCP_MIMIC)
    tuned_all = _ordered_methods(tuned_all)

    summary_dir = nonlinear_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        coverage_target = coverage_targets.get(task)

        vanilla_task = [m for m in vanilla_all if m in task_df["method"].unique()]
        if vanilla_task:
            summary_vanilla_path = summary_dir / f"{task}_overall_vanilla.pdf"
            _plot_summary(task_df, task, vanilla_task, coverage_target, summary_vanilla_path)
            print(f"Saved {task} overall vanilla summary figure: {summary_vanilla_path}")
        else:
            print(f"No methods available for {task} vanilla summary figure; skipped.")

        tuned_task = [m for m in tuned_all if m in task_df["method"].unique()]
        if METHOD_MDCP_MIMIC in tuned_task and tuned_task:
            summary_tuned_path = summary_dir / f"{task}_overall_tuned.pdf"
            _plot_summary(task_df, task, tuned_task, coverage_target, summary_tuned_path)
            print(f"Saved {task} overall tuned summary figure: {summary_tuned_path}")
        else:
            print(f"No mimic-selected MDCP observed for {task}; tuned summary figure skipped.")

    summary_csv_path = summary_dir / "overall_summary.csv"
    summary_df = (
        df.groupby(["task", "term", "method", "metric"], dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    )
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved overall summary table: {summary_csv_path}")


if __name__ == "__main__":
    main()

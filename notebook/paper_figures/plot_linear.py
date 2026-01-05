"""Generate publication-ready plots for the MDCP linear evaluation.

This script focuses on two visual comparisons using the evaluation
artifacts stored under ``eval_out/linear``:

1. Baselines vs. non-penalized MDCP.
2. Baselines vs. non-penalized MDCP vs. trial-wise penalized MDCP where
    the penalty is chosen using mimic test metrics and then evaluated on
    the true test set.

The resulting figures follow the specifications outlined in ``AGENTS.md``
by using bar + dot plots, omitting titles, and emitting PDF files under
``eval_out/final_paper_ready``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ensure numpy can unpickle PosixPath objects serialized on Unix systems
import pathlib as _pathlib

try:  # pragma: no cover - platform dependent guard
    if hasattr(_pathlib, "PosixPath") and hasattr(_pathlib, "WindowsPath"):
        _pathlib.PosixPath = _pathlib.WindowsPath  # type: ignore[attr-defined]
except (AttributeError, TypeError):  # pragma: no cover - safety fallback
    pass

sns.set_theme(style="whitegrid")

LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11

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
WORST_CASE_COVERAGE_KEY = "worst_case_coverage"
CLASSIFICATION_EFF_KEY = "avg_set_size"
REGRESSION_EFF_KEY = "avg_width"

METHOD_BASELINE_MAX = "baseline_max"
METHOD_BASELINE_SOURCE_PREFIX = "baseline_source_"
METHOD_MDCP_NONPEN = "mdcp_nonpen"
METHOD_MDCP_MIMIC = "mdcp_mimic_selected"

METHOD_LABELS: Dict[str, str] = {
    METHOD_BASELINE_MAX: "Baseline agg",
    METHOD_MDCP_NONPEN: "MDCP",
    METHOD_MDCP_MIMIC: "MDCP tuned",
}

METHOD_COLORS: Dict[str, str] = {
    METHOD_BASELINE_MAX: "#5E6674",
    METHOD_MDCP_NONPEN: "#5C8FD4",
    METHOD_MDCP_MIMIC: "#81ABDE",
}

BASELINE_SOURCE_COLORS = [
    "#6B7381",
    "#78808E",
    "#858D9A",
    "#929AA7",
    "#9FA7B3",
    "#ACB4BF",
    "#B9C1CB",
]

_COLOR_CACHE: Dict[str, str] = {}

MIN_COVERAGE_Y = 0.7
JITTER_MAX = 0.18
COVERAGE_COMPRESS_FACTOR = 0.25
COVERAGE_PADDING = 0.01

COVERAGE_METRICS = {COVERAGE_KEY, WORST_CASE_COVERAGE_KEY}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return float("nan")
        if arr.size == 1:
            try:
                return float(arr.item())
            except Exception:
                return float("nan")
        return float("nan")


def _extract_worst_case(metrics: Optional[Dict[str, object]]) -> float:
    if not isinstance(metrics, dict):
        return float("nan")
    if WORST_CASE_COVERAGE_KEY in metrics:
        value = _to_float(metrics.get(WORST_CASE_COVERAGE_KEY))
        if not np.isnan(value):
            return value
    indiv = metrics.get("individual_coverage")
    if isinstance(indiv, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(indiv, dtype=float)
        except Exception:
            return float("nan")
        if arr.size:
            arr = arr[~np.isnan(arr)]
            if arr.size:
                return float(arr.min())
    return float("nan")


def _register_source_method(method: str) -> None:
    if method in METHOD_LABELS:
        return
    try:
        idx = int(method.replace(METHOD_BASELINE_SOURCE_PREFIX, ""))
    except ValueError:
        idx = 0
    METHOD_LABELS[method] = f"Baseline src {idx}"


def _color_for_method(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    if method in _COLOR_CACHE:
        return _COLOR_CACHE[method]
    color: Optional[str]
    color = None
    if method.startswith(METHOD_BASELINE_SOURCE_PREFIX):
        try:
            idx = int(method.replace(METHOD_BASELINE_SOURCE_PREFIX, ""))
        except ValueError:
            idx = 0
        color = BASELINE_SOURCE_COLORS[idx % len(BASELINE_SOURCE_COLORS)]
    if color is None:
        palette = sns.color_palette("tab10")
        idx = len(_COLOR_CACHE) % len(palette)
        rgb = palette[idx]
        color = mcolors.to_hex(rgb)
    _COLOR_CACHE[method] = color
    return color


def _efficiency_key(task: str) -> str:
    return CLASSIFICATION_EFF_KEY if task == CLASSIFICATION else REGRESSION_EFF_KEY


def _transform_metric_value(metric: str, value: float) -> float:
    if metric not in COVERAGE_METRICS or np.isnan(value):
        return value
    if value >= MIN_COVERAGE_Y:
        return value
    return MIN_COVERAGE_Y - (MIN_COVERAGE_Y - value) * COVERAGE_COMPRESS_FACTOR

def _coverage_axis_limits(
    values: np.ndarray, coverage_target: Optional[float], metric: str
) -> tuple[float, float, List[float], List[str]]:
    valid = values[~np.isnan(values)]
    ticks: List[float] = []
    labels: List[str] = []

    if valid.size == 0:
        bottom = MIN_COVERAGE_Y - COVERAGE_PADDING
        top = MIN_COVERAGE_Y + COVERAGE_PADDING
        ticks = [MIN_COVERAGE_Y]
        labels = [f"{MIN_COVERAGE_Y:.2f}"]
        return bottom, top, ticks, labels

    min_val = float(valid.min())
    max_val = float(valid.max())

    bottom = _transform_metric_value(metric, min_val) if min_val < MIN_COVERAGE_Y else MIN_COVERAGE_Y
    bottom = min(bottom, MIN_COVERAGE_Y - COVERAGE_PADDING)

    top_candidates = [MIN_COVERAGE_Y, _transform_metric_value(metric, max_val)]
    if coverage_target is not None:
        top_candidates.append(_transform_metric_value(metric, float(coverage_target)))
    top = max(top_candidates)
    top = max(top, MIN_COVERAGE_Y + COVERAGE_PADDING)

    ordered_actuals: List[float] = []
    if min_val < MIN_COVERAGE_Y:
        ordered_actuals.append(min_val)
    ordered_actuals.append(MIN_COVERAGE_Y)
    if coverage_target is not None:
        ordered_actuals.append(float(coverage_target))
    ordered_actuals.append(max_val)

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

_GAMMA_VALUE_PATTERN = re.compile(r"g\d+_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _is_zero_gamma(name: Optional[str]) -> bool:
    if not name:
        return False
    values = [float(match) for match in _GAMMA_VALUE_PATTERN.findall(name)]
    if not values:
        tokens = [tok for tok in name.split("_") if tok]
        raw_values: List[float] = []
        for token in tokens:
            try:
                raw_values.append(float(token))
            except ValueError:
                continue
        values = raw_values
    if not values:
        return False
    return all(abs(val) < 1e-10 for val in values)

def _select_best_gamma(entries: Iterable[Dict[str, object]], metric_key: str, coverage_target: float) -> Optional[Dict[str, object]]:
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

def _record_metrics(
    records: List[Dict[str, object]],
    task: str,
    method: str,
    run_id: str,
    metrics: Dict[str, object],
    coverage_target: float,
    gamma_name: Optional[str] = None,
) -> None:
    eff_key = _efficiency_key(task)
    coverage = _to_float(metrics.get(COVERAGE_KEY, np.nan)) if isinstance(metrics, dict) else float("nan")
    worst_case = _extract_worst_case(metrics)
    efficiency = _to_float(metrics.get(eff_key, np.nan)) if isinstance(metrics, dict) else float("nan")

    if not np.isnan(coverage):
        records.append(
            {
                "task": task,
                "method": method,
                "metric": COVERAGE_KEY,
                "value": float(coverage),
                "run_id": run_id,
                "gamma": gamma_name,
                "coverage_target": coverage_target,
            }
        )
    if not np.isnan(worst_case):
        records.append(
            {
                "task": task,
                "method": method,
                "metric": WORST_CASE_COVERAGE_KEY,
                "value": float(worst_case),
                "run_id": run_id,
                "gamma": gamma_name,
                "coverage_target": coverage_target,
            }
        )
    if not np.isnan(efficiency):
        records.append(
            {
                "task": task,
                "method": method,
                "metric": eff_key,
                "value": float(efficiency),
                "run_id": run_id,
                "gamma": gamma_name,
                "coverage_target": coverage_target,
            }
        )

def _process_npz(path: Path, task: str, records: List[Dict[str, object]], targets: Dict[str, List[float]]) -> None:
    data = np.load(path, allow_pickle=True)
    metadata = data["metadata"].item()
    run_id = f"trial{metadata.get('trial_index', 0):03d}_seed{metadata.get('trial_seed', 'na')}"
    coverage_target = 1.0 - float(metadata.get("alpha", 0.1))
    targets.setdefault(task, []).append(coverage_target)

    results = data["results"].item()
    baselines = {
        METHOD_BASELINE_MAX: results.get("Max Aggregation"),
    }
    for method, metrics in baselines.items():
        if isinstance(metrics, dict):
            _record_metrics(records, task, method, run_id, metrics, coverage_target)

    baseline_comp = data.get("baseline_comprehensive")
    if isinstance(baseline_comp, np.ndarray) and baseline_comp.size == 1:
        baseline_comp = baseline_comp.item()
    if isinstance(baseline_comp, dict):
        eff_key = _efficiency_key(task)
        for key, subset_map in baseline_comp.items():
            if not key.startswith("Source_"):
                continue
            try:
                src_idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                src_idx = 0
            method_name = f"{METHOD_BASELINE_SOURCE_PREFIX}{src_idx}"
            _register_source_method(method_name)
            overall_metrics = subset_map.get("Overall") if isinstance(subset_map, dict) else None
            if not isinstance(overall_metrics, dict):
                continue
            cov_val = overall_metrics.get(COVERAGE_KEY)
            eff_val = overall_metrics.get(eff_key)
            metrics_payload = dict(overall_metrics)
            metrics_payload[COVERAGE_KEY] = _to_float(cov_val)
            metrics_payload[eff_key] = _to_float(eff_val)
            _record_metrics(records, task, method_name, run_id, metrics_payload, coverage_target)

    gamma_entries = list(data.get("gamma_results", []))
    if not gamma_entries:
        return

    zero_entry = None
    for entry in gamma_entries:
        if isinstance(entry, dict) and _is_zero_gamma(entry.get("gamma_name")):
            zero_entry = entry
            break
    if zero_entry is not None:
        true_metrics = zero_entry.get("true_metrics")
        if isinstance(true_metrics, dict):
            _record_metrics(records, task, METHOD_MDCP_NONPEN, run_id, true_metrics, coverage_target, zero_entry.get("gamma_name"))

    metric_key = _efficiency_key(task)
    chosen_entry = _select_best_gamma(gamma_entries, metric_key, coverage_target)
    if chosen_entry is None:
        return
    true_metrics = chosen_entry.get("true_metrics")
    if not isinstance(true_metrics, dict):
        return
    _record_metrics(
        records,
        task,
        METHOD_MDCP_MIMIC,
        run_id,
        true_metrics,
        coverage_target,
        chosen_entry.get("gamma_name"),
    )

def _collect_records(eval_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records: List[Dict[str, object]] = []
    targets: Dict[str, List[float]] = {}

    for task in (CLASSIFICATION, REGRESSION):
        task_dir = eval_dir / task
        if not task_dir.exists():
            continue
        for path in sorted(task_dir.glob("*.npz")):
            _process_npz(path, task, records, targets)

    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame.from_records(records)
    target_map = {task: float(np.mean(values)) for task, values in targets.items() if values}
    return df, target_map

def _ordered_methods(methods: Iterable[str]) -> List[str]:
    methods_set = set(methods)
    ordered: List[str] = []
    if METHOD_BASELINE_MAX in methods_set:
        ordered.append(METHOD_BASELINE_MAX)
    source_methods = sorted(
        (m for m in methods_set if m.startswith(METHOD_BASELINE_SOURCE_PREFIX)),
        key=lambda name: int(name.replace(METHOD_BASELINE_SOURCE_PREFIX, "")) if name.replace(METHOD_BASELINE_SOURCE_PREFIX, "").isdigit() else 0,
    )
    ordered.extend(source_methods)
    if METHOD_MDCP_NONPEN in methods_set:
        ordered.append(METHOD_MDCP_NONPEN)
    if METHOD_MDCP_MIMIC in methods_set:
        ordered.append(METHOD_MDCP_MIMIC)
    return ordered

def _draw_bars_with_points(ax: plt.Axes, df: pd.DataFrame, method_order: List[str], coverage_target: Optional[float], metric: str) -> None:
    means = df.groupby("method")["value"].mean()
    x_positions = range(len(method_order))
    for idx, method in enumerate(method_order):
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        mean_val = means.get(method, np.nan)
        color = _color_for_method(method)
        display_mean = _transform_metric_value(metric, float(mean_val)) if not np.isnan(mean_val) else np.nan
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
            display_values = np.array([_transform_metric_value(metric, val) for val in values])
            offsets = np.linspace(-JITTER_MAX, JITTER_MAX, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.shape, idx, dtype=float) + offsets,
                display_values,
                color=color,
                s=10,
                zorder=3,
                linewidths=0.0,
                clip_on=False,
            )

    ax.set_xticks(list(x_positions))
    labels = [METHOD_LABELS.get(method, method) for method in method_order]
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    if method_order:
        ax.set_xlim(-0.5 - JITTER_MAX, len(method_order) - 0.5 + JITTER_MAX)
    if metric in COVERAGE_METRICS:
        bottom, top, ticks, labels = _coverage_axis_limits(
            df["value"].to_numpy(dtype=float), coverage_target, metric
        )
        ax.set_ylim(bottom, top)
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        if coverage_target is not None:
            ax.axhline(
                _transform_metric_value(metric, float(coverage_target)),
                color="#666666",
                linestyle="--",
                linewidth=1.0,
            )
    else:
        ax.set_ylim(bottom=0.0)


def _plot_summary(
    df: pd.DataFrame,
    task: str,
    methods: List[str],
    coverage_target: Optional[float],
    output_path: Path,
) -> None:
    if df.empty or not methods:
        return

    metrics = [COVERAGE_KEY, WORST_CASE_COVERAGE_KEY, _efficiency_key(task)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.4), squeeze=False)

    for col_idx, metric in enumerate(metrics):
        ax = axes[0, col_idx]
        subset = df[(df["metric"] == metric) & df["method"].isin(methods)]
        if subset.empty:
            ax.axis("off")
            continue
        method_order = [m for m in methods if m in subset["method"].unique()]
        _draw_bars_with_points(ax, subset, method_order, coverage_target, metric)
        if metric in COVERAGE_METRICS:
            ylabel = "Overall coverage" if metric == COVERAGE_KEY else "Worst-case coverage"
        else:
            ylabel = "Avg set size" if task == CLASSIFICATION else "Avg interval width"
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold")
        ax.set_xlabel("Method", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_methods = [m for m in methods if m in df["method"].unique()]
    if legend_methods:
        handles = [
            mpatches.Patch(facecolor=_color_for_method(m), edgecolor="black", label=METHOD_LABELS.get(m, m))
            for m in legend_methods
        ]
        fig.legend(
            handles,
            [METHOD_LABELS.get(m, m) for m in legend_methods],
            loc="upper center",
            ncol=max(1, len(handles)),
            frameon=False,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )
    # fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.tight_layout(rect=[0, -0.02, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready MDCP linear evaluation plots")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "linear",
        help="Directory containing classification/regression NPZ files",
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

    df, target_map = _collect_records(args.eval_dir)
    if df.empty:
        raise RuntimeError("No evaluation records were collected.")

    tasks_available = [task for task in (CLASSIFICATION, REGRESSION) if task in df["task"].unique()]

    for task in tasks_available:
        task_df = df[df["task"] == task]
        present_methods = set(task_df["method"].unique())
        if not present_methods:
            continue

        coverage_target = target_map.get(task)
        if coverage_target is None or np.isnan(coverage_target):
            coverage_target = float(task_df["coverage_target"].dropna().mean()) if "coverage_target" in task_df else None

        methods_nonpen = _ordered_methods(m for m in present_methods if m != METHOD_MDCP_MIMIC)
        if methods_nonpen:
            nonpen_path = args.output_dir / f"linear_{task}_vanilla.pdf"
            _plot_summary(task_df, task, methods_nonpen, coverage_target, nonpen_path)
            print(f"Saved {task} vanilla comparison figure: {nonpen_path}")
        else:
            print(f"No methods available for {task} vanilla comparison; skipped.")

        if METHOD_MDCP_MIMIC in present_methods:
            methods_tuned = _ordered_methods(present_methods)
        else:
            methods_tuned = methods_nonpen

        if methods_tuned:
            mimic_path = args.output_dir / f"linear_{task}_tuned.pdf"
            _plot_summary(task_df, task, methods_tuned, coverage_target, mimic_path)
            print(f"Saved {task} tuned comparison figure: {mimic_path}")
        else:
            print(f"No methods available for {task} tuned comparison; skipped.")

    summary_path = args.output_dir / "linear_overall_summary.csv"
    summary = (
        df.groupby(["task", "method", "metric"], dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved overall summary statistics: {summary_path}")

if __name__ == "__main__":
    main()

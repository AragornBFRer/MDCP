"""Paper-ready plots for covariate shift experiments (delta_x sweep).

This script consumes covariate-shift evaluation artifacts produced by
`notebook/eval_cov_shift.py` (expects an `eval-root` that contains a
`summaries/` subfolder).

It generates two sets of figures:

- Vanilla figures (exclude tuned MDCP)
- Tuned figures (include mimic-selected MDCP)

Outputs are stored beneath `eval_out/final_paper_ready/cov_shift/` by
default.

MDCP tuning logic (important):
For each trial and delta_x, we select the MDCP penalty (gamma) by
optimizing mimic-test performance (coverage constraint, then efficiency),
then plot the corresponding true-test metrics. This matches
`archive/mimic_calibtest_logic.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

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

METRICS = ["overall_coverage", "worst_case_coverage", "efficiency"]
METRIC_LABELS = {
    "overall_coverage": "Overall coverage",
    "worst_case_coverage": "Worst-case coverage",
    "efficiency": {
        CLASSIFICATION: "Avg set size",
        REGRESSION: "Avg interval width",
    },
}

COVERAGE_MIN = 0.7
COVERAGE_PADDING = 0.015

EFF_KEY = {CLASSIFICATION: "avg_set_size", REGRESSION: "avg_width"}

# Paper plotting requirement: do not plot beyond this x-value.
MAX_DELTA_X = 4.5

# Method labels required by AGENTS.md
METHOD_BASELINE_AGG = "Baseline agg"
METHOD_BASELINE_SRC_PREFIX = "Baseline src "
METHOD_MDCP = "MDCP"
METHOD_MDCP_TUNED = "MDCP tuned"

PALETTE: Dict[str, str] = {
    METHOD_BASELINE_AGG: "#5E6674",
    f"{METHOD_BASELINE_SRC_PREFIX}0": "#8C94A0",
    f"{METHOD_BASELINE_SRC_PREFIX}1": "#9EA6B1",
    f"{METHOD_BASELINE_SRC_PREFIX}2": "#AFB7C1",
    METHOD_MDCP: "#5C8FD4",
    METHOD_MDCP_TUNED: "#81ABDE",
}

LINE_STYLES: Dict[str, Dict[str, object]] = {
    METHOD_BASELINE_AGG: {
        "linestyle": "-",
        "linewidth": 1.6,
        "alpha": 0.85,
        "marker": "o",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.35,
    },
    f"{METHOD_BASELINE_SRC_PREFIX}0": {
        "linestyle": ":",
        "linewidth": 1.1,
        "alpha": 0.7,
        "marker": "P",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.3,
    },
    f"{METHOD_BASELINE_SRC_PREFIX}1": {
        "linestyle": "-.",
        "linewidth": 1.1,
        "alpha": 0.7,
        "marker": "s",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.3,
    },
    f"{METHOD_BASELINE_SRC_PREFIX}2": {
        "linestyle": "--",
        "linewidth": 1.1,
        "alpha": 0.7,
        "marker": "^",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.3,
    },
    METHOD_MDCP: {
        "linestyle": "-",
        "linewidth": 2.3,
        "alpha": 0.95,
        "marker": "o",
        "markersize": 5,
        "markeredgecolor": "#1D3B5B",
        "markeredgewidth": 0.45,
    },
    METHOD_MDCP_TUNED: {
        "linestyle": "-",
        "linewidth": 2.3,
        "alpha": 0.95,
        "marker": "D",
        "markersize": 4,
        "markeredgecolor": "#1D3B5B",
        "markeredgewidth": 0.45,
    },
}


def _baseline_label(raw_name: str) -> Optional[str]:
    if raw_name == "Max_Aggregated":
        return METHOD_BASELINE_AGG
    if raw_name.startswith("Source_"):
        suffix = raw_name.split("_", 1)[1]
        try:
            suffix = str(int(suffix))
        except ValueError:
            pass
        return f"{METHOD_BASELINE_SRC_PREFIX}{suffix}"
    return None


def _append_metric_records(
    records: List[Dict[str, object]],
    base_info: Dict[str, object],
    method: str,
    variant: str,
    metrics: Dict[str, object],
    task: str,
    run_id: str,
) -> None:
    if not isinstance(metrics, dict):
        return

    overall_cov = float(metrics.get("coverage", np.nan))
    eff_value = float(metrics.get(EFF_KEY[task], np.nan))

    worst_cov = np.nan
    indiv_cov = metrics.get("individual_coverage")
    if indiv_cov is not None:
        try:
            cov_arr = np.asarray(indiv_cov, dtype=float)
        except Exception:
            cov_arr = None
        if cov_arr is not None:
            cov_arr = cov_arr[np.isfinite(cov_arr)]
            if cov_arr.size:
                worst_cov = float(cov_arr.min())

    bundles = {
        "overall_coverage": overall_cov,
        "worst_case_coverage": worst_cov,
        "efficiency": eff_value,
    }

    for metric, value in bundles.items():
        if value is None or np.isnan(float(value)):
            continue
        records.append(
            {
                **base_info,
                "task": task,
                "method": method,
                "metric": metric,
                "value": float(value),
                "variant": variant,
                "run_id": run_id,
            }
        )


def _choose_gamma_entry(
    entries: Iterable[Dict[str, object]],
    task: str,
    coverage_target: float,
) -> Optional[Dict[str, object]]:
    """Pick best gamma using mimic metrics (coverage constraint then efficiency)."""

    eff_key = EFF_KEY[task]

    best_feasible: Optional[Dict[str, object]] = None
    best_feasible_eff = float("inf")

    best_fallback: Optional[Dict[str, object]] = None
    best_fallback_cov = -float("inf")
    best_fallback_eff = float("inf")

    for entry in entries:
        mimic_metrics = entry.get("mimic_metrics") or entry.get("mimic")
        if not isinstance(mimic_metrics, dict):
            continue
        try:
            coverage = float(mimic_metrics.get("coverage", np.nan))
            efficiency = float(mimic_metrics.get(eff_key, np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(coverage) or not np.isfinite(efficiency):
            continue

        if coverage >= coverage_target and efficiency < best_feasible_eff:
            best_feasible = entry
            best_feasible_eff = efficiency

        if coverage > best_fallback_cov or (
            np.isclose(coverage, best_fallback_cov) and efficiency < best_fallback_eff
        ):
            best_fallback = entry
            best_fallback_cov = coverage
            best_fallback_eff = efficiency

    return best_feasible or best_fallback


def _find_gamma_zero_true_metrics(entries: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
    for entry in entries:
        try:
            gamma1 = float(entry.get("gamma1", np.nan))
        except (TypeError, ValueError):
            continue
        if abs(gamma1) < 1e-12:
            tm = entry.get("true_metrics") or entry.get("true")
            if isinstance(tm, dict):
                return tm
    return None


def _collect_records_from_summaries(eval_root: Path) -> pd.DataFrame:
    summaries_dir = eval_root / "summaries"
    if not summaries_dir.exists():
        raise FileNotFoundError(f"Missing summaries directory: {summaries_dir}")

    summary_paths = sorted(summaries_dir.glob("trial_*_summary.npz"))
    if not summary_paths:
        raise FileNotFoundError(f"No trial summaries found under: {summaries_dir}")

    records: List[Dict[str, object]] = []

    for path in summary_paths:
        try:
            with np.load(path, allow_pickle=True) as payload:
                trial_summary = payload["trial_summary"]
                trial_obj = trial_summary[0]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load {path}: {exc}")
            continue

        alpha = float(trial_obj.get("alpha", 0.1))
        coverage_target = 1.0 - alpha
        seed = trial_obj.get("trial_seed", "na")

        per_delta = trial_obj.get("per_delta_x", [])
        for delta_entry in per_delta:
            delta_x = float(delta_entry.get("delta_x", np.nan))
            if not np.isfinite(delta_x) or delta_x > MAX_DELTA_X:
                continue
            base_info = {
                "delta_x": delta_x,
                "coverage_target": coverage_target,
            }

            for task in (CLASSIFICATION, REGRESSION):
                payload_task = delta_entry.get(task)
                if not isinstance(payload_task, dict):
                    continue

                gamma_entries = payload_task.get("gamma_results") or []
                if not isinstance(gamma_entries, list):
                    try:
                        gamma_entries = list(gamma_entries)
                    except Exception:
                        gamma_entries = []

                run_id = f"{task}_seed{seed}_dx{delta_x:g}"

                # Baselines from baseline_comprehensive
                baseline_comp = payload_task.get("baseline_comprehensive")
                if isinstance(baseline_comp, np.ndarray) and baseline_comp.size == 1:
                    baseline_comp = baseline_comp.item()
                if isinstance(baseline_comp, dict):
                    for raw_name, subset_map in baseline_comp.items():
                        label = _baseline_label(str(raw_name))
                        if label is None:
                            continue
                        overall = subset_map.get("Overall") if isinstance(subset_map, dict) else None
                        if isinstance(overall, dict):
                            _append_metric_records(
                                records,
                                base_info,
                                label,
                                "baseline",
                                overall,
                                task,
                                run_id,
                            )

                # Vanilla MDCP = gamma=0 true metrics
                gamma_zero_true = _find_gamma_zero_true_metrics(gamma_entries)
                if gamma_zero_true is not None:
                    _append_metric_records(
                        records,
                        base_info,
                        METHOD_MDCP,
                        "mdcp",
                        gamma_zero_true,
                        task,
                        run_id,
                    )

                # Tuned MDCP = mimic-selected gamma, plotted with true metrics
                best_entry = _choose_gamma_entry(gamma_entries, task, coverage_target)
                if best_entry is not None:
                    true_metrics = best_entry.get("true_metrics") or best_entry.get("true")
                    if isinstance(true_metrics, dict):
                        _append_metric_records(
                            records,
                            base_info,
                            METHOD_MDCP_TUNED,
                            "mdcp_tuned",
                            true_metrics,
                            task,
                            run_id,
                        )

    return pd.DataFrame.from_records(records)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["task", "metric", "method", "delta_x"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _prepare_method_list(methods_present: Iterable[str], include_tuned: bool) -> List[str]:
    methods = []
    unique = set(methods_present)

    if METHOD_BASELINE_AGG in unique:
        methods.append(METHOD_BASELINE_AGG)

    baseline_src = sorted(
        (m for m in unique if m.startswith(METHOD_BASELINE_SRC_PREFIX)),
        key=lambda s: int(s.replace(METHOD_BASELINE_SRC_PREFIX, "").strip() or "0"),
    )
    methods.extend(baseline_src)

    if METHOD_MDCP in unique:
        methods.append(METHOD_MDCP)

    if include_tuned and METHOD_MDCP_TUNED in unique:
        methods.append(METHOD_MDCP_TUNED)

    return methods


def _line_with_band(
    ax: plt.Axes,
    xs: Sequence[float],
    mean_values: Sequence[float],
    std_values: Sequence[float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    marker: str,
    markersize: float,
    markeredgecolor: str,
    markeredgewidth: float,
    fill_alpha: float = 0.18,
) -> None:
    ax.plot(
        xs,
        mean_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth,
    )
    mean_arr = np.asarray(mean_values, dtype=float)
    std_arr = np.asarray(std_values, dtype=float)
    if std_arr.size and np.all(np.isfinite(std_arr)):
        ax.fill_between(
            xs,
            mean_arr - std_arr,
            mean_arr + std_arr,
            color=color,
            alpha=fill_alpha,
            linewidth=0,
        )


def _metric_ylabel(metric: str, task: str) -> str:
    label = METRIC_LABELS.get(metric, metric)
    if isinstance(label, dict):
        return str(label.get(task, metric))
    return str(label)


def _format_delta_axis(ax: plt.Axes, deltas: Sequence[float]) -> None:
    ordered = sorted(
        set(
            float(d)
            for d in deltas
            if np.isfinite(d) and float(d) <= MAX_DELTA_X
        )
    )
    ax.set_xticks(ordered)
    ax.set_xticklabels([f"{d:g}" for d in ordered])
    if ordered:
        xmin = float(min(ordered))
        xmax = float(MAX_DELTA_X)
        span = max(xmax - xmin, 1e-9)
        pad = span * 0.05
        if len(ordered) >= 2:
            diffs = np.diff(np.asarray(sorted(ordered), dtype=float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                pad = max(pad, float(diffs.min()) * 0.25)
        if xmin >= xmax:
            pad = max(pad, 0.25)
        ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xlabel(r"Shift magnitude", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _metric_title(metric: str, task: str) -> str:
    title = METRIC_LABELS.get(metric, metric)
    if isinstance(title, dict):
        return str(title.get(task, metric))
    return str(title)


def _coverage_axis_limits(values: Sequence[float], coverage_target: float) -> Tuple[float, float]:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return COVERAGE_MIN - COVERAGE_PADDING, 1.02
    lower = min(clean)
    upper = max(clean + [float(coverage_target)])
    lower = min(lower, COVERAGE_MIN - COVERAGE_PADDING)
    return lower - COVERAGE_PADDING, min(max(upper + COVERAGE_PADDING, COVERAGE_MIN + 0.1), 1.02)


def _configure_metric_axis(ax: plt.Axes, metric: str, task: str, coverage_target: float) -> None:
    if metric in {"overall_coverage", "worst_case_coverage"} and np.isfinite(coverage_target):
        values: List[float] = []
        for line in ax.get_lines():
            ydata = getattr(line, "get_ydata", lambda: [])()
            if ydata is None:
                continue
            values.extend(float(v) for v in ydata if np.isfinite(v))
        ymin, ymax = _coverage_axis_limits(values, coverage_target)
        ax.set_ylim(ymin, ymax)
        ax.axhline(coverage_target, color="#7A7A7A", linestyle="--", linewidth=1.0)
    else:
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, 1e-3)
        buffer = span * 0.05
        ax.set_ylim(max(0.0, ymin - buffer), ymax + buffer)


def _plot_variant_panels(
    summary: pd.DataFrame,
    *,
    task: str,
    include_tuned: bool,
    coverage_target: float,
    output_path: Path,
) -> None:
    task_slice = summary[summary["task"] == task]
    if task_slice.empty:
        return

    methods = _prepare_method_list(task_slice["method"].unique(), include_tuned=include_tuned)
    if not methods:
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.0 * len(METRICS), 2.9), squeeze=False)
    legend_handles: Dict[str, plt.Line2D] = {}

    for col_idx, metric in enumerate(METRICS):
        ax = axes[0, col_idx]
        metric_slice = task_slice[task_slice["metric"] == metric]
        if metric_slice.empty:
            ax.axis("off")
            continue

        for method in methods:
            method_slice = metric_slice[metric_slice["method"] == method].sort_values("delta_x")
            if method_slice.empty:
                continue
            style = LINE_STYLES.get(method, {"linestyle": "-", "linewidth": 1.4, "alpha": 0.9})
            color = PALETTE.get(method, "#444444")
            marker = str(style.get("marker", "o"))
            markersize = float(style.get("markersize", 3.4))
            markeredgecolor = str(style.get("markeredgecolor", "black"))
            markeredgewidth = float(style.get("markeredgewidth", 0.35))
            _line_with_band(
                ax,
                method_slice["delta_x"].to_numpy(dtype=float),
                method_slice["mean"].to_numpy(dtype=float),
                method_slice["std"].fillna(0.0).to_numpy(dtype=float),
                color=color,
                linestyle=str(style.get("linestyle", "-")),
                linewidth=float(style.get("linewidth", 1.4)),
                alpha=float(style.get("alpha", 0.9)),
                marker=marker,
                markersize=markersize,
                markeredgecolor=markeredgecolor,
                markeredgewidth=markeredgewidth,
                fill_alpha=0.18 if method != METHOD_MDCP_TUNED else 0.22,
            )
            if method not in legend_handles and col_idx == 0:
                legend_handles[method] = plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=str(style.get("linestyle", "-")),
                    linewidth=float(style.get("linewidth", 1.4)),
                    alpha=float(style.get("alpha", 0.9)),
                    marker=marker,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    label=method,
                )

        _configure_metric_axis(ax, metric, task, coverage_target)
        _format_delta_axis(ax, metric_slice["delta_x"].unique())
        ax.set_title(_metric_title(metric, task), fontsize=LABEL_FONT_SIZE, fontweight="bold")
        ax.set_ylabel("")

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            [h.get_label() for h in legend_handles.values()],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            frameon=False,
            ncol=max(1, len(legend_handles)),
            columnspacing=0.9,
            handlelength=1.2,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )

    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.08, right=0.995, hspace=0.25, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready MDCP covariate shift (delta_x) plots")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "cov_shift",
        help="Directory containing covariate shift evaluation artifacts (expects a 'summaries/' subfolder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "final_paper_ready",
        help="Directory where publication-ready figures will be saved.",
    )
    args = parser.parse_args()

    if not args.eval_root.exists():
        raise FileNotFoundError(f"Evaluation root not found: {args.eval_root}")

    records_df = _collect_records_from_summaries(args.eval_root)
    if records_df.empty:
        raise RuntimeError("No evaluation records found for covariate shift plots.")

    summary_df = _summarize(records_df)

    coverage_by_task = (
        records_df.groupby("task")["coverage_target"].mean().to_dict()
        if "task" in records_df
        else {}
    )

    output_root = args.output_dir / "cov_shift"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save summary table
    summary_table_path = output_root / "cov_shift_summary.csv"
    summary_df.to_csv(summary_table_path, index=False)
    print(f"Saved summary table: {summary_table_path}")

    for task in (CLASSIFICATION, REGRESSION):
        coverage_target = float(coverage_by_task.get(task, np.nan))
        for include_tuned in (False, True):
            suffix = "tuned" if include_tuned else "vanilla"
            out = output_root / f"cov_shift_{task}_metrics_{suffix}.pdf"
            _plot_variant_panels(
                summary_df,
                task=task,
                include_tuned=include_tuned,
                coverage_target=coverage_target,
                output_path=out,
            )


if __name__ == "__main__":
    main()

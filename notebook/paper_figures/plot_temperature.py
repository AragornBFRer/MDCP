from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
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
METRIC_TITLES = {
    "overall_coverage": "Overall coverage",
    "worst_case_coverage": "Worst-case coverage",
    "efficiency": {
        CLASSIFICATION: "Avg set size",
        REGRESSION: "Avg interval width",
    },
}

EFF_KEY = {CLASSIFICATION: "avg_set_size", REGRESSION: "avg_width"}

BASELINE_MAX = "Max_Aggregated"
METHOD_BASELINE_PREFIX = "Source_"
METHOD_BASELINE_LABEL_PREFIX = "Baseline src "
METHOD_BASELINE_MAX_LABEL = "Baseline (max agg)"
METHOD_MDCP = "MDCP"
METHOD_MDCP_TUNED = "MDCP tuned"

PALETTE: Dict[str, str] = {
    METHOD_BASELINE_MAX_LABEL: "#5F6672",
    f"{METHOD_BASELINE_LABEL_PREFIX}0": "#8C94A0",
    f"{METHOD_BASELINE_LABEL_PREFIX}1": "#9EA6B1",
    f"{METHOD_BASELINE_LABEL_PREFIX}2": "#AFB7C1",
    METHOD_MDCP: "#5C8FD4",
    METHOD_MDCP_TUNED: "#81ABDE",
}

LINE_STYLES: Dict[str, Dict[str, object]] = {
    METHOD_BASELINE_MAX_LABEL: {
        "linestyle": "-",
        "linewidth": 1.6,
        "alpha": 0.85,
        "marker": "o",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.35,
    },
    f"{METHOD_BASELINE_LABEL_PREFIX}0": {
        "linestyle": ":",
        "linewidth": 1.1,
        "alpha": 0.7,
        "marker": "P",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.3,
    },
    f"{METHOD_BASELINE_LABEL_PREFIX}1": {
        "linestyle": "-.",
        "linewidth": 1.1,
        "alpha": 0.7,
        "marker": "s",
        "markersize": 4,
        "markeredgecolor": "black",
        "markeredgewidth": 0.3,
    },
    f"{METHOD_BASELINE_LABEL_PREFIX}2": {
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

JITTER = 0.08
COVERAGE_MIN = 0.7
COVERAGE_PADDING = 0.015

# Paper plotting requirement: do not plot beyond this x-value.
MAX_TEMPERATURE = 4.5


def _baseline_label(raw_name: str) -> Optional[str]:
    if raw_name == BASELINE_MAX:
        return METHOD_BASELINE_MAX_LABEL
    if raw_name.startswith(METHOD_BASELINE_PREFIX):
        suffix = raw_name[len(METHOD_BASELINE_PREFIX) :]
        try:
            suffix = str(int(suffix))
        except ValueError:
            suffix = suffix
        return f"{METHOD_BASELINE_LABEL_PREFIX}{suffix}"
    return None


def _coverage_axis_limits(values: Iterable[float], coverage_target: float) -> Tuple[float, float]:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return COVERAGE_MIN - COVERAGE_PADDING, 1.02
    lower = min(clean)
    upper = max(clean + [coverage_target])
    lower = min(lower, COVERAGE_MIN - COVERAGE_PADDING)
    return lower - COVERAGE_PADDING, max(upper + COVERAGE_PADDING, COVERAGE_MIN + 0.1)


def _metric_title(metric: str, task: str) -> str:
    title = METRIC_TITLES.get(metric)
    if isinstance(title, dict):
        return str(title.get(task, metric.replace("_", " ").title()))
    if title is not None:
        return str(title)
    return metric.replace("_", " ").title()


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
    bundles = {
        "overall_coverage": float(metrics.get("coverage", np.nan)),
        "efficiency": float(metrics.get(EFF_KEY[task], np.nan)),
    }
    worst_cov = np.nan
    worst_eff = np.nan
    indiv_cov = metrics.get("individual_coverage")
    indiv_eff = metrics.get("individual_widths")
    if indiv_cov is not None:
        cov_arr = np.asarray(indiv_cov, dtype=float)
        cov_arr = cov_arr[np.isfinite(cov_arr)]
        if cov_arr.size:
            worst_cov = float(cov_arr.min())
    if indiv_eff is not None:
        eff_arr = np.asarray(indiv_eff, dtype=float)
        eff_arr = eff_arr[np.isfinite(eff_arr)]
        if eff_arr.size and (worst_cov is np.nan or not np.isnan(worst_cov)):
            worst_eff = float(eff_arr.max())
    bundles["worst_case_coverage"] = worst_cov
    bundles["worst_case_efficiency"] = worst_eff

    for metric, value in bundles.items():
        if np.isnan(value):
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


def _collect_records(eval_root: Path) -> Tuple[pd.DataFrame, float]:
    records: List[Dict[str, object]] = []
    coverage_targets: List[float] = []

    for task in (CLASSIFICATION, REGRESSION):
        task_dir = eval_root / task
        if not task_dir.exists():
            continue
        for npz_path in sorted(task_dir.glob("*.npz")):
            try:
                with np.load(npz_path, allow_pickle=True) as payload:
                    metadata = payload["metadata"][()]
                    results = payload["results"][()]
                    baseline_comprehensive = payload.get("baseline_comprehensive")
                    gamma_entries = list(payload.get("gamma_results", []))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Failed to load {npz_path}: {exc}")
                continue

            alpha = float(metadata.get("alpha", 0.1))
            coverage_target = 1.0 - alpha
            coverage_targets.append(coverage_target)
            temperature = float(metadata.get("temperature", np.nan))
            if not np.isfinite(temperature) or temperature > MAX_TEMPERATURE:
                continue
            seed = metadata.get("trial_seed", "na")
            run_id = f"{task}_seed{seed}_temp{temperature:g}"
            base_info = {"temperature": temperature, "coverage_target": coverage_target}

            if isinstance(baseline_comprehensive, np.ndarray) and baseline_comprehensive.size == 1:
                baseline_comprehensive = baseline_comprehensive.item()
            if isinstance(baseline_comprehensive, dict):
                for raw_name, subset_map in baseline_comprehensive.items():
                    label = _baseline_label(raw_name)
                    if label is None:
                        continue
                    overall = subset_map.get("Overall") if isinstance(subset_map, dict) else None
                    _append_metric_records(records, base_info, label, "baseline", overall or {}, task, run_id)

            mdcp_dict = results.get("MDCP", {}) if isinstance(results, dict) else {}
            gamma_zero_metrics = None
            for candidate in ("g1_0_g2_0_g3_0.0", "g1_0_g2_0_g3_0", "g1_0.0_g2_0.0_g3_0.0"):
                if candidate in mdcp_dict:
                    gamma_zero_metrics = mdcp_dict[candidate]
                    break
            if isinstance(gamma_zero_metrics, dict):
                _append_metric_records(records, base_info, METHOD_MDCP, "mdcp", gamma_zero_metrics, task, run_id)

            best_entry = _choose_gamma_entry(gamma_entries, task, coverage_target)
            if best_entry is not None:
                true_metrics = best_entry.get("true_metrics") or best_entry.get("true")
                if isinstance(true_metrics, dict):
                    _append_metric_records(records, base_info, METHOD_MDCP_TUNED, "mdcp_tuned", true_metrics, task, run_id)

    if not records:
        return pd.DataFrame(), float("nan")

    df = pd.DataFrame.from_records(records)
    coverage_reference = float(np.nanmean(df["coverage_target"]))
    return df, coverage_reference


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    summary = (
        df.groupby(["task", "metric", "method", "temperature"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary


def _prepare_method_list(df: pd.DataFrame, include_tuned: bool) -> List[str]:
    methods = []
    baseline_methods = sorted(
        m for m in df["method"].unique() if m.startswith(METHOD_BASELINE_LABEL_PREFIX)
    )
    if METHOD_BASELINE_MAX_LABEL in df["method"].unique():
        methods.append(METHOD_BASELINE_MAX_LABEL)
    methods.extend(baseline_methods)
    if METHOD_MDCP in df["method"].unique():
        methods.append(METHOD_MDCP)
    if include_tuned and METHOD_MDCP_TUNED in df["method"].unique():
        methods.append(METHOD_MDCP_TUNED)
    return methods


def _line_with_band(
    ax: plt.Axes,
    temperatures: Sequence[float],
    mean_values: Sequence[float],
    std_values: Sequence[float],
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    marker: str = "o",
    markersize: float = 4.0,
    markeredgecolor: str = "black",
    markeredgewidth: float = 0.4,
    fill_alpha: float = 0.18,
) -> None:
    ax.plot(
        temperatures,
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
    if std_values is not None:
        mean_arr = np.asarray(mean_values, dtype=float)
        std_arr = np.asarray(std_values, dtype=float)
        if np.all(np.isfinite(std_arr)):
            ax.fill_between(
                temperatures,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=fill_alpha,
                linewidth=0,
            )


def _configure_metric_axis(ax: plt.Axes, metric: str, task: str, coverage_target: float) -> str:
    if metric in {"overall_coverage", "worst_case_coverage"}:
        values: List[float] = []
        for line in ax.get_lines():
            ydata = getattr(line, "get_ydata", lambda: [])()
            if ydata is None:
                continue
            values.extend(float(v) for v in ydata if np.isfinite(v))
        ymin, ymax = _coverage_axis_limits(values, coverage_target)
        ax.set_ylim(ymin, min(ymax, 1.02))
        ax.axhline(coverage_target, color="#7A7A7A", linestyle="--", linewidth=1.0)
    elif metric in {"efficiency", "worst_case_efficiency"}:
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, 1e-3)
        buffer = span * 0.05
        ax.set_ylim(max(0.0, ymin - buffer), ymax + buffer)
    ylabel = METRIC_LABELS.get(metric, metric)
    if isinstance(ylabel, dict):
        ylabel = ylabel[task]
    return str(ylabel)


def _format_temperature_axis(ax: plt.Axes, temperatures: Sequence[float], *, xlabel: bool) -> None:
    ordered = sorted(
        set(
            float(t)
            for t in temperatures
            if np.isfinite(t) and float(t) <= MAX_TEMPERATURE
        )
    )
    ax.set_xticks(ordered)
    ax.set_xticklabels([f"{t:g}" for t in ordered])
    if ordered:
        xmin = float(min(ordered))
        xmax = float(MAX_TEMPERATURE)
        span = max(xmax - xmin, 1e-9)
        pad = span * 0.05
        if len(ordered) >= 2:
            diffs = np.diff(np.asarray(sorted(ordered), dtype=float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                pad = max(pad, float(diffs.min()) * 0.25)
        if xmin >= xmax:
            # Degenerate case (e.g., only x==4.5). Still show a small window.
            pad = max(pad, 0.25)
        ax.set_xlim(xmin - pad, xmax + pad)
    if xlabel:
        ax.set_xlabel("Temperature", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    else:
        ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _plot_variant_panels(
    summary: pd.DataFrame,
    task: str,
    methods: Sequence[str],
    coverage_target: float,
    output_path: Path,
) -> None:
    task_slice = summary[summary["task"] == task]
    if task_slice.empty:
        print(f"No records available for task '{task}' variant panels")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.0 * len(METRICS), 2.9), squeeze=False)
    legend_handles: List[plt.Line2D] = []

    for col_idx, metric in enumerate(METRICS):
        ax = axes[0, col_idx]
        metric_slice = task_slice[task_slice["metric"] == metric]
        if metric_slice.empty:
            ax.axis("off")
            continue
        temps = sorted(metric_slice["temperature"].unique())
        for method in methods:
            method_slice = metric_slice[metric_slice["method"] == method]
            if method_slice.empty:
                continue
            method_slice = method_slice.sort_values("temperature")
            style = LINE_STYLES.get(method, {"linestyle": "-", "linewidth": 1.4, "alpha": 0.9})
            color = PALETTE.get(method, "#444444")
            marker = str(style.get("marker", "o"))
            markersize = float(style.get("markersize", 3.2))
            markeredgecolor = style.get("markeredgecolor", "black")
            markeredgewidth = float(style.get("markeredgewidth", 0.35))
            _line_with_band(
                ax,
                method_slice["temperature"].to_numpy(dtype=float),
                method_slice["mean"].to_numpy(dtype=float),
                method_slice["std"].to_numpy(dtype=float),
                color=color,
                linestyle=str(style.get("linestyle", "-")),
                linewidth=float(style.get("linewidth", 1.4)),
                alpha=float(style.get("alpha", 0.9)),
                marker=marker,
                markersize=markersize,
                markeredgecolor=markeredgecolor,
                markeredgewidth=markeredgewidth,
            )
            if col_idx == 0:
                handle = plt.Line2D(
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
                legend_handles.append(handle)
        ylabel = _configure_metric_axis(ax, metric, task, coverage_target)
        _format_temperature_axis(ax, temps, xlabel=True)
        # No need for y-label to repeat, titles are sufficient
        # if col_idx == 0:
        #     ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold")
        # else:
        #     ax.set_ylabel("")
        title = _metric_title(metric, task)
        ax.set_title(title, fontsize=LABEL_FONT_SIZE, fontweight="bold")

    if legend_handles:
        unique_handles = {handle.get_label(): handle for handle in legend_handles}
        fig.legend(
            unique_handles.values(),
            list(unique_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            frameon=False,
            ncol=max(1, len(unique_handles)),
            columnspacing=0.9,
            handlelength=1.2,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.12, right=0.995, hspace=0.25, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _plot_mdcp_comparison(
    summary: pd.DataFrame,
    task: str,
    coverage_target: float,
    output_path: Path,
) -> None:
    mdcp_slice = summary[(summary["task"] == task) & (summary["method"].isin({METHOD_MDCP, METHOD_MDCP_TUNED}))]
    if mdcp_slice.empty:
        print(f"No MDCP records found for comparison plot ({task})")
        return

    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.0 * len(METRICS), 3.8), squeeze=False)
    legend_handles: Dict[str, plt.Line2D] = {}

    for col_idx, metric in enumerate(METRICS):
        top_ax = axes[0, col_idx]
        bottom_ax = axes[1, col_idx]
        metric_slice = mdcp_slice[mdcp_slice["metric"] == metric]
        if metric_slice.empty:
            top_ax.axis("off")
            bottom_ax.axis("off")
            continue

        temps = sorted(metric_slice["temperature"].unique())
        pivot = {}
        for method in (METHOD_MDCP, METHOD_MDCP_TUNED):
            method_slice = metric_slice[metric_slice["method"] == method].sort_values("temperature")
            if method_slice.empty:
                continue
            pivot[method] = method_slice
            style = LINE_STYLES.get(method, {"linestyle": "-", "linewidth": 2.0, "alpha": 0.95})
            color = PALETTE.get(method, "#2A6D70")
            marker = str(style.get("marker", "o"))
            markersize = float(style.get("markersize", 3.4))
            markeredgecolor = style.get("markeredgecolor", "black")
            markeredgewidth = float(style.get("markeredgewidth", 0.4))
            _line_with_band(
                top_ax,
                method_slice["temperature"].to_numpy(dtype=float),
                method_slice["mean"].to_numpy(dtype=float),
                method_slice["std"].to_numpy(dtype=float),
                color=color,
                linestyle=str(style.get("linestyle", "-")),
                linewidth=float(style.get("linewidth", 2.0)),
                alpha=float(style.get("alpha", 0.95)),
                marker=marker,
                markersize=markersize,
                markeredgecolor=markeredgecolor,
                markeredgewidth=markeredgewidth,
                fill_alpha=0.22,
            )
            if method not in legend_handles:
                legend_handles[method] = plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=str(style.get("linestyle", "-")),
                    linewidth=float(style.get("linewidth", 2.0)),
                    alpha=float(style.get("alpha", 0.95)),
                    marker=marker,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    label=method,
                )

        _configure_metric_axis(top_ax, metric, task, coverage_target)
        _format_temperature_axis(top_ax, temps, xlabel=False)
        title = _metric_title(metric, task)
        top_ax.set_title(title, fontsize=LABEL_FONT_SIZE, fontweight="bold")
        if col_idx == 0:
            top_ax.set_ylabel(f"Raw", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        else:
            top_ax.set_ylabel("")

        mdcp_means = pivot.get(METHOD_MDCP)
        tuned_means = pivot.get(METHOD_MDCP_TUNED)
        if mdcp_means is not None and tuned_means is not None:
            merged = pd.merge(
                mdcp_means[["temperature", "mean"]],
                tuned_means[["temperature", "mean"]],
                on="temperature",
                how="inner",
                suffixes=("_mdcp", "_tuned"),
            )
            merged = merged.sort_values("temperature")
            diff = merged["mean_tuned"].to_numpy(dtype=float) - merged["mean_mdcp"].to_numpy(dtype=float)
            temps_arr = merged["temperature"].to_numpy(dtype=float)
            bottom_ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
            bottom_ax.plot(
                temps_arr,
                diff,
                color="#444444",
                linestyle="-",
                linewidth=1.6,
                marker="o",
                markersize=4,
            )
            bottom_ax.fill_between(temps_arr, 0, diff, color="#444444", alpha=0.15)
            diff_label = "Diff = tuned - vanilla"
            bottom_ax.set_ylabel(diff_label, fontsize=LABEL_FONT_SIZE, fontweight="bold")
            span = max(abs(diff).max(), 1e-3)
            bottom_ax.set_ylim(-span * 1.25, span * 1.25)
        else:
            bottom_ax.axis("off")

        _format_temperature_axis(bottom_ax, temps, xlabel=True)
        if col_idx > 0:
            bottom_ax.set_ylabel("")

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            [h.get_label() for h in legend_handles.values()],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            ncol=len(legend_handles),
            columnspacing=0.9,
            handlelength=1.2,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )

    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.12, right=0.995, hspace=0.32, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    summary = (
        df.groupby(["task", "metric", "method", "temperature"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary table: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready MDCP temperature plots")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval_out" / "temperature",
        help="Directory containing temperature evaluation NPZ files.",
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

    records_df, coverage_target = _collect_records(args.eval_root)
    if records_df.empty:
        raise RuntimeError("No evaluation records found for temperature plots.")

    summary_df = _summarize(records_df)

    coverage_overall = float(np.nanmean(records_df["coverage_target"])) if not records_df["coverage_target"].dropna().empty else float("nan")
    coverage_by_task = (
        records_df.groupby("task")["coverage_target"].mean().to_dict()
        if "task" in records_df
        else {}
    )

    output_root = args.output_dir / "temperature"
    output_root.mkdir(parents=True, exist_ok=True)

    for task in (CLASSIFICATION, REGRESSION):
        task_summary = summary_df[summary_df["task"] == task]
        if task_summary.empty:
            print(f"No records found for task '{task}'; skipping temperature plots")
            continue

        coverage_task = float(coverage_by_task.get(task, coverage_overall)) if coverage_by_task else coverage_target
        if np.isnan(coverage_task):
            coverage_task = coverage_target

        # Non-penalized figure (task-specific)
        nonpen_methods = _prepare_method_list(task_summary, include_tuned=False)
        if nonpen_methods:
            nonpen_path = output_root / f"temperature_{task}_nonpenalized.pdf"
            _plot_variant_panels(task_summary, task, nonpen_methods, coverage_task, nonpen_path)
            print(f"Saved {task} non-penalized temperature figure: {nonpen_path}")
        else:
            print(f"No methods available for {task} non-penalized figure; skipped.")

        # Penalized figure (task-specific, includes tuned if available)
        penalized_methods = _prepare_method_list(task_summary, include_tuned=True)
        if penalized_methods:
            penalized_path = output_root / f"temperature_{task}_penalized.pdf"
            _plot_variant_panels(task_summary, task, penalized_methods, coverage_task, penalized_path)
            print(f"Saved {task} penalized temperature figure: {penalized_path}")
        else:
            print(f"No methods available for {task} penalized figure; skipped.")

        # MDCP comparison figure
        mdcp_task_slice = task_summary[task_summary["method"].isin({METHOD_MDCP, METHOD_MDCP_TUNED})]
        if not mdcp_task_slice.empty:
            comparison_path = output_root / f"temperature_{task}_mdcp_comparison.pdf"
            _plot_mdcp_comparison(task_summary, task, coverage_task, comparison_path)
            print(f"Saved {task} MDCP comparison figure: {comparison_path}")
        else:
            print(f"No MDCP records found for {task}; comparison figure skipped.")

    # Summary table
    summary_table_path = output_root / "temperature_summary.csv"
    _write_summary(records_df, summary_table_path)


if __name__ == "__main__":
    main()

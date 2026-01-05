"""Run multi-trial MDCP and baseline conformal evaluations on MEPS panels.

This script evaluates the Multi-Distribution Conformal Prediction (MDCP)
method and baseline conformal predictors on each MEPS regression panel stored
under ``meps/data``. Each racial group (``RACE`` column) is treated as a
separate source distribution. Results are persisted as ``.npz`` payloads that
capture per-gamma MDCP metrics, baseline comprehensive metrics, and basic split
summaries to facilitate downstream aggregation experiments.

Example usage (from repository root)::

    python -m meps.evaluate_meps_rand_avg --panels 19 20 21 --num-trials 5

The script purposely separates data loading, splitting, and evaluation logic so
that individual components can be imported and unit-tested in isolation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebook"
MODEL_ROOT = REPO_ROOT / "model"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.append(str(NOTEBOOK_ROOT))
if str(MODEL_ROOT) not in sys.path:
    sys.path.append(str(MODEL_ROOT))

from notebook.data_utils import reconstruct_source_data  # type: ignore  # noqa:E402
from notebook.eval_utils import (  # type: ignore  # noqa:E402
    _format_gamma_name,
    _run_mdcp_for_gamma,
    _score_gamma_candidate,
    _split_mimic_sets,
    _summarize_metrics_for_logging,
    evaluate_baseline_regression_comprehensive,
    generate_y_grid_regression,
)
from notebook.baseline import BaselineConformalPredictor  # type: ignore  # noqa:E402
from model.MDCP import (  # type: ignore  # noqa:E402
    SourceModelRegressionGaussian,
    PooledConditionalRegressionGaussian,
    compute_source_weights_from_sizes,
)


TARGET_COL = "UTILIZATION_reg"
SENSITIVE_COL = "RACE"
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA_GRID: List[float] = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DATA_ROOT = REPO_ROOT / "meps" / "data"
OUTPUT_ROOT = REPO_ROOT / "eval_out" / "meps"
TARGET_TRANSFORM_NONE = "none"
TARGET_TRANSFORM_LOG1P = "log1p"


@dataclass
class SplitPayload:
    X_train: np.ndarray
    X_cal: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_cal: np.ndarray
    Y_test: np.ndarray
    source_train: np.ndarray
    source_cal: np.ndarray
    source_test: np.ndarray


@dataclass
class PanelMetadata:
    panel_name: str
    alpha: float
    random_seed: int
    gamma_grid: List[float]
    total_samples: int
    feature_columns: List[str]
    drop_sensitive_feature: bool
    source_mapping: Dict[int, str]
    mimic_cal_ratio: float
    target_transform: str = TARGET_TRANSFORM_NONE


def apply_target_transform(y: Union[np.ndarray, pd.Series], transform: str) -> np.ndarray:
    """Apply the requested target transform (default: log1p)."""

    transform_key = (transform or TARGET_TRANSFORM_NONE).strip().lower()
    values = np.asarray(y, dtype=float)

    if transform_key in {"", TARGET_TRANSFORM_NONE}:
        return values
    if transform_key == TARGET_TRANSFORM_LOG1P:
        if np.any(values < -1.0):
            raise ValueError(
                "log1p transform requires target values to be >= -1. Encountered minimum "
                f"value {float(np.min(values)):.6f}."
            )
        return np.log1p(values)

    raise ValueError(f"Unsupported target transform: {transform}")


def load_panel_dataframe(
    panel_name: str,
    max_samples: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    csv_path = DATA_ROOT / f"meps_{panel_name}_reg.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset for panel {panel_name}: {csv_path}")

    df = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"], errors="ignore")
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    drop_sensitive_feature: bool,
    target_transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if TARGET_COL not in df.columns or SENSITIVE_COL not in df.columns:
        missing = {TARGET_COL, SENSITIVE_COL} - set(df.columns)
        raise KeyError(f"Dataset missing expected columns: {missing}")

    y = apply_target_transform(df[TARGET_COL], target_transform)
    sources = df[SENSITIVE_COL].to_numpy(dtype=int)

    feature_cols = [col for col in df.columns if col not in {TARGET_COL}]
    if drop_sensitive_feature and SENSITIVE_COL in feature_cols:
        feature_cols.remove(SENSITIVE_COL)

    X = df[feature_cols].to_numpy(dtype=float)
    return X, y, sources, feature_cols


def three_way_split(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    train_ratio: float,
    cal_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitPayload:
    if not np.isclose(train_ratio + cal_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + cal_ratio + test_ratio must equal 1.0")

    # First split: train vs remainder (cal+test)
    X_train, X_rest, y_train, y_rest, src_train, src_rest = train_test_split(
        X,
        y,
        sources,
        test_size=(cal_ratio + test_ratio),
        stratify=sources,
        random_state=seed,
    )

    cal_fraction = cal_ratio / (cal_ratio + test_ratio)
    X_cal, X_test, y_cal, y_test, src_cal, src_test = train_test_split(
        X_rest,
        y_rest,
        src_rest,
        test_size=(1.0 - cal_fraction),
        stratify=src_rest,
        random_state=seed + 1,
    )

    return SplitPayload(
        X_train=X_train,
        X_cal=X_cal,
        X_test=X_test,
        Y_train=y_train,
        Y_cal=y_cal,
        Y_test=y_test,
        source_train=src_train,
        source_cal=src_cal,
        source_test=src_test,
    )


def instantiate_sources(
    X_sources_train: List[np.ndarray],
    Y_sources_train: List[np.ndarray],
) -> List[SourceModelRegressionGaussian]:
    sources: List[SourceModelRegressionGaussian] = []
    for idx, (X_src, Y_src) in enumerate(zip(X_sources_train, Y_sources_train)):
        if len(X_src) == 0:
            raise ValueError(f"Source {idx} has no training samples; adjust splits.")
        model = SourceModelRegressionGaussian(X_src, Y_src)
        sources.append(model)
    return sources


def _make_json_serializable(value: object) -> object:
    """Recursively convert numpy containers into JSON-friendly objects."""

    if isinstance(value, dict):
        return {key: _make_json_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):  # type: ignore[arg-type]
        return int(value)
    if isinstance(value, (np.floating,)):  # type: ignore[arg-type]
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sample_lambda_payload(
    lambda_model,
    X_test: np.ndarray,
    sample_limit: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    if lambda_model is None or len(X_test) == 0 or sample_limit <= 0:
        return None

    sample_size = min(sample_limit, len(X_test))
    indices = rng.choice(len(X_test), size=sample_size, replace=False)
    lambda_vals = np.asarray(lambda_model.lambda_at_x(X_test[indices]))
    return {
        "indices": indices.astype(int),
        "lambda_values": lambda_vals,
        "sample_size": int(sample_size),
    }


def evaluate_panel(
    panel_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    X, y, sources, feature_cols = build_feature_matrix(
        df,
        drop_sensitive_feature=args.drop_sensitive,
        target_transform=args.target_transform,
    )

    split = three_way_split(
        X,
        y,
        sources,
        train_ratio=args.train_ratio,
        cal_ratio=args.cal_ratio,
        test_ratio=args.test_ratio,
        seed=args.random_seed,
    )

    n_sources = len(np.unique(sources))
    X_sources_train, Y_sources_train = reconstruct_source_data(
        split.X_train,
        split.Y_train,
        split.source_train,
        n_sources,
    )
    X_sources_cal, Y_sources_cal = reconstruct_source_data(
        split.X_cal,
        split.Y_cal,
        split.source_cal,
        n_sources,
    )

    train_sizes = [len(arr) for arr in X_sources_train]
    cal_sizes = [len(arr) for arr in X_sources_cal]
    source_weights = compute_source_weights_from_sizes(train_sizes)

    sources_models = instantiate_sources(X_sources_train, Y_sources_train)

    # Fit pooled p_data(y|x) on the pooled *training* split only.
    pooled_p_data_model = PooledConditionalRegressionGaussian(
        split.X_train,
        split.Y_train,
        learner='gbm',
    )
    p_data_conditional_train = pooled_p_data_model.conditional_at_pairs(split.X_train, split.Y_train)

    mimic_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    mimic_error: Optional[str] = None
    mimic_summary: Optional[Dict[str, object]] = None
    X_sources_mimic_cal: List[np.ndarray] = []
    Y_sources_mimic_cal: List[np.ndarray] = []
    X_mimic_test = np.empty((0, split.X_train.shape[1]))
    Y_mimic_test = np.empty((0,))
    source_mimic_test = np.empty((0,), dtype=int)
    y_grid_mimic: Optional[np.ndarray] = None
    if len(sources_models) >= 2:
        try:
            mimic_components = _split_mimic_sets(
                split.X_train,
                split.Y_train,
                split.source_train,
                args.mimic_cal_ratio,
                seed=args.random_seed + 303,
                stratify=False,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            mimic_error = str(exc)
    else:
        mimic_error = "Insufficient sources for mimic evaluation"

    if mimic_components is not None:
        (
            X_mimic_cal,
            X_mimic_test,
            Y_mimic_cal,
            Y_mimic_test,
            source_mimic_cal,
            source_mimic_test,
        ) = mimic_components

        X_sources_mimic_cal, Y_sources_mimic_cal = reconstruct_source_data(
            X_mimic_cal,
            Y_mimic_cal,
            source_mimic_cal,
            n_sources,
        )
        candidate_grid = generate_y_grid_regression(
            Y_sources_train + Y_sources_mimic_cal
        )
        if candidate_grid.size == 0:
            mimic_error = "Failed to generate y_grid for mimic evaluation"
            y_grid_mimic = None
        else:
            y_grid_mimic = candidate_grid
            mimic_summary = {
                "cal_samples": int(len(X_mimic_cal)),
                "test_samples": int(len(X_mimic_test)),
                "cal_per_source": [len(arr) for arr in X_sources_mimic_cal],
                "test_per_source": [int(np.sum(source_mimic_test == idx)) for idx in range(n_sources)],
            }

    y_grid = generate_y_grid_regression(Y_sources_train + Y_sources_cal)
    if y_grid.size == 0:
        raise ValueError("Failed to generate y_grid for MDCP evaluation.")

    gamma_entries: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)
    for gamma_value in args.gamma_grid:
        entry: Dict[str, object] = {
            "gamma": float(gamma_value),
            "gamma_name": _format_gamma_name(gamma_value),
        }

        if mimic_components is not None and y_grid_mimic is not None:
            try:
                mimic_metrics, _ = _run_mdcp_for_gamma(
                    gamma_value,
                    sources_models,
                    split.X_train,
                    split.Y_train,
                    X_sources_mimic_cal,
                    Y_sources_mimic_cal,
                    X_mimic_test,
                    Y_mimic_test,
                    y_grid_mimic,
                    args.alpha,
                    source_weights,
                    "regression",
                    source_mimic_test,
                    verbose=args.verbose,
                    p_data_conditional=p_data_conditional_train,
                )
                entry["mimic_metrics"] = mimic_metrics
                entry["mimic_efficiency"] = _score_gamma_candidate(
                    mimic_metrics, "regression"
                )
                entry["mimic_summary"] = _summarize_metrics_for_logging(
                    mimic_metrics, "regression"
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                entry["mimic_error"] = str(exc)
        else:
            entry["mimic_error"] = mimic_error

        metrics, lambda_model = _run_mdcp_for_gamma(
            gamma_value,
            sources_models,
            split.X_train,
            split.Y_train,
            X_sources_cal,
            Y_sources_cal,
            split.X_test,
            split.Y_test,
            y_grid,
            args.alpha,
            source_weights,
            "regression",
            split.source_test,
            verbose=args.verbose,
            p_data_conditional=p_data_conditional_train,
        )
        entry["metrics"] = metrics
        entry["true_efficiency"] = _score_gamma_candidate(metrics, "regression")
        entry["true_summary"] = _summarize_metrics_for_logging(
            metrics, "regression"
        )

        lambda_payload = sample_lambda_payload(
            lambda_model,
            split.X_test,
            args.lambda_sample_limit,
            rng,
        )
        entry["lambda_sample"] = lambda_payload
        gamma_entries.append(entry)

    baseline_results = {}
    baseline_comprehensive = None
    baseline_predictor = BaselineConformalPredictor(
        random_seed=args.random_seed,
        enable_cqr=bool(getattr(args, "enable_cqr_baseline", False)),
    )
    try:
        baseline_predictor.train_source_models(
            X_sources_train,
            Y_sources_train,
            task="regression",
            cqr_alpha=args.alpha,
        )
        baseline_predictor.calibrate(
            X_sources_cal,
            Y_sources_cal,
            alpha=args.alpha,
        )
        baseline_comprehensive = evaluate_baseline_regression_comprehensive(
            baseline_predictor,
            split.X_test,
            split.Y_test,
            split.source_test,
            args.alpha,
        )
        if baseline_comprehensive:
            max_agg = baseline_comprehensive.get("Max_Aggregated", {}).get("Overall")
            single = baseline_comprehensive.get("Source_0", {}).get("Overall")
            if max_agg:
                baseline_results["Max Aggregation"] = max_agg
            if single:
                baseline_results["Single Source"] = single

            cqr_max = baseline_comprehensive.get("CQR_Max_Aggregated", {}).get("Overall")
            if cqr_max:
                baseline_results["CQR Max Aggregation"] = cqr_max
            for key, subset_map in baseline_comprehensive.items():
                if not key.startswith("CQR_Source_"):
                    continue
                overall_metrics = subset_map.get("Overall") if isinstance(subset_map, dict) else None
                if overall_metrics:
                    baseline_results[key.replace("_", " ")] = overall_metrics
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"Baseline evaluation failed on panel {panel_name}: {exc}")

    metadata = PanelMetadata(
        panel_name=panel_name,
        alpha=args.alpha,
        random_seed=args.random_seed,
        gamma_grid=list(map(float, args.gamma_grid)),
        total_samples=len(df),
        feature_columns=feature_cols,
        drop_sensitive_feature=bool(args.drop_sensitive),
        source_mapping={0: "non-white", 1: "white"},
        mimic_cal_ratio=float(args.mimic_cal_ratio),
        target_transform=str(args.target_transform),
    )

    split_summary = {
        "train_samples": int(len(split.X_train)),
        "cal_samples": int(len(split.X_cal)),
        "test_samples": int(len(split.X_test)),
        "train_per_source": train_sizes,
        "cal_per_source": cal_sizes,
    }

    return {
        "panel": panel_name,
        "metadata": metadata,
        "split_summary": split_summary,
        "mimic_summary": mimic_summary,
        "baseline": baseline_results,
        "baseline_comprehensive": baseline_comprehensive,
        "mdcp_gamma_results": gamma_entries,
    }


def save_payload(payload: Dict[str, object], output_dir: Path) -> Path:
    panel = payload.get("panel", "unknown")
    metadata: PanelMetadata = payload["metadata"]  # type: ignore[assignment]
    filename = (
        f"meps_panel_{panel}_seed_{metadata.random_seed}_alpha_{metadata.alpha:.3f}.npz"
    )
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, payload=np.array(payload, dtype=object))
    return output_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MDCP and baseline conformal methods on MEPS panels.",
    )
    parser.add_argument(
        "--panels",
        nargs="*",
        default=["19", "20", "21"],
        help="List of panel suffixes to evaluate (e.g., 19 20 21).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=123,
        help="Random seed for reproducible data splits and sampling.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of independent trials to execute.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Base random seed used to derive per-trial seeds when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list. Length must match --num-trials if supplied.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Target miscoverage level.",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=DEFAULT_GAMMA_GRID,
        help="Gamma values to scan for MDCP spline regularization.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of rows per panel (for quick experiments).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where evaluation payloads are stored.",
    )
    parser.add_argument(
        "--drop-sensitive",
        action="store_true",
        help="Drop the sensitive attribute (RACE) from feature matrix before training.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of data used for training pool.",
    )
    parser.add_argument(
        "--cal-ratio",
        type=float,
        default=0.2,
        help="Fraction of data used for calibration pool.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data used for final evaluation.",
    )
    parser.add_argument(
        "--mimic-cal-ratio",
        type=float,
        default=0.5,
        help="Fraction of the training pool reserved for mimic calibration (remainder for mimic test).",
    )
    parser.add_argument(
        "--lambda-sample-limit",
        type=int,
        default=200,
        help="Maximum number of test points to snapshot lambda weights for.",
    )
    parser.add_argument(
        "--enable-cqr-baseline",
        action="store_true",
        help="Enable CQR-based baseline intervals in addition to the default Gaussian baseline.",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default=TARGET_TRANSFORM_LOG1P,
        choices=[TARGET_TRANSFORM_NONE, TARGET_TRANSFORM_LOG1P],
        help="Target transform applied prior to model fitting (default: log1p).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose MDCP fitting diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1.")

    if args.seeds is not None and len(args.seeds) != args.num_trials:
        raise ValueError("Length of --seeds must match --num-trials.")

    base_seed = args.base_seed if args.base_seed is not None else args.random_seed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds is not None:
        trial_seeds = list(args.seeds)
    else:
        trial_seeds = [int(base_seed + idx) for idx in range(args.num_trials)]

    all_trial_summaries: List[Dict[str, object]] = []

    for trial_index, trial_seed in enumerate(trial_seeds, start=1):
        print(
            f"\n##### Trial {trial_index}/{args.num_trials} -- seed {trial_seed} #####"
        )
        np.random.seed(trial_seed)

        trial_output_dir = args.output_dir / f"trial_{trial_index:03d}_seed_{trial_seed}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        trial_args = argparse.Namespace(**vars(args))
        trial_args.random_seed = trial_seed
        trial_args.output_dir = trial_output_dir

        payloads = []
        for panel in args.panels:
            print(f"\n=== Evaluating MEPS panel {panel} (trial {trial_index}) ===")
            df = load_panel_dataframe(panel, trial_args.max_samples, trial_seed)
            payload = evaluate_panel(panel, df, trial_args)
            saved_path = save_payload(payload, trial_output_dir)
            print(f"  Saved results to {saved_path}")
            payloads.append(payload)

        summary_filename = (
            f"meps_eval_summary_trial_{trial_index:03d}_seed_{trial_seed}.json"
        )
        summary_path = trial_output_dir / summary_filename
        summary_payload = _make_json_serializable({
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "panels": [
                {
                    "panel": p["panel"],
                    "split_summary": p["split_summary"],
                    "mimic_summary": p.get("mimic_summary"),
                    "metadata": asdict(p["metadata"]),
                    "mdcp_gamma_results": [
                        {
                            "gamma": entry["gamma"],
                            "metrics": entry["metrics"],
                            "mimic_metrics": entry.get("mimic_metrics"),
                            "mimic_summary": entry.get("mimic_summary"),
                            "mimic_efficiency": entry.get("mimic_efficiency"),
                            "true_efficiency": entry.get("true_efficiency"),
                        }
                        for entry in p["mdcp_gamma_results"]
                    ],
                }
                for p in payloads
            ],
        })
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)
        print(f"  Trial summary written to {summary_path}")

        try:
            relative_summary = summary_path.relative_to(args.output_dir)
        except ValueError:
            relative_summary = summary_path

        all_trial_summaries.append(_make_json_serializable(
            {
                "trial_index": trial_index,
                "trial_seed": trial_seed,
                "summary_path": str(relative_summary),
                "panels": [
                    {
                        "panel": p["panel"],
                        "split_summary": p["split_summary"],
                        "mimic_summary": p.get("mimic_summary"),
                    }
                    for p in payloads
                ],
            }
        ))

    multi_summary_path = args.output_dir / (
        f"meps_eval_multi_trial_summary_{args.num_trials}_trials.json"
    )
    multi_summary_payload = _make_json_serializable({
        "num_trials": args.num_trials,
        "panels": list(args.panels),
        "trial_summaries": all_trial_summaries,
        "base_seed": base_seed,
        "explicit_seeds": args.seeds,
    })
    with multi_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(multi_summary_payload, fp, indent=2)
    print(f"\nMulti-trial summary written to {multi_summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate MDCP and baselines on PovertyMap pending multihead artifacts.

This is the CPU-side stage of the "predict once, split many" workflow.

Expected prediction directory contents:
  - pending_embeddings.npz
      - embedding: (n, d)
      - target: (n,)
      - source_id: (n,)  (rural=0, urban=1)
      - index: (n,)      dataset indices (for traceability)
  - pending_multihead_heads.pt
      - pooled_head weights
      - per-source head weights

The script runs many randomized per-source splits and fits LambdaNN (lambda_mode='nn')
using pooled p_data(y|x) from the saved pooled head.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_ROOT = REPO_ROOT / "notebook"
MODEL_ROOT = REPO_ROOT / "model"

for path in (REPO_ROOT, NOTEBOOK_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from notebook.data_utils import reconstruct_source_data  # type: ignore  # noqa: E402
from notebook.eval_utils import (  # type: ignore  # noqa: E402
    _run_mdcp_for_gamma,
    _score_gamma_candidate,
    _split_mimic_sets,
    _summarize_metrics_for_logging,
    evaluate_baseline_regression_comprehensive,
    evaluate_regression_performance_with_individual_sets,
    generate_y_grid_regression,
)
from notebook.baseline import BaselineConformalPredictor  # type: ignore  # noqa: E402
from model.MDCP import (  # type: ignore  # noqa: E402
    NNSourceModelRegressionGaussian,
    PooledConditionalRegressionGaussian,
    compute_source_weights_from_sizes,
)
from wilds.poverty.multihead_models import GaussianParamHead  # noqa: E402


MIMIC_CAL_RATIO = 0.5


@dataclass
class SplitArrays:
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
class TrialPayload:
    trial_index: int
    random_seed: int
    split_sizes: Dict[str, int]
    source_sizes: Dict[str, Dict[str, int]]
    mdcp_results: List[Dict[str, Any]]
    baseline_results: Dict[str, Any]
    lambda_snapshots: List[Dict[str, Any]]
    mimic_info: Dict[str, Any]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MDCP and baseline conformal methods on PovertyMap pending multihead artifacts.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing pending_embeddings.npz and pending_multihead_heads.pt",
    )
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.125)
    parser.add_argument("--cal-frac", type=float, default=0.375)
    parser.add_argument("--test-frac", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--y-grid-size", type=int, default=512)
    parser.add_argument("--y-margin", type=float, default=0.05)
    parser.add_argument("--lambda-sample", type=int, default=64)

    # Optional PCA for LambdaNN only (source heads and pooled head still consume the original embeddings).
    # PCA is fit on the pooled training split (split_arrays.X_train) per trial.
    parser.add_argument(
        "--lambda-use-pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit a PCA projection (k=--lambda-pca-dim) on the pooled training split per trial and train LambdaNN on PCA features "
            "(default: enabled). Disable with --no-lambda-use-pca."
        ),
    )
    parser.add_argument(
        "--lambda-pca-dim",
        type=int,
        default=16,
        help="PCA dimension k used for LambdaNN when --lambda-use-pca is enabled.",
    )
    parser.add_argument(
        "--lambda-pca-whiten",
        action="store_true",
        help="Whether to whiten PCA components used by LambdaNN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the evaluation artifact (.npz). Defaults to eval_out/poverty/mdcp/.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_embeddings(pred_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    emb_path = pred_dir / "pending_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    payload = np.load(emb_path)
    X = np.asarray(payload["embedding"], dtype=float)
    Y = np.asarray(payload["target"], dtype=float).reshape(-1)
    src = np.asarray(payload["source_id"], dtype=int).reshape(-1)
    idx = np.asarray(payload["index"], dtype=int).reshape(-1)
    if not (len(X) == len(Y) == len(src) == len(idx)):
        raise ValueError("Embedding payload arrays must share the same length")
    return X, Y, src, idx


def load_heads(pred_dir: Path) -> Dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load head weights") from _TORCH_IMPORT_ERROR
    head_path = pred_dir / "pending_multihead_heads.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Missing {head_path}")
    return torch.load(head_path, map_location="cpu")


def allocate_counts(total: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    weights = np.asarray(fractions, dtype=float)
    if not np.all(weights >= 0):
        raise ValueError("Split fractions must be non-negative")
    if float(np.sum(weights)) <= 0:
        raise ValueError("At least one split fraction must be positive")
    normalized = weights / float(np.sum(weights))
    raw = normalized * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total - int(np.sum(counts)))
    if remainder > 0:
        fractional = raw - counts
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def sample_split_indices(
    source_ids: np.ndarray,
    rng: np.random.Generator,
    train_frac: float,
    cal_frac: float,
    test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    if not math.isclose(train_frac + cal_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + cal_frac + test_frac must equal 1.0")
    unique_sources = sorted(set(int(s) for s in np.asarray(source_ids, dtype=int).tolist()))
    train_idx: List[int] = []
    cal_idx: List[int] = []
    test_idx: List[int] = []
    per_source_counts: Dict[str, Dict[str, int]] = {}

    for sid in unique_sources:
        src_indices = np.where(source_ids == int(sid))[0]
        if src_indices.size == 0:
            raise ValueError(f"Source {sid} has no samples")
        perm = rng.permutation(src_indices)
        n_train, n_cal, n_test = allocate_counts(
            perm.size,
            (train_frac, cal_frac, test_frac),
        )
        if n_cal <= 0 or n_test <= 0:
            raise ValueError(
                f"Source {sid} split yields insufficient cal/test (cal={n_cal}, test={n_test})."
            )
        train_part = perm[:n_train]
        cal_part = perm[n_train:n_train + n_cal]
        test_part = perm[n_train + n_cal:n_train + n_cal + n_test]
        train_idx.extend(train_part.tolist())
        cal_idx.extend(cal_part.tolist())
        test_idx.extend(test_part.tolist())
        per_source_counts[str(sid)] = {
            "train": int(train_part.size),
            "cal": int(cal_part.size),
            "test": int(test_part.size),
            "total": int(src_indices.size),
        }

    return (
        np.sort(np.asarray(train_idx, dtype=int)),
        np.sort(np.asarray(cal_idx, dtype=int)),
        np.sort(np.asarray(test_idx, dtype=int)),
        per_source_counts,
    )


def build_split_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    source_ids: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> SplitArrays:
    return SplitArrays(
        X_train=X[train_idx],
        X_cal=X[cal_idx],
        X_test=X[test_idx],
        Y_train=Y[train_idx],
        Y_cal=Y[cal_idx],
        Y_test=Y[test_idx],
        source_train=source_ids[train_idx],
        source_cal=source_ids[cal_idx],
        source_test=source_ids[test_idx],
    )


def instantiate_sources_from_heads(
    head_payload: Dict[str, Any],
    contig_to_raw: Sequence[int],
    *,
    device: str = "cpu",
    variance_floor: float = 1e-6,
    pdf_floor: float = 1e-12,
) -> Tuple[List[NNSourceModelRegressionGaussian], Any]:
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for evaluation") from _TORCH_IMPORT_ERROR

    if not head_payload:
        raise ValueError("Missing head payload")
    if not head_payload.get("multihead", False):
        raise ValueError("Head payload does not indicate multihead=True")
    if str(head_payload.get("task")) != "regression":
        raise ValueError("Expected regression head payload")

    embedding_dim = int(head_payload.get("embedding_dim"))
    scale_floor = float(head_payload.get("scale_floor", 1e-3))
    payload_source_ids = [int(s) for s in head_payload.get("source_ids", [])]
    if not payload_source_ids:
        raise ValueError("Head payload missing source_ids")

    pooled_head = GaussianParamHead(embedding_dim, scale_floor=scale_floor)
    pooled_head.load_state_dict(head_payload["pooled_head"])
    p_data_model = PooledConditionalRegressionGaussian(
        model=pooled_head,
        device=device,
        variance_floor=variance_floor,
        pdf_floor=pdf_floor,
    )

    sources: List[NNSourceModelRegressionGaussian] = []
    available = set(payload_source_ids)
    for raw_id in contig_to_raw:
        if int(raw_id) not in available:
            raise ValueError(f"Missing head for raw source_id={raw_id}; available={payload_source_ids}")
        head_state = head_payload["source_heads"][str(int(raw_id))]
        head = GaussianParamHead(embedding_dim, scale_floor=scale_floor)
        head.load_state_dict(head_state)
        src = NNSourceModelRegressionGaussian(
            head,
            device=device,
            variance_floor=variance_floor,
            pdf_floor=pdf_floor,
        )
        sources.append(src)

    return sources, p_data_model


def snapshot_lambda_values(
    lambda_model: Any,
    X_test: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if lambda_model is None or len(X_test) == 0 or limit <= 0:
        return {}
    sample_size = min(limit, len(X_test))
    indices = rng.choice(len(X_test), size=sample_size, replace=False)
    lambda_values = lambda_model.lambda_at_x(X_test[indices])
    return {"indices": indices, "lambda_values": lambda_values}


def fit_lambda_pca_state(
    X_train: np.ndarray,
    *,
    pca_dim: int,
    whiten: bool,
    seed: int,
) -> Dict[str, Any]:
    """Fit PCA on the pooled training split and return a serializable projection state.

    This projection is intended ONLY for LambdaNN; density heads remain in the original
    embedding space.
    """

    from sklearn.decomposition import PCA  # local import to keep dependency optional

    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")

    n, d = int(X_train.shape[0]), int(X_train.shape[1])
    k = int(pca_dim)
    if k <= 0:
        raise ValueError("pca_dim must be positive")
    if k > min(n, d):
        raise ValueError(f"pca_dim={k} must be <= min(n, d)={min(n, d)} for PCA")

    pca = PCA(
        n_components=k,
        whiten=bool(whiten),
        random_state=int(seed),
        svd_solver="auto",
    )
    pca.fit(X_train)

    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "whiten": bool(whiten),
    }


def run_mdcp_trial(
    *,
    gamma_grid: List[float],
    alpha: float,
    sources: List[Any],
    X_sources_train: List[np.ndarray],
    Y_sources_train: List[np.ndarray],
    X_sources_cal: List[np.ndarray],
    Y_sources_cal: List[np.ndarray],
    split_arrays: SplitArrays,
    y_grid_size: int,
    y_margin: float,
    lambda_sample: int,
    rng: np.random.Generator,
    p_data_model: Any,
    lambda_pca_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_weights = compute_source_weights_from_sizes([len(X) for X in X_sources_train])
    y_grid_true = generate_y_grid_regression(
        Y_sources_train + Y_sources_cal,
        n_grid_points=y_grid_size,
        margin_factor=y_margin,
    )

    num_sources = len(X_sources_train)
    feature_dim = split_arrays.X_train.shape[1] if split_arrays.X_train.ndim > 1 else 1

    mimic_seed = int(rng.integers(0, 2**32 - 1))
    mimic_components: Tuple[np.ndarray, ...] | None = None
    mimic_error: str | None = None

    if num_sources >= 2:
        try:
            mimic_components = _split_mimic_sets(
                split_arrays.X_train,
                split_arrays.Y_train,
                split_arrays.source_train,
                MIMIC_CAL_RATIO,
                mimic_seed,
                stratify=False,
            )
        except Exception as exc:  # pragma: no cover
            mimic_error = str(exc)
    else:
        mimic_error = "insufficient_sources"

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
            num_sources,
        )
        y_grid_mimic = generate_y_grid_regression(
            Y_sources_train + Y_sources_mimic_cal,
            n_grid_points=y_grid_size,
            margin_factor=y_margin,
        )
        mimic_available = True
        mimic_cal_counts = np.bincount(source_mimic_cal, minlength=num_sources).astype(int).tolist()
        mimic_test_counts = np.bincount(source_mimic_test, minlength=num_sources).astype(int).tolist()
    else:
        X_mimic_cal = np.empty((0, feature_dim))
        X_mimic_test = np.empty((0, feature_dim))
        Y_mimic_cal = np.empty((0,), dtype=split_arrays.Y_train.dtype)
        Y_mimic_test = np.empty((0,), dtype=split_arrays.Y_train.dtype)
        source_mimic_cal = np.empty((0,), dtype=split_arrays.source_train.dtype)
        source_mimic_test = np.empty((0,), dtype=split_arrays.source_train.dtype)
        X_sources_mimic_cal = []
        Y_sources_mimic_cal = []
        y_grid_mimic = None
        mimic_available = False
        mimic_cal_counts: List[int] = []
        mimic_test_counts: List[int] = []

    gamma_results: List[Dict[str, Any]] = []
    lambda_snapshots: List[Dict[str, Any]] = []
    lambda_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

    for gamma_value in gamma_grid:
        entry: Dict[str, Any] = {"gamma": float(gamma_value)}

        pca_nn_kwargs: Dict[str, Any] = {}
        if lambda_pca_state is not None:
            pca_nn_kwargs = {"pca_state": lambda_pca_state}

        if mimic_available and y_grid_mimic is not None:
            try:
                mimic_metrics, _ = _run_mdcp_for_gamma(
                    gamma_value=gamma_value,
                    sources=sources,
                    X_train=split_arrays.X_train,
                    Y_train=split_arrays.Y_train,
                    X_cal_list=X_sources_mimic_cal,  # type: ignore[arg-type]
                    Y_cal_list=Y_sources_mimic_cal,  # type: ignore[arg-type]
                    X_test=X_mimic_test,
                    Y_test=Y_mimic_test,
                    y_grid=y_grid_mimic,
                    alpha_val=alpha,
                    source_weights=source_weights,
                    task_type="regression",
                    source_test=source_mimic_test,
                    verbose=False,
                    lambda_mode="nn",
                    p_data_model=p_data_model,
                    # small hidden layers for small sample sizes
                    nn_kwargs={"hidden_sizes": (4, 4), "device": "cpu", **pca_nn_kwargs},
                )
                entry["mimic_metrics"] = mimic_metrics
                entry["mimic_efficiency"] = _score_gamma_candidate(mimic_metrics, "regression")
                entry["mimic_summary"] = _summarize_metrics_for_logging(mimic_metrics, "regression")
            except Exception as exc:  # pragma: no cover
                entry["mimic_error"] = str(exc)
        elif mimic_error is not None:
            entry["mimic_error"] = mimic_error

        try:
            true_metrics, lambda_model = _run_mdcp_for_gamma(
                gamma_value=gamma_value,
                sources=sources,
                X_train=split_arrays.X_train,
                Y_train=split_arrays.Y_train,
                X_cal_list=X_sources_cal,
                Y_cal_list=Y_sources_cal,
                X_test=split_arrays.X_test,
                Y_test=split_arrays.Y_test,
                y_grid=y_grid_true,
                alpha_val=alpha,
                source_weights=source_weights,
                task_type="regression",
                source_test=split_arrays.source_test,
                verbose=False,
                lambda_mode="nn",
                p_data_model=p_data_model,
                # small hidden layers for small sample sizes
                nn_kwargs={"hidden_sizes": (4, 4), "device": "cpu", **pca_nn_kwargs},
            )
            entry["metrics"] = true_metrics
            entry["efficiency"] = _score_gamma_candidate(true_metrics, "regression")
            entry["summary"] = _summarize_metrics_for_logging(true_metrics, "regression")

            snapshot = snapshot_lambda_values(lambda_model, split_arrays.X_test, lambda_sample, lambda_rng)
            if snapshot:
                snapshot.update({"gamma": float(gamma_value)})
                lambda_snapshots.append(snapshot)
        except Exception as exc:  # pragma: no cover
            entry["true_error"] = str(exc)

        gamma_results.append(entry)

    mimic_info = {
        "ratio": float(MIMIC_CAL_RATIO),
        "seed": int(mimic_seed),
        "available": bool(mimic_available),
        "error": mimic_error,
        "cal_counts": mimic_cal_counts,
        "test_counts": mimic_test_counts,
        "n_mimic_cal": int(X_mimic_cal.shape[0]),
        "n_mimic_test": int(X_mimic_test.shape[0]),
    }
    return gamma_results, lambda_snapshots, mimic_info


def run_baseline_trial(
    *,
    alpha: float,
    sources: List[Any],
    X_sources_cal: List[np.ndarray],
    Y_sources_cal: List[np.ndarray],
    split_arrays: SplitArrays,
    trial_seed: int,
) -> Dict[str, Any]:
    baseline = BaselineConformalPredictor(random_seed=trial_seed)
    baseline.task = "regression"
    baseline.source_models = sources
    baseline.calibrate(X_sources_cal, Y_sources_cal, alpha=alpha)
    return evaluate_baseline_regression_comprehensive(
        baseline,
        split_arrays.X_test,
        split_arrays.Y_test,
        split_arrays.source_test,
        alpha,
    )


def summarize_mdcp_trials(trials: List[TrialPayload], gamma_grid: List[float]) -> Dict[str, Any]:
    coverage_by_gamma: Dict[float, List[float]] = defaultdict(list)
    width_by_gamma: Dict[float, List[float]] = defaultdict(list)

    for trial in trials:
        for entry in trial.mdcp_results:
            gamma = float(entry["gamma"])
            summary = entry.get("summary", {})
            coverage_by_gamma[gamma].append(float(summary.get("coverage", np.nan)))
            width_by_gamma[gamma].append(float(summary.get("avg_width", np.nan)))

    aggregated: Dict[str, Dict[str, float]] = {}
    for gamma in gamma_grid:
        cov_values = coverage_by_gamma.get(float(gamma), [])
        wid_values = width_by_gamma.get(float(gamma), [])
        aggregated[str(gamma)] = {
            "coverage_mean": float(np.mean(cov_values)) if cov_values else float("nan"),
            "coverage_std": float(np.std(cov_values)) if cov_values else float("nan"),
            "avg_width_mean": float(np.mean(wid_values)) if wid_values else float("nan"),
            "avg_width_std": float(np.std(wid_values)) if wid_values else float("nan"),
        }
    return aggregated


def summarize_baseline_trials(trials: List[TrialPayload]) -> Dict[str, Any]:
    coverage_store: Dict[str, List[float]] = defaultdict(list)
    width_store: Dict[str, List[float]] = defaultdict(list)
    for trial in trials:
        for method, eval_dict in trial.baseline_results.items():
            overall = eval_dict.get("Overall")
            if not overall:
                continue
            coverage_store[method].append(float(overall.get("coverage", np.nan)))
            width_store[method].append(float(overall.get("avg_width", np.nan)))

    aggregated: Dict[str, Dict[str, float]] = {}
    for method in sorted(coverage_store.keys()):
        cov_vals = coverage_store[method]
        wid_vals = width_store[method]
        aggregated[method] = {
            "coverage_mean": float(np.mean(cov_vals)) if cov_vals else float("nan"),
            "coverage_std": float(np.std(cov_vals)) if cov_vals else float("nan"),
            "avg_width_mean": float(np.mean(wid_vals)) if wid_vals else float("nan"),
            "avg_width_std": float(np.std(wid_vals)) if wid_vals else float("nan"),
        }
    return aggregated


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)

    if args.num_trials <= 0:
        raise ValueError("num-trials must be positive")
    if not math.isclose(args.train_frac + args.cal_frac + args.test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train-frac + cal-frac + test-frac must equal 1.0")

    X, Y, raw_source_ids, dataset_indices = load_embeddings(args.prediction_dir)
    head_payload = load_heads(args.prediction_dir)

    # Remap raw source IDs -> contiguous indices for downstream utilities.
    unique_raw = sorted(set(int(v) for v in raw_source_ids.tolist()))
    raw_to_contig = {raw: idx for idx, raw in enumerate(unique_raw)}
    contig_to_raw = unique_raw
    source_ids = np.asarray([raw_to_contig[int(v)] for v in raw_source_ids], dtype=int)

    sources, p_data_model = instantiate_sources_from_heads(
        head_payload=head_payload,
        contig_to_raw=contig_to_raw,
        device="cpu",
    )

    trials: List[TrialPayload] = []
    n_sources = len(contig_to_raw)
    total_samples = len(X)

    for trial_idx in range(args.num_trials):
        trial_seed = int(args.base_seed + trial_idx)
        rng = np.random.default_rng(trial_seed)
        train_idx, cal_idx, test_idx, per_source_counts = sample_split_indices(
            source_ids,
            rng,
            float(args.train_frac),
            float(args.cal_frac),
            float(args.test_frac),
        )
        split_arrays = build_split_arrays(X, Y, source_ids, train_idx, cal_idx, test_idx)

        lambda_pca_state: Optional[Dict[str, Any]] = None
        if bool(getattr(args, "lambda_use_pca", False)):
            lambda_pca_state = fit_lambda_pca_state(
                split_arrays.X_train,
                pca_dim=int(args.lambda_pca_dim),
                whiten=bool(args.lambda_pca_whiten),
                seed=trial_seed,
            )

        X_sources_train, Y_sources_train = reconstruct_source_data(
            split_arrays.X_train,
            split_arrays.Y_train,
            split_arrays.source_train,
            n_sources,
        )
        X_sources_cal, Y_sources_cal = reconstruct_source_data(
            split_arrays.X_cal,
            split_arrays.Y_cal,
            split_arrays.source_cal,
            n_sources,
        )

        mdcp_results, lambda_snapshots, mimic_info = run_mdcp_trial(
            gamma_grid=[float(g) for g in args.gamma_grid],
            alpha=float(args.alpha),
            sources=sources,
            X_sources_train=X_sources_train,
            Y_sources_train=Y_sources_train,
            X_sources_cal=X_sources_cal,
            Y_sources_cal=Y_sources_cal,
            split_arrays=split_arrays,
            y_grid_size=int(args.y_grid_size),
            y_margin=float(args.y_margin),
            lambda_sample=int(args.lambda_sample),
            rng=rng,
            p_data_model=p_data_model,
            lambda_pca_state=lambda_pca_state,
        )

        baseline_results = run_baseline_trial(
            alpha=float(args.alpha),
            sources=sources,
            X_sources_cal=X_sources_cal,
            Y_sources_cal=Y_sources_cal,
            split_arrays=split_arrays,
            trial_seed=trial_seed,
        )

        split_sizes = {
            "train": int(len(train_idx)),
            "cal": int(len(cal_idx)),
            "test": int(len(test_idx)),
        }
        trials.append(
            TrialPayload(
                trial_index=trial_idx,
                random_seed=trial_seed,
                split_sizes=split_sizes,
                source_sizes=per_source_counts,
                mdcp_results=mdcp_results,
                baseline_results=baseline_results,
                lambda_snapshots=lambda_snapshots,
                mimic_info=mimic_info,
            )
        )

        print(
            f"Trial {trial_idx + 1}/{args.num_trials} | seed={trial_seed} | "
            f"cal={split_sizes['cal']} test={split_sizes['test']}"
        )
        for entry in mdcp_results:
            summary_line = entry.get("summary", {})
            print(
                f"  MDCP gamma={entry['gamma']}: coverage={float(summary_line.get('coverage', np.nan)):.3f}, "
                f"avg_width={float(summary_line.get('avg_width', np.nan)):.3f}"
            )
        baseline_overall = baseline_results.get("Max_Aggregated", {}).get("Overall", {})
        if baseline_overall:
            print(
                "  Baseline Max_Aggregated: "
                f"coverage={float(baseline_overall.get('coverage', np.nan)):.3f}, "
                f"avg_width={float(baseline_overall.get('avg_width', np.nan)):.3f}"
            )

    mdcp_summary = summarize_mdcp_trials(trials, [float(g) for g in args.gamma_grid])
    baseline_summary = summarize_baseline_trials(trials)

    metadata_payload: Dict[str, Any] = {
        "prediction_dir": str(args.prediction_dir),
        "alpha": float(args.alpha),
        "gamma_grid": [float(g) for g in args.gamma_grid],
        "num_trials": int(args.num_trials),
        "base_seed": int(args.base_seed),
        "train_frac": float(args.train_frac),
        "cal_frac": float(args.cal_frac),
        "test_frac": float(args.test_frac),
        "y_grid_size": int(args.y_grid_size),
        "y_margin": float(args.y_margin),
        "total_samples": int(total_samples),
        "n_sources": int(n_sources),
        "raw_source_id_to_contig": raw_to_contig,
        "contig_to_raw_source_id": contig_to_raw,
        "lambda_sample": int(args.lambda_sample),
        "lambda_use_pca": bool(args.lambda_use_pca),
        "lambda_pca_dim": int(args.lambda_pca_dim) if bool(args.lambda_use_pca) else None,
        "lambda_pca_whiten": bool(args.lambda_pca_whiten) if bool(args.lambda_use_pca) else None,
        "dataset_indices": dataset_indices.tolist(),
        "head_payload_meta": {
            "task": head_payload.get("task"),
            "head_type": head_payload.get("head_type"),
            "embedding_dim": head_payload.get("embedding_dim"),
            "source_ids": head_payload.get("source_ids"),
        },
    }

    output_path = args.output
    if output_path is None:
        default_dir = REPO_ROOT / "eval_out" / "poverty" / "mdcp"
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / f"poverty_mdcp_trials_trials{args.num_trials}_seed{args.base_seed}.npz"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": np.array([metadata_payload], dtype=object),
        "trials": np.array([asdict(trial) for trial in trials], dtype=object),
        "mdcp_summary": np.array([mdcp_summary], dtype=object),
        "baseline_summary": np.array([baseline_summary], dtype=object),
    }
    np.savez(output_path, **payload)

    print(f"Saved evaluation artifact to {output_path}")
    return {
        "output_path": output_path,
        "metadata": metadata_payload,
        "mdcp_summary": mdcp_summary,
        "baseline_summary": baseline_summary,
    }


if __name__ == "__main__":  # pragma: no cover
    main()

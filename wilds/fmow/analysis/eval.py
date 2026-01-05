from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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
    _summarize_metrics_for_logging,
    _split_mimic_sets,
    evaluate_baseline_classification_comprehensive,
    generate_y_grid_classification,
)
from notebook.baseline import BaselineConformalPredictor  # type: ignore  # noqa: E402
from model.MDCP import (  # type: ignore  # noqa: E402
    compute_source_weights_from_sizes,
    NNSourceModelClassification,
    PooledConditionalClassifier,
)


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
        description="Evaluate MDCP and baseline conformal methods on FMoW holdout predictions.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing holdout_probabilities.npy and holdout_metadata.csv",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of randomized calibration/test splits to evaluate (default: 100).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed; individual trial seeds are derived deterministically.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.125,
        help="Fraction of holdout samples reserved for MDCP lambda training (per region).",
    )
    parser.add_argument(
        "--cal-frac",
        type=float,
        default=0.375,
        help="Fraction of holdout samples reserved for calibration (per region).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.5,
        help="Fraction of holdout samples reserved for evaluation (per region).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Target miscoverage level for conformal prediction.",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        help="Candidate gamma values used when fitting MDCP lambda splines.",
    )
    parser.add_argument(
        "--lambda-sample",
        type=int,
        default=64,
        help="Number of test points per trial to snapshot lambda values for diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the evaluation artifact (.npz). Default saves under eval_out/fmow/mdcp/",
    )

    # PCA for LambdaNN only (enabled by default).
    parser.add_argument(
        "--lambda-use-pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply a PCA projection to the input features used by LambdaNN (lambda(x)) only (default: enabled). "
            "Disable with --no-lambda-use-pca. This does NOT change the embedding space used by the per-region heads or pooled head."
        ),
    )
    parser.add_argument(
        "--lambda-pca-dim",
        type=int,
        default=16,
        help="Number of PCA components for LambdaNN when --lambda-use-pca is set (default: 16).",
    )
    parser.add_argument(
        "--lambda-pca-whiten",
        action="store_true",
        help="Use whitened PCA components for LambdaNN inputs (default: False).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def fit_lambda_pca_state(
    X_train: np.ndarray,
    *,
    pca_dim: int,
    whiten: bool,
    seed: int,
) -> Dict[str, Any]:
    """Fit PCA on the pooled LambdaNN training split and return a serializable state.

    Notes
    -----
    - This PCA state is meant to be consumed by `model.MDCP.LambdaNN(pca_state=...)`.
    - Fit is performed on `split_arrays.X_train` only (per trial) to avoid leakage.
    - We intentionally error out (rather than silently shrinking k) if `pca_dim` is infeasible.
    """

    X_train = np.asarray(X_train)
    if X_train.ndim != 2:
        raise ValueError(f"Expected X_train to be 2D, got shape={X_train.shape}")
    n, d = int(X_train.shape[0]), int(X_train.shape[1])
    k = int(pca_dim)
    if k <= 0:
        raise ValueError("lambda_pca_dim must be positive")
    max_k = int(min(n, d))
    if k > max_k:
        raise ValueError(
            f"lambda_pca_dim={k} is too large for this trial's training split (n_train={n}, d={d}); "
            f"must be <= min(n_train, d)={max_k}. "
            "Try a smaller --lambda-pca-dim, or increase --train-frac."
        )
    if k == d:
        raise ValueError(
            f"lambda_pca_dim={k} equals the embedding dimension d={d}; PCA would be an identity transform. "
            "Disable PCA or choose a smaller k."
        )

    pca = PCA(n_components=k, whiten=bool(whiten), random_state=int(seed), svd_solver='auto')
    pca.fit(X_train.astype(np.float32))
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "whiten": bool(whiten),
    }


def load_prediction_bundle(pred_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    meta_path = pred_dir / "holdout_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            "Prediction directory must contain holdout_metadata.csv",
        )
    metadata = pd.read_csv(meta_path)

    summary_path = pred_dir / "prediction_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

    emb_path = pred_dir / "holdout_embeddings.npz"
    embeddings: Optional[np.ndarray] = None
    emb_indices: Optional[np.ndarray] = None
    if emb_path.exists():
        emb_payload = np.load(emb_path)
        embeddings = emb_payload.get("embedding")
        emb_indices = emb_payload.get("index")
        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=float)
        if emb_indices is not None:
            emb_indices = np.asarray(emb_indices, dtype=int)

    return metadata, summary, embeddings, emb_indices


def load_embedding_meta(pred_dir: Path) -> Dict[str, Any]:
    meta_path = pred_dir / "holdout_embeddings_meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_multihead_heads(pred_dir: Path) -> Dict[str, Any]:
    heads_path = pred_dir / "holdout_multihead_heads.pt"
    if not heads_path.exists():
        return {}
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to load multihead heads but could not be imported."
        ) from _TORCH_IMPORT_ERROR
    return torch.load(heads_path, map_location="cpu")


def build_feature_arrays(
    metadata: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    emb_indices: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int], List[int]]:
    df = metadata.copy()
    if "region_id" not in df.columns:
        raise ValueError("metadata must contain a 'region_id' column")

    # raw region_id values come from the dataset; we remap to contiguous [0..K-1]
    raw_region_ids = df["region_id"].astype(int).to_numpy()
    unique_raw = sorted(set(int(v) for v in raw_region_ids.tolist()))
    raw_to_contig = {raw: idx for idx, raw in enumerate(unique_raw)}
    contig_ids = np.asarray([raw_to_contig[int(v)] for v in raw_region_ids], dtype=int)

    df["region_name"] = df.get("region", df["region_id"]).astype(str)
    df["row_id"] = np.arange(len(df), dtype=int)
    df = df.set_index("row_id", drop=False)

    labels = df["label"].astype(int).to_numpy()
    source_ids = contig_ids

    if embeddings is not None and emb_indices is not None:
        # Align embeddings to metadata order via dataset_idx when available
        if "dataset_idx" not in df.columns:
            df["dataset_idx"] = df.index
        index_to_pos = {int(idx): pos for pos, idx in enumerate(emb_indices)}
        ordered_emb: List[np.ndarray] = []
        for idx in df["dataset_idx"]:
            pos = index_to_pos.get(int(idx))
            if pos is None:
                raise ValueError(f"Missing embedding for dataset_idx {idx}")
            ordered_emb.append(embeddings[pos])
        feature_values = np.stack(ordered_emb, axis=0)
    else:
        raise ValueError("Embeddings are required to build feature arrays.")
        # scale = float(max(len(df) - 1, 1))
        # feature_values = (df["row_id"].to_numpy(dtype=float) / scale).reshape(-1, 1)

    return feature_values, labels, source_ids, raw_to_contig, unique_raw


def _train_cal_test_split(
    region_df: pd.DataFrame,
    rng: np.random.Generator,
    train_frac: float,
    cal_frac: float,
    test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(train_frac + cal_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + cal_frac + test_frac must equal 1.0")

    indices = region_df.index.to_numpy()
    labels = region_df["label"].to_numpy()
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    test_size = np.clip(test_frac, 1e-4, 1.0 - 1e-4)
    seed = int(rng.integers(0, 2**32 - 1))
    try:
        train_val_idx, test_idx, train_val_labels, _ = train_test_split(
            indices,
            labels,
            test_size=test_size,
            stratify=labels if len(np.unique(labels)) > 1 else None,
            random_state=seed,
        )
    except ValueError:
        train_val_idx, test_idx, train_val_labels, _ = train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=seed,
        )

    if len(train_val_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), indices

    cal_ratio = cal_frac / max(1.0 - test_size, 1e-8)
    cal_ratio = np.clip(cal_ratio, 1e-4, 1.0 - 1e-4)
    seed = int(rng.integers(0, 2**32 - 1))
    try:
        train_idx, cal_idx = train_test_split(
            train_val_idx,
            test_size=cal_ratio,
            stratify=train_val_labels if len(np.unique(train_val_labels)) > 1 else None,
            random_state=seed,
        )
    except ValueError:
        train_idx, cal_idx = train_test_split(
            train_val_idx,
            test_size=cal_ratio,
            random_state=seed,
        )

    if len(cal_idx) == 0 and len(train_idx) > 1:
        cal_idx = np.array([train_idx[-1]])
        train_idx = train_idx[:-1]
    if len(train_idx) == 0 and len(cal_idx) > 1:
        train_idx = np.array([cal_idx[-1]])
        cal_idx = cal_idx[:-1]
    if len(test_idx) == 0 and len(cal_idx) > 0:
        test_idx = np.array([cal_idx[-1]])
        cal_idx = cal_idx[:-1]
    if len(test_idx) == 0 and len(train_idx) > 0:
        test_idx = np.array([train_idx[-1]])
        train_idx = train_idx[:-1]

    return (
        np.sort(train_idx.astype(int)),
        np.sort(cal_idx.astype(int)),
        np.sort(test_idx.astype(int)),
    )


def sample_split_indices(
    df: pd.DataFrame,
    rng: np.random.Generator,
    train_frac: float,
    cal_frac: float,
    test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    train_indices: List[int] = []
    cal_indices: List[int] = []
    test_indices: List[int] = []
    per_source_counts: Dict[str, Dict[str, int]] = {}

    for region_name, region_df in df.groupby("region_name", sort=True):
        train_idx, cal_idx, test_idx = _train_cal_test_split(
            region_df,
            rng,
            train_frac,
            cal_frac,
            test_frac,
        )
        train_indices.extend(train_idx.tolist())
        cal_indices.extend(cal_idx.tolist())
        test_indices.extend(test_idx.tolist())
        per_source_counts[region_name] = {
            "train": int(len(train_idx)),
            "cal": int(len(cal_idx)),
            "test": int(len(test_idx)),
            "total": int(len(region_df)),
        }

    return (
        np.sort(np.asarray(train_indices, dtype=int)),
        np.sort(np.asarray(cal_indices, dtype=int)),
        np.sort(np.asarray(test_indices, dtype=int)),
        per_source_counts,
    )


def build_split_arrays(
    feature_values: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> SplitArrays:
    return SplitArrays(
        X_train=feature_values[train_idx],
        X_cal=feature_values[cal_idx],
        X_test=feature_values[test_idx],
        Y_train=labels[train_idx],
        Y_cal=labels[cal_idx],
        Y_test=labels[test_idx],
        source_train=source_ids[train_idx],
        source_cal=source_ids[cal_idx],
        source_test=source_ids[test_idx],
    )


def instantiate_sources_from_multihead_heads(
    head_payload: Dict[str, Any],
    embedding_dim: int,
    num_classes: int,
    contig_to_raw: List[int],
    device: str = "cpu",
) -> Tuple[List[NNSourceModelClassification], Any]:
    """Create per-source models (one per source) and pooled p_data_model from saved multihead heads."""

    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for multihead-head evaluation but could not be imported."
        ) from _TORCH_IMPORT_ERROR

    if not head_payload:
        raise ValueError("Missing multihead head payload; expected holdout_multihead_heads.pt")
    if not head_payload.get("multihead", False):
        raise ValueError("Head payload does not indicate multihead=True")

    payload_source_ids = [int(s) for s in head_payload.get("source_ids", [])]
    if not payload_source_ids:
        raise ValueError("Head payload missing source_ids")

    pooled_head = nn.Linear(int(embedding_dim), int(num_classes))
    pooled_head.load_state_dict(head_payload["pooled_head"])

    # Wrap pooled head as p_data(y|x)
    p_data_model = PooledConditionalClassifier(model=pooled_head, device=device)

    sources: List[NNSourceModelClassification] = []
    for contig_idx, raw_id in enumerate(contig_to_raw):
        if int(raw_id) not in set(payload_source_ids):
            raise ValueError(
                f"Missing head for raw region_id={raw_id} (contig={contig_idx}). "
                f"Available source_ids={payload_source_ids}"
            )
        state = head_payload["source_heads"][str(int(raw_id))]
        head = nn.Linear(int(embedding_dim), int(num_classes))
        head.load_state_dict(state)
        src = NNSourceModelClassification(head, device=device)
        # Baseline utilities look for `classes` attribute.
        src.classes = np.arange(int(num_classes), dtype=int)  # type: ignore[attr-defined]
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
    return {
        "indices": indices,
        "lambda_values": lambda_values,
    }


def run_mdcp_trial(
    gamma_grid: List[float],
    alpha: float,
    sources: List[Any],
    X_sources_train: List[np.ndarray],
    Y_sources_train: List[np.ndarray],
    X_sources_cal: List[np.ndarray],
    Y_sources_cal: List[np.ndarray],
    split_arrays: SplitArrays,
    lambda_sample: int,
    rng: np.random.Generator,
    p_data_model: Any,
    lambda_pca_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_weights = compute_source_weights_from_sizes([len(X) for X in X_sources_train])
    y_grid_true = generate_y_grid_classification(Y_sources_train + Y_sources_cal)

    num_sources = len(X_sources_train)
    feature_dim = split_arrays.X_train.shape[1] if split_arrays.X_train.ndim > 1 else 1

    mimic_ratio = 0.5
    mimic_seed = int(rng.integers(0, 2**32 - 1))
    mimic_components: Tuple[np.ndarray, ...] | None = None
    mimic_error: str | None = None

    if num_sources >= 2:
        try:
            mimic_components = _split_mimic_sets(
                split_arrays.X_train,
                split_arrays.Y_train,
                split_arrays.source_train,
                mimic_ratio,
                mimic_seed,
                stratify=True,
            )
        except Exception as exc:  # pragma: no cover - defensive path
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
        y_grid_mimic = generate_y_grid_classification(Y_sources_train + Y_sources_mimic_cal)
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

        mimic_nn_kwargs: Dict[str, Any] = {"hidden_sizes": (4, 4), "device": "cpu"}
        true_nn_kwargs: Dict[str, Any] = {"hidden_sizes": (4, 4), "device": "cpu"}
        if lambda_pca_state is not None:
            mimic_nn_kwargs["pca_state"] = lambda_pca_state
            true_nn_kwargs["pca_state"] = lambda_pca_state

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
                    task_type="classification",
                    source_test=source_mimic_test,
                    verbose=False,
                    lambda_mode="nn",
                    p_data_model=p_data_model,
                    nn_kwargs=mimic_nn_kwargs,
                )
                entry["mimic_metrics"] = mimic_metrics
                entry["mimic_efficiency"] = _score_gamma_candidate(mimic_metrics, "classification")
                entry["mimic_summary"] = _summarize_metrics_for_logging(mimic_metrics, "classification")
            except Exception as exc:  # pragma: no cover - defensive path
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
                task_type="classification",
                source_test=split_arrays.source_test,
                verbose=False,
                lambda_mode="nn",
                p_data_model=p_data_model,
                nn_kwargs=true_nn_kwargs,
            )
            entry["metrics"] = true_metrics
            entry["efficiency"] = _score_gamma_candidate(true_metrics, "classification")
            entry["summary"] = _summarize_metrics_for_logging(true_metrics, "classification")

            snapshot = snapshot_lambda_values(lambda_model, split_arrays.X_test, lambda_sample, lambda_rng)
            if snapshot:
                snapshot.update({"gamma": float(gamma_value)})
                lambda_snapshots.append(snapshot)
        except Exception as exc:  # pragma: no cover - defensive path
            entry["true_error"] = str(exc)

        gamma_results.append(entry)

    mimic_info = {
        "ratio": mimic_ratio,
        "seed": mimic_seed,
        "available": mimic_available,
        "error": mimic_error,
        "cal_counts": mimic_cal_counts,
        "test_counts": mimic_test_counts,
        "n_mimic_cal": int(X_mimic_cal.shape[0]),
        "n_mimic_test": int(X_mimic_test.shape[0]),
    }

    return gamma_results, lambda_snapshots, mimic_info


def run_baseline_trial(
    alpha: float,
    sources: List[Any],
    X_sources_cal: List[np.ndarray],
    Y_sources_cal: List[np.ndarray],
    split_arrays: SplitArrays,
    trial_seed: int,
) -> Dict[str, Any]:
    baseline = BaselineConformalPredictor(random_seed=trial_seed)
    baseline.task = "classification"
    baseline.source_models = sources
    baseline.calibrate(X_sources_cal, Y_sources_cal, alpha=alpha)

    return evaluate_baseline_classification_comprehensive(
        baseline,
        split_arrays.X_test,
        split_arrays.Y_test,
        split_arrays.source_test,
        alpha,
    )


def summarize_mdcp_trials(
    trials: List[TrialPayload],
    gamma_grid: List[float],
) -> Dict[str, Any]:
    coverage_by_gamma: Dict[float, List[float]] = defaultdict(list)
    size_by_gamma: Dict[float, List[float]] = defaultdict(list)

    for trial in trials:
        for entry in trial.mdcp_results:
            gamma = float(entry["gamma"])
            summary = entry["summary"]
            coverage_by_gamma[gamma].append(float(summary["coverage"]))
            size_by_gamma[gamma].append(float(summary.get("avg_set_size", np.nan)))

    aggregated: Dict[str, Dict[str, float]] = {}
    for gamma in gamma_grid:
        cov_values = coverage_by_gamma.get(float(gamma), [])
        size_values = size_by_gamma.get(float(gamma), [])
        aggregated[str(gamma)] = {
            "coverage_mean": float(np.mean(cov_values)) if cov_values else float("nan"),
            "coverage_std": float(np.std(cov_values)) if cov_values else float("nan"),
            "avg_set_size_mean": float(np.mean(size_values)) if size_values else float("nan"),
            "avg_set_size_std": float(np.std(size_values)) if size_values else float("nan"),
        }
    return aggregated


def summarize_baseline_trials(trials: List[TrialPayload]) -> Dict[str, Any]:
    coverage_store: Dict[str, List[float]] = defaultdict(list)
    set_size_store: Dict[str, List[float]] = defaultdict(list)

    for trial in trials:
        for method_name, eval_dict in trial.baseline_results.items():
            overall = eval_dict.get("Overall")
            if not overall:
                continue
            coverage_store[method_name].append(float(overall.get("coverage", np.nan)))
            set_size_store[method_name].append(float(overall.get("avg_set_size", np.nan)))

    aggregated: Dict[str, Dict[str, float]] = {}
    for method_name in sorted(coverage_store.keys()):
        cov_vals = coverage_store[method_name]
        size_vals = set_size_store[method_name]
        aggregated[method_name] = {
            "coverage_mean": float(np.mean(cov_vals)) if cov_vals else float("nan"),
            "coverage_std": float(np.std(cov_vals)) if cov_vals else float("nan"),
            "avg_set_size_mean": float(np.mean(size_vals)) if size_vals else float("nan"),
            "avg_set_size_std": float(np.std(size_vals)) if size_vals else float("nan"),
        }
    return aggregated


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)

    if args.num_trials <= 0:
        raise ValueError("num-trials must be positive")
    if not math.isclose(args.train_frac + args.cal_frac + args.test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train-frac + cal-frac + test-frac must equal 1.0")

    metadata, summary, embeddings, emb_indices = load_prediction_bundle(args.prediction_dir)
    emb_meta = load_embedding_meta(args.prediction_dir)
    head_payload = load_multihead_heads(args.prediction_dir)

    feature_values, labels, source_ids, raw_to_contig, contig_to_raw = build_feature_arrays(
        metadata,
        embeddings,
        emb_indices,
    )

    trials: List[TrialPayload] = []
    n_sources = len(contig_to_raw)
    total_samples = len(feature_values)

    is_multihead = bool(emb_meta.get("multihead", False)) or bool(head_payload.get("multihead", False))
    if not is_multihead:
        raise ValueError(
            "This evaluation script now expects multihead prediction artifacts (holdout_multihead_heads.pt + embeddings). "
            "Re-run prediction with a multihead checkpoint and --save-embeddings."
        )
    if embeddings is None:
        raise ValueError("Embeddings are required for multihead evaluation. Re-run prediction with --save-embeddings.")
    if not head_payload:
        raise ValueError("Missing holdout_multihead_heads.pt in prediction-dir")

    num_classes = head_payload.get("num_classes")
    if num_classes is None:
        # Hard fail rather than silently guessing.
        raise ValueError("Multihead head payload missing 'num_classes'")

    sources, p_data_model_global = instantiate_sources_from_multihead_heads(
        head_payload=head_payload,
        embedding_dim=int(embeddings.shape[1]),
        num_classes=int(num_classes),
        contig_to_raw=contig_to_raw,
        device="cpu",
    )

    for trial_idx in range(args.num_trials):
        trial_seed = int(args.base_seed + trial_idx)
        rng = np.random.default_rng(trial_seed)
        # use metadata frame for splits (contains region info)
        meta_for_split = metadata.copy()
        meta_for_split["region_name"] = meta_for_split["region"].astype(str)
        meta_for_split = meta_for_split.reset_index(drop=True)
        train_idx, cal_idx, test_idx, per_source_counts = sample_split_indices(
            meta_for_split,
            rng,
            args.train_frac,
            args.cal_frac,
            args.test_frac,
        )
        split_arrays = build_split_arrays(
            feature_values,
            labels,
            source_ids,
            train_idx,
            cal_idx,
            test_idx,
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

        p_data_model = p_data_model_global

        lambda_pca_state = None
        if bool(args.lambda_use_pca):
            lambda_pca_state = fit_lambda_pca_state(
                split_arrays.X_train,
                pca_dim=int(args.lambda_pca_dim),
                whiten=bool(args.lambda_pca_whiten),
                seed=trial_seed,
            )

        mdcp_results, lambda_snapshots, mimic_info = run_mdcp_trial(
            gamma_grid=[float(g) for g in args.gamma_grid],
            alpha=args.alpha,
            sources=sources,
            X_sources_train=X_sources_train,
            Y_sources_train=Y_sources_train,
            X_sources_cal=X_sources_cal,
            Y_sources_cal=Y_sources_cal,
            split_arrays=split_arrays,
            lambda_sample=args.lambda_sample,
            rng=rng,
            p_data_model=p_data_model,
            lambda_pca_state=lambda_pca_state,
        )

        baseline_results = run_baseline_trial(
            alpha=args.alpha,
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
            ),
        )

        print(
            f"Trial {trial_idx + 1}/{args.num_trials} | seed={trial_seed} | "
            f"cal={split_sizes['cal']} test={split_sizes['test']}"
        )
        for entry in mdcp_results:
            summary_line = entry["summary"]
            print(
                f"  MDCP gamma={entry['gamma']}: coverage={summary_line['coverage']:.3f}, "
                f"avg_set_size={summary_line.get('avg_set_size', np.nan):.3f}"
            )
        baseline_overall = baseline_results.get("Max_Aggregated", {}).get("Overall", {})
        if baseline_overall:
            print(
                "  Baseline Max_Aggregated: "
                f"coverage={baseline_overall.get('coverage', np.nan):.3f}, "
                f"avg_set_size={baseline_overall.get('avg_set_size', np.nan):.3f}"
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
        "total_samples": int(total_samples),
        "n_sources": int(n_sources),
        "raw_region_id_to_contig": raw_to_contig,
        "contig_to_raw_region_id": contig_to_raw,
        "lambda_sample": int(args.lambda_sample),
        "prediction_summary": summary,
        "embeddings_meta": emb_meta,
        "lambda_use_pca": bool(args.lambda_use_pca),
        "lambda_pca_dim": int(args.lambda_pca_dim),
        "lambda_pca_whiten": bool(args.lambda_pca_whiten),
    }

    output_path = args.output
    if output_path is None:
        default_dir = REPO_ROOT / "eval_out" / "fmow" / "mdcp"
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / (
            f"fmow_mdcp_baseline_trials_"
            f"trials{args.num_trials}_seed{args.base_seed}.npz"
        )
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
    print("MDCP aggregate summary (coverage / avg set size means):")
    for gamma, stats in mdcp_summary.items():
        print(
            f"  gamma={gamma}: coverage_mean={stats['coverage_mean']:.3f}, "
            f"avg_set_size_mean={stats['avg_set_size_mean']:.3f}"
        )
    print("Baseline aggregate summary (coverage / avg set size means):")
    for method, stats in baseline_summary.items():
        print(
            f"  {method}: coverage_mean={stats['coverage_mean']:.3f}, "
            f"avg_set_size_mean={stats['avg_set_size_mean']:.3f}"
        )

    return {
        "output_path": output_path,
        "metadata": metadata_payload,
        "mdcp_summary": mdcp_summary,
        "baseline_summary": baseline_summary,
    }


if __name__ == "__main__":  # pragma: no cover
    main()

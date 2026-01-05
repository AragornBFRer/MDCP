from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.const import DATA_DENSITY_FOLDER, ensure_project_dir, prefer_relative_path

# Default upper bound on how many samples per source we persist in the snapshot.
_DEFAULT_SAMPLE_LIMIT_PER_SOURCE = 256


def _stack_samples(
    X_sources: Sequence[np.ndarray],
    Y_sources: Sequence[np.ndarray],
    sample_limit_per_source: int,
    rng: np.random.Generator,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_features = []
    sample_targets = []
    sample_source_ids = []

    for source_id, (X_src, Y_src) in enumerate(zip(X_sources, Y_sources)):
        if len(X_src) == 0:
            continue

        limit = min(sample_limit_per_source, len(X_src))
        if limit < len(X_src):
            indices = rng.choice(len(X_src), size=limit, replace=False)
        else:
            indices = np.arange(len(X_src))

        sample_features.append(X_src[indices])
        sample_targets.append(Y_src[indices])
        sample_source_ids.append(np.full(len(indices), source_id, dtype=int))

    n_features = X_sources[0].shape[1] if X_sources else 0

    if not sample_features:
        empty_features = np.empty((0, n_features))
        empty_targets = np.empty((0,), dtype=float)
        empty_source_ids = np.empty((0,), dtype=int)
        return empty_features, empty_targets, empty_source_ids

    stacked_features = np.vstack(sample_features)
    stacked_targets = np.concatenate(sample_targets)
    if task == 'classification':
        stacked_targets = stacked_targets.astype(int, copy=False)
    stacked_source_ids = np.concatenate(sample_source_ids)

    return stacked_features, stacked_targets, stacked_source_ids


def _compute_source_stats(
    X_sources: Sequence[np.ndarray],
    Y_sources: Sequence[np.ndarray],
    task: str,
) -> List[Dict[str, Any]]:
    if not X_sources:
        return []

    n_features = X_sources[0].shape[1]
    stats: list[Dict[str, Any]] = []

    for source_id, (X_src, Y_src) in enumerate(zip(X_sources, Y_sources)):
        entry: Dict[str, Any] = {
            'source_id': source_id,
            'n_samples': int(len(X_src)),
            'feature_mean': np.mean(X_src, axis=0) if len(X_src) else np.zeros(n_features),
            'feature_std': np.std(X_src, axis=0) if len(X_src) else np.zeros(n_features),
            'feature_min': np.min(X_src, axis=0) if len(X_src) else np.zeros(n_features),
            'feature_max': np.max(X_src, axis=0) if len(X_src) else np.zeros(n_features),
        }

        if len(X_src) > 1:
            entry['feature_cov'] = np.cov(X_src, rowvar=False)
        elif len(X_src) == 1:
            entry['feature_cov'] = np.zeros((n_features, n_features))
        else:
            entry['feature_cov'] = np.zeros((n_features, n_features))

        if task == 'classification':
            classes, counts = np.unique(Y_src.astype(int, copy=False), return_counts=True)
            entry['class_counts'] = {
                int(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())
            }
        else:
            entry['target_mean'] = float(np.mean(Y_src)) if len(Y_src) else float('nan')
            entry['target_std'] = float(np.std(Y_src)) if len(Y_src) else float('nan')
            entry['target_min'] = float(np.min(Y_src)) if len(Y_src) else float('nan')
            entry['target_max'] = float(np.max(Y_src)) if len(Y_src) else float('nan')

        stats.append(entry)

    return stats


def save_density_snapshot(
    script_tag: str,
    task: str,
    dataset_id: str,
    X_sources: Sequence[np.ndarray],
    Y_sources: Sequence[np.ndarray],
    simulation_params: Optional[Sequence[Any]],
    random_seed: int,
    temperature: float,
    *,
    sample_limit_per_source: int = _DEFAULT_SAMPLE_LIMIT_PER_SOURCE,
    extra_metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Path:
    """Persist a compact snapshot describing the simulated source distributions."""
    if task not in {'classification', 'regression'}:
        raise ValueError(f"Unsupported task '{task}' supplied to save_density_snapshot.")

    safe_dataset_id = dataset_id.replace(' ', '_')
    base_dir = ensure_project_dir(DATA_DENSITY_FOLDER / script_tag / task)
    snapshot_path = base_dir / f"{safe_dataset_id}.npz"

    if snapshot_path.exists() and not overwrite:
        return snapshot_path

    rng = rng or np.random.default_rng(random_seed)

    metadata: Dict[str, Any] = {
        'script_tag': script_tag,
        'task': task,
        'dataset_id': dataset_id,
        'random_seed': int(random_seed),
        'temperature': float(temperature),
        'n_sources': len(X_sources),
        'n_features': int(X_sources[0].shape[1]) if X_sources else 0,
        'n_samples_per_source': [int(len(X)) for X in X_sources],
        'n_total_samples': int(sum(len(X) for X in X_sources)),
        'sample_limit_per_source': int(sample_limit_per_source),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    if task == 'classification':
        all_targets = np.concatenate(Y_sources) if Y_sources else np.array([], dtype=int)
        metadata['unique_classes'] = np.unique(all_targets.astype(int, copy=False)).tolist()
    else:
        total_targets = np.concatenate(Y_sources) if Y_sources else np.array([], dtype=float)
        metadata['target_mean_overall'] = float(np.mean(total_targets)) if len(total_targets) else float('nan')
        metadata['target_std_overall'] = float(np.std(total_targets)) if len(total_targets) else float('nan')

    sample_features, sample_targets, sample_source_ids = _stack_samples(
        X_sources,
        Y_sources,
        sample_limit_per_source,
        rng,
        task,
    )

    metadata['sample_size'] = int(len(sample_source_ids))
    if len(sample_source_ids) > 0:
        metadata['sample_counts_per_source'] = np.bincount(
            sample_source_ids, minlength=len(X_sources)
        ).astype(int).tolist()
    else:
        metadata['sample_counts_per_source'] = [0 for _ in range(len(X_sources))]

    source_stats = list(_compute_source_stats(X_sources, Y_sources, task))

    payload: Dict[str, Any] = {
        'metadata': np.array([metadata], dtype=object),
        'samples_X': sample_features,
        'samples_Y': sample_targets,
        'samples_source_ids': sample_source_ids,
        'source_stats': np.array(source_stats, dtype=object),
        'simulation_params': np.array(list(simulation_params) if simulation_params else [], dtype=object),
    }

    np.savez_compressed(snapshot_path, **payload)
    print(f"Saved density snapshot to {prefer_relative_path(snapshot_path)}")
    return snapshot_path

#!/usr/bin/env python3
"""Export PovertyMap multihead artifacts for MDCP evaluation.

This is the GPU-heavy stage.

Given a single multi-head checkpoint (shared backbone + pooled head + domain heads),
we run the forward pass once on the MDCP pending split and save:
- cached embeddings (NPZ)
- head-only weights (PT)

Evaluation can then run CPU-only for many randomized splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
if _DEFAULT_REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _DEFAULT_REPO_ROOT.as_posix())

from wilds.poverty.train_resnet import (
    _extend_sys_path,
    build_transforms,
    gaussian_nll,
)
from wilds.poverty.multihead_models import MultiHeadGaussianRegressor, export_head_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multihead embeddings + heads on MDCP PovertyMap pending split")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="MDCP repository root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (best_model.pth)")
    parser.add_argument("--split-csv", type=Path, default=None, help="Path to mdcp_split.csv with train/pending labels")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store predictions and summaries. "
            "Defaults to eval_out/poverty/predictions/<checkpoint_parent>/ under --repo-root."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of pending samples (for smoke tests)")
    return parser.parse_args()


def _default_output_dir(repo_root: Path, checkpoint_path: Path) -> Path:
    tag = checkpoint_path.resolve().parent.name
    if not tag:
        tag = "run"
    return repo_root / "eval_out" / "poverty" / "predictions" / tag


def compute_summary(
    *,
    pooled_mean: np.ndarray,
    pooled_scale: np.ndarray,
    target: np.ndarray,
    summary_path: Path,
) -> None:
    target = np.asarray(target, dtype=float)
    pooled_mean = np.asarray(pooled_mean, dtype=float)
    pooled_scale = np.asarray(pooled_scale, dtype=float)

    target_t = torch.from_numpy(target)
    mean_t = torch.from_numpy(pooled_mean)
    scale_t = torch.from_numpy(pooled_scale)

    nll = float(gaussian_nll(target_t, mean_t, scale_t).mean().item())
    rmse = float(np.sqrt(np.mean((pooled_mean - target) ** 2)))
    pearson = float(np.corrcoef(pooled_mean, target)[0, 1]) if target.size > 1 else float("nan")

    summary = {
        "num_samples": int(target.size),
        "avg_nll_pooled": nll,
        "rmse_pooled": rmse,
        "pearson_pooled": pearson,
        "head_type": "gaussian_multihead",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    _extend_sys_path(repo_root)

    data_root = repo_root / "data"
    split_csv = args.split_csv or (data_root / "poverty_v1.1" / "mdcp_split.csv")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV {split_csv} not found")

    import pandas as pd  # defer heavy import until needed

    split_df = pd.read_csv(split_csv)
    pending_indices = split_df.loc[split_df["split"] == "pending", "index"].to_numpy(dtype=np.int64)
    if pending_indices.size == 0:
        raise RuntimeError("No pending indices found in split CSV")

    from wilds.datasets.poverty_dataset import PovertyMapDataset  # type: ignore
    from examples.models.resnet_multispectral import ResNet18  # type: ignore

    dataset = PovertyMapDataset(version="1.1", root_dir=data_root.as_posix(), download=False, split_scheme="official")

    pending_filtered = pending_indices
    if args.limit is not None:
        pending_filtered = pending_filtered[: args.limit]
    if len(pending_filtered) == 0:
        raise RuntimeError("No pending samples remain")

    transforms = build_transforms()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    backbone = ResNet18(num_classes=None, num_channels=8)
    model = MultiHeadGaussianRegressor(
        backbone=backbone,
        embedding_dim=int(backbone.d_out),
        source_ids=(0, 1),
        scale_floor=1e-3,
    ).to(device)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint  # assume raw state dict
    else:
        raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = (
        _default_output_dir(repo_root, checkpoint_path)
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)


    # Export embeddings + pooled predictions (for a quick sanity-check summary).
    from wilds.datasets.wilds_dataset import WILDSSubset  # type: ignore
    subset = WILDSSubset(dataset, pending_filtered, transform=transforms["val"])
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataset_indices = np.array(subset.indices, dtype=np.int64)
    cursor = 0
    idx_batches = []
    emb_batches = []
    tgt_batches = []
    src_batches = []
    pooled_mean_batches = []
    pooled_scale_batches = []

    with torch.no_grad():
        for inputs, targets, metadata in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().squeeze(1)
            metadata = metadata.to(device)
            source_ids = (metadata[:, 0] > 0.5).long()
            out = model(inputs, source_ids=source_ids)

            batch_len = inputs.size(0)
            batch_indices = dataset_indices[cursor:cursor + batch_len]
            cursor += batch_len

            idx_batches.append(batch_indices)
            emb_batches.append(out["embedding"].detach().cpu().numpy().astype(np.float32))
            tgt_batches.append(targets.detach().cpu().numpy().astype(np.float32))
            src_batches.append(source_ids.detach().cpu().numpy().astype(np.int64))
            pooled_mean_batches.append(out["pooled_mean"].detach().cpu().numpy().astype(np.float32))
            pooled_scale_batches.append(out["pooled_scale"].detach().cpu().numpy().astype(np.float32))

    if cursor != len(dataset_indices):
        raise RuntimeError("Did not consume all subset indices when exporting embeddings")

    indices_arr = np.concatenate(idx_batches, axis=0)
    embedding_arr = np.concatenate(emb_batches, axis=0)
    target_arr = np.concatenate(tgt_batches, axis=0)
    source_arr = np.concatenate(src_batches, axis=0)
    pooled_mean_arr = np.concatenate(pooled_mean_batches, axis=0)
    pooled_scale_arr = np.concatenate(pooled_scale_batches, axis=0)

    np.savez_compressed(
        output_dir / "pending_embeddings.npz",
        embedding=embedding_arr,
        target=target_arr,
        source_id=source_arr,
        index=indices_arr,
    )

    head_payload = export_head_weights(model)
    head_payload["checkpoint"] = checkpoint_path.as_posix()
    torch.save(head_payload, output_dir / "pending_multihead_heads.pt")

    summary_path = output_dir / "prediction_summary.json"
    compute_summary(
        pooled_mean=pooled_mean_arr,
        pooled_scale=pooled_scale_arr,
        target=target_arr,
        summary_path=summary_path,
    )

    print(f"Saved embeddings to {output_dir / 'pending_embeddings.npz'}")
    print(f"Saved head-only weights to {output_dir / 'pending_multihead_heads.pt'}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

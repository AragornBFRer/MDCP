#!/usr/bin/env python3
"""Train a multi-head ResNet18 multispectral regressor on the MDCP PovertyMap split.

Architecture
-----------
- Shared backbone (ResNet18 multispectral) produces penultimate embeddings e(x).
- Heads:
    - pooled head: Gaussian (mu, scale)
    - per-source heads: one Gaussian head per domain (rural=0, urban=1)

Training objective
------------------
We optimize a weighted sum of negative log-likelihood losses:
    pooled_loss_weight * NLL(pooled_head) + source_loss_weight * NLL(domain_head)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as T

_REPO_ROOT = Path(__file__).resolve().parents[2]
# When executed as a script (e.g. `python wilds/poverty/train_resnet.py`), Python
# sets sys.path[0] to `wilds/poverty/`, so the repository root is not importable.
# Insert the repo root to ensure we import the *local* `wilds/` package rather
# than the external `wilds` (WILDS) dependency.
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from wilds.poverty.multihead_models import MultiHeadGaussianRegressor, export_head_weights


def _extend_sys_path(repo_root: Path) -> None:
    candidates = [
        repo_root / "external" / "wilds_upstream",  # clones to external/wilds_upstream
        repo_root / "get_data" / "get_wilds_data",  # alternative
    ]
    for wilds_repo in candidates:
        if wilds_repo.exists() and wilds_repo.as_posix() not in sys.path:
            sys.path.insert(0, wilds_repo.as_posix())


@torch.no_grad()
def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1 or y.numel() <= 1:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum())
    if denom.item() == 0:
        return float("nan")
    return float((vx * vy).sum() / denom)


def split_train_val(indices: Sequence[int], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=np.int64)
    perm = rng.permutation(idx)
    n_val = max(1, int(math.floor(val_frac * len(perm))))
    n_val = min(n_val, len(perm) - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


@dataclass
class TrainingConfig:
    repo_root: str
    data_root: str
    output_dir: str
    train_indices_path: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_frac: float
    seed: int
    num_workers: int
    betas: Tuple[float, float]
    density_head: str
    subset: str
    scale_floor: float
    nu_floor: Optional[float]


def urban_flag_to_source_id(metadata: torch.Tensor) -> torch.Tensor:
    """Map PovertyMap metadata -> source IDs.

    WILDS PovertyMap uses metadata[:, 0] as an urban indicator.
    We map: rural -> 0, urban -> 1.
    """

    if metadata.ndim != 2 or metadata.size(1) < 1:
        raise ValueError("Metadata must be a 2D tensor with an urban flag in column 0")
    return (metadata[:, 0] > 0.5).long()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gaussian_nll(targets: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Gaussian negative log-likelihood."""

    if not (targets.shape == mean.shape == scale.shape):
        raise ValueError("Targets, mean, and scale must share shape")
    scale = torch.clamp(scale, min=1e-6)
    var = scale ** 2
    log_term = torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
    sq_term = 0.5 * ((targets - mean) ** 2) / var
    return log_term + sq_term


def build_transforms() -> Dict[str, T.Compose]:
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])
    val_transform = T.Compose([])
    return {"train": train_transform, "val": val_transform}


def prepare_dataloaders(
    dataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    transforms: Dict[str, T.Compose],
) -> Tuple[DataLoader, DataLoader]:
    from wilds.datasets.wilds_dataset import WILDSSubset  # type: ignore

    train_subset = WILDSSubset(dataset, train_indices, transform=transforms["train"])
    val_subset = WILDSSubset(dataset, val_indices, transform=transforms["val"])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(
    path: Path,
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, torch.Tensor],
    epoch: int,
    best_metric: float,
    *,
    config: Dict[str, object],
) -> None:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "best_val_nll": best_metric,
        **config,
    }
    torch.save(payload, path)


def train_loop(
    model: MultiHeadGaussianRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    output_dir: Path,
    pooled_loss_weight: float,
    source_loss_weight: float,
    *,
    checkpoint_config: Dict[str, object],
) -> List[Dict[str, float]]:
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict[str, float]] = []
    best_val_nll = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        train_nll_sum = 0.0
        train_sq_error_sum = 0.0
        train_samples = 0
        train_preds: List[torch.Tensor] = []
        train_targets: List[torch.Tensor] = []

        for inputs, targets, metadata in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().squeeze(1)

            metadata = metadata.to(device)
            source_ids = urban_flag_to_source_id(metadata)

            optimizer.zero_grad()
            out = model(inputs, source_ids=source_ids)

            pooled_nll = gaussian_nll(targets, out["pooled_mean"], out["pooled_scale"])
            source_nll = gaussian_nll(targets, out["source_mean"], out["source_scale"])
            batch_nll = pooled_loss_weight * pooled_nll + source_loss_weight * source_nll
            loss = batch_nll.mean()
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            train_nll_sum += float(batch_nll.sum().item())
            mean_detached = out["pooled_mean"].detach()
            train_sq_error_sum += float(torch.sum((mean_detached - targets) ** 2).item())
            train_samples += batch_size
            train_preds.append(mean_detached.cpu())
            train_targets.append(targets.detach().cpu())

        train_nll = train_nll_sum / max(1, train_samples)
        train_rmse = math.sqrt(train_sq_error_sum / max(1, train_samples)) if train_samples > 0 else float("nan")
        if train_preds and train_targets:
            train_pearson = pearsonr_torch(torch.cat(train_preds), torch.cat(train_targets))
        else:
            train_pearson = float("nan")

        model.eval()
        val_nll_sum = 0.0
        val_sq_error_sum = 0.0
        val_samples = 0
        val_preds: List[torch.Tensor] = []
        val_targets: List[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets, metadata in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().squeeze(1)
                metadata = metadata.to(device)

                source_ids = urban_flag_to_source_id(metadata)
                out = model(inputs, source_ids=source_ids)

                pooled_nll = gaussian_nll(targets, out["pooled_mean"], out["pooled_scale"])
                source_nll = gaussian_nll(targets, out["source_mean"], out["source_scale"])
                batch_nll = pooled_loss_weight * pooled_nll + source_loss_weight * source_nll
                val_nll_sum += float(batch_nll.sum().item())
                mean_detached = out["pooled_mean"].detach()
                val_sq_error_sum += float(torch.sum((mean_detached - targets) ** 2).item())
                batch_size = targets.size(0)
                val_samples += batch_size
                val_preds.append(mean_detached.cpu())
                val_targets.append(targets.detach().cpu())

        val_nll = val_nll_sum / max(1, val_samples)
        val_rmse = math.sqrt(val_sq_error_sum / max(1, val_samples)) if val_samples > 0 else float("nan")
        if val_preds and val_targets:
            val_pearson = pearsonr_torch(torch.cat(val_preds), torch.cat(val_targets))
        else:
            val_pearson = float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "train_rmse": train_rmse,
                "train_pearson": train_pearson,
                "val_nll": val_nll,
                "val_rmse": val_rmse,
                "val_pearson": val_pearson,
            }
        )

        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(
                output_dir / "best_model.pth",
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                best_val_nll,
                config=checkpoint_config,
            )

        scheduler.step()

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_checkpoint(
            output_dir / "best_model.pth",
            model.state_dict(),
            optimizer.state_dict(),
            epochs - 1,
            history[-1]["val_nll"],
            config=checkpoint_config,
        )

    final_ckpt = {
        "model_state": best_state["model_state_dict"],
        "optimizer_state": best_state["optimizer_state_dict"],
        "history": history,
    }
    torch.save(final_ckpt, output_dir / "final_checkpoint.pth")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train resnet18_ms on MDCP poverty split")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-indices", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--pooled-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-loss-weight", type=float, default=1.0)
    parser.add_argument("--scale-floor", type=float, default=1e-3, help="Minimum scale (added after softplus)")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _extend_sys_path(repo_root)

    from wilds.datasets.poverty_dataset import PovertyMapDataset  # type: ignore
    from examples.models.resnet_multispectral import ResNet18  # type: ignore

    data_root = repo_root / "data"
    default_run_dir = repo_root / "eval_out" / "poverty" / "training" / "run_multihead_gaussian"
    output_dir = args.output_dir or default_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices_path = args.train_indices or (repo_root / "eval_out" / "poverty" / "splits" / "train_indices.json")
    train_indices = json.loads(train_indices_path.read_text(encoding="utf-8"))
    if not train_indices:
        raise RuntimeError("No training indices found")

    set_seed(args.seed)

    dataset = PovertyMapDataset(version="1.1", root_dir=data_root.as_posix(), download=False, split_scheme="official")

    if len(train_indices) < 2:
        raise RuntimeError(f"Training split yielded too few samples ({len(train_indices)})")

    train_arr, val_arr = split_train_val(train_indices, val_frac=args.val_frac, seed=args.seed)
    print(f"Multihead training: {len(train_arr)} train / {len(val_arr)} val samples (from {len(train_indices)} eligible indices)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = ResNet18(num_classes=None, num_channels=8)
    model = MultiHeadGaussianRegressor(
        backbone=backbone,
        embedding_dim=int(backbone.d_out),
        source_ids=(0, 1),
        scale_floor=float(args.scale_floor),
    )
    model.to(device)

    transforms = build_transforms()
    train_loader, val_loader = prepare_dataloaders(
        dataset=dataset,
        train_indices=train_arr,
        val_indices=val_arr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
    )

    checkpoint_config = {
        "multihead": True,
        "task": "regression",
        "head_type": "gaussian",
        "source_ids": [0, 1],
        "scale_floor": float(args.scale_floor),
        "pooled_loss_weight": float(args.pooled_loss_weight),
        "source_loss_weight": float(args.source_loss_weight),
    }

    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        output_dir=output_dir,
        pooled_loss_weight=float(args.pooled_loss_weight),
        source_loss_weight=float(args.source_loss_weight),
        checkpoint_config=checkpoint_config,
    )

    config = TrainingConfig(
        repo_root=str(repo_root),
        data_root=str(data_root),
        output_dir=str(output_dir),
        train_indices_path=str(train_indices_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        betas=tuple(args.betas),
        density_head="gaussian_multihead",
        subset="all",
        scale_floor=args.scale_floor,
        nu_floor=None,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    # Head-only weights are exported during prediction; training stores the full checkpoint.

    print(f"Training complete. History length: {len(history)} epochs")
    print(f"Artifacts stored in {output_dir}")


if __name__ == "__main__":
    main()

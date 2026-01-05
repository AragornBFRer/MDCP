from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

try:
    from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]
    _autocast_signature = inspect.signature(torch_autocast)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore
    _autocast_signature = None

try:
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
    _grad_scaler_signature = inspect.signature(TorchGradScaler.__init__)
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    _grad_scaler_signature = None

_AUTocast_SUPPORTS_DEVICE_TYPE = _autocast_signature is not None and "device_type" in _autocast_signature.parameters
_GRAD_SCALER_SUPPORTS_DEVICE_TYPE = (
    _grad_scaler_signature is not None and "device_type" in _grad_scaler_signature.parameters
)

from .data import DataLoaders, create_dataloaders
from .models import ArchName, build_model, load_checkpoint
from .multihead_models import build_multihead_model


def _resolve_region_ids_for_grouping(region_id_to_name: Sequence[str]) -> tuple[int, int]:
    """Return (europe_id, americas_id) from a WILDS region mapping list.

    The WILDS FMoW dataset encodes regions as integer ids with a corresponding
    `dataset.metadata_map['region']` list that maps id -> name.

    Raises on ambiguity or missing names (no silent fallback).
    """

    europe_candidates = [i for i, name in enumerate(region_id_to_name) if str(name).strip().lower() == "europe"]
    americas_candidates = [
        i
        for i, name in enumerate(region_id_to_name)
        if str(name).strip().lower() in {"americas", "america"}
    ]

    if len(europe_candidates) != 1:
        raise ValueError(
            "Expected exactly one 'Europe' entry in dataset.metadata_map['region'], "
            f"found indices={europe_candidates} names={list(region_id_to_name)}"
        )
    if len(americas_candidates) != 1:
        raise ValueError(
            "Expected exactly one 'Americas' entry in dataset.metadata_map['region'], "
            f"found indices={americas_candidates} names={list(region_id_to_name)}"
        )
    return int(europe_candidates[0]), int(americas_candidates[0])


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def _to_device(batch, device: torch.device):
    if len(batch) == 3:
        inputs, targets, metadata = batch
        indices = None
    else:
        inputs, targets, metadata, indices = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    metadata = metadata.to(device, non_blocking=True)
    if indices is not None:
        indices = torch.as_tensor(indices, device=device)
    return inputs, targets, metadata, indices


def _split_for_validation(indices: Sequence[int], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.array(indices, dtype=np.int64)
    perm = rng.permutation(indices)
    n_val = max(1, int(math.floor(val_frac * len(perm))))
    if n_val >= len(perm):
        n_val = max(1, len(perm) // 5)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _make_grad_scaler(device_type: str, enabled: bool):
    scaler_kwargs = {"enabled": enabled}
    if _GRAD_SCALER_SUPPORTS_DEVICE_TYPE:
        scaler_kwargs["device_type"] = device_type
    return TorchGradScaler(**scaler_kwargs)


class Trainer:
    def __init__(
        self,
        dataset,
        train_indices: Sequence[int],
        holdout_indices: Optional[Sequence[int]],
        output_dir: Path,
        arch: ArchName = "densenet121",
        batch_size: int = 32,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        step_size: int = 20,
        gamma: float = 0.1,
        val_frac: float = 0.1,
        seed: int = 0,
        num_workers: int = 4,
        balance_regions: bool = True,
        use_amp: bool = True,
        resume_checkpoint: Optional[Path] = None,
        wandb_run: Optional[Any] = None,
        wandb_watch: bool = False,
        multihead: bool = False,
        pooled_loss_weight: float = 1.0,
        source_loss_weight: float = 1.0,
        group_others_head: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.wandb_run = wandb_run
        self._wandb_watch = wandb_watch

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        train_idx, val_idx = _split_for_validation(train_indices, val_frac, seed)
        np.save(self.output_dir / "train_idx_used.npy", train_idx)
        np.save(self.output_dir / "val_idx.npy", val_idx)

        self.multihead = bool(multihead)
        self.pooled_loss_weight = float(pooled_loss_weight)
        self.source_loss_weight = float(source_loss_weight)
        self.group_others_head = bool(group_others_head)
        self.region_id_to_name: Optional[List[str]] = None

        resume_payload: Optional[Dict[str, Any]] = None
        if resume_checkpoint is not None and resume_checkpoint.exists():
            # Load metadata early so we can rebuild a compatible model (grouped vs ungrouped heads).
            resume_payload = torch.load(resume_checkpoint, map_location="cpu")
            if bool(resume_payload.get("multihead", False)) != bool(self.multihead):
                raise ValueError(
                    "Resume checkpoint multihead flag does not match current run. "
                    f"checkpoint.multihead={resume_payload.get('multihead')} current.multihead={self.multihead}"
                )
            if "group_others_head" in resume_payload:
                self.group_others_head = bool(resume_payload.get("group_others_head"))
            if resume_payload.get("region_id_to_name") is not None:
                self.region_id_to_name = list(resume_payload.get("region_id_to_name"))

        if self.multihead:
            if resume_payload is not None and resume_payload.get("source_ids") is not None:
                region_ids = [int(s) for s in resume_payload.get("source_ids")]
            else:
                region_ids = np.unique(dataset.metadata_array[:, 0].cpu().numpy()).astype(int).tolist()

            source_id_to_group = None
            if resume_payload is not None and resume_payload.get("source_id_to_group") is not None:
                source_id_to_group = {int(k): str(v) for k, v in resume_payload.get("source_id_to_group").items()}
                self.group_others_head = True
            elif self.group_others_head:
                if not getattr(dataset, "metadata_map", None) or dataset.metadata_map.get("region") is None:
                    raise ValueError(
                        "Grouped-head training requires dataset.metadata_map['region'] to resolve region names."
                    )
                region_id_to_name = list(dataset.metadata_map["region"])
                self.region_id_to_name = list(region_id_to_name)
                europe_id, americas_id = _resolve_region_ids_for_grouping(region_id_to_name)
                source_id_to_group = {}
                for sid in region_ids:
                    if int(sid) == int(europe_id):
                        source_id_to_group[int(sid)] = "Europe"
                    elif int(sid) == int(americas_id):
                        source_id_to_group[int(sid)] = "Americas"
                    else:
                        source_id_to_group[int(sid)] = "Other"

            self.model = build_multihead_model(
                arch=arch,
                num_classes=int(dataset.n_classes),
                source_ids=region_ids,
                pretrained_backbone=True,
                source_id_to_group=source_id_to_group,
            )
        else:
            self.model = build_model(arch=arch, num_classes=int(dataset.n_classes), pretrained=True)
        self.model.to(self.device)
        if self.wandb_run is not None and self._wandb_watch:
            try:
                self.wandb_run.watch(self.model, log="gradients", log_freq=100)
            except Exception:
                pass

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        scaler_device_type = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
        self.scaler = _make_grad_scaler(device_type=scaler_device_type, enabled=self.use_amp)

        loaders = create_dataloaders(
            dataset=dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            holdout_indices=holdout_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            balance_regions=balance_regions,
        )
        self.loaders = loaders
        self.epochs = epochs
        self.arch = arch

        self.start_epoch = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_acc = 0.0

        if resume_checkpoint is not None and resume_checkpoint.exists():
            self._load_resume_state(resume_checkpoint)

    def _load_resume_state(self, checkpoint_path: Path) -> None:
        checkpoint = load_checkpoint(self.model, checkpoint_path.as_posix(), self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        default_scaler_state = _make_grad_scaler(
            device_type=self.device.type if self.device.type in {"cuda", "cpu"} else "cuda",
            enabled=self.use_amp,
        ).state_dict()
        self.scaler.load_state_dict(checkpoint.get("scaler_state", default_scaler_state))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))

    def _run_epoch(self, loader: torch.utils.data.DataLoader, training: bool) -> EpochMetrics:
        mode = "train" if training else "eval"
        self.model.train(mode == "train")
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for batch in loader:
            inputs, targets, metadata, _ = _to_device(batch, self.device)
            amp_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
            autocast_kwargs = {"enabled": self.use_amp}
            if _AUTocast_SUPPORTS_DEVICE_TYPE:
                autocast_kwargs["device_type"] = amp_device
            with torch_autocast(**autocast_kwargs):
                if self.multihead:
                    out = self.model(inputs)
                    pooled_logits = out["pooled_logits"]
                    emb = out["embedding"]
                    region_ids = metadata[:, 0].long().reshape(-1)
                    region_logits = self.model.forward_selected_source_logits(emb, region_ids)
                    loss_pooled = self.criterion(pooled_logits, targets)
                    loss_region = self.criterion(region_logits, targets)
                    loss = self.pooled_loss_weight * loss_pooled + self.source_loss_weight * loss_region
                    logits_for_acc = pooled_logits
                else:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
                    logits_for_acc = logits

            if training:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits_for_acc.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return EpochMetrics(loss=avg_loss, accuracy=acc)

    def fit(self) -> Dict[str, float]:
        history: List[Dict[str, float]] = []
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._run_epoch(self.loaders.train, training=True)
            val_metrics = None
            if self.loaders.val is not None:
                val_metrics = self._run_epoch(self.loaders.val, training=False)
                if val_metrics.accuracy > self.best_val_acc:
                    self.best_val_acc = val_metrics.accuracy
                    self.best_state = {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                        "epoch": epoch,
                        "best_val_acc": self.best_val_acc,
                        "arch": self.arch,
                        "multihead": self.multihead,
                        "pooled_loss_weight": self.pooled_loss_weight,
                        "source_loss_weight": self.source_loss_weight,
                        "source_ids": getattr(self.model, 'source_ids', None),
                        "group_others_head": self.group_others_head,
                        "source_id_to_group": getattr(self.model, 'source_id_to_group', None),
                        "region_id_to_name": self.region_id_to_name,
                    }
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
            }
            if val_metrics is not None:
                record.update({
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "best_val_accuracy": self.best_val_acc,
                })
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_metrics.loss,
                "train/accuracy": train_metrics.accuracy,
            }
            if val_metrics is not None:
                wandb_log.update({
                    "val/loss": val_metrics.loss,
                    "val/accuracy": val_metrics.accuracy,
                    "val/best_accuracy": self.best_val_acc,
                })
            history.append(record)
            log_path = self.output_dir / "training_log.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            if self.wandb_run is not None:
                self.wandb_run.log(wandb_log, step=epoch)

        if self.best_state is None:
            self.best_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                "epoch": self.epochs - 1,
                "best_val_acc": self.best_val_acc,
                "arch": self.arch,
                "multihead": self.multihead,
                "pooled_loss_weight": self.pooled_loss_weight,
                "source_loss_weight": self.source_loss_weight,
                "source_ids": getattr(self.model, 'source_ids', None),
                "group_others_head": self.group_others_head,
                "source_id_to_group": getattr(self.model, 'source_id_to_group', None),
                "region_id_to_name": self.region_id_to_name,
            }

        checkpoint_path = self.output_dir / "checkpoint_best.pt"
        torch.save(self.best_state, checkpoint_path)
        if self.wandb_run is not None:
            self.wandb_run.summary["best_val_accuracy"] = self.best_val_acc
            if self.best_state is not None:
                self.wandb_run.summary["best_epoch"] = self.best_state.get("epoch", self.epochs - 1)
        return {"checkpoint": checkpoint_path.as_posix(), "best_val_accuracy": self.best_val_acc}


def run_training(
    dataset,
    train_indices: Sequence[int],
    holdout_indices: Optional[Sequence[int]],
    output_dir: Path,
    arch: ArchName = "densenet121",
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    step_size: int = 20,
    gamma: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 4,
    balance_regions: bool = True,
    use_amp: bool = True,
    resume_checkpoint: Optional[Path] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_notes: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_watch: bool = False,
    wandb_config: Optional[Dict[str, Any]] = None,
    multihead: bool = False,
    pooled_loss_weight: float = 1.0,
    source_loss_weight: float = 1.0,
    group_others_head: bool = True,
) -> Dict[str, float]:
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Weights & Biases not installed. Install it inside the mdcp environment with 'pip install wandb'."
            ) from exc

        wandb_kwargs: Dict[str, Any] = {}
        if wandb_project is not None:
            wandb_kwargs["project"] = wandb_project
        if wandb_entity is not None:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_run_name is not None:
            wandb_kwargs["name"] = wandb_run_name
        if wandb_tags is not None:
            wandb_kwargs["tags"] = list(wandb_tags)
        if wandb_notes is not None:
            wandb_kwargs["notes"] = wandb_notes
        if wandb_mode is not None:
            wandb_kwargs["mode"] = wandb_mode

        wandb_run = wandb.init(**wandb_kwargs)
        config_payload: Dict[str, Any] = {
            "arch": arch,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "step_size": step_size,
            "lr_gamma": gamma,
            "val_frac": val_frac,
            "seed": seed,
            "num_workers": num_workers,
            "balance_regions": balance_regions,
            "use_amp": use_amp,
            "multihead": bool(multihead),
            "pooled_loss_weight": float(pooled_loss_weight),
            "source_loss_weight": float(source_loss_weight),
            "group_others_head": bool(group_others_head),
        }
        if wandb_config is not None:
            config_payload.update(wandb_config)
        wandb_run.config.update(config_payload, allow_val_change=True)

    trainer = Trainer(
        dataset=dataset,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        output_dir=output_dir,
        arch=arch,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        val_frac=val_frac,
        seed=seed,
        num_workers=num_workers,
        balance_regions=balance_regions,
        use_amp=use_amp,
        resume_checkpoint=resume_checkpoint,
        wandb_run=wandb_run,
        wandb_watch=wandb_watch,
        multihead=multihead,
        pooled_loss_weight=pooled_loss_weight,
        source_loss_weight=source_loss_weight,
        group_others_head=group_others_head,
    )
    try:
        result = trainer.fit()
        return result
    finally:
        if wandb_run is not None:
            wandb_run.finish()

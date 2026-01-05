from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
import torch

if __package__ is None or __package__ == "":  # script execution fallback
    package_root = Path(__file__).resolve().parents[1]
    if package_root.as_posix() not in sys.path:
        sys.path.insert(0, package_root.as_posix())
    from pipeline.common import load_fmow_dataset  # type: ignore
    from pipeline.predict import run_holdout_prediction  # type: ignore
    from pipeline.split import create_year_filtered_split  # type: ignore
    from pipeline.train import run_training  # type: ignore
else:
    from .common import load_fmow_dataset
    from .predict import run_holdout_prediction
    from .split import create_year_filtered_split
    from .train import run_training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMoW 2016 pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Create 2016-only splits")
    split_parser.add_argument("--root", type=Path, required=True, help="Path to fmow_v1.1 root")
    split_parser.add_argument("--wilds-repo", type=Path, default=None, help="Path to get_wilds_data repository (optional)")
    split_parser.add_argument("--output", type=Path, required=True, help="Directory to store split indices")
    split_parser.add_argument("--train-frac", type=float, default=0.375, help="Per-group training fraction")
    split_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    split_parser.add_argument("--target-year", type=int, default=2016, help="Year to filter on")
    split_parser.add_argument("--group-field", type=str, default="region", help="Metadata column used for grouping")

    train_parser = subparsers.add_parser("train", help="Train a classifier on the 2016 subset")
    train_parser.add_argument("--root", type=Path, required=True, help="Path to fmow_v1.1 root")
    train_parser.add_argument("--wilds-repo", type=Path, default=None)
    train_parser.add_argument("--train-idx", type=Path, required=True, help="Path to train_idx.npy")
    train_parser.add_argument("--holdout-idx", type=Path, default=None, help="Path to holdout_idx.npy")
    train_parser.add_argument("--output", type=Path, required=True, help="Training output directory")
    train_parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "resnet50"])
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--step-size", type=int, default=20)
    train_parser.add_argument("--gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay")
    train_parser.add_argument("--val-frac", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers - how many subprocesses to use for data loading")
    train_parser.add_argument("--balance-regions", action="store_true", default=False)
    train_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU")
    train_parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb-project", type=str, default=None)
    train_parser.add_argument("--wandb-entity", type=str, default=None)
    train_parser.add_argument("--wandb-name", type=str, default=None)
    train_parser.add_argument("--wandb-tags", nargs="*", default=None)
    train_parser.add_argument("--wandb-notes", type=str, default=None)
    train_parser.add_argument("--wandb-mode", type=str, choices=["online", "offline"], default=None, help="Set wandb.init mode")
    train_parser.add_argument("--wandb-watch", action="store_true", help="Log gradients/parameters via wandb.watch")
    train_parser.add_argument(
        "--multihead",
        action="store_true",
        help="Train a multi-head model (per-region heads + pooled head) instead of a single-head classifier",
    )
    train_parser.add_argument(
        "--pooled-loss-weight",
        type=float,
        default=1.0,
        help="Weight for pooled-head cross entropy loss when --multihead is enabled",
    )
    train_parser.add_argument(
        "--source-loss-weight",
        type=float,
        default=1.0,
        help="Weight for per-source-head cross entropy loss when --multihead is enabled",
    )
    train_parser.add_argument(
        "--group-others-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --multihead is enabled, use a grouped-head design: keep 'Europe' and 'Americas' as separate heads "
            "and share a single head across all other regions. Disable with --no-group-others-head."
        ),
    )

    predict_parser = subparsers.add_parser("predict", help="Run holdout predictions")
    predict_parser.add_argument("--root", type=Path, required=True)
    predict_parser.add_argument("--wilds-repo", type=Path, default=None)
    predict_parser.add_argument("--holdout-idx", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--arch", type=str, default=None, choices=["densenet121", "resnet50"])
    predict_parser.add_argument("--batch-size", type=int, default=64)
    predict_parser.add_argument("--num-workers", type=int, default=4)
    predict_parser.add_argument("--save-embeddings", action="store_true", help="Save penultimate embeddings to NPZ")
    predict_parser.add_argument("--embeddings-npz", type=Path, default=None, help="Optional path for embeddings NPZ")

    return parser


def cmd_split(args: argparse.Namespace) -> dict:
    dataset = load_fmow_dataset(args.root, args.wilds_repo)
    result = create_year_filtered_split(
        dataset=dataset,
        target_year=args.target_year,
        train_frac=args.train_frac,
        seed=args.seed,
        group_field=args.group_field,
        output_dir=args.output,
    )
    print(json.dumps(result["summary"], indent=2))
    return result


def cmd_train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    dataset = load_fmow_dataset(args.root, args.wilds_repo)
    train_idx = np.load(args.train_idx)
    holdout_idx = np.load(args.holdout_idx) if args.holdout_idx is not None else None
    metrics = run_training(
        dataset=dataset,
        train_indices=train_idx,
        holdout_indices=holdout_idx,
        output_dir=args.output,
        arch=args.arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        balance_regions=args.balance_regions,
        use_amp=not args.no_amp,
        resume_checkpoint=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_watch=args.wandb_watch,
        multihead=args.multihead,
        pooled_loss_weight=args.pooled_loss_weight,
        source_loss_weight=args.source_loss_weight,
        group_others_head=args.group_others_head,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_predict(args: argparse.Namespace) -> dict:
    dataset = load_fmow_dataset(args.root, args.wilds_repo)
    holdout_idx = np.load(args.holdout_idx)
    summary = run_holdout_prediction(
        dataset=dataset,
        holdout_indices=holdout_idx,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        arch=args.arch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_embeddings=args.save_embeddings,
        embeddings_npz=args.embeddings_npz,
    )
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[Iterable[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "split":
        return cmd_split(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import build_transforms, make_subset
from .models import ArchName, build_model, load_checkpoint
from .multihead_models import build_multihead_model


def _build_holdout_loader(dataset, holdout_indices: Sequence[int], batch_size: int, num_workers: int) -> DataLoader:
    transform = build_transforms(train=False)
    subset = make_subset(dataset, holdout_indices, transform, return_index=True)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
def run_holdout_prediction(
    dataset,
    holdout_indices: Sequence[int],
    checkpoint_path: Path,
    output_dir: Path,
    arch: Optional[ArchName] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    save_embeddings: bool = False,
    embeddings_npz: Optional[Path] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch_name = arch or checkpoint.get("arch", "densenet121")
    is_multihead = bool(checkpoint.get("multihead", False))
    source_ids: Optional[list[int]] = None
    source_id_to_group = checkpoint.get("source_id_to_group")

    if save_embeddings and not is_multihead:
        raise ValueError(
            "Embedding export is supported only for multihead checkpoints. "
            "Train with --multihead so embeddings are emitted directly by the model."
        )

    if is_multihead:
        source_ids = checkpoint.get("source_ids")
        if source_ids is None:
            source_ids = np.unique(dataset.metadata_array[:, 0].cpu().numpy()).astype(int).tolist()
        source_ids = [int(s) for s in source_ids]
        model = build_multihead_model(
            arch=arch_name,
            num_classes=int(dataset.n_classes),
            source_ids=source_ids,
            pretrained_backbone=False,
            source_id_to_group=source_id_to_group,
        )
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = build_model(arch=arch_name, num_classes=int(dataset.n_classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    loader = _build_holdout_loader(dataset, holdout_indices, batch_size=batch_size, num_workers=num_workers)

    embedding_batches: list[torch.Tensor] = []

    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_indices: list[int] = []
    all_regions: list[int] = []
    metadata_records = []

    region_map = dataset.metadata_map.get("region") if dataset.metadata_map else None

    with torch.no_grad():
        for batch in loader:
            inputs, targets, metadata, indices = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            if is_multihead:
                out = model(inputs)
                logits = out["pooled_logits"]
                if save_embeddings:
                    embedding_batches.append(out["embedding"].detach().cpu())
            else:
                logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(targets.cpu().tolist())
            all_indices.extend(indices.cpu().tolist())
            all_regions.extend(metadata[:, 0].cpu().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)

    embeddings_arr = None
    if save_embeddings:
        if not embedding_batches:
            raise RuntimeError("Embedding extraction requested but no embeddings were captured")
        embeddings_arr = torch.cat(embedding_batches, dim=0).numpy()

    for idx, label, region in zip(all_indices, all_labels, all_regions):
        metadata_idx = int(dataset.full_idxs[int(idx)])
        row = dataset.metadata.iloc[metadata_idx]
        metadata_records.append({
            "dataset_idx": int(idx),
            "metadata_idx": metadata_idx,
            "label": int(label),
            "region_id": int(region),
            "region": region_map[region] if region_map is not None and region < len(region_map) else str(region),
            "timestamp": row["timestamp"],
            "country_code": row.get("country_code", ""),
            "lon": row.get("longitude"),
            "lat": row.get("latitude"),
        })

    metadata_df = pd.DataFrame.from_records(metadata_records)
    preds = probs_arr.argmax(axis=1)
    accuracy = float((preds == metadata_df["label"].to_numpy()).mean())

    metadata_df.to_csv(output_dir / "holdout_metadata.csv", index=False)
    np.save(output_dir / "holdout_probabilities.npy", probs_arr)
    np.save(output_dir / "holdout_logits.npy", logits_arr)

    # Save lightweight head-only weights for CPU-side evaluation on cached embeddings.
    if is_multihead:
        if source_ids is None:
            raise RuntimeError("Multihead checkpoint detected but source_ids could not be resolved")

        region_id_to_name = None
        if getattr(dataset, "metadata_map", None) is not None:
            region_id_to_name = dataset.metadata_map.get("region")

        if hasattr(model, "state_dict_per_source_head"):
            per_source_heads = model.state_dict_per_source_head()  # type: ignore[attr-defined]
        else:  # pragma: no cover
            per_source_heads = {str(int(sid)): model.source_heads[f"source_{int(sid)}"].state_dict() for sid in source_ids}

        head_payload = {
            "arch": arch_name,
            "checkpoint": checkpoint_path.as_posix(),
            "multihead": True,
            "num_classes": int(dataset.n_classes),
            "source_ids": [int(s) for s in source_ids],
            "group_others_head": bool(checkpoint.get("group_others_head", False)),
            "source_id_to_group": source_id_to_group,
            "region_id_to_name": list(region_id_to_name) if region_id_to_name is not None else None,
            "pooled_head": model.pooled_head.state_dict(),
            "source_heads": per_source_heads,
        }
        torch.save(head_payload, output_dir / "holdout_multihead_heads.pt")

    if embeddings_arr is not None:
        emb_path = embeddings_npz or (output_dir / "holdout_embeddings.npz")
        np.savez_compressed(
            emb_path,
            embedding=embeddings_arr.astype(np.float32),
            label=np.asarray(all_labels, dtype=np.int64),
            source_id=np.asarray(all_regions, dtype=np.int64),
            index=np.asarray(all_indices, dtype=np.int64),
        )
        meta = {
            "arch": arch_name,
            "checkpoint": checkpoint_path.as_posix(),
            "multihead": bool(is_multihead),
            "source_ids": source_ids if source_ids is not None else checkpoint.get("source_ids"),
            "group_others_head": bool(checkpoint.get("group_others_head", False)),
            "source_id_to_group": source_id_to_group,
            "region_id_to_name": list(region_map) if region_map is not None else None,
            "n_embeddings": int(embeddings_arr.shape[0]),
            "embedding_dim": int(embeddings_arr.shape[1]),
            "num_classes": int(dataset.n_classes),
        }
        with (output_dir / "holdout_embeddings_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "arch": arch_name,
        "n_examples": int(len(metadata_df)),
        "accuracy": accuracy,
    }
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

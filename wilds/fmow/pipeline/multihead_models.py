from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence

import torch
from torch import nn
from torchvision import models

ArchName = Literal["densenet121", "resnet50"]


def _instantiate_with_weights(model_fn, weight_qualname: str, pretrained: bool):
    if pretrained:
        try:
            module_name, enum_name = weight_qualname.split(".")
            weights_enum = getattr(models, module_name)
            for part in enum_name.split("."):
                weights_enum = getattr(weights_enum, part)
            return model_fn(weights=weights_enum)
        except (AttributeError, ValueError):
            return model_fn(pretrained=True)
    try:
        return model_fn(weights=None)
    except TypeError:
        return model_fn(pretrained=False)


def build_backbone(arch: ArchName, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (backbone, embedding_dim) where backbone(x) -> embedding."""

    if arch == "densenet121":
        base = _instantiate_with_weights(models.densenet121, "DenseNet121_Weights.IMAGENET1K_V1", pretrained)
        emb_dim = int(base.classifier.in_features)
        base.classifier = nn.Identity()
        return base, emb_dim
    if arch == "resnet50":
        if pretrained:
            try:
                weights_enum = models.ResNet50_Weights.IMAGENET1K_V2  # type: ignore[attr-defined]
                base = models.resnet50(weights=weights_enum)
            except AttributeError:
                base = _instantiate_with_weights(models.resnet50, "ResNet50_Weights.IMAGENET1K_V1", True)
        else:
            base = _instantiate_with_weights(models.resnet50, "ResNet50_Weights.IMAGENET1K_V1", False)
        emb_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
        return base, emb_dim

    raise ValueError(f"Unsupported architecture: {arch}")


@dataclass(frozen=True)
class HeadSpec:
    name: str
    source_id: Optional[int]


class MultiHeadClassifier(nn.Module):
    """Shared backbone + per-source heads + pooled head.

    Notes
    -----
    - `forward_embeddings(x)` returns embeddings e(x).
    - `forward_all_heads_from_embeddings(e)` returns pooled logits and per-head logits.
    - During training, use `forward_selected_source_logits(e, source_ids)` for the per-source loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        num_classes: int,
        source_ids: Sequence[int],
        source_id_to_group: Optional[Mapping[int, str]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.source_ids = list(int(s) for s in source_ids)
        if len(self.source_ids) == 0:
            raise ValueError("source_ids must be non-empty")

        normalized_group: Optional[Dict[int, str]] = None
        if source_id_to_group is not None:
            normalized_group = {int(k): str(v) for k, v in source_id_to_group.items()}
            missing = set(self.source_ids) - set(normalized_group.keys())
            extra = set(normalized_group.keys()) - set(self.source_ids)
            if missing:
                raise ValueError(f"source_id_to_group missing mappings for source ids: {sorted(missing)}")
            if extra:
                raise ValueError(f"source_id_to_group contains unknown source ids: {sorted(extra)}")
        self.source_id_to_group = normalized_group

        # Heads
        self.pooled_head = nn.Linear(self.embedding_dim, self.num_classes)

        if self.source_id_to_group is None:
            self.source_heads = nn.ModuleDict(
                {f"source_{sid}": nn.Linear(self.embedding_dim, self.num_classes) for sid in self.source_ids}
            )
            self.group_heads: Optional[nn.ModuleDict] = None
        else:
            # Preserve a stable group order based on the order of `source_ids`.
            group_names: List[str] = []
            for sid in self.source_ids:
                group = self.source_id_to_group[int(sid)]
                if group not in group_names:
                    group_names.append(group)
            self.group_heads = nn.ModuleDict(
                {f"group_{name}": nn.Linear(self.embedding_dim, self.num_classes) for name in group_names}
            )
            self.source_heads = nn.ModuleDict()

    def _head_for_source_id(self, sid: int) -> nn.Module:
        sid = int(sid)
        if self.source_id_to_group is None:
            return self.source_heads[f"source_{sid}"]
        group = self.source_id_to_group[sid]
        if self.group_heads is None:  # pragma: no cover
            raise RuntimeError("Grouped heads requested but group_heads not initialized")
        return self.group_heads[f"group_{group}"]

    def state_dict_per_source_head(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a per-raw-source state_dict mapping.

        This keeps downstream tooling (CPU evaluation, artifact formats) stable even when
        multiple source ids share a single underlying head during training.
        """

        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for sid in self.source_ids:
            out[str(int(sid))] = self._head_for_source_id(int(sid)).state_dict()
        return out

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_all_heads_from_embeddings(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_logits = self.pooled_head(emb)
        per_head = []
        for sid in self.source_ids:
            per_head.append(self._head_for_source_id(int(sid))(emb))
        source_logits = torch.stack(per_head, dim=1)  # (n, K, C)
        return pooled_logits, source_logits

    def forward_selected_source_logits(self, emb: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for the head matching each sample's `source_ids`.

        emb: (n, d)
        source_ids: (n,)
        returns: (n, C)
        """

        if source_ids.ndim != 1:
            source_ids = source_ids.reshape(-1)
        n = emb.size(0)
        out = emb.new_zeros((n, self.num_classes))
        for sid in self.source_ids:
            mask = source_ids == int(sid)
            if torch.any(mask):
                head = self._head_for_source_id(int(sid))
                out[mask] = head(emb[mask])
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.forward_embeddings(x)
        pooled_logits, source_logits = self.forward_all_heads_from_embeddings(emb)
        return {
            "embedding": emb,
            "pooled_logits": pooled_logits,
            "source_logits": source_logits,
        }


def build_multihead_model(
    arch: ArchName,
    num_classes: int,
    source_ids: Sequence[int],
    pretrained_backbone: bool = True,
    source_id_to_group: Optional[Mapping[int, str]] = None,
) -> MultiHeadClassifier:
    backbone, emb_dim = build_backbone(arch=arch, pretrained=pretrained_backbone)
    return MultiHeadClassifier(
        backbone=backbone,
        embedding_dim=emb_dim,
        num_classes=int(num_classes),
        source_ids=list(source_ids),
        source_id_to_group=source_id_to_group,
    )

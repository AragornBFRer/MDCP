from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GaussianParams:
    mean: torch.Tensor
    scale: torch.Tensor


class GaussianParamHead(nn.Module):
    """Linear head producing (mu, scale) with a positive scale.

    The raw scale output is transformed via softplus + scale_floor.
    """

    def __init__(self, in_features: int, *, scale_floor: float = 1e-3) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_features), 2)
        self.softplus = nn.Softplus()
        self.scale_floor = float(scale_floor)

    def forward(self, emb: torch.Tensor) -> GaussianParams:
        raw = self.linear(emb)
        mu = raw[:, 0]
        raw_scale = raw[:, 1]
        scale = self.softplus(raw_scale) + self.scale_floor
        return GaussianParams(mean=mu, scale=scale)


class MultiHeadGaussianRegressor(nn.Module):
    """Shared backbone + per-domain Gaussian heads + pooled head.

    Sources are encoded by integer IDs (e.g., rural=0, urban=1).

    Forward returns:
      - embedding: (n, d)
      - pooled: GaussianParams for pooled head
      - source: GaussianParams for the matching per-source head
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        *,
        source_ids: Sequence[int],
        scale_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.source_ids = [int(s) for s in source_ids]
        if not self.source_ids:
            raise ValueError("source_ids must be non-empty")

        self.pooled_head = GaussianParamHead(self.embedding_dim, scale_floor=scale_floor)
        self.source_heads = nn.ModuleDict(
            {f"source_{sid}": GaussianParamHead(self.embedding_dim, scale_floor=scale_floor) for sid in self.source_ids}
        )

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_pooled_from_embeddings(self, emb: torch.Tensor) -> GaussianParams:
        return self.pooled_head(emb)

    def forward_selected_source_from_embeddings(self, emb: torch.Tensor, source_ids: torch.Tensor) -> GaussianParams:
        if source_ids.ndim != 1:
            source_ids = source_ids.reshape(-1)
        n = emb.size(0)
        mean = emb.new_zeros((n,))
        scale = emb.new_zeros((n,))
        for sid in self.source_ids:
            mask = source_ids == int(sid)
            if torch.any(mask):
                params = self.source_heads[f"source_{sid}"](emb[mask])
                mean[mask] = params.mean
                scale[mask] = params.scale
        return GaussianParams(mean=mean, scale=scale)

    def forward(self, x: torch.Tensor, source_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        emb = self.forward_embeddings(x)
        pooled = self.forward_pooled_from_embeddings(emb)
        out: Dict[str, torch.Tensor] = {
            "embedding": emb,
            "pooled_mean": pooled.mean,
            "pooled_scale": pooled.scale,
        }
        if source_ids is not None:
            src = self.forward_selected_source_from_embeddings(emb, source_ids)
            out.update({"source_mean": src.mean, "source_scale": src.scale})
        return out


def export_head_weights(model: MultiHeadGaussianRegressor) -> Dict[str, object]:
    """Export CPU-friendly head-only state dicts."""

    return {
        "multihead": True,
        "task": "regression",
        "head_type": "gaussian",
        "embedding_dim": int(model.embedding_dim),
        "scale_floor": float(model.pooled_head.scale_floor),
        "source_ids": [int(s) for s in model.source_ids],
        "pooled_head": model.pooled_head.state_dict(),
        "source_heads": {
            str(int(sid)): model.source_heads[f"source_{int(sid)}"].state_dict() for sid in model.source_ids
        },
    }

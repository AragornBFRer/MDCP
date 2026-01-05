from __future__ import annotations

from typing import Literal

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


def build_model(arch: ArchName = "densenet121", num_classes: int = 62, pretrained: bool = True) -> nn.Module:
    if arch == "densenet121":
        model = _instantiate_with_weights(models.densenet121, "DenseNet121_Weights.IMAGENET1K_V1", pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        if pretrained:
            try:
                weights_enum = models.ResNet50_Weights.IMAGENET1K_V2  # type: ignore[attr-defined]
                model = models.resnet50(weights=weights_enum)
            except AttributeError:
                model = _instantiate_with_weights(models.resnet50, "ResNet50_Weights.IMAGENET1K_V1", True)
        else:
            model = _instantiate_with_weights(models.resnet50, "ResNet50_Weights.IMAGENET1K_V1", False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - guard for unsupported arch
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def freeze_backbone(model: nn.Module, train_last_layer_only: bool = False) -> None:
    if not train_last_layer_only:
        return
    for name, param in model.named_parameters():
        requires_grad = name.endswith("weight") or name.endswith("bias")
        if "classifier" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint

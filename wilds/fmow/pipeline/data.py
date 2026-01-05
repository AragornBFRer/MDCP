from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wilds.datasets.wilds_dataset import WILDSSubset

# RGB channel-wise mean and standard deviation computed over the ImageNet-1K training set. 
# They are applied in data.py to normalize each image ((x - mean) / std), 
# so that inputs match the scale the ImageNet-pretrained DenseNet/ResNet backbones expect; 
# skipping them typically harms convergence when fine-tuning.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class IndexedSubset(Dataset):
    """Wrap a WILDSSubset to also emit the underlying dataset index."""

    def __init__(self, subset: WILDSSubset, return_index: bool = False):
        self.subset = subset
        self.return_index = return_index

    def __getitem__(self, idx: int):
        x, y, metadata = self.subset[idx]
        if self.return_index:
            return x, y, metadata, int(self.subset.indices[idx])
        return x, y, metadata

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def indices(self) -> np.ndarray:
        return self.subset.indices


@dataclass
class DataLoaders:
    train: DataLoader
    val: Optional[DataLoader]
    holdout: Optional[DataLoader]


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_subset(dataset, indices: Sequence[int], transform, return_index: bool = False) -> IndexedSubset:
    subset = WILDSSubset(dataset, np.array(indices, dtype=np.int64), transform)
    return IndexedSubset(subset, return_index=return_index)


def build_weighted_sampler(dataset, indices: Sequence[int]) -> torch.utils.data.WeightedRandomSampler:
    metadata = dataset.metadata_array[np.array(indices, dtype=np.int64)]
    regions = metadata[:, 0].cpu().numpy()
    counts = np.bincount(regions, minlength=int(regions.max()) + 1)
    weights = 1.0 / counts[regions]
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)


def create_dataloaders(
    dataset,
    train_indices: Sequence[int],
    val_indices: Optional[Sequence[int]] = None,
    holdout_indices: Optional[Sequence[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_regions: bool = True,
) -> DataLoaders:
    train_subset = make_subset(dataset, train_indices, build_transforms(train=True))
    sampler = None
    if balance_regions:
        sampler = build_weighted_sampler(dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = None
    if val_indices is not None and len(val_indices) > 0:
        val_subset = make_subset(dataset, val_indices, build_transforms(train=False))
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    holdout_loader = None
    if holdout_indices is not None and len(holdout_indices) > 0:
        holdout_subset = make_subset(dataset, holdout_indices, build_transforms(train=False), return_index=True)
        holdout_loader = DataLoader(
            holdout_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoaders(train=train_loader, val=val_loader, holdout=holdout_loader)

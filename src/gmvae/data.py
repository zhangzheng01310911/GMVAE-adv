from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class TrafficArrays:
    values: np.ndarray
    adjacency: np.ndarray
    timestamps: np.ndarray | None = None


def load_npz(path: str | Path) -> TrafficArrays:
    archive = np.load(path, allow_pickle=False)
    if "values" not in archive or "adjacency" not in archive:
        raise ValueError("NPZ must contain values [time,nodes,channels] and adjacency [nodes,nodes]")
    values = archive["values"].astype(np.float32)
    adjacency = archive["adjacency"].astype(np.float32)
    timestamps = archive["timestamps"] if "timestamps" in archive else None
    if values.ndim != 3 or adjacency.shape != (values.shape[1], values.shape[1]):
        raise ValueError("inconsistent traffic array shapes")
    return TrafficArrays(values, adjacency, timestamps)


class WindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        input_length: int,
        horizon: int,
        target_channel: int | None = None,
    ) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.input_length = input_length
        self.horizon = horizon
        self.target_channel = target_channel

    def __len__(self) -> int:
        return max(0, len(self.values) - self.input_length - self.horizon + 1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        split = index + self.input_length
        target = self.values[split:split + self.horizon]
        if self.target_channel is not None:
            target = target[..., self.target_channel:self.target_channel + 1]
        return self.values[index:split], target


def chronological_split(values: np.ndarray, ratios=(0.7, 0.2, 0.1)):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    n = len(values)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    return values[:train_end], values[train_end:val_end], values[val_end:]


def fit_zscore(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def impute_and_normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(values), mean, values)
    return ((filled - mean) / std).astype(np.float32)

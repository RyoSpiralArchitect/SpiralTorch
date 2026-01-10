"""Composable augmentation primitives for self-supervised pipelines."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray


def _ensure_array(x: ArrayLike | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, received shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def normalize_batch(batch: ArrayLike | Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize each sample to zero mean and unit variance."""

    batch_arr = _ensure_array(batch)
    mean = batch_arr.mean(axis=1, keepdims=True)
    std = batch_arr.std(axis=1, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (batch_arr - mean) / std


def gaussian_noise(
    batch: ArrayLike | Sequence[Sequence[float]],
    std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Additive zero-mean Gaussian noise."""

    if std < 0:
        raise ValueError("std must be non-negative")
    rng = np.random.default_rng(seed)
    batch_arr = _ensure_array(batch)
    return batch_arr + rng.normal(0.0, std, size=batch_arr.shape).astype(np.float32)


def random_crop(
    image: ArrayLike | Sequence[Sequence[float]],
    crop: Tuple[int, int],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Random spatial crop for 2D image-like arrays."""

    if crop[0] <= 0 or crop[1] <= 0:
        raise ValueError("crop sizes must be positive")
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError("expected (C, H, W) shaped array")
    c, h, w = arr.shape
    if crop[0] > h or crop[1] > w:
        raise ValueError("crop cannot exceed spatial dimensions")
    rng = np.random.default_rng(seed)
    top = rng.integers(0, h - crop[0] + 1)
    left = rng.integers(0, w - crop[1] + 1)
    return arr[:, top : top + crop[0], left : left + crop[1]].copy()


def random_mask(
    sequence: ArrayLike | Sequence[float],
    mask_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly mask a fraction of positions in the input sequence."""

    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0, 1]")
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("random_mask expects a 1D sequence")
    rng = np.random.default_rng(seed)
    num_mask = max(1, int(math.ceil(mask_ratio * arr.shape[0])))
    indices = rng.choice(arr.shape[0], size=num_mask, replace=False)
    mask = np.zeros_like(arr, dtype=bool)
    mask[indices] = True
    return mask, arr[mask]


def solarize(
    image: ArrayLike | Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply a solarize effect by inverting values above a threshold."""

    arr = np.asarray(image, dtype=np.float32)
    return np.where(arr >= threshold, 1.0 - arr, arr)


__all__ = [
    "gaussian_noise",
    "random_crop",
    "random_mask",
    "solarize",
    "normalize_batch",
]


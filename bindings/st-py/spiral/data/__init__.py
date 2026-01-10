"""Data utilities for self-supervised learning experiments.

This subpackage is optional: it depends on NumPy.  When NumPy is missing, the
exports remain present but raise a friendly error on use.
"""

from __future__ import annotations

import sys
import types
from typing import Any, NoReturn

__all__ = [
    "augment",
    "gaussian_noise",
    "random_crop",
    "random_mask",
    "solarize",
    "normalize_batch",
]


def _missing_numpy(*_: Any, **__: Any) -> NoReturn:
    raise RuntimeError("spiral.data requires NumPy. Install it with `pip install numpy`.")


try:
    from . import augment as augment
    from .augment import (
        gaussian_noise,
        normalize_batch,
        random_crop,
        random_mask,
        solarize,
    )
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    augment = types.ModuleType(f"{__name__}.augment")
    gaussian_noise = _missing_numpy
    random_crop = _missing_numpy
    random_mask = _missing_numpy
    solarize = _missing_numpy
    normalize_batch = _missing_numpy
    sys.modules[f"{__name__}.augment"] = augment

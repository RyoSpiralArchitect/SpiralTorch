"""Unified SpiralTorch high-level helpers.

The SpiralTorch wheel ships two related namespaces:

- `spiraltorch`: the low-level Rust-backed primitives (Tensor, Hypergrad, Z-space, ...)
- `spiral`: optional high-level helpers (sessions, hypergrad utilities, chat prompt helpers, ...)

The `spiral` package is designed to remain importable even when optional
dependencies (e.g. NumPy for `spiral.data`) are missing.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, NoReturn

from .export import DeploymentTarget, ExportConfig, ExportPipeline, load_benchmark_report

__all__: list[str] = [
    "DeploymentTarget",
    "ExportConfig",
    "ExportPipeline",
    "load_benchmark_report",
    "cli",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly via imports
    if name == "cli":
        module = importlib.import_module(f"{__name__}.cli")
        globals()["cli"] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - import nicety
    exported = set(__all__)
    exported.update(globals().keys())
    return sorted(exported)


def _missing(feature: str) -> NoReturn:
    raise RuntimeError(
        f"spiral.{feature} is unavailable in this build. "
        "Install the SpiralTorch wheel (or build the native extension) to enable it."
    )


# --- Hypergrad helpers -------------------------------------------------------
try:
    from . import hypergrad as hypergrad  # noqa: F401
    from .hypergrad import hypergrad_session, hypergrad_summary_dict, suggest_hypergrad_operator
except Exception:  # pragma: no cover - defensive: missing native extension / broken install
    hypergrad = types.ModuleType(f"{__name__}.hypergrad")
    sys.modules.setdefault(hypergrad.__name__, hypergrad)

    def hypergrad_session(*_: Any, **__: Any) -> NoReturn:
        _missing("hypergrad_session")

    def hypergrad_summary_dict(*_: Any, **__: Any) -> NoReturn:
        _missing("hypergrad_summary_dict")

    def suggest_hypergrad_operator(*_: Any, **__: Any) -> NoReturn:
        _missing("suggest_hypergrad_operator")

    hypergrad.hypergrad_session = hypergrad_session  # type: ignore[attr-defined]
    hypergrad.hypergrad_summary_dict = hypergrad_summary_dict  # type: ignore[attr-defined]
    hypergrad.suggest_hypergrad_operator = suggest_hypergrad_operator  # type: ignore[attr-defined]
    hypergrad.__all__ = [
        "hypergrad_session",
        "hypergrad_summary_dict",
        "suggest_hypergrad_operator",
    ]
else:
    __all__ += [
        "hypergrad",
        "hypergrad_session",
        "hypergrad_summary_dict",
        "suggest_hypergrad_operator",
    ]


# --- Inference helpers -------------------------------------------------------
try:
    from . import inference as inference  # noqa: F401
    from .inference import (
        AuditEvent,
        AuditLog,
        ChatMessage,
        ChatPrompt,
        InferenceClient,
        InferenceResult,
        SafetyEvent,
        SafetyVerdict,
        SafetyViolation,
        format_chat_prompt,
    )
except Exception:  # pragma: no cover - defensive: missing native extension / broken install
    inference = types.ModuleType(f"{__name__}.inference")
    sys.modules.setdefault(inference.__name__, inference)

    def format_chat_prompt(*_: Any, **__: Any) -> NoReturn:
        _missing("format_chat_prompt")

    inference.format_chat_prompt = format_chat_prompt  # type: ignore[attr-defined]
    inference.__all__ = ["format_chat_prompt"]
else:
    __all__ += [
        "inference",
        "AuditEvent",
        "AuditLog",
        "ChatMessage",
        "ChatPrompt",
        "InferenceClient",
        "InferenceResult",
        "SafetyEvent",
        "SafetyVerdict",
        "SafetyViolation",
        "format_chat_prompt",
    ]


# --- Optional NumPy data helpers --------------------------------------------
try:
    from . import data as data  # noqa: F401
    from .data import augment as augment  # noqa: F401
    from .data import gaussian_noise, normalize_batch, random_crop, random_mask, solarize
except Exception:
    data = types.ModuleType(f"{__name__}.data")
    sys.modules.setdefault(data.__name__, data)

    augment = types.ModuleType(f"{__name__}.data.augment")
    sys.modules.setdefault(augment.__name__, augment)
    sys.modules.setdefault(f"{__name__}.augment", augment)

    def gaussian_noise(*_: Any, **__: Any) -> NoReturn:
        _missing("gaussian_noise")

    def random_crop(*_: Any, **__: Any) -> NoReturn:
        _missing("random_crop")

    def random_mask(*_: Any, **__: Any) -> NoReturn:
        _missing("random_mask")

    def solarize(*_: Any, **__: Any) -> NoReturn:
        _missing("solarize")

    def normalize_batch(*_: Any, **__: Any) -> NoReturn:
        _missing("normalize_batch")

    data.augment = augment  # type: ignore[attr-defined]
    data.gaussian_noise = gaussian_noise  # type: ignore[attr-defined]
    data.random_crop = random_crop  # type: ignore[attr-defined]
    data.random_mask = random_mask  # type: ignore[attr-defined]
    data.solarize = solarize  # type: ignore[attr-defined]
    data.normalize_batch = normalize_batch  # type: ignore[attr-defined]
    data.__all__ = [
        "augment",
        "gaussian_noise",
        "normalize_batch",
        "random_crop",
        "random_mask",
        "solarize",
    ]
else:
    __all__ += [
        "data",
        "augment",
        "gaussian_noise",
        "normalize_batch",
        "random_crop",
        "random_mask",
        "solarize",
    ]
    sys.modules.setdefault(f"{__name__}.augment", augment)

"""Hugging Face fine-tuning bridge helpers for SpiralTorch."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .runtime_imports import (
    csv_label,
    runtime_import_preflight_report,
    runtime_import_preflight_summary_lines,
    write_runtime_import_preflight_report,
)

__all__ = [
    "HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS",
    "HF_GPT2_FT_REQUIRED_PYTHON_PACKAGES",
    "HF_GPT2_FT_REQUIRED_RUST_SURFACES",
    "hf_gpt2_finetune_preflight_report",
    "hf_gpt2_finetune_rust_dependency_report",
    "hf_gpt2_finetune_summary_lines",
    "hf_gpt2_finetune_zspace_probe",
    "write_hf_gpt2_finetune_run_card",
]


HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS = ["wgpu", "cpu"]
HF_GPT2_FT_REQUIRED_PYTHON_PACKAGES = [
    "transformers",
    "torch",
    "tokenizers",
    "datasets",
    "accelerate",
    "safetensors",
    "pyarrow",
    "tqdm",
    "evaluate",
    "peft",
]
HF_GPT2_FT_REQUIRED_RUST_SURFACES = [
    {
        "crate": "st-tensor",
        "python_surface": "spiraltorch.Tensor / DLPack / runtime device reports",
        "why": (
            "Owns the tensor boundary that lets PyTorch-side values be audited "
            "beside SpiralTorch/WGPU telemetry."
        ),
    },
    {
        "crate": "st-nn",
        "python_surface": "spiraltorch.nn.ZSpaceProjector / trainer trace helpers",
        "why": (
            "Carries the Z-Space projection and training instrumentation that "
            "can be attached to GPT-2 fine-tune probes."
        ),
    },
    {
        "crate": "st-text",
        "python_surface": "spiraltorch.text / language-wave encoders",
        "why": (
            "Keeps tokenizer and language-wave semantics close to the HF text "
            "pipeline instead of leaving them as a separate demo layer."
        ),
    },
    {
        "crate": "st-logic",
        "python_surface": "spiraltorch.OpenTopos / topos control signals",
        "why": (
            "Provides the open-topos control vocabulary used to gate or trace "
            "runtime geometry while training."
        ),
    },
    {
        "crate": "st-frac",
        "python_surface": "Z-Space geometry, Mellin/log/fractal probes",
        "why": (
            "Supplies the geometric probes that make the FT run more than a "
            "plain Transformers wrapper."
        ),
    },
    {
        "crate": "st-spiral-rl",
        "python_surface": "spiraltorch.rl.stAgent route policy hooks",
        "why": (
            "Lets trained route policies or runtime decisions be replayed "
            "against FT telemetry later."
        ),
    },
    {
        "crate": "st-backend-wgpu",
        "python_surface": "describe_runtime_devices('wgpu') / WGPU-first reports",
        "why": (
            "Makes GPU readiness explicit and keeps MPS as an honest placeholder "
            "rather than an implied working backend."
        ),
    },
]


def _unique(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = values.split(",")
    elif isinstance(values, Iterable):
        raw_values = [str(value) for value in values]
    else:
        raw_values = [str(values)]
    return list(dict.fromkeys(value.strip() for value in raw_values if value.strip()))


def hf_gpt2_finetune_rust_dependency_report() -> dict[str, object]:
    """Describe the Rust crate surfaces that matter for GPT-2-scale local FT."""

    crates = [dict(row) for row in HF_GPT2_FT_REQUIRED_RUST_SURFACES]
    packages = list(HF_GPT2_FT_REQUIRED_PYTHON_PACKAGES)
    return {
        "row_type": "hf_gpt2_finetune_rust_dependency_report",
        "rust_surfaces": crates,
        "rust_surface_crates": csv_label(row["crate"] for row in crates),
        "python_packages": packages,
        "python_package_label": csv_label(packages),
        "position": (
            "For local GPT-2 small fine-tuning, SpiralTorch should keep the "
            "Rust wheel focused on tensor/nn/text/logic/frac/rl/wgpu surfaces, "
            "while Python explicitly brings the Hugging Face model, data, "
            "adapter, and evaluation stack."
        ),
    }


def hf_gpt2_finetune_preflight_report(
    *,
    model_name: str = "gpt2",
    dataset_name: str | None = "wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    train_split: str = "train",
    eval_split: str | None = "validation",
    text_column: str = "text",
    runtime_device_backends: object = None,
    required_runtime_device_ready_backends: object = None,
    require_hf_gpt2_ft: bool = True,
    describe_runtime_devices=None,
) -> dict[str, object]:
    """Build a strict preflight report for local GPT-2 fine-tuning."""

    requested_backends = _unique(runtime_device_backends)
    if not requested_backends:
        requested_backends = list(HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS)
    required_presets = ["hf-gpt2-ft"] if require_hf_gpt2_ft else []
    report = runtime_import_preflight_report(
        runtime_import_presets=["hf-gpt2-ft"],
        required_runtime_import_presets=required_presets,
        runtime_device_backends=requested_backends,
        required_runtime_device_ready_backends=required_runtime_device_ready_backends,
        describe_runtime_devices=describe_runtime_devices,
    )
    dependency_report = hf_gpt2_finetune_rust_dependency_report()
    report.update(
        {
            "row_type": "hf_gpt2_finetune_preflight",
            "hf_model_name": str(model_name),
            "hf_dataset_name": dataset_name,
            "hf_dataset_config": dataset_config,
            "hf_train_split": str(train_split),
            "hf_eval_split": eval_split,
            "hf_text_column": str(text_column),
            "hf_gpt2_ft_required": bool(require_hf_gpt2_ft),
            "hf_gpt2_ft_python_packages": dependency_report["python_package_label"],
            "hf_gpt2_ft_rust_surfaces": dependency_report["rust_surface_crates"],
            "hf_gpt2_ft_rust_dependency_report": dependency_report,
        }
    )
    return report


def hf_gpt2_finetune_summary_lines(report: Mapping[str, object]) -> list[str]:
    """Return concise human-readable lines for a GPT-2 FT preflight report."""

    lines = [
        (
            "hf_gpt2_finetune "
            f"model={report.get('hf_model_name')} "
            f"dataset={report.get('hf_dataset_name')} "
            f"config={report.get('hf_dataset_config')} "
            f"train_split={report.get('hf_train_split')} "
            f"text_column={report.get('hf_text_column')}"
        ),
        (
            "hf_gpt2_finetune_surfaces "
            f"rust={report.get('hf_gpt2_ft_rust_surfaces', 'none')} "
            f"python={report.get('hf_gpt2_ft_python_packages', 'none')}"
        ),
    ]
    lines.extend(runtime_import_preflight_summary_lines(report))
    return lines


def _token_probe_values(
    token_ids: Sequence[int | float],
    *,
    dim: int,
    vocab_size: int | None,
) -> tuple[list[float], int]:
    clipped = [float(value) for value in token_ids[:dim]]
    observed = len(clipped)
    if not clipped:
        return [], observed
    scale = float(vocab_size) if vocab_size and vocab_size > 0 else max(
        1.0,
        max(abs(value) for value in clipped),
    )
    values = [value / scale for value in clipped]
    if len(values) < dim:
        values.extend(0.0 for _ in range(dim - len(values)))
    return values, observed


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def hf_gpt2_finetune_zspace_probe(
    token_ids: Sequence[int | float],
    *,
    dim: int = 64,
    vocab_size: int | None = None,
    curvature: float = -0.04,
    frequency: float = 0.65,
    strength: float = 1.0,
    require: bool = False,
) -> dict[str, object]:
    """Project a token-id preview through SpiralTorch Z-Space for FT audit cards."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    values, observed = _token_probe_values(token_ids, dim=dim, vocab_size=vocab_size)
    row: dict[str, object] = {
        "row_type": "hf_gpt2_finetune_zspace_probe",
        "zspace_probe_requested": True,
        "zspace_probe_status": "missing_tokens" if not values else "pending",
        "zspace_probe_error": None,
        "zspace_probe_dim": int(dim),
        "zspace_probe_observed_token_count": observed,
        "zspace_probe_vocab_size": vocab_size,
        "zspace_probe_curvature": float(curvature),
        "zspace_probe_frequency": float(frequency),
        "zspace_probe_strength": float(strength),
        "zspace_probe_input_l2": _l2(values) if values else None,
        "zspace_probe_output_l2": None,
        "zspace_probe_delta_l2": None,
        "zspace_probe_delta_input_l2_ratio": None,
    }
    if not values:
        if require:
            raise RuntimeError("Z-Space token probe requires at least one token id")
        return row
    try:
        import spiraltorch as st
        from spiraltorch.nn import ZSpaceProjector

        tensor = st.Tensor(1, len(values), values)
        topos = st.OpenTopos(float(curvature))
        encoder = st.LanguageWaveEncoder(topos.curvature(), float(frequency))
        projector = ZSpaceProjector(topos, encoder, strength=float(strength))
        projected = projector.forward(tensor)
        projected_values = [float(value) for value in projected.data()]
    except Exception as exc:  # pragma: no cover - depends on native runtime.
        row.update(
            {
                "zspace_probe_status": "error",
                "zspace_probe_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        if require:
            raise RuntimeError(row["zspace_probe_error"]) from exc
        return row

    input_l2 = float(row["zspace_probe_input_l2"] or 0.0)
    output_l2 = _l2(projected_values)
    delta_l2 = _l2([after - before for before, after in zip(values, projected_values)])
    row.update(
        {
            "zspace_probe_status": "ok",
            "zspace_probe_output_l2": output_l2,
            "zspace_probe_delta_l2": delta_l2,
            "zspace_probe_delta_input_l2_ratio": (
                None if input_l2 == 0.0 else delta_l2 / input_l2
            ),
        }
    )
    return row


def write_hf_gpt2_finetune_run_card(
    report: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write a GPT-2 FT run card JSON artifact and return its path."""

    if report.get("row_type") == "hf_gpt2_finetune_preflight":
        return write_runtime_import_preflight_report(report, path)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(output_path)

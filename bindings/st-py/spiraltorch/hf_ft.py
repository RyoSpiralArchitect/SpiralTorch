"""Hugging Face fine-tuning bridge helpers for SpiralTorch."""

from __future__ import annotations

import json
import math
import time
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
    "hf_gpt2_finetune_trainer_trace_callback",
    "hf_gpt2_finetune_trainer_trace_event",
    "hf_gpt2_finetune_zspace_probe",
    "load_hf_gpt2_finetune_trainer_trace",
    "summarize_hf_gpt2_finetune_trainer_trace",
    "write_hf_gpt2_finetune_run_card",
    "write_hf_gpt2_finetune_trainer_trace_event",
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


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _safe_attr(value: object, name: str, default: object = None) -> object:
    if value is None:
        return default
    return getattr(value, name, default)


def _safe_number(value: object) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _metric_fields(values: Mapping[str, object] | None) -> dict[str, object]:
    if not values:
        return {}
    return {
        str(key): _json_safe(value)
        for key, value in values.items()
        if not str(key).startswith("_")
    }


def hf_gpt2_finetune_trainer_trace_event(
    event: str,
    *,
    args: object = None,
    state: object = None,
    control: object = None,
    logs: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
    run_id: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one JSON-safe HF Trainer trace row for SpiralTorch run cards."""

    metric_payload = _metric_fields(metrics if metrics is not None else logs)
    global_step = _safe_number(_safe_attr(state, "global_step"))
    epoch = _safe_number(_safe_attr(state, "epoch"))
    row: dict[str, object] = {
        "row_type": "hf_gpt2_finetune_trainer_trace",
        "event": str(event),
        "time_unix_s": time.time(),
        "run_id": run_id,
        "global_step": global_step,
        "epoch": epoch,
        "max_steps": _safe_number(_safe_attr(state, "max_steps")),
        "num_train_epochs": _safe_number(_safe_attr(args, "num_train_epochs")),
        "output_dir": _json_safe(_safe_attr(args, "output_dir")),
        "learning_rate": _safe_number(_safe_attr(args, "learning_rate")),
        "per_device_train_batch_size": _safe_number(
            _safe_attr(args, "per_device_train_batch_size")
        ),
        "gradient_accumulation_steps": _safe_number(
            _safe_attr(args, "gradient_accumulation_steps")
        ),
        "log_history_count": _safe_number(
            len(_safe_attr(state, "log_history", []) or [])
        ),
        "should_training_stop": bool(
            _safe_attr(control, "should_training_stop", False)
        ),
        "should_evaluate": bool(_safe_attr(control, "should_evaluate", False)),
        "should_save": bool(_safe_attr(control, "should_save", False)),
        "metrics": metric_payload,
        "metric_keys": csv_label(sorted(metric_payload)),
    }
    if extra:
        row.update({str(key): _json_safe(value) for key, value in extra.items()})
    return row


def write_hf_gpt2_finetune_trainer_trace_event(
    row: Mapping[str, object],
    path: str | Path,
) -> str:
    """Append one HF Trainer trace event as JSONL and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return str(output_path)


def load_hf_gpt2_finetune_trainer_trace(path: str | Path) -> list[dict[str, object]]:
    """Load SpiralTorch HF Trainer trace JSONL rows."""

    rows = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_no} invalid JSONL: {exc}") from exc
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _last_metric(rows: Sequence[Mapping[str, object]], key: str) -> object:
    for row in reversed(rows):
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping) and key in metrics:
            return metrics[key]
    return None


def _min_numeric_metric(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        number = _safe_number(metrics.get(key))
        if number is not None:
            values.append(float(number))
    return min(values) if values else None


def summarize_hf_gpt2_finetune_trainer_trace(
    path_or_rows: str | Path | Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize HF Trainer trace rows for a GPT-2 fine-tune run card."""

    if isinstance(path_or_rows, (str, Path)):
        rows = load_hf_gpt2_finetune_trainer_trace(path_or_rows)
    else:
        rows = [dict(row) for row in path_or_rows]
    event_counts: dict[str, int] = {}
    max_step: int | float | None = None
    for row in rows:
        event = str(row.get("event") or "unknown")
        event_counts[event] = event_counts.get(event, 0) + 1
        step = _safe_number(row.get("global_step"))
        if step is not None and (max_step is None or step > max_step):
            max_step = step
    return {
        "row_type": "hf_gpt2_finetune_trainer_trace_summary",
        "trace_event_count": len(rows),
        "trace_event_counts": event_counts,
        "trace_events": csv_label(sorted(event_counts)),
        "trace_max_global_step": max_step,
        "trace_last_event": rows[-1].get("event") if rows else None,
        "trace_last_loss": _last_metric(rows, "loss"),
        "trace_min_loss": _min_numeric_metric(rows, "loss"),
        "trace_last_eval_loss": _last_metric(rows, "eval_loss"),
        "trace_min_eval_loss": _min_numeric_metric(rows, "eval_loss"),
        "trace_last_learning_rate": _last_metric(rows, "learning_rate"),
    }


def hf_gpt2_finetune_trainer_trace_callback(
    path: str | Path,
    *,
    run_id: str | None = None,
    reset: bool = True,
    zspace_probe_tokens: Sequence[int | float] | None = None,
    zspace_probe_kwargs: Mapping[str, object] | None = None,
):
    """Create a Transformers TrainerCallback that writes SpiralTorch JSONL."""

    try:
        import importlib

        transformers = importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise RuntimeError(
            "hf_gpt2_finetune_trainer_trace_callback requires transformers"
        ) from exc
    base_cls = getattr(transformers, "TrainerCallback", object)
    trace_path = Path(path)
    probe_kwargs = dict(zspace_probe_kwargs or {})
    probe_tokens = list(zspace_probe_tokens or [])

    class SpiralTorchHFTrainerTraceCallback(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.path = trace_path
            self.run_id = run_id
            self.event_count = 0
            if reset:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text("", encoding="utf-8")

        def _emit(
            self,
            event: str,
            args: object,
            state: object,
            control: object,
            *,
            logs: Mapping[str, object] | None = None,
            metrics: Mapping[str, object] | None = None,
            extra: Mapping[str, object] | None = None,
        ) -> object:
            row = hf_gpt2_finetune_trainer_trace_event(
                event,
                args=args,
                state=state,
                control=control,
                logs=logs,
                metrics=metrics,
                run_id=self.run_id,
                extra=extra,
            )
            write_hf_gpt2_finetune_trainer_trace_event(row, self.path)
            self.event_count += 1
            return control

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            extra = {}
            if probe_tokens:
                extra["zspace_probe"] = hf_gpt2_finetune_zspace_probe(
                    probe_tokens,
                    **probe_kwargs,
                )
            return self._emit("train_begin", args, state, control, extra=extra)

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            return self._emit("log", args, state, control, logs=logs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[no-untyped-def]
            return self._emit("evaluate", args, state, control, metrics=metrics)

        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            return self._emit("save", args, state, control)

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            return self._emit("train_end", args, state, control)

    return SpiralTorchHFTrainerTraceCallback()


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

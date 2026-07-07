"""Hugging Face fine-tuning bridge helpers for SpiralTorch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .hf_generation import zspace_generation_control_bridge_cli_args
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
    "hf_gpt2_finetune_corpus_file_report",
    "hf_gpt2_finetune_corpus_scan_report",
    "hf_gpt2_finetune_dataset_fit_report",
    "hf_gpt2_finetune_eval_report",
    "hf_gpt2_finetune_generation_curve_lines",
    "hf_gpt2_finetune_generation_curve_report",
    "hf_gpt2_finetune_generation_report",
    "hf_gpt2_finetune_inference_distortion_handoff_report",
    "hf_gpt2_finetune_inference_distortion_handoff_lines",
    "hf_gpt2_finetune_rust_dependency_report",
    "hf_gpt2_finetune_scale_up_command",
    "hf_gpt2_finetune_scale_up_preflight_lines",
    "hf_gpt2_finetune_scale_up_preflight_report",
    "hf_gpt2_finetune_summary_lines",
    "hf_gpt2_finetune_training_telemetry_frame",
    "hf_gpt2_finetune_trainer_trace_callback",
    "hf_gpt2_finetune_trainer_trace_event",
    "hf_gpt2_finetune_zspace_probe",
    "compare_hf_gpt2_finetune_run_cards",
    "load_hf_gpt2_finetune_run_card",
    "load_hf_gpt2_finetune_sweep_report",
    "load_hf_gpt2_finetune_trainer_trace",
    "summarize_hf_gpt2_finetune_run_card",
    "summarize_hf_gpt2_finetune_sweep_report",
    "summarize_hf_gpt2_finetune_sweep_report_lines",
    "summarize_hf_gpt2_finetune_trainer_trace",
    "write_hf_gpt2_finetune_run_card",
    "write_hf_gpt2_finetune_trainer_trace_event",
]


HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS = ["wgpu", "cpu"]
HF_GPT2_FT_DISTORTION_EVAL_PENALTY_WEIGHT = 0.1
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


def _path_values(values: object) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        raw_values = [values]
    elif isinstance(values, Iterable):
        raw_values = list(values)
    else:
        raw_values = [values]
    return [Path(value) for value in raw_values if str(value)]


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


def hf_gpt2_finetune_corpus_file_report(
    *,
    train_files: object = None,
    validation_files: object = None,
    dataset_format: str = "text",
    text_column: str = "text",
) -> dict[str, object]:
    """Build a lightweight manifest for local GPT-2 FT corpus files."""

    rows = []
    fingerprint = hashlib.sha256()
    total_bytes = 0
    missing = []
    readable = []
    for split, paths in (
        ("train", _path_values(train_files)),
        ("validation", _path_values(validation_files)),
    ):
        for path in paths:
            label = str(path)
            row: dict[str, object] = {
                "split": split,
                "path": label,
                "exists": path.is_file(),
                "bytes": None,
                "mtime_ns": None,
            }
            if not path.is_file():
                missing.append(label)
            else:
                stat = path.stat()
                size = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
                row.update({"bytes": size, "mtime_ns": mtime_ns})
                readable.append(label)
                total_bytes += size
                fingerprint.update(
                    f"{split}\0{path.resolve()}\0{size}\0{mtime_ns}\n".encode()
                )
            rows.append(row)
    return {
        "row_type": "hf_gpt2_finetune_corpus_file_report",
        "dataset_source": "local_files" if rows else "hf_dataset",
        "dataset_format": str(dataset_format),
        "text_column": str(text_column),
        "file_count": len(rows),
        "train_file_count": sum(1 for row in rows if row["split"] == "train"),
        "validation_file_count": sum(
            1 for row in rows if row["split"] == "validation"
        ),
        "total_bytes": total_bytes,
        "readable_files": csv_label(readable),
        "missing_files": csv_label(missing),
        "all_files_available": not missing,
        "fingerprint": fingerprint.hexdigest() if rows else None,
        "files": rows,
    }


def hf_gpt2_finetune_corpus_scan_report(
    *,
    train_files: object = None,
    validation_files: object = None,
    dataset_format: str = "text",
    text_column: str = "text",
    sample_line_limit: int = 8,
    sample_preview_chars: int = 160,
    max_bytes_per_file: int | None = None,
    encoding: str = "utf-8",
) -> dict[str, object]:
    """Stream local corpus files and summarize text shape before long FT runs."""

    sample_limit = max(0, int(sample_line_limit))
    preview_chars = max(0, int(sample_preview_chars))
    max_bytes = None
    if max_bytes_per_file is not None and int(max_bytes_per_file) > 0:
        max_bytes = int(max_bytes_per_file)

    rows = []
    missing = []
    truncated = []
    scan_errors = []
    total_scanned_bytes = 0
    total_line_count = 0
    total_nonempty_line_count = 0
    total_empty_line_count = 0
    total_nonempty_line_bytes = 0
    max_line_bytes = 0
    fingerprint = hashlib.sha256()

    for split, paths in (
        ("train", _path_values(train_files)),
        ("validation", _path_values(validation_files)),
    ):
        for path in paths:
            label = str(path)
            row_hash = hashlib.sha256()
            row: dict[str, object] = {
                "split": split,
                "path": label,
                "exists": path.is_file(),
                "scan_truncated": False,
                "scanned_bytes": 0,
                "line_count": 0,
                "nonempty_line_count": 0,
                "empty_line_count": 0,
                "nonempty_line_bytes": 0,
                "max_line_bytes": 0,
                "rough_gpt2_token_estimate": 0,
                "mean_nonempty_line_bytes": None,
                "scanned_content_sha256": None,
                "sample_texts": [],
                "error": None,
            }
            if not path.is_file():
                missing.append(label)
                row["error"] = "missing_file"
                rows.append(row)
                continue

            try:
                with path.open("rb") as handle:
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        if (
                            max_bytes is not None
                            and int(row["scanned_bytes"]) + len(line) > max_bytes
                        ):
                            row["scan_truncated"] = True
                            truncated.append(label)
                            break

                        row_hash.update(line)
                        row["scanned_bytes"] = int(row["scanned_bytes"]) + len(line)
                        row["line_count"] = int(row["line_count"]) + 1
                        stripped = line.strip()
                        line_length = len(line)
                        row["max_line_bytes"] = max(
                            int(row["max_line_bytes"]),
                            line_length,
                        )
                        if stripped:
                            row["nonempty_line_count"] = (
                                int(row["nonempty_line_count"]) + 1
                            )
                            row["nonempty_line_bytes"] = (
                                int(row["nonempty_line_bytes"]) + len(stripped)
                            )
                            samples = row["sample_texts"]
                            if (
                                isinstance(samples, list)
                                and len(samples) < sample_limit
                            ):
                                text = line.decode(encoding, errors="replace").strip()
                                samples.append(text[:preview_chars])
                        else:
                            row["empty_line_count"] = int(row["empty_line_count"]) + 1
            except OSError as exc:
                scan_errors.append(label)
                row["error"] = f"{exc.__class__.__name__}: {exc}"

            nonempty_count = int(row["nonempty_line_count"])
            nonempty_bytes = int(row["nonempty_line_bytes"])
            if nonempty_count:
                row["mean_nonempty_line_bytes"] = nonempty_bytes / nonempty_count
            row["rough_gpt2_token_estimate"] = int(math.ceil(nonempty_bytes / 4.0))
            row["scanned_content_sha256"] = (
                row_hash.hexdigest() if int(row["scanned_bytes"]) else None
            )

            total_scanned_bytes += int(row["scanned_bytes"])
            total_line_count += int(row["line_count"])
            total_nonempty_line_count += nonempty_count
            total_empty_line_count += int(row["empty_line_count"])
            total_nonempty_line_bytes += nonempty_bytes
            max_line_bytes = max(max_line_bytes, int(row["max_line_bytes"]))
            fingerprint.update(
                (
                    f"{split}\0{path.resolve()}\0{row['scanned_bytes']}\0"
                    f"{row['line_count']}\0{row['scanned_content_sha256']}\0"
                    f"{row['scan_truncated']}\n"
                ).encode()
            )
            rows.append(row)

    return {
        "row_type": "hf_gpt2_finetune_corpus_scan_report",
        "dataset_source": "local_files" if rows else "hf_dataset",
        "dataset_format": str(dataset_format),
        "text_column": str(text_column),
        "encoding": str(encoding),
        "sample_line_limit": sample_limit,
        "sample_preview_chars": preview_chars,
        "max_bytes_per_file": max_bytes,
        "scan_mode": "bounded" if max_bytes is not None else "full",
        "file_count": len(rows),
        "scanned_bytes": total_scanned_bytes,
        "line_count": total_line_count,
        "nonempty_line_count": total_nonempty_line_count,
        "empty_line_count": total_empty_line_count,
        "nonempty_line_bytes": total_nonempty_line_bytes,
        "max_line_bytes": max_line_bytes,
        "mean_nonempty_line_bytes": (
            None
            if total_nonempty_line_count == 0
            else total_nonempty_line_bytes / total_nonempty_line_count
        ),
        "rough_gpt2_token_estimate": int(math.ceil(total_nonempty_line_bytes / 4.0)),
        "scan_truncated_file_count": len(truncated),
        "scan_truncated_files": csv_label(truncated),
        "scan_error_count": len(scan_errors),
        "scan_error_files": csv_label(scan_errors),
        "missing_files": csv_label(missing),
        "all_files_available": not missing,
        "fingerprint": fingerprint.hexdigest() if rows else None,
        "files": rows,
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hf_gpt2_finetune_dataset_fit_report(
    *,
    raw_train_rows: object = None,
    raw_eval_rows: object = None,
    tokenized_train_rows: object = None,
    tokenized_eval_rows: object = None,
    block_size: int = 128,
    min_train_blocks: int = 1,
    min_eval_blocks: int = 1,
) -> dict[str, object]:
    """Summarize whether tokenized local FT splits are train/eval ready."""

    raw_train = _optional_int(raw_train_rows)
    raw_eval = _optional_int(raw_eval_rows)
    train_blocks = _optional_int(tokenized_train_rows)
    eval_blocks = _optional_int(tokenized_eval_rows)
    min_train = max(1, int(min_train_blocks))
    min_eval = max(0, int(min_eval_blocks))
    train_ready = train_blocks is not None and train_blocks >= min_train
    eval_requested = raw_eval is not None and raw_eval > 0
    eval_ready = (
        None
        if not eval_requested
        else eval_blocks is not None and eval_blocks >= min_eval
    )
    warnings = []
    if raw_train is not None and raw_train == 0:
        warnings.append("raw_train_empty")
    if train_blocks is None:
        warnings.append("tokenized_train_unknown")
    elif train_blocks < min_train:
        warnings.append("tokenized_train_too_small")
    if eval_requested:
        if eval_blocks is None:
            warnings.append("tokenized_eval_unknown")
        elif eval_blocks < min_eval:
            warnings.append("tokenized_eval_too_small")
    verdict = "train_eval_ready"
    if not train_ready:
        verdict = "not_trainable"
    elif eval_ready is False:
        verdict = "train_ready_eval_unusable"
    elif eval_ready is None:
        verdict = "train_ready_eval_not_requested"
    return {
        "row_type": "hf_gpt2_finetune_dataset_fit_report",
        "raw_train_rows": raw_train,
        "raw_eval_rows": raw_eval,
        "tokenized_train_rows": train_blocks,
        "tokenized_eval_rows": eval_blocks,
        "block_size": int(block_size),
        "min_train_blocks": min_train,
        "min_eval_blocks": min_eval,
        "train_ready": bool(train_ready),
        "eval_requested": bool(eval_requested),
        "eval_ready": eval_ready,
        "eval_dropped_empty": bool(eval_requested and eval_ready is False),
        "verdict": verdict,
        "warnings": csv_label(warnings),
    }


def hf_gpt2_finetune_generation_report(
    *,
    stage: str,
    prompt: object = "",
    generated_text: object = None,
    generated_continuation_text: object = None,
    input_token_count: object = None,
    output_token_count: object = None,
    max_new_tokens: int = 32,
    generation_method: object = None,
    fallback_error: object = None,
    generation_control: Mapping[str, object] | None = None,
    error: object = None,
) -> dict[str, object]:
    """Summarize a GPT-2 FT before/after generation sample for run cards."""

    prompt_text = "" if prompt is None else str(prompt)
    text = "" if generated_text is None else str(generated_text)
    continuation = (
        "" if generated_continuation_text is None else str(generated_continuation_text)
    )
    input_tokens = _optional_int(input_token_count)
    output_tokens = _optional_int(output_token_count)
    new_tokens = None
    if input_tokens is not None and output_tokens is not None:
        new_tokens = max(0, output_tokens - input_tokens)
    method_text = None if generation_method is None else str(generation_method)
    fallback_error_text = None if fallback_error is None else str(fallback_error)
    error_text = None if error is None else str(error)
    status = "error" if error_text else ("ok" if text else "empty")
    control_payload = (
        _json_safe(generation_control)
        if isinstance(generation_control, Mapping)
        else None
    )
    return {
        "row_type": "hf_gpt2_finetune_generation_report",
        "stage": str(stage),
        "status": status,
        "prompt": prompt_text,
        "generated_text": text,
        "generated_continuation_text": continuation,
        "prompt_char_count": len(prompt_text),
        "generated_char_count": len(text),
        "generated_continuation_char_count": len(continuation),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "generated_text_sha256": (
            hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        ),
        "generated_continuation_sha256": (
            hashlib.sha256(continuation.encode("utf-8")).hexdigest()
            if continuation
            else None
        ),
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "new_token_count": new_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generation_method": method_text,
        "generation_control": control_payload,
        "fallback_error": fallback_error_text,
        "error": error_text,
    }


def hf_gpt2_finetune_inference_distortion_handoff_report(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 3,
) -> dict[str, object]:
    """Summarize the inference-distortion probe/sweep that motivated an FT run."""

    from .hf_generation import (
        load_zspace_inference_distortion_sweep,
        summarize_zspace_inference_distortion_sweep,
        zspace_inference_distortion_sweep_report_from_probes,
    )

    input_row_type = None
    if isinstance(report_or_path, Mapping):
        input_row_type = report_or_path.get("row_type")
    elif isinstance(report_or_path, (str, Path)):
        try:
            raw_payload = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
            if isinstance(raw_payload, Mapping):
                input_row_type = raw_payload.get("row_type")
        except Exception:
            input_row_type = None
    source_kind = (
        "probe"
        if input_row_type == "zspace_inference_distortion_probe"
        else "sweep"
    )
    summary_source: str | Path | Mapping[str, object] = report_or_path
    if source_kind == "probe":
        label = None
        if isinstance(report_or_path, (str, Path)):
            label = Path(report_or_path).stem
        summary_source = zspace_inference_distortion_sweep_report_from_probes(
            report_or_path,
            labels=None if label is None else [label],
            report_path=report_or_path if isinstance(report_or_path, (str, Path)) else None,
            top_n=top_n,
        )
    summary = summarize_zspace_inference_distortion_sweep(
        summary_source,
        top_n=top_n,
    )
    if isinstance(summary_source, Mapping):
        report = dict(summary_source)
        source_path = str(
            report.get("report_path")
            or summary.get("sweep_path")
            or (
                report_or_path
                if isinstance(report_or_path, (str, Path))
                else report.get("source_path") or ""
            )
        )
    elif isinstance(report_or_path, (str, Path)):
        source_path = str(report_or_path)
        try:
            report = load_zspace_inference_distortion_sweep(report_or_path)
        except (OSError, ValueError):
            report = {}
    else:
        report = dict(report_or_path)
        source_path = str(report.get("report_path") or summary.get("sweep_path") or "")
    runtime = _mapping_item(report, "runtime")
    config = _mapping_item(summary, "recommended_config")
    request = _mapping_item(summary, "recommended_request")
    request_filter = _mapping_item(summary, "recommended_request_filter")
    processor_kwargs = _mapping_item(summary, "recommended_processor_kwargs")
    activation_hook = _mapping_item(summary, "recommended_activation_hook")
    dropped_keys = _unique(summary.get("recommended_api_request_dropped_keys"))
    retry_dropped_keys = _unique(
        summary.get("recommended_api_request_retry_dropped_keys")
    )
    sent_keys = _unique(summary.get("recommended_api_request_sent_keys"))
    bridge_cli_args = _generation_inference_bridge_cli_args(processor_kwargs)
    source_path_value = source_path or None
    if source_kind == "probe":
        source_cli_args = (
            ["--inference-distortion-probe", source_path_value]
            if source_path_value
            else []
        )
    else:
        source_cli_args = (
            ["--inference-distortion-sweep-report", source_path_value]
            if source_path_value
            else []
        )
    generation_handoff_cli_args = (
        [*source_cli_args, "--generation-from-inference-distortion"]
        if source_cli_args
        else []
    )
    explicit_generation_bridge_cli_args = (
        [*source_cli_args, *bridge_cli_args]
        if source_cli_args and bridge_cli_args
        else []
    )
    status = "ok" if config else "missing_recommendation"
    return {
        "row_type": "hf_gpt2_finetune_inference_distortion_handoff",
        "status": status,
        "source_path": source_path_value,
        "source_kind": source_kind,
        "source_row_type": input_row_type,
        "sweep_status": summary.get("status"),
        "prompt": summary.get("prompt"),
        "recommended_probe": summary.get("recommended_probe"),
        "recommendation_reason": summary.get("recommendation_reason"),
        "recommended_effect_score": _safe_number(
            summary.get("recommended_effect_score")
        ),
        "recommended_risk_score": _safe_number(summary.get("recommended_risk_score")),
        "recommended_api_compatibility_score": _safe_number(
            summary.get("recommended_api_compatibility_score")
        ),
        "recommended_probe_path": summary.get("recommended_probe_path"),
        "recommended_config": config or None,
        "recommended_request": request or None,
        "recommended_request_filter": request_filter or None,
        "recommended_processor_kwargs": processor_kwargs or None,
        "recommended_bridge_cli_args": bridge_cli_args,
        "recommended_bridge_cli_display": _shell_join_args(bridge_cli_args),
        "recommended_source_cli_args": source_cli_args,
        "recommended_source_cli_display": _shell_join_args(source_cli_args),
        "recommended_generation_handoff_cli_args": generation_handoff_cli_args,
        "recommended_generation_handoff_cli_display": _shell_join_args(
            generation_handoff_cli_args
        ),
        "recommended_explicit_generation_bridge_cli_args": (
            explicit_generation_bridge_cli_args
        ),
        "recommended_explicit_generation_bridge_cli_display": _shell_join_args(
            explicit_generation_bridge_cli_args
        ),
        "recommended_activation_hook": activation_hook or None,
        "recommended_probe_cli_args": list(
            summary.get("recommended_probe_cli_args") or []
        ),
        "recommended_sweep_cli_args": list(
            summary.get("recommended_sweep_cli_args") or []
        ),
        "runtime": runtime or None,
        "local_model": runtime.get("local_model"),
        "api_provider": runtime.get("api_provider"),
        "api_model": runtime.get("api_model"),
        "api_request_dropped_key_count": _safe_number(
            summary.get("recommended_api_request_dropped_key_count")
        ),
        "api_request_dropped_keys": dropped_keys,
        "api_request_retry_dropped_key_count": _safe_number(
            summary.get("recommended_api_request_retry_dropped_key_count")
        ),
        "api_request_retry_dropped_keys": retry_dropped_keys,
        "api_request_sent_keys": sent_keys,
        "desire_pressure": _safe_number(config.get("desire_pressure")),
        "desire_stability": _safe_number(config.get("desire_stability")),
        "psi_total": _safe_number(config.get("psi_total")),
        "coherence": _safe_number(config.get("coherence")),
        "distortion_strength": _safe_number(config.get("distortion_strength")),
        "base_temperature": _safe_number(config.get("base_temperature")),
        "base_top_p": _safe_number(config.get("base_top_p")),
        "include_penalties": config.get("include_penalties"),
    }


def hf_gpt2_finetune_inference_distortion_handoff_lines(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 3,
    replay_arg_limit: int = 24,
) -> list[str]:
    """Render compact human-readable lines for an inference-distortion handoff."""

    if (
        isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_gpt2_finetune_inference_distortion_handoff"
    ):
        handoff = dict(report_or_path)
    else:
        handoff = hf_gpt2_finetune_inference_distortion_handoff_report(
            report_or_path,
            top_n=top_n,
        )
    replay_args = [
        str(item) for item in list(handoff.get("recommended_bridge_cli_args") or [])
    ]
    generation_handoff_args = _handoff_generation_handoff_cli_args(handoff)
    replay_limit = max(0, int(replay_arg_limit))
    replay_preview = _cli_arg_preview(replay_args, limit=replay_limit)
    generation_preview = _cli_arg_preview(
        generation_handoff_args,
        limit=replay_limit,
    )
    lines = [
        (
            "hf_gpt2_ft_inference_handoff "
            f"status={handoff.get('status')} "
            f"source={handoff.get('source_kind')} "
            f"probe={handoff.get('recommended_probe')} "
            f"effect={handoff.get('recommended_effect_score')} "
            f"risk={handoff.get('recommended_risk_score')} "
            f"api_compat={handoff.get('recommended_api_compatibility_score')} "
            f"desire={handoff.get('desire_pressure')} "
            f"psi={handoff.get('psi_total')} "
            f"api={handoff.get('api_provider')} "
            f"dropped={handoff.get('api_request_dropped_key_count')}"
            f" retry_dropped={handoff.get('api_request_retry_dropped_key_count')}"
        )
    ]
    if replay_args:
        lines.append(
            "hf_gpt2_ft_inference_handoff_replay "
            f"arg_count={len(replay_args)} "
            f"args={replay_preview}"
        )
    if generation_handoff_args:
        lines.append(
            "hf_gpt2_ft_inference_handoff_generation "
            f"arg_count={len(generation_handoff_args)} "
            f"args={generation_preview}"
        )
    return lines


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
    corpus = report.get("corpus_file_report")
    if isinstance(corpus, Mapping):
        lines.append(
            "hf_gpt2_corpus_files "
            f"source={corpus.get('dataset_source')} "
            f"files={corpus.get('file_count')} "
            f"bytes={corpus.get('total_bytes')} "
            f"missing={corpus.get('missing_files', 'none')}"
        )
    scan = report.get("corpus_scan_report")
    if isinstance(scan, Mapping):
        lines.append(
            "hf_gpt2_corpus_scan "
            f"mode={scan.get('scan_mode')} "
            f"lines={scan.get('line_count')} "
            f"nonempty={scan.get('nonempty_line_count')} "
            f"rough_tokens={scan.get('rough_gpt2_token_estimate')} "
            f"truncated={scan.get('scan_truncated_files', 'none')} "
            f"errors={scan.get('scan_error_files', 'none')}"
        )
    handoff_lines = report.get("inference_distortion_handoff_lines")
    if isinstance(handoff_lines, Sequence) and not isinstance(
        handoff_lines,
        (str, bytes),
    ):
        lines.extend(str(line) for line in handoff_lines)
    else:
        handoff = report.get("inference_distortion_handoff")
        if isinstance(handoff, Mapping):
            lines.extend(hf_gpt2_finetune_inference_distortion_handoff_lines(handoff))
    if report.get("trainer_telemetry_enabled") is not None:
        lines.append(
            "hf_gpt2_ft_trainer_telemetry "
            f"requested={report.get('trainer_telemetry_requested')} "
            f"enabled={report.get('trainer_telemetry_enabled')} "
            f"auto={report.get('trainer_telemetry_auto_reason')} "
            f"prefix={report.get('trainer_telemetry_prefix')}"
        )
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


def _flatten_numeric_values(value: object) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("numeric tensor values cannot be text")
    if isinstance(value, Iterable):
        flattened: list[float] = []
        for item in value:
            if isinstance(item, Iterable) and not isinstance(
                item,
                (str, bytes, bytearray),
            ):
                flattened.extend(_flatten_numeric_values(item))
            else:
                flattened.append(float(item))
        return flattened
    raise TypeError("numeric tensor values must be iterable")


def _projected_tensor_values(value: object) -> list[float]:
    for name in ("data", "tolist"):
        exporter = getattr(value, name, None)
        if callable(exporter):
            return _flatten_numeric_values(exporter())
    return _flatten_numeric_values(value)


def _hf_gpt2_ft_default_topos(
    st_module: object,
    *,
    curvature: float,
    observed_token_count: int,
) -> object:
    max_depth = max(1, int(observed_token_count))
    max_volume = max(64, max_depth * 8)
    factory = getattr(st_module, "hypergrad_topos", None)
    if callable(factory):
        return factory(
            curvature=float(curvature),
            tolerance=1e-3,
            saturation=1.0,
            max_depth=max_depth,
            max_volume=max_volume,
        )

    topos_cls = getattr(st_module, "OpenCartesianTopos", None)
    if not callable(topos_cls):
        topos_cls = getattr(st_module, "OpenTopos", None)
    if not callable(topos_cls):
        raise AttributeError("SpiralTorch native OpenCartesianTopos is unavailable")

    try:
        return topos_cls(float(curvature), 1e-3, 1.0, max_depth, max_volume)
    except TypeError as new_exc:
        try:
            return topos_cls(float(curvature))
        except TypeError:
            raise new_exc


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
        topos = _hf_gpt2_ft_default_topos(
            st,
            curvature=float(curvature),
            observed_token_count=observed,
        )
        encoder = st.LanguageWaveEncoder(topos.curvature(), float(frequency))
        projector = ZSpaceProjector(topos, encoder, strength=float(strength))
        projected = projector.forward(tensor)
        projected_values = _projected_tensor_values(projected)
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


def _first_safe_number(*values: object) -> int | float | None:
    for value in values:
        number = _safe_number(value)
        if number is not None:
            return number
    return None


def _metric_fields(values: Mapping[str, object] | None) -> dict[str, object]:
    if not values:
        return {}
    return {
        str(key): _json_safe(value)
        for key, value in values.items()
        if not str(key).startswith("_")
    }


def _safe_exp(value: int | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(math.exp(float(value)))
    except (OverflowError, ValueError):
        return None


def _bounded01(value: object) -> float | None:
    number = _safe_number(value)
    if number is None:
        return None
    finite = float(number)
    if not math.isfinite(finite):
        return None
    magnitude = abs(finite)
    return magnitude / (1.0 + magnitude)


def _finite_non_negative(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _prefixed_numeric_payload(
    prefix: str,
    values: Mapping[str, object],
) -> dict[str, float]:
    clean_prefix = str(prefix or "hf_ft").strip() or "hf_ft"
    payload: dict[str, float] = {}
    for key, value in values.items():
        number = _safe_number(value)
        if number is None:
            continue
        payload[f"{clean_prefix}.{key}"] = float(number)
    return payload


def _inference_distortion_telemetry_values(
    handoff: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(handoff, Mapping):
        return {}
    config = _mapping_item(handoff, "recommended_config")
    request = _mapping_item(handoff, "recommended_request")
    processor = _mapping_item(handoff, "recommended_processor_kwargs")
    include_penalties = handoff.get("include_penalties")
    if include_penalties is None:
        include_penalties = config.get("include_penalties")
    values = {
        "inference_distortion.handoff_present": 1.0,
        "inference_distortion.effect_score": handoff.get("recommended_effect_score"),
        "inference_distortion.risk_score": handoff.get("recommended_risk_score"),
        "inference_distortion.api_compatibility_score": handoff.get(
            "recommended_api_compatibility_score"
        ),
        "inference_distortion.desire_pressure": _first_safe_number(
            handoff.get("desire_pressure"),
            config.get("desire_pressure"),
        ),
        "inference_distortion.desire_stability": _first_safe_number(
            handoff.get("desire_stability"),
            config.get("desire_stability"),
        ),
        "inference_distortion.psi_total": _first_safe_number(
            handoff.get("psi_total"),
            config.get("psi_total"),
        ),
        "inference_distortion.coherence": _first_safe_number(
            handoff.get("coherence"),
            config.get("coherence"),
        ),
        "inference_distortion.distortion_strength": _first_safe_number(
            handoff.get("distortion_strength"),
            config.get("distortion_strength"),
        ),
        "inference_distortion.base_temperature": _first_safe_number(
            handoff.get("base_temperature"),
            config.get("base_temperature"),
        ),
        "inference_distortion.base_top_p": _first_safe_number(
            handoff.get("base_top_p"),
            config.get("base_top_p"),
        ),
        "inference_distortion.request_temperature": request.get("temperature"),
        "inference_distortion.request_top_p": request.get("top_p"),
        "inference_distortion.api_request_dropped_key_count": handoff.get(
            "api_request_dropped_key_count"
        ),
        "inference_distortion.api_request_retry_dropped_key_count": handoff.get(
            "api_request_retry_dropped_key_count"
        ),
        "inference_distortion.logits_repression_strength": processor.get(
            "repression_strength"
        ),
        "inference_distortion.logits_ngram_repression_strength": processor.get(
            "ngram_repression_strength"
        ),
    }
    if isinstance(include_penalties, bool):
        values["inference_distortion.include_penalties"] = (
            1.0 if include_penalties else 0.0
        )
    return values


def hf_gpt2_finetune_training_telemetry_frame(
    event: str,
    *,
    logs: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
    state: object = None,
    previous_loss: object = None,
    telemetry_prefix: str = "hf_ft",
    desire_gain: float = 1.0,
    psi_gain: float = 1.0,
    inference_distortion_handoff: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Derive desire/psi telemetry from HF Trainer logs during FT.

    The frame is intentionally bounded and JSON-safe so it can be injected into
    trainer traces, run cards, or Z-space partial telemetry without depending on
    live native PSI hooks.
    """

    desire_gain_value = _finite_non_negative(desire_gain, label="desire_gain")
    psi_gain_value = _finite_non_negative(psi_gain, label="psi_gain")
    clean_prefix = str(telemetry_prefix or "hf_ft").strip() or "hf_ft"
    metric_payload = _metric_fields(metrics if metrics is not None else logs)
    global_step = _safe_number(_safe_attr(state, "global_step"))
    max_steps = _safe_number(_safe_attr(state, "max_steps"))
    epoch = _safe_number(_safe_attr(state, "epoch"))
    progress = None
    if global_step is not None and max_steps is not None and float(max_steps) > 0.0:
        progress = max(0.0, min(1.0, float(global_step) / float(max_steps)))

    loss_key = None
    loss = None
    for candidate in ("eval_loss", "loss", "train_loss"):
        candidate_loss = _safe_number(metric_payload.get(candidate))
        if candidate_loss is not None:
            loss_key = candidate
            loss = float(candidate_loss)
            break
    previous = _safe_number(previous_loss)
    loss_delta = None if loss is None or previous is None else loss - float(previous)
    loss_improvement = None if loss_delta is None else -loss_delta
    loss_pressure = _bounded01(loss)
    grad_norm = _safe_number(metric_payload.get("grad_norm"))
    grad_pressure = _bounded01(grad_norm)
    learning_rate = _safe_number(metric_payload.get("learning_rate"))
    lr_pressure = None
    if learning_rate is not None:
        lr_pressure = _bounded01(float(learning_rate) * 10_000.0)
    stability = None if loss_delta is None else 1.0 / (1.0 + abs(float(loss_delta)))
    improvement_pressure = _bounded01(loss_improvement)
    desire_pressure = (
        None
        if loss_pressure is None
        else min(1.0, desire_gain_value * float(loss_pressure))
    )
    saturation_terms = [
        value
        for value in (loss_pressure, grad_pressure)
        if value is not None
    ]
    desire_saturation = (
        None
        if not saturation_terms
        else min(1.0, desire_gain_value * sum(saturation_terms) / len(saturation_terms))
    )
    psi_components = [
        value
        for value in (loss_pressure, grad_pressure, lr_pressure)
        if value is not None
    ]
    psi_total = (
        None
        if not psi_components
        else min(
            1.0,
            psi_gain_value
            * sum(float(value) for value in psi_components)
            / len(psi_components),
        )
    )
    desire = {
        "gain": desire_gain_value,
        "pressure": desire_pressure,
        "stability": stability,
        "saturation": desire_saturation,
        "improvement_pressure": improvement_pressure,
    }
    psi = {
        "gain": psi_gain_value,
        "total": psi_total,
        "loss_component": loss_pressure,
        "gradient_component": grad_pressure,
        "learning_rate_component": lr_pressure,
    }
    telemetry_values = {
        "step": global_step,
        "max_steps": max_steps,
        "epoch": epoch,
        "progress": progress,
        "loss": loss,
        "loss_delta": loss_delta,
        "loss_improvement": loss_improvement,
        "grad_norm": grad_norm,
        "learning_rate": learning_rate,
        "desire.gain": desire_gain_value,
        "desire.pressure": desire_pressure,
        "desire.stability": stability,
        "desire.saturation": desire_saturation,
        "desire.improvement_pressure": improvement_pressure,
        "psi.gain": psi_gain_value,
        "psi.total": psi_total,
        "psi.loss_component": loss_pressure,
        "psi.gradient_component": grad_pressure,
        "psi.learning_rate_component": lr_pressure,
    }
    inference_handoff = (
        _json_safe(inference_distortion_handoff)
        if isinstance(inference_distortion_handoff, Mapping)
        else None
    )
    telemetry_values.update(
        _inference_distortion_telemetry_values(
            inference_handoff if isinstance(inference_handoff, Mapping) else None
        )
    )
    telemetry = _prefixed_numeric_payload(clean_prefix, telemetry_values)
    status = "ok" if telemetry else "empty"
    return {
        "row_type": "hf_gpt2_finetune_training_telemetry",
        "status": status,
        "event": str(event),
        "telemetry_prefix": clean_prefix,
        "metric_keys": csv_label(sorted(metric_payload)),
        "loss_key": loss_key,
        "loss": loss,
        "previous_loss": previous,
        "loss_delta": loss_delta,
        "loss_improvement": loss_improvement,
        "global_step": global_step,
        "epoch": epoch,
        "progress": progress,
        "desire": {key: value for key, value in desire.items() if value is not None},
        "psi": {key: value for key, value in psi.items() if value is not None},
        "inference_distortion_handoff": inference_handoff,
        "telemetry": telemetry,
    }


def hf_gpt2_finetune_eval_report(
    *,
    stage: str,
    metrics: Mapping[str, object] | None = None,
    loss_key: str = "eval_loss",
    error: object = None,
    skipped_reason: object = None,
) -> dict[str, object]:
    """Summarize a GPT-2 FT eval pass for before/after run cards."""

    metric_fields = _metric_fields(metrics)
    loss = _safe_number(metric_fields.get(loss_key))
    error_text = None if error is None else str(error)
    skipped_text = None if skipped_reason is None else str(skipped_reason)
    if error_text:
        status = "error"
    elif skipped_text:
        status = "skipped"
    elif metric_fields:
        status = "ok"
    else:
        status = "empty"
    return {
        "row_type": "hf_gpt2_finetune_eval_report",
        "stage": str(stage),
        "status": status,
        "loss_key": str(loss_key),
        "eval_loss": loss,
        "eval_perplexity": _safe_exp(loss),
        "metric_count": len(metric_fields),
        "metric_keys": csv_label(sorted(metric_fields)),
        "metrics": metric_fields,
        "error": error_text,
        "skipped_reason": skipped_text,
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


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _trace_time_bounds(
    rows: Sequence[Mapping[str, object]],
) -> tuple[float | None, float | None, float | None]:
    times = [
        float(value)
        for row in rows
        if (value := _safe_number(row.get("time_unix_s"))) is not None
    ]
    if not times:
        return None, None, None
    first_time = min(times)
    last_time = max(times)
    return first_time, last_time, last_time - first_time


def _trace_log_step_rates(rows: Sequence[Mapping[str, object]]) -> list[float]:
    log_rows: list[tuple[float, float]] = []
    for row in rows:
        if row.get("event") != "log":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or "loss" not in metrics:
            continue
        step = _safe_number(row.get("global_step"))
        timestamp = _safe_number(row.get("time_unix_s"))
        if step is None or timestamp is None:
            continue
        log_rows.append((float(step), float(timestamp)))

    rates = []
    for (prev_step, prev_time), (step, timestamp) in zip(log_rows, log_rows[1:]):
        step_delta = step - prev_step
        time_delta = timestamp - prev_time
        if step_delta > 0.0 and time_delta > 0.0:
            rates.append(step_delta / time_delta)
    return rates


def _trace_eval_loss_points(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    points = []
    for row in rows:
        metrics = row.get("metrics")
        if row.get("event") != "evaluate" or not isinstance(metrics, Mapping):
            continue
        loss = _safe_number(metrics.get("eval_loss"))
        if loss is None:
            continue
        points.append(
            {
                "step": _safe_number(row.get("global_step")),
                "eval_loss": loss,
                "eval_runtime": _safe_number(metrics.get("eval_runtime")),
                "time_unix_s": _safe_number(row.get("time_unix_s")),
            }
        )
    return points


def _trace_training_telemetry_values(
    rows: Sequence[Mapping[str, object]],
    *,
    section: str,
    key: str,
) -> list[float]:
    values = []
    for row in rows:
        source = row.get(section)
        if not isinstance(source, Mapping):
            telemetry = row.get("training_telemetry")
            source = (
                telemetry.get(section)
                if isinstance(telemetry, Mapping)
                else None
            )
        if not isinstance(source, Mapping):
            continue
        number = _safe_number(source.get(key))
        if number is not None:
            values.append(float(number))
    return values


def _trace_training_telemetry_count(rows: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for row in rows:
        telemetry = row.get("training_telemetry")
        if isinstance(telemetry, Mapping) and telemetry.get("status") == "ok":
            count += 1
    return count


def _trace_numeric_telemetry_values(
    rows: Sequence[Mapping[str, object]],
    key_suffix: str,
) -> list[float]:
    suffix = str(key_suffix)
    values = []
    for row in rows:
        source = row.get("telemetry")
        if not isinstance(source, Mapping):
            frame = row.get("training_telemetry")
            source = frame.get("telemetry") if isinstance(frame, Mapping) else None
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            key_text = str(key)
            if key_text == suffix or key_text.endswith(f".{suffix}"):
                number = _safe_number(value)
                if number is not None:
                    values.append(float(number))
                break
    return values


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
    first_time, last_time, duration_s = _trace_time_bounds(rows)
    step_rates = _trace_log_step_rates(rows)
    eval_loss_points = _trace_eval_loss_points(rows)
    eval_runtimes = [
        float(runtime)
        for point in eval_loss_points
        if (runtime := _safe_number(point.get("eval_runtime"))) is not None
    ]
    desire_pressures = _trace_training_telemetry_values(
        rows,
        section="desire",
        key="pressure",
    )
    desire_stabilities = _trace_training_telemetry_values(
        rows,
        section="desire",
        key="stability",
    )
    psi_totals = _trace_training_telemetry_values(
        rows,
        section="psi",
        key="total",
    )
    inference_desire_pressures = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.desire_pressure",
    )
    inference_psi_totals = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.psi_total",
    )
    inference_effect_scores = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.effect_score",
    )
    inference_risk_scores = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.risk_score",
    )
    inference_api_compatibility_scores = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.api_compatibility_score",
    )
    inference_api_dropped_counts = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.api_request_dropped_key_count",
    )
    inference_api_retry_dropped_counts = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.api_request_retry_dropped_key_count",
    )
    inference_repression_strengths = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.logits_repression_strength",
    )
    inference_ngram_repression_strengths = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.logits_ngram_repression_strength",
    )
    inference_include_penalties = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.include_penalties",
    )
    inference_handoff_present = _trace_numeric_telemetry_values(
        rows,
        "inference_distortion.handoff_present",
    )
    return {
        "row_type": "hf_gpt2_finetune_trainer_trace_summary",
        "trace_event_count": len(rows),
        "trace_event_counts": event_counts,
        "trace_events": csv_label(sorted(event_counts)),
        "trace_max_global_step": max_step,
        "trace_first_time_unix_s": first_time,
        "trace_last_time_unix_s": last_time,
        "trace_duration_s": duration_s,
        "trace_last_event": rows[-1].get("event") if rows else None,
        "trace_last_loss": _last_metric(rows, "loss"),
        "trace_min_loss": _min_numeric_metric(rows, "loss"),
        "trace_last_eval_loss": _last_metric(rows, "eval_loss"),
        "trace_min_eval_loss": _min_numeric_metric(rows, "eval_loss"),
        "trace_last_learning_rate": _last_metric(rows, "learning_rate"),
        "trace_log_interval_count": len(step_rates),
        "trace_log_steps_per_second_min": min(step_rates) if step_rates else None,
        "trace_log_steps_per_second_mean": _mean(step_rates),
        "trace_log_steps_per_second_max": max(step_rates) if step_rates else None,
        "trace_eval_loss_points": eval_loss_points,
        "trace_eval_loss_series": csv_label(
            f"{point.get('step')}={point.get('eval_loss')}"
            for point in eval_loss_points
        ),
        "trace_eval_runtime_min": min(eval_runtimes) if eval_runtimes else None,
        "trace_eval_runtime_mean": _mean(eval_runtimes),
        "trace_eval_runtime_max": max(eval_runtimes) if eval_runtimes else None,
        "trace_training_telemetry_count": _trace_training_telemetry_count(rows),
        "trace_last_desire_pressure": (
            desire_pressures[-1] if desire_pressures else None
        ),
        "trace_max_desire_pressure": (
            max(desire_pressures) if desire_pressures else None
        ),
        "trace_mean_desire_stability": _mean(desire_stabilities),
        "trace_last_psi_total": psi_totals[-1] if psi_totals else None,
        "trace_max_psi_total": max(psi_totals) if psi_totals else None,
        "trace_mean_psi_total": _mean(psi_totals),
        "trace_inference_distortion_telemetry_count": len(inference_handoff_present),
        "trace_last_inference_distortion_desire_pressure": (
            inference_desire_pressures[-1] if inference_desire_pressures else None
        ),
        "trace_last_inference_distortion_psi_total": (
            inference_psi_totals[-1] if inference_psi_totals else None
        ),
        "trace_last_inference_distortion_effect_score": (
            inference_effect_scores[-1] if inference_effect_scores else None
        ),
        "trace_last_inference_distortion_risk_score": (
            inference_risk_scores[-1] if inference_risk_scores else None
        ),
        "trace_last_inference_distortion_api_compatibility_score": (
            inference_api_compatibility_scores[-1]
            if inference_api_compatibility_scores
            else None
        ),
        "trace_last_inference_distortion_api_request_dropped_key_count": (
            inference_api_dropped_counts[-1]
            if inference_api_dropped_counts
            else None
        ),
        "trace_last_inference_distortion_api_request_retry_dropped_key_count": (
            inference_api_retry_dropped_counts[-1]
            if inference_api_retry_dropped_counts
            else None
        ),
        "trace_last_inference_distortion_logits_repression_strength": (
            inference_repression_strengths[-1]
            if inference_repression_strengths
            else None
        ),
        "trace_last_inference_distortion_logits_ngram_repression_strength": (
            inference_ngram_repression_strengths[-1]
            if inference_ngram_repression_strengths
            else None
        ),
        "trace_last_inference_distortion_include_penalties": (
            inference_include_penalties[-1] if inference_include_penalties else None
        ),
    }


def load_hf_gpt2_finetune_run_card(path: str | Path) -> dict[str, object]:
    """Load one GPT-2 FT run-card JSON artifact."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} did not contain a JSON object")
    return dict(payload)


def load_hf_gpt2_finetune_sweep_report(path: str | Path) -> dict[str, object]:
    """Load one GPT-2 FT sweep report JSON artifact."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} did not contain a JSON object")
    return dict(payload)


def _run_card_payload(
    card_or_path: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if isinstance(card_or_path, (str, Path)):
        path = str(card_or_path)
        return load_hf_gpt2_finetune_run_card(card_or_path), path
    if isinstance(card_or_path, Mapping):
        return dict(card_or_path), None
    raise TypeError("run card must be a Mapping or path")


def _sweep_report_payload(
    report_or_path: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if isinstance(report_or_path, (str, Path)):
        path = str(report_or_path)
        return load_hf_gpt2_finetune_sweep_report(report_or_path), path
    if isinstance(report_or_path, Mapping):
        return dict(report_or_path), None
    raise TypeError("sweep report must be a Mapping or path")


def _mapping_item(
    row: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    value = row.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _trainer_trace_summary_for_card(
    card: Mapping[str, object],
) -> dict[str, object]:
    summary = _mapping_item(card, "trainer_trace_summary")
    trace_path = card.get("trainer_trace_jsonl")
    if not isinstance(trace_path, (str, Path)) or not str(trace_path):
        return summary
    try:
        trace_summary = summarize_hf_gpt2_finetune_trainer_trace(trace_path)
    except (OSError, ValueError):
        return summary
    merged = dict(trace_summary)
    merged.update(summary)
    return merged


def _numeric_delta(
    after: object,
    before: object,
) -> float | None:
    after_number = _safe_number(after)
    before_number = _safe_number(before)
    if after_number is None or before_number is None:
        return None
    return float(after_number) - float(before_number)


def _metric_number(
    row: Mapping[str, object],
    key: str,
) -> int | float | None:
    return _safe_number(row.get(key))


def _bounded_pressure(value: object, *, scale: float = 1.0) -> float | None:
    number = _safe_number(value)
    if number is None:
        return None
    if scale <= 0.0 or not math.isfinite(scale):
        scale = 1.0
    return max(0.0, min(1.0, float(number) / scale))


def _compatibility_pressure(value: object) -> float | None:
    number = _safe_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(number)))


def _distortion_pressure_index(
    *,
    risk_score: object = None,
    api_compatibility_score: object = None,
    api_request_dropped_key_count: object = None,
    api_request_retry_dropped_key_count: object = None,
    logits_repression_strength: object = None,
    logits_ngram_repression_strength: object = None,
) -> float | None:
    terms = [
        _bounded_pressure(risk_score),
        _compatibility_pressure(api_compatibility_score),
        _bounded_pressure(api_request_dropped_key_count, scale=4.0),
        _bounded_pressure(api_request_retry_dropped_key_count, scale=2.0),
        _bounded_pressure(logits_repression_strength, scale=4.0),
        _bounded_pressure(logits_ngram_repression_strength, scale=4.0),
    ]
    values = [value for value in terms if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _distortion_adjusted_eval_loss(
    eval_loss: object,
    distortion_pressure_index: object,
) -> float | None:
    loss = _safe_number(eval_loss)
    if loss is None:
        return None
    pressure = _safe_number(distortion_pressure_index)
    if pressure is None:
        return float(loss)
    return float(loss) + (
        HF_GPT2_FT_DISTORTION_EVAL_PENALTY_WEIGHT * float(pressure)
    )


def _generation_control(
    row: Mapping[str, object],
) -> dict[str, object]:
    return _mapping_item(row, "generation_control")


def _generation_control_backend(
    control: Mapping[str, object],
) -> str | None:
    backend = control.get("backend")
    if backend:
        return str(backend)
    rows = control.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("backend"):
                return str(row.get("backend"))
    return None


def _effective_eval_after_loss(
    eval_after: Mapping[str, object],
    trainer_trace: Mapping[str, object],
) -> tuple[int | float | None, str | None]:
    loss = _metric_number(eval_after, "eval_loss")
    if loss is not None:
        return loss, "eval_after_train"
    if (
        eval_after.get("status") == "skipped"
        and eval_after.get("skipped_reason") == "final_step_eval_already_requested"
    ):
        trace_loss = _metric_number(trainer_trace, "trace_last_eval_loss")
        if trace_loss is not None:
            return trace_loss, "trainer_trace_last_eval_loss"
    return None, None


def _run_label(
    card: Mapping[str, object],
    *,
    source_path: str | None,
    run_label: str | None,
) -> str:
    if run_label:
        return str(run_label)
    for key in ("run_id", "output_dir", "model_name"):
        value = card.get(key)
        if value:
            return str(value)
    return source_path or "run"


def _generation_inference_bridge_cli_args(
    processor_kwargs: Mapping[str, object],
) -> list[str]:
    return zspace_generation_control_bridge_cli_args(
        processor_kwargs,
        include_enable_flag=True,
    )


def _string_sequence(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _inference_handoff_lines_from_payload(
    payload: Mapping[str, object],
    handoff: Mapping[str, object],
) -> list[str]:
    lines = _string_sequence(payload.get("inference_distortion_handoff_lines"))
    if lines:
        return lines
    if handoff:
        return hf_gpt2_finetune_inference_distortion_handoff_lines(handoff)
    return []


def _handoff_source_cli_args(handoff: Mapping[str, object]) -> list[str]:
    existing = _string_sequence(handoff.get("recommended_source_cli_args"))
    if existing:
        return existing
    source_path = handoff.get("source_path")
    if source_path is None:
        return []
    source_kind = handoff.get("source_kind")
    flag = (
        "--inference-distortion-probe"
        if source_kind == "probe"
        else "--inference-distortion-sweep-report"
    )
    return [flag, str(source_path)]


def _handoff_bridge_cli_args(handoff: Mapping[str, object]) -> list[str]:
    return _string_sequence(handoff.get("recommended_bridge_cli_args"))


def _handoff_generation_handoff_cli_args(
    handoff: Mapping[str, object],
) -> list[str]:
    existing = _string_sequence(
        handoff.get("recommended_generation_handoff_cli_args")
    )
    if existing:
        return existing
    source_args = _handoff_source_cli_args(handoff)
    if not source_args:
        return []
    return [*source_args, "--generation-from-inference-distortion"]


def _handoff_explicit_generation_bridge_cli_args(
    handoff: Mapping[str, object],
) -> list[str]:
    existing = _string_sequence(
        handoff.get("recommended_explicit_generation_bridge_cli_args")
    )
    if existing:
        return existing
    source_args = _handoff_source_cli_args(handoff)
    bridge_args = _handoff_bridge_cli_args(handoff)
    if not source_args or not bridge_args:
        return []
    return [*source_args, *bridge_args]


def _cli_arg_preview(args: Sequence[object], *, limit: int = 24) -> str:
    values = [str(item) for item in args]
    preview = _shell_join_args(values[: max(0, int(limit))])
    if limit and len(values) > int(limit):
        return f"{preview} ..."
    return preview


def _shell_join_args(args: Sequence[object]) -> str:
    values = [str(item) for item in args]
    if not values:
        return ""
    return shlex.join(values)


def _command_flag_value(command: Sequence[object], flag: str) -> str | None:
    values = [str(item) for item in command]
    for index, item in enumerate(values):
        if item == flag and index + 1 < len(values):
            return values[index + 1]
    return None


def _command_flag_values(command: Sequence[object], flag: str) -> list[str]:
    values = [str(item) for item in command]
    found = []
    for index, item in enumerate(values):
        if item == flag and index + 1 < len(values):
            found.append(values[index + 1])
    return found


def _scaled_int_flag_value(
    command: Sequence[object],
    flag: str,
    *,
    explicit: int | None,
    multiplier: float | None,
) -> int | None:
    if explicit is not None:
        return int(explicit)
    if multiplier is None:
        return None
    current = _safe_number(_command_flag_value(command, flag))
    if current is None or float(current) <= 0.0:
        return None
    return max(1, int(math.ceil(float(current) * float(multiplier))))


def _replace_or_append_command_flag(
    command: Sequence[object],
    flag: str,
    value: object,
) -> list[str]:
    values = [str(item) for item in command]
    replacement = str(value)
    for index, item in enumerate(values):
        if item != flag:
            continue
        if index + 1 < len(values):
            values[index + 1] = replacement
        else:
            values.append(replacement)
        return values
    values.extend([flag, replacement])
    return values


def _path_with_suffix(path: object, suffix: str) -> str | None:
    if path is None:
        return None
    text = str(path)
    if not text:
        return None
    clean_suffix = str(suffix or "scaleup").strip() or "scaleup"
    value = Path(text)
    return str(value.with_name(f"{value.name}-{clean_suffix}"))


def _command_with_overrides(
    command: Sequence[object],
    overrides: Mapping[str, object],
) -> list[str]:
    updated = [str(item) for item in command]
    for flag, value in overrides.items():
        if value is None:
            continue
        updated = _replace_or_append_command_flag(updated, str(flag), value)
    return updated


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current if current.exists() else None


def hf_gpt2_finetune_scale_up_command(
    report_or_summary: str | Path | Mapping[str, object],
    *,
    model_name: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    max_steps_multiplier: float | None = 2.0,
    max_train_samples: int | None = None,
    max_train_samples_multiplier: float | None = 2.0,
    max_eval_samples: int | None = None,
    output_dir: str | Path | None = None,
    output_suffix: str = "scaleup",
    run_card: str | Path | None = None,
    trainer_trace_jsonl: str | Path | None = None,
) -> dict[str, object]:
    """Build a longer FT command from the distortion-adjusted scale-up candidate."""

    if isinstance(report_or_summary, Mapping) and report_or_summary.get(
        "row_type"
    ) == "hf_gpt2_finetune_sweep_report_summary":
        summary = dict(report_or_summary)
    else:
        summary = summarize_hf_gpt2_finetune_sweep_report(report_or_summary)
    command_value = summary.get("scale_up_candidate_command")
    if not isinstance(command_value, Sequence) or isinstance(
        command_value,
        (str, bytes),
    ):
        return {
            "row_type": "hf_gpt2_finetune_scale_up_command",
            "status": "missing_candidate_command",
            "scale_up_candidate_label": summary.get("scale_up_candidate_label"),
        }
    base_command = [str(item) for item in command_value]
    resolved_output_dir = (
        str(output_dir)
        if output_dir is not None
        else _path_with_suffix(
            summary.get("scale_up_candidate_output_dir")
            or summary.get("scale_up_candidate_run_dir")
            or _command_flag_value(base_command, "--output-dir"),
            output_suffix,
        )
    )
    resolved_run_card = (
        str(run_card)
        if run_card is not None
        else (
            str(Path(resolved_output_dir) / "spiraltorch-hf-gpt2-ft-run-card.json")
            if resolved_output_dir
            else None
        )
    )
    resolved_trace = (
        str(trainer_trace_jsonl)
        if trainer_trace_jsonl is not None
        else (
            str(
                Path(resolved_output_dir)
                / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            )
            if resolved_output_dir
            else None
        )
    )
    resolved_max_steps = _scaled_int_flag_value(
        base_command,
        "--max-steps",
        explicit=max_steps,
        multiplier=max_steps_multiplier,
    )
    resolved_max_train_samples = _scaled_int_flag_value(
        base_command,
        "--max-train-samples",
        explicit=max_train_samples,
        multiplier=max_train_samples_multiplier,
    )
    overrides = {
        "--model-name": None if model_name is None else str(model_name),
        "--resume-from-checkpoint": (
            None if resume_from_checkpoint is None else str(resume_from_checkpoint)
        ),
        "--output-dir": resolved_output_dir,
        "--run-card": resolved_run_card,
        "--trainer-trace-jsonl": resolved_trace,
        "--max-steps": resolved_max_steps,
        "--max-train-samples": resolved_max_train_samples,
        "--max-eval-samples": max_eval_samples,
    }
    command = _command_with_overrides(base_command, overrides)
    applied = {key: value for key, value in overrides.items() if value is not None}
    return {
        "row_type": "hf_gpt2_finetune_scale_up_command",
        "status": "ok",
        "scale_up_candidate_label": summary.get("scale_up_candidate_label"),
        "scale_up_candidate_reason": summary.get("scale_up_candidate_reason"),
        "scale_up_candidate_distortion_adjusted_eval_loss": summary.get(
            "scale_up_candidate_distortion_adjusted_eval_loss"
        ),
        "scale_up_candidate_distortion_pressure_index": summary.get(
            "scale_up_candidate_distortion_pressure_index"
        ),
        "base_command": base_command,
        "base_command_display": _shell_join_args(base_command),
        "command": command,
        "command_display": _shell_join_args(command),
        "command_preview": _cli_arg_preview(command),
        "applied_overrides": applied,
        "applied_override_count": len(applied),
    }


def _scale_up_preflight_command_from_source(
    command_or_artifact: str | Path | Mapping[str, object] | Sequence[object],
) -> tuple[list[str] | None, str | None]:
    if isinstance(command_or_artifact, (str, Path)):
        path = Path(command_or_artifact)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            command, _ = _scale_up_preflight_command_from_source(payload)
            return command, str(path)
        return None, str(path)
    if isinstance(command_or_artifact, Mapping):
        command_value = command_or_artifact.get("command")
        if not isinstance(command_value, Sequence) or isinstance(
            command_value,
            (str, bytes),
        ):
            command_value = command_or_artifact.get("scale_up_candidate_command")
        if isinstance(command_value, Sequence) and not isinstance(
            command_value,
            (str, bytes),
        ):
            return [str(item) for item in command_value], None
        scale_up_command = hf_gpt2_finetune_scale_up_command(command_or_artifact)
        command_value = scale_up_command.get("command")
        if isinstance(command_value, Sequence) and not isinstance(
            command_value,
            (str, bytes),
        ):
            return [str(item) for item in command_value], None
        return None, None
    if isinstance(command_or_artifact, Sequence) and not isinstance(
        command_or_artifact,
        (str, bytes),
    ):
        return [str(item) for item in command_or_artifact], None
    return None, None


def hf_gpt2_finetune_scale_up_preflight_report(
    command_or_artifact: str | Path | Mapping[str, object] | Sequence[object],
) -> dict[str, object]:
    """Preflight a resolved scale-up command before a longer FT run."""

    command, source_path = _scale_up_preflight_command_from_source(command_or_artifact)
    if not command:
        return {
            "row_type": "hf_gpt2_finetune_scale_up_preflight",
            "status": "blocked",
            "ready": False,
            "source_path": source_path,
            "command": command,
            "error_count": 1,
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "field": "command",
                    "message": "scale-up artifact does not contain a command list",
                }
            ],
            "inputs": [],
            "outputs": [],
        }

    issues: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    executable = command[0] if command else None
    executable_resolved = None
    if executable:
        if os.sep in executable:
            executable_path = Path(executable)
            executable_resolved = str(executable_path)
            if not executable_path.is_file():
                issues.append(
                    {
                        "severity": "error",
                        "field": "executable",
                        "path": executable,
                        "message": "command executable does not exist",
                    }
                )
        else:
            executable_resolved = shutil.which(executable)
            if executable_resolved is None:
                issues.append(
                    {
                        "severity": "error",
                        "field": "executable",
                        "path": executable,
                        "message": "command executable is not on PATH",
                    }
                )
    else:
        issues.append(
            {
                "severity": "error",
                "field": "executable",
                "message": "command is empty",
            }
        )

    bridge_script = (
        command[1] if len(command) > 1 and not command[1].startswith("-") else None
    )
    if (
        bridge_script
        and bridge_script.endswith(".py")
        and not Path(bridge_script).is_file()
    ):
        issues.append(
            {
                "severity": "error",
                "field": "bridge_script",
                "path": bridge_script,
                "message": "bridge script does not exist",
            }
        )

    for flag in (
        "--train-file",
        "--validation-file",
        "--inference-distortion-sweep-report",
        "--inference-distortion-probe",
    ):
        for value in _command_flag_values(command, flag):
            path = Path(value)
            exists = path.is_file()
            inputs.append({"flag": flag, "path": str(path), "exists": exists})
            if not exists:
                issues.append(
                    {
                        "severity": "error",
                        "field": flag,
                        "path": str(path),
                        "message": "input file does not exist",
                    }
                )

    for flag in ("--resume-from-checkpoint",):
        for value in _command_flag_values(command, flag):
            path = Path(value)
            exists = path.is_dir()
            inputs.append({"flag": flag, "path": str(path), "exists": exists})
            if not exists:
                issues.append(
                    {
                        "severity": "error",
                        "field": flag,
                        "path": str(path),
                        "message": "checkpoint directory does not exist",
                    }
                )

    for flag in ("--output-dir", "--run-card", "--trainer-trace-jsonl"):
        value = _command_flag_value(command, flag)
        if value is None:
            continue
        path = Path(value)
        parent = path if flag == "--output-dir" else path.parent
        nearest = _nearest_existing_parent(parent)
        writable = bool(nearest and os.access(nearest, os.W_OK))
        outputs.append(
            {
                "flag": flag,
                "path": str(path),
                "exists": path.exists(),
                "parent": str(parent),
                "parent_exists": parent.exists(),
                "nearest_existing_parent": None if nearest is None else str(nearest),
                "nearest_existing_parent_writable": writable,
            }
        )
        if nearest is None or not writable:
            issues.append(
                {
                    "severity": "error",
                    "field": flag,
                    "path": str(path),
                    "message": "output parent is not writable",
                }
            )
        elif path.exists():
            issues.append(
                {
                    "severity": "warning",
                    "field": flag,
                    "path": str(path),
                    "message": "output target already exists",
                }
            )

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return {
        "row_type": "hf_gpt2_finetune_scale_up_preflight",
        "status": "ready" if error_count == 0 else "blocked",
        "ready": error_count == 0,
        "source_path": source_path,
        "command": command,
        "command_display": _shell_join_args(command),
        "command_preview": _cli_arg_preview(command),
        "error_count": error_count,
        "warning_count": warning_count,
        "executable": executable,
        "executable_resolved": executable_resolved,
        "bridge_script": bridge_script,
        "inputs": inputs,
        "outputs": outputs,
        "issues": issues,
    }


def hf_gpt2_finetune_scale_up_preflight_lines(
    report_or_command: str | Path | Mapping[str, object] | Sequence[object],
) -> list[str]:
    """Render concise lines for a scale-up command preflight report."""

    report = (
        dict(report_or_command)
        if isinstance(report_or_command, Mapping)
        and report_or_command.get("row_type") == "hf_gpt2_finetune_scale_up_preflight"
        else hf_gpt2_finetune_scale_up_preflight_report(report_or_command)
    )
    lines = [
        (
            "hf_gpt2_ft_scale_up_preflight "
            f"status={report.get('status')} "
            f"errors={report.get('error_count')} "
            f"warnings={report.get('warning_count')} "
            f"executable={report.get('executable_resolved') or report.get('executable')}"
        )
    ]
    for issue in report.get("issues", []):
        if not isinstance(issue, Mapping):
            continue
        lines.append(
            "hf_gpt2_ft_scale_up_preflight_issue "
            f"severity={issue.get('severity')} "
            f"field={issue.get('field')} "
            f"path={issue.get('path')} "
            f"message={issue.get('message')}"
        )
    return lines


def summarize_hf_gpt2_finetune_run_card(
    card_or_path: str | Path | Mapping[str, object],
    *,
    run_label: str | None = None,
) -> dict[str, object]:
    """Flatten one GPT-2 FT run card into comparison-friendly metrics."""

    card, source_path = _run_card_payload(card_or_path)
    dataset_fit = _mapping_item(card, "dataset_fit_report")
    eval_before = _mapping_item(card, "eval_before_train")
    eval_after = _mapping_item(card, "eval_after_train")
    generation_before = _mapping_item(card, "generation_before_train")
    generation_after = _mapping_item(card, "generation_after_train")
    generation_before_control = _generation_control(generation_before)
    generation_after_control = _generation_control(generation_after)
    generation_inference = _mapping_item(
        card,
        "generation_from_inference_distortion_applied",
    )
    generation_inference_processor = _mapping_item(
        generation_inference,
        "processor_kwargs",
    )
    generation_inference_bridge_cli_args = _generation_inference_bridge_cli_args(
        generation_inference_processor,
    )
    trainer_metrics = _mapping_item(card, "trainer_metrics")
    trainer_trace = _trainer_trace_summary_for_card(card)
    corpus_scan = _mapping_item(card, "corpus_scan_report")
    inference_handoff = _mapping_item(card, "inference_distortion_handoff")
    inference_handoff_lines = _inference_handoff_lines_from_payload(
        card,
        inference_handoff,
    )
    inference_bridge_cli_args = _handoff_bridge_cli_args(inference_handoff)
    inference_source_cli_args = _handoff_source_cli_args(inference_handoff)
    inference_generation_handoff_cli_args = _handoff_generation_handoff_cli_args(
        inference_handoff,
    )
    inference_explicit_generation_bridge_cli_args = (
        _handoff_explicit_generation_bridge_cli_args(inference_handoff)
    )
    inference_bridge_cli_display = _shell_join_args(inference_bridge_cli_args)
    inference_source_cli_display = _shell_join_args(inference_source_cli_args)
    inference_generation_handoff_cli_display = _shell_join_args(
        inference_generation_handoff_cli_args
    )
    inference_explicit_generation_bridge_cli_display = _shell_join_args(
        inference_explicit_generation_bridge_cli_args
    )

    effective_eval_after_loss, effective_eval_after_source = (
        _effective_eval_after_loss(eval_after, trainer_trace)
    )
    eval_loss_delta = _numeric_delta(
        effective_eval_after_loss,
        eval_before.get("eval_loss"),
    )
    eval_perplexity_delta = _numeric_delta(
        eval_after.get("eval_perplexity"),
        eval_before.get("eval_perplexity"),
    )
    before_continuation_hash = generation_before.get("generated_continuation_sha256")
    after_continuation_hash = generation_after.get("generated_continuation_sha256")
    generation_changed = (
        None
        if not before_continuation_hash or not after_continuation_hash
        else before_continuation_hash != after_continuation_hash
    )
    distortion_pressure_index = _distortion_pressure_index(
        risk_score=_first_safe_number(
            trainer_trace.get("trace_last_inference_distortion_risk_score"),
            inference_handoff.get("recommended_risk_score"),
        ),
        api_compatibility_score=_first_safe_number(
            trainer_trace.get(
                "trace_last_inference_distortion_api_compatibility_score"
            ),
            inference_handoff.get("recommended_api_compatibility_score"),
        ),
        api_request_dropped_key_count=_first_safe_number(
            trainer_trace.get(
                "trace_last_inference_distortion_api_request_dropped_key_count"
            ),
            inference_handoff.get("api_request_dropped_key_count"),
        ),
        api_request_retry_dropped_key_count=_first_safe_number(
            trainer_trace.get(
                "trace_last_inference_distortion_api_request_retry_dropped_key_count"
            ),
            inference_handoff.get("api_request_retry_dropped_key_count"),
        ),
        logits_repression_strength=_first_safe_number(
            trainer_trace.get(
                "trace_last_inference_distortion_logits_repression_strength"
            ),
            generation_inference_processor.get("repression_strength"),
            _mapping_item(
                inference_handoff,
                "recommended_processor_kwargs",
            ).get("repression_strength"),
        ),
        logits_ngram_repression_strength=_first_safe_number(
            trainer_trace.get(
                "trace_last_inference_distortion_logits_ngram_repression_strength"
            ),
            generation_inference_processor.get("ngram_repression_strength"),
            _mapping_item(
                inference_handoff,
                "recommended_processor_kwargs",
            ).get("ngram_repression_strength"),
        ),
    )
    distortion_adjusted_eval_loss = _distortion_adjusted_eval_loss(
        effective_eval_after_loss,
        distortion_pressure_index,
    )

    return {
        "row_type": "hf_gpt2_finetune_run_card_summary",
        "run_label": _run_label(card, source_path=source_path, run_label=run_label),
        "run_card_path": source_path,
        "model_name": card.get("model_name"),
        "dataset_name": card.get("dataset_name"),
        "dataset_source": card.get("dataset_source"),
        "dataset_format": card.get("dataset_format"),
        "block_size": _safe_number(card.get("block_size")),
        "max_train_samples": _safe_number(card.get("max_train_samples")),
        "max_eval_samples": _safe_number(card.get("max_eval_samples")),
        "load_status": card.get("load_status"),
        "failure_stage": card.get("failure_stage"),
        "failure_error": card.get("failure_error"),
        "model_saved": card.get("model_saved"),
        "raw_train_rows": _safe_number(card.get("raw_train_rows")),
        "raw_eval_rows": _safe_number(card.get("raw_eval_rows")),
        "tokenized_train_rows": _safe_number(card.get("tokenized_train_rows")),
        "tokenized_eval_rows": _safe_number(card.get("tokenized_eval_rows")),
        "dataset_fit_verdict": dataset_fit.get("verdict"),
        "dataset_fit_warnings": dataset_fit.get("warnings"),
        "train_ready": dataset_fit.get("train_ready"),
        "eval_ready": dataset_fit.get("eval_ready"),
        "eval_before_status": eval_before.get("status"),
        "eval_before_loss": _metric_number(eval_before, "eval_loss"),
        "eval_before_perplexity": _metric_number(eval_before, "eval_perplexity"),
        "eval_after_status": eval_after.get("status"),
        "eval_after_loss": _metric_number(eval_after, "eval_loss"),
        "eval_after_perplexity": _metric_number(eval_after, "eval_perplexity"),
        "effective_eval_after_loss": effective_eval_after_loss,
        "effective_eval_after_loss_source": effective_eval_after_source,
        "eval_loss_delta": eval_loss_delta,
        "eval_perplexity_delta": eval_perplexity_delta,
        "distortion_pressure_index": distortion_pressure_index,
        "distortion_adjusted_eval_loss": distortion_adjusted_eval_loss,
        "distortion_adjusted_eval_penalty_weight": (
            HF_GPT2_FT_DISTORTION_EVAL_PENALTY_WEIGHT
        ),
        "eval_loss_improved": (
            None if eval_loss_delta is None else bool(eval_loss_delta < 0.0)
        ),
        "eval_perplexity_improved": (
            None
            if eval_perplexity_delta is None
            else bool(eval_perplexity_delta < 0.0)
        ),
        "trainer_train_loss": _metric_number(trainer_metrics, "train_loss"),
        "trainer_runtime": _metric_number(trainer_metrics, "train_runtime"),
        "trainer_samples_per_second": _metric_number(
            trainer_metrics,
            "train_samples_per_second",
        ),
        "trainer_steps_per_second": _metric_number(
            trainer_metrics,
            "train_steps_per_second",
        ),
        "trainer_metric_keys": csv_label(sorted(trainer_metrics)),
        "trainer_telemetry_requested": card.get("trainer_telemetry_requested"),
        "trainer_telemetry_enabled": card.get("trainer_telemetry_enabled"),
        "trainer_telemetry_auto_reason": card.get("trainer_telemetry_auto_reason"),
        "inference_distortion_handoff_status": inference_handoff.get("status"),
        "inference_distortion_sweep_path": inference_handoff.get("source_path"),
        "inference_distortion_recommended_probe": inference_handoff.get(
            "recommended_probe"
        ),
        "inference_distortion_recommendation_reason": inference_handoff.get(
            "recommendation_reason"
        ),
        "inference_distortion_effect_score": _metric_number(
            inference_handoff,
            "recommended_effect_score",
        ),
        "inference_distortion_risk_score": _metric_number(
            inference_handoff,
            "recommended_risk_score",
        ),
        "inference_distortion_api_compatibility_score": _metric_number(
            inference_handoff,
            "recommended_api_compatibility_score",
        ),
        "inference_distortion_desire_pressure": _metric_number(
            inference_handoff,
            "desire_pressure",
        ),
        "inference_distortion_desire_stability": _metric_number(
            inference_handoff,
            "desire_stability",
        ),
        "inference_distortion_psi_total": _metric_number(
            inference_handoff,
            "psi_total",
        ),
        "inference_distortion_coherence": _metric_number(
            inference_handoff,
            "coherence",
        ),
        "inference_distortion_api_provider": inference_handoff.get("api_provider"),
        "inference_distortion_api_model": inference_handoff.get("api_model"),
        "inference_distortion_api_request_dropped_key_count": _metric_number(
            inference_handoff,
            "api_request_dropped_key_count",
        ),
        "inference_distortion_api_request_dropped_keys": csv_label(
            _unique(inference_handoff.get("api_request_dropped_keys"))
        ),
        "inference_distortion_api_request_retry_dropped_key_count": _metric_number(
            inference_handoff,
            "api_request_retry_dropped_key_count",
        ),
        "inference_distortion_api_request_retry_dropped_keys": csv_label(
            _unique(inference_handoff.get("api_request_retry_dropped_keys"))
        ),
        "inference_distortion_api_request_sent_keys": csv_label(
            _unique(inference_handoff.get("api_request_sent_keys"))
        ),
        "inference_distortion_handoff_lines": inference_handoff_lines,
        "inference_distortion_handoff_line_count": len(inference_handoff_lines),
        "inference_distortion_bridge_cli_args": inference_bridge_cli_args,
        "inference_distortion_bridge_cli_display": inference_bridge_cli_display,
        "inference_distortion_replay_arg_count": len(inference_bridge_cli_args),
        "inference_distortion_replay_cli_preview": _cli_arg_preview(
            inference_bridge_cli_args,
        ),
        "inference_distortion_source_cli_args": inference_source_cli_args,
        "inference_distortion_source_cli_display": inference_source_cli_display,
        "inference_distortion_generation_handoff_cli_args": (
            inference_generation_handoff_cli_args
        ),
        "inference_distortion_generation_handoff_cli_display": (
            inference_generation_handoff_cli_display
        ),
        "inference_distortion_generation_handoff_cli_preview": _cli_arg_preview(
            inference_generation_handoff_cli_args,
        ),
        "inference_distortion_explicit_generation_bridge_cli_args": (
            inference_explicit_generation_bridge_cli_args
        ),
        "inference_distortion_explicit_generation_bridge_cli_display": (
            inference_explicit_generation_bridge_cli_display
        ),
        "inference_distortion_explicit_generation_bridge_cli_preview": (
            _cli_arg_preview(inference_explicit_generation_bridge_cli_args)
        ),
        "trace_event_count": _metric_number(trainer_trace, "trace_event_count"),
        "trace_last_loss": _metric_number(trainer_trace, "trace_last_loss"),
        "trace_min_eval_loss": _metric_number(trainer_trace, "trace_min_eval_loss"),
        "trace_duration_s": _metric_number(trainer_trace, "trace_duration_s"),
        "trace_log_steps_per_second_min": _metric_number(
            trainer_trace,
            "trace_log_steps_per_second_min",
        ),
        "trace_log_steps_per_second_mean": _metric_number(
            trainer_trace,
            "trace_log_steps_per_second_mean",
        ),
        "trace_log_steps_per_second_max": _metric_number(
            trainer_trace,
            "trace_log_steps_per_second_max",
        ),
        "trace_eval_runtime_max": _metric_number(
            trainer_trace,
            "trace_eval_runtime_max",
        ),
        "trace_eval_loss_series": trainer_trace.get("trace_eval_loss_series"),
        "trace_training_telemetry_count": _metric_number(
            trainer_trace,
            "trace_training_telemetry_count",
        ),
        "trace_last_desire_pressure": _metric_number(
            trainer_trace,
            "trace_last_desire_pressure",
        ),
        "trace_max_desire_pressure": _metric_number(
            trainer_trace,
            "trace_max_desire_pressure",
        ),
        "trace_mean_desire_stability": _metric_number(
            trainer_trace,
            "trace_mean_desire_stability",
        ),
        "trace_last_psi_total": _metric_number(
            trainer_trace,
            "trace_last_psi_total",
        ),
        "trace_max_psi_total": _metric_number(
            trainer_trace,
            "trace_max_psi_total",
        ),
        "trace_mean_psi_total": _metric_number(
            trainer_trace,
            "trace_mean_psi_total",
        ),
        "trace_inference_distortion_telemetry_count": _metric_number(
            trainer_trace,
            "trace_inference_distortion_telemetry_count",
        ),
        "trace_last_inference_distortion_desire_pressure": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_desire_pressure",
        ),
        "trace_last_inference_distortion_psi_total": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_psi_total",
        ),
        "trace_last_inference_distortion_effect_score": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_effect_score",
        ),
        "trace_last_inference_distortion_risk_score": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_risk_score",
        ),
        "trace_last_inference_distortion_api_compatibility_score": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_api_compatibility_score",
        ),
        "trace_last_inference_distortion_api_request_dropped_key_count": (
            _metric_number(
                trainer_trace,
                "trace_last_inference_distortion_api_request_dropped_key_count",
            )
        ),
        "trace_last_inference_distortion_api_request_retry_dropped_key_count": (
            _metric_number(
                trainer_trace,
                "trace_last_inference_distortion_api_request_retry_dropped_key_count",
            )
        ),
        "trace_last_inference_distortion_logits_repression_strength": (
            _metric_number(
                trainer_trace,
                "trace_last_inference_distortion_logits_repression_strength",
            )
        ),
        "trace_last_inference_distortion_logits_ngram_repression_strength": (
            _metric_number(
                trainer_trace,
                "trace_last_inference_distortion_logits_ngram_repression_strength",
            )
        ),
        "trace_last_inference_distortion_include_penalties": _metric_number(
            trainer_trace,
            "trace_last_inference_distortion_include_penalties",
        ),
        "generation_before_status": generation_before.get("status"),
        "generation_before_method": generation_before.get("generation_method"),
        "generation_from_inference_distortion": card.get(
            "generation_from_inference_distortion"
        ),
        "generation_from_inference_distortion_status": generation_inference.get(
            "status"
        ),
        "generation_from_inference_distortion_source_kind": generation_inference.get(
            "source_kind"
        ),
        "generation_from_inference_distortion_probe": generation_inference.get(
            "recommended_probe"
        ),
        "generation_from_inference_distortion_applied_arg_count": _metric_number(
            generation_inference,
            "applied_arg_count",
        ),
        "generation_from_inference_distortion_bridge_cli_args": (
            generation_inference_bridge_cli_args
        ),
        "generation_from_inference_distortion_top_k": _metric_number(
            generation_inference_processor,
            "top_k",
        ),
        "generation_from_inference_distortion_temperature": _metric_number(
            generation_inference_processor,
            "temperature",
        ),
        "generation_from_inference_distortion_entropy_target": _metric_number(
            generation_inference_processor,
            "entropy_target",
        ),
        "generation_from_inference_distortion_repression_strength": _metric_number(
            generation_inference_processor,
            "repression_strength",
        ),
        "generation_from_inference_distortion_ngram_repression_strength": (
            _metric_number(
                generation_inference_processor,
                "ngram_repression_strength",
            )
        ),
        "generation_before_control_status": generation_before_control.get("status"),
        "generation_before_control_calls": _metric_number(
            generation_before_control,
            "calls",
        ),
        "generation_before_control_top_token_changed_count": _metric_number(
            generation_before_control,
            "top_token_changed_count",
        ),
        "generation_before_control_temperature_min": _metric_number(
            generation_before_control,
            "temperature_min",
        ),
        "generation_before_control_temperature_max": _metric_number(
            generation_before_control,
            "temperature_max",
        ),
        "generation_before_control_entropy_min": _metric_number(
            generation_before_control,
            "entropy_min",
        ),
        "generation_before_control_entropy_max": _metric_number(
            generation_before_control,
            "entropy_max",
        ),
        "generation_before_control_backend": _generation_control_backend(
            generation_before_control
        ),
        "generation_before_control_native_error": generation_before_control.get(
            "native_error"
        ),
        "generation_after_status": generation_after.get("status"),
        "generation_after_method": generation_after.get("generation_method"),
        "generation_after_control_status": generation_after_control.get("status"),
        "generation_after_control_calls": _metric_number(
            generation_after_control,
            "calls",
        ),
        "generation_after_control_top_token_changed_count": _metric_number(
            generation_after_control,
            "top_token_changed_count",
        ),
        "generation_after_control_temperature_min": _metric_number(
            generation_after_control,
            "temperature_min",
        ),
        "generation_after_control_temperature_max": _metric_number(
            generation_after_control,
            "temperature_max",
        ),
        "generation_after_control_entropy_min": _metric_number(
            generation_after_control,
            "entropy_min",
        ),
        "generation_after_control_entropy_max": _metric_number(
            generation_after_control,
            "entropy_max",
        ),
        "generation_after_control_backend": _generation_control_backend(
            generation_after_control
        ),
        "generation_after_control_native_error": generation_after_control.get(
            "native_error"
        ),
        "generation_after_new_token_count": _metric_number(
            generation_after,
            "new_token_count",
        ),
        "generation_after_continuation_char_count": _metric_number(
            generation_after,
            "generated_continuation_char_count",
        ),
        "generation_continuation_changed": generation_changed,
        "generation_before_continuation_sha256": before_continuation_hash,
        "generation_after_continuation_sha256": after_continuation_hash,
        "corpus_scan_line_count": _metric_number(corpus_scan, "line_count"),
        "corpus_scan_rough_gpt2_token_estimate": _metric_number(
            corpus_scan,
            "rough_gpt2_token_estimate",
        ),
    }


def _best_summary(
    summaries: Sequence[Mapping[str, object]],
    key: str,
) -> Mapping[str, object] | None:
    candidates = []
    for summary in summaries:
        value = _safe_number(summary.get(key))
        if value is not None:
            candidates.append((float(value), summary))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _sweep_summary_rows(
    comparison: Mapping[str, object],
) -> list[dict[str, object]]:
    summaries = comparison.get("summaries")
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        return []
    rows = []
    for row in summaries:
        if not isinstance(row, Mapping):
            continue
        summary = dict(row)
        run_card_path = summary.get("run_card_path")
        if isinstance(run_card_path, (str, Path)) and str(run_card_path):
            try:
                refreshed = summarize_hf_gpt2_finetune_run_card(
                    run_card_path,
                    run_label=(
                        str(summary["run_label"]) if summary.get("run_label") else None
                    ),
                )
            except (OSError, TypeError, ValueError):
                pass
            else:
                summary.update(refreshed)
        rows.append(summary)
    return rows


def _ranked_sweep_rows(
    summaries: Sequence[Mapping[str, object]],
    *,
    top_n: int,
) -> list[dict[str, object]]:
    def sort_key(row: Mapping[str, object]) -> tuple[float, float, str]:
        eval_after = _safe_number(row.get("effective_eval_after_loss"))
        eval_delta = _safe_number(row.get("eval_loss_delta"))
        return (
            math.inf if eval_after is None else float(eval_after),
            math.inf if eval_delta is None else float(eval_delta),
            str(row.get("run_label") or ""),
        )

    ranked = sorted(summaries, key=sort_key)
    if top_n >= 0:
        ranked = ranked[:top_n]
    rows = []
    for index, row in enumerate(ranked, 1):
        rows.append(
            {
                "rank": index,
                "run_label": row.get("run_label"),
                "run_card_path": row.get("run_card_path"),
                "eval_after_loss": _safe_number(row.get("eval_after_loss")),
                "effective_eval_after_loss": _safe_number(
                    row.get("effective_eval_after_loss")
                ),
                "effective_eval_after_loss_source": row.get(
                    "effective_eval_after_loss_source"
                ),
                "eval_loss_delta": _safe_number(row.get("eval_loss_delta")),
                "distortion_pressure_index": _safe_number(
                    row.get("distortion_pressure_index")
                ),
                "distortion_adjusted_eval_loss": _safe_number(
                    row.get("distortion_adjusted_eval_loss")
                ),
                "eval_loss_improved": row.get("eval_loss_improved"),
                "generation_continuation_changed": row.get(
                    "generation_continuation_changed"
                ),
                "generation_from_inference_distortion": row.get(
                    "generation_from_inference_distortion"
                ),
                "generation_from_inference_distortion_status": row.get(
                    "generation_from_inference_distortion_status"
                ),
                "generation_from_inference_distortion_probe": row.get(
                    "generation_from_inference_distortion_probe"
                ),
                "generation_from_inference_distortion_applied_arg_count": (
                    _safe_number(
                        row.get(
                            "generation_from_inference_distortion_applied_arg_count"
                        )
                    )
                ),
                "generation_from_inference_distortion_bridge_cli_args": list(
                    row.get("generation_from_inference_distortion_bridge_cli_args")
                    or []
                ),
                "generation_from_inference_distortion_entropy_target": _safe_number(
                    row.get("generation_from_inference_distortion_entropy_target")
                ),
                "generation_from_inference_distortion_repression_strength": (
                    _safe_number(
                        row.get(
                            "generation_from_inference_distortion_repression_strength"
                        )
                    )
                ),
                "generation_from_inference_distortion_ngram_repression_strength": (
                    _safe_number(
                        row.get(
                            "generation_from_inference_distortion_ngram_repression_strength"
                        )
                    )
                ),
                "generation_after_control_top_token_changed_count": _safe_number(
                    row.get("generation_after_control_top_token_changed_count")
                ),
                "generation_after_control_temperature_min": _safe_number(
                    row.get("generation_after_control_temperature_min")
                ),
                "generation_after_control_temperature_max": _safe_number(
                    row.get("generation_after_control_temperature_max")
                ),
                "generation_after_control_entropy_min": _safe_number(
                    row.get("generation_after_control_entropy_min")
                ),
                "generation_after_control_entropy_max": _safe_number(
                    row.get("generation_after_control_entropy_max")
                ),
                "generation_after_control_backend": row.get(
                    "generation_after_control_backend"
                ),
                "trainer_train_loss": _safe_number(row.get("trainer_train_loss")),
                "trainer_runtime": _safe_number(row.get("trainer_runtime")),
                "trainer_steps_per_second": _safe_number(
                    row.get("trainer_steps_per_second")
                ),
                "trainer_telemetry_requested": row.get(
                    "trainer_telemetry_requested"
                ),
                "trainer_telemetry_enabled": row.get("trainer_telemetry_enabled"),
                "trainer_telemetry_auto_reason": row.get(
                    "trainer_telemetry_auto_reason"
                ),
                "trace_event_count": _safe_number(row.get("trace_event_count")),
                "trace_duration_s": _safe_number(row.get("trace_duration_s")),
                "trace_log_steps_per_second_min": _safe_number(
                    row.get("trace_log_steps_per_second_min")
                ),
                "trace_log_steps_per_second_mean": _safe_number(
                    row.get("trace_log_steps_per_second_mean")
                ),
                "trace_log_steps_per_second_max": _safe_number(
                    row.get("trace_log_steps_per_second_max")
                ),
                "trace_eval_runtime_max": _safe_number(
                    row.get("trace_eval_runtime_max")
                ),
                "trace_eval_loss_series": row.get("trace_eval_loss_series"),
                "trace_training_telemetry_count": _safe_number(
                    row.get("trace_training_telemetry_count")
                ),
                "trace_last_desire_pressure": _safe_number(
                    row.get("trace_last_desire_pressure")
                ),
                "trace_max_desire_pressure": _safe_number(
                    row.get("trace_max_desire_pressure")
                ),
                "trace_mean_desire_stability": _safe_number(
                    row.get("trace_mean_desire_stability")
                ),
                "trace_last_psi_total": _safe_number(row.get("trace_last_psi_total")),
                "trace_max_psi_total": _safe_number(row.get("trace_max_psi_total")),
                "trace_mean_psi_total": _safe_number(row.get("trace_mean_psi_total")),
                "trace_inference_distortion_telemetry_count": _safe_number(
                    row.get("trace_inference_distortion_telemetry_count")
                ),
                "trace_last_inference_distortion_desire_pressure": _safe_number(
                    row.get("trace_last_inference_distortion_desire_pressure")
                ),
                "trace_last_inference_distortion_psi_total": _safe_number(
                    row.get("trace_last_inference_distortion_psi_total")
                ),
                "trace_last_inference_distortion_effect_score": _safe_number(
                    row.get("trace_last_inference_distortion_effect_score")
                ),
                "trace_last_inference_distortion_risk_score": _safe_number(
                    row.get("trace_last_inference_distortion_risk_score")
                ),
                "trace_last_inference_distortion_api_compatibility_score": _safe_number(
                    row.get("trace_last_inference_distortion_api_compatibility_score")
                ),
                "trace_last_inference_distortion_api_request_dropped_key_count": (
                    _safe_number(
                        row.get(
                            "trace_last_inference_distortion_api_request_dropped_key_count"
                        )
                    )
                ),
                "trace_last_inference_distortion_api_request_retry_dropped_key_count": (
                    _safe_number(
                        row.get(
                            "trace_last_inference_distortion_api_request_retry_dropped_key_count"
                        )
                    )
                ),
                "trace_last_inference_distortion_logits_repression_strength": (
                    _safe_number(
                        row.get(
                            "trace_last_inference_distortion_logits_repression_strength"
                        )
                    )
                ),
                "trace_last_inference_distortion_logits_ngram_repression_strength": (
                    _safe_number(
                        row.get(
                            "trace_last_inference_distortion_logits_ngram_repression_strength"
                        )
                    )
                ),
                "trace_last_inference_distortion_include_penalties": _safe_number(
                    row.get("trace_last_inference_distortion_include_penalties")
                ),
                "inference_distortion_recommended_probe": row.get(
                    "inference_distortion_recommended_probe"
                ),
                "inference_distortion_effect_score": _safe_number(
                    row.get("inference_distortion_effect_score")
                ),
                "inference_distortion_risk_score": _safe_number(
                    row.get("inference_distortion_risk_score")
                ),
                "inference_distortion_api_compatibility_score": _safe_number(
                    row.get("inference_distortion_api_compatibility_score")
                ),
                "inference_distortion_desire_pressure": _safe_number(
                    row.get("inference_distortion_desire_pressure")
                ),
                "inference_distortion_psi_total": _safe_number(
                    row.get("inference_distortion_psi_total")
                ),
                "inference_distortion_api_provider": row.get(
                    "inference_distortion_api_provider"
                ),
                "inference_distortion_api_request_dropped_key_count": _safe_number(
                    row.get("inference_distortion_api_request_dropped_key_count")
                ),
                "inference_distortion_api_request_dropped_keys": row.get(
                    "inference_distortion_api_request_dropped_keys"
                ),
                "inference_distortion_api_request_retry_dropped_key_count": _safe_number(
                    row.get("inference_distortion_api_request_retry_dropped_key_count")
                ),
                "inference_distortion_api_request_retry_dropped_keys": row.get(
                    "inference_distortion_api_request_retry_dropped_keys"
                ),
                "inference_distortion_api_request_sent_keys": row.get(
                    "inference_distortion_api_request_sent_keys"
                ),
                "inference_distortion_bridge_cli_args": list(
                    row.get("inference_distortion_bridge_cli_args") or []
                ),
                "dataset_fit_verdict": row.get("dataset_fit_verdict"),
                "failure_stage": row.get("failure_stage"),
            }
        )
    return rows


def _selected_sweep_run(
    runs: Sequence[Mapping[str, object]],
    label: object,
) -> dict[str, object]:
    if label is None:
        return {}
    label_text = str(label)
    for row in runs:
        if str(row.get("name")) == label_text:
            return dict(row)
    return {}


def _sweep_run_command(row: Mapping[str, object]) -> list[object] | None:
    command = row.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        safe = _json_safe(command)
        return list(safe) if isinstance(safe, list) else None
    return None


def summarize_hf_gpt2_finetune_sweep_report(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 5,
) -> dict[str, object]:
    """Summarize a GPT-2 FT sweep report into a scale-up decision surface."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    report, source_path = _sweep_report_payload(report_or_path)
    comparison = _mapping_item(report, "comparison")
    summaries = _sweep_summary_rows(comparison)
    inference_handoff = _mapping_item(report, "inference_distortion_handoff")
    inference_handoff_lines = _inference_handoff_lines_from_payload(
        report,
        inference_handoff,
    )
    inference_bridge_cli_args = _handoff_bridge_cli_args(inference_handoff)
    inference_source_cli_args = _handoff_source_cli_args(inference_handoff)
    inference_generation_handoff_cli_args = _handoff_generation_handoff_cli_args(
        inference_handoff,
    )
    inference_explicit_generation_bridge_cli_args = (
        _handoff_explicit_generation_bridge_cli_args(inference_handoff)
    )
    inference_bridge_cli_display = _shell_join_args(inference_bridge_cli_args)
    inference_source_cli_display = _shell_join_args(inference_source_cli_args)
    inference_generation_handoff_cli_display = _shell_join_args(
        inference_generation_handoff_cli_args
    )
    inference_explicit_generation_bridge_cli_display = _shell_join_args(
        inference_explicit_generation_bridge_cli_args
    )
    generation_inference_plan = _mapping_item(
        report,
        "generation_from_inference_distortion_plan",
    )
    scale_up_command_payload = _mapping_item(report, "scale_up_command")
    generation_inference_plan_processor = _mapping_item(
        generation_inference_plan,
        "processor_kwargs",
    )
    generation_inference_plan_bridge_cli_args = (
        _generation_inference_bridge_cli_args(
            generation_inference_plan_processor,
        )
    )
    runs_value = report.get("runs")
    runs = (
        [dict(row) for row in runs_value if isinstance(row, Mapping)]
        if isinstance(runs_value, Sequence) and not isinstance(runs_value, (str, bytes))
        else []
    )
    best_delta = _safe_number(comparison.get("best_eval_loss_delta"))
    best_delta_label = comparison.get("best_eval_loss_delta_run_label")
    best_after_label = comparison.get("best_eval_after_run_label")
    scale_up_candidate = _best_summary(summaries, "distortion_adjusted_eval_loss")
    selected_reason = "best_eval_after_loss"
    selected_label = best_after_label
    if best_delta is not None and best_delta < 0.0 and best_delta_label:
        selected_reason = "best_eval_loss_delta"
        selected_label = best_delta_label
    selected_run = _selected_sweep_run(runs, selected_label)
    scale_up_candidate_label = (
        None if scale_up_candidate is None else scale_up_candidate.get("run_label")
    )
    scale_up_candidate_run = _selected_sweep_run(runs, scale_up_candidate_label)

    completed = _safe_number(report.get("completed_run_count"))
    failed = _safe_number(report.get("failed_run_count"))
    dry_run = bool(report.get("dry_run"))
    if dry_run:
        status = "planned"
    elif completed is None or int(completed) <= 0:
        status = "no_completed_runs"
    elif failed is not None and int(failed) > 0:
        status = "partial"
    else:
        status = "complete"

    return {
        "row_type": "hf_gpt2_finetune_sweep_report_summary",
        "sweep_report_path": source_path,
        "status": status,
        "dry_run": dry_run,
        "run_count": _safe_number(report.get("run_count")),
        "attempted_run_count": _safe_number(report.get("attempted_run_count")),
        "completed_run_count": completed,
        "failed_run_count": failed,
        "reused_run_count": _safe_number(report.get("reused_run_count")),
        "skipped_run_count": _safe_number(report.get("skipped_run_count")),
        "comparison_run_count": _safe_number(comparison.get("run_count")),
        "successful_run_count": _safe_number(comparison.get("successful_run_count")),
        "eval_after_ok_count": _safe_number(comparison.get("eval_after_ok_count")),
        "eval_loss_improved_count": _safe_number(
            comparison.get("eval_loss_improved_count")
        ),
        "generation_changed_count": _safe_number(
            comparison.get("generation_changed_count")
        ),
        "generation_from_inference_distortion_count": _safe_number(
            comparison.get("generation_from_inference_distortion_count")
        ),
        "generation_from_inference_distortion_requested": report.get(
            "generation_from_inference_distortion"
        ),
        "generation_from_inference_distortion_plan_status": (
            generation_inference_plan.get("status")
        ),
        "generation_from_inference_distortion_plan_probe": (
            generation_inference_plan.get("recommended_probe")
        ),
        "generation_from_inference_distortion_plan_source_kind": (
            generation_inference_plan.get("source_kind")
        ),
        "generation_from_inference_distortion_plan_applied_arg_count": (
            _metric_number(generation_inference_plan, "applied_arg_count")
        ),
        "generation_from_inference_distortion_plan_bridge_cli_args": (
            generation_inference_plan_bridge_cli_args
        ),
        "generation_from_inference_distortion_plan_top_k": _metric_number(
            generation_inference_plan_processor,
            "top_k",
        ),
        "generation_from_inference_distortion_plan_temperature": _metric_number(
            generation_inference_plan_processor,
            "temperature",
        ),
        "generation_from_inference_distortion_plan_entropy_target": _metric_number(
            generation_inference_plan_processor,
            "entropy_target",
        ),
        "generation_from_inference_distortion_plan_repression_strength": (
            _metric_number(
                generation_inference_plan_processor,
                "repression_strength",
            )
        ),
        "generation_from_inference_distortion_plan_ngram_repression_strength": (
            _metric_number(
                generation_inference_plan_processor,
                "ngram_repression_strength",
            )
        ),
        "trainer_telemetry_requested": report.get("trainer_telemetry_requested"),
        "trainer_telemetry_enabled": report.get("trainer_telemetry_enabled"),
        "trainer_telemetry_auto_reason": report.get("trainer_telemetry_auto_reason"),
        "best_eval_after_run_label": best_after_label,
        "best_eval_after_loss": _safe_number(comparison.get("best_eval_after_loss")),
        "best_eval_after_loss_source": comparison.get("best_eval_after_loss_source"),
        "best_eval_loss_delta_run_label": best_delta_label,
        "best_eval_loss_delta": best_delta,
        "selected_run_label": selected_label,
        "selected_reason": selected_reason if selected_label else None,
        "scale_up_candidate_label": scale_up_candidate_label,
        "scale_up_candidate_reason": (
            None
            if scale_up_candidate is None
            else "lowest_distortion_adjusted_eval_loss"
        ),
        "scale_up_candidate_distortion_adjusted_eval_loss": (
            None
            if scale_up_candidate is None
            else _safe_number(scale_up_candidate.get("distortion_adjusted_eval_loss"))
        ),
        "scale_up_candidate_distortion_pressure_index": (
            None
            if scale_up_candidate is None
            else _safe_number(scale_up_candidate.get("distortion_pressure_index"))
        ),
        "scale_up_candidate_effective_eval_after_loss": (
            None
            if scale_up_candidate is None
            else _safe_number(scale_up_candidate.get("effective_eval_after_loss"))
        ),
        "scale_up_candidate_run_name": scale_up_candidate_run.get("name"),
        "scale_up_candidate_status": scale_up_candidate_run.get("status"),
        "scale_up_candidate_reused": scale_up_candidate_run.get("reused"),
        "scale_up_candidate_returncode": _safe_number(
            scale_up_candidate_run.get("returncode")
        ),
        "scale_up_candidate_run_dir": scale_up_candidate_run.get("run_dir"),
        "scale_up_candidate_output_dir": scale_up_candidate_run.get("output_dir")
        or scale_up_candidate_run.get("run_dir"),
        "scale_up_candidate_run_card": scale_up_candidate_run.get("run_card"),
        "scale_up_candidate_trainer_trace_jsonl": scale_up_candidate_run.get(
            "trainer_trace_jsonl"
        ),
        "scale_up_candidate_command_display": scale_up_candidate_run.get(
            "command_display"
        ),
        "scale_up_candidate_command": _sweep_run_command(scale_up_candidate_run),
        "scale_up_command_path": report.get("scale_up_command_path")
        or scale_up_command_payload.get("artifact_path"),
        "scale_up_command_status": report.get("scale_up_command_status")
        or scale_up_command_payload.get("status"),
        "scale_up_command_applied_override_count": _safe_number(
            scale_up_command_payload.get("applied_override_count")
        ),
        "scale_up_command_preview": report.get("scale_up_command_preview")
        or scale_up_command_payload.get("command_preview"),
        "scale_up_command_display": scale_up_command_payload.get("command_display"),
        "scale_up_command_base_display": scale_up_command_payload.get(
            "base_command_display"
        ),
        "selected_run_name": selected_run.get("name"),
        "selected_run_status": selected_run.get("status"),
        "selected_run_reused": selected_run.get("reused"),
        "selected_run_returncode": _safe_number(selected_run.get("returncode")),
        "selected_run_dir": selected_run.get("run_dir"),
        "selected_output_dir": selected_run.get("output_dir")
        or selected_run.get("run_dir"),
        "selected_run_card": selected_run.get("run_card"),
        "selected_trainer_trace_jsonl": selected_run.get("trainer_trace_jsonl"),
        "selected_command_display": selected_run.get("command_display"),
        "selected_command": _sweep_run_command(selected_run),
        "inference_distortion_sweep_report": report.get(
            "inference_distortion_sweep_report"
        ),
        "inference_distortion_handoff_status": inference_handoff.get("status"),
        "inference_distortion_recommended_probe": inference_handoff.get(
            "recommended_probe"
        ),
        "inference_distortion_recommendation_reason": inference_handoff.get(
            "recommendation_reason"
        ),
        "inference_distortion_effect_score": _metric_number(
            inference_handoff,
            "recommended_effect_score",
        ),
        "inference_distortion_risk_score": _metric_number(
            inference_handoff,
            "recommended_risk_score",
        ),
        "inference_distortion_api_compatibility_score": _metric_number(
            inference_handoff,
            "recommended_api_compatibility_score",
        ),
        "inference_distortion_desire_pressure": _metric_number(
            inference_handoff,
            "desire_pressure",
        ),
        "inference_distortion_psi_total": _metric_number(
            inference_handoff,
            "psi_total",
        ),
        "inference_distortion_api_provider": inference_handoff.get("api_provider"),
        "inference_distortion_api_request_dropped_key_count": _metric_number(
            inference_handoff,
            "api_request_dropped_key_count",
        ),
        "inference_distortion_api_request_dropped_keys": csv_label(
            _unique(inference_handoff.get("api_request_dropped_keys"))
        ),
        "inference_distortion_api_request_retry_dropped_key_count": _metric_number(
            inference_handoff,
            "api_request_retry_dropped_key_count",
        ),
        "inference_distortion_api_request_retry_dropped_keys": csv_label(
            _unique(inference_handoff.get("api_request_retry_dropped_keys"))
        ),
        "inference_distortion_api_request_sent_keys": csv_label(
            _unique(inference_handoff.get("api_request_sent_keys"))
        ),
        "inference_distortion_handoff_lines": inference_handoff_lines,
        "inference_distortion_handoff_line_count": len(inference_handoff_lines),
        "inference_distortion_bridge_cli_args": inference_bridge_cli_args,
        "inference_distortion_bridge_cli_display": inference_bridge_cli_display,
        "inference_distortion_replay_arg_count": len(inference_bridge_cli_args),
        "inference_distortion_replay_cli_preview": _cli_arg_preview(
            inference_bridge_cli_args,
        ),
        "inference_distortion_source_cli_args": inference_source_cli_args,
        "inference_distortion_source_cli_display": inference_source_cli_display,
        "inference_distortion_generation_handoff_cli_args": (
            inference_generation_handoff_cli_args
        ),
        "inference_distortion_generation_handoff_cli_display": (
            inference_generation_handoff_cli_display
        ),
        "inference_distortion_generation_handoff_cli_preview": _cli_arg_preview(
            inference_generation_handoff_cli_args,
        ),
        "inference_distortion_explicit_generation_bridge_cli_args": (
            inference_explicit_generation_bridge_cli_args
        ),
        "inference_distortion_explicit_generation_bridge_cli_display": (
            inference_explicit_generation_bridge_cli_display
        ),
        "inference_distortion_explicit_generation_bridge_cli_preview": (
            _cli_arg_preview(inference_explicit_generation_bridge_cli_args)
        ),
        "top_runs": _ranked_sweep_rows(summaries, top_n=top_n),
        "failed_runs": [
            {
                "name": row.get("name"),
                "run_card": row.get("run_card"),
                "returncode": row.get("returncode"),
                "command_display": row.get("command_display"),
            }
            for row in runs
            if row.get("returncode") is not None
            and _safe_number(row.get("returncode")) != 0
        ],
    }


def summarize_hf_gpt2_finetune_sweep_report_lines(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 3,
) -> list[str]:
    """Render a compact human-readable summary for a GPT-2 FT sweep report."""

    summary = summarize_hf_gpt2_finetune_sweep_report(
        report_or_path,
        top_n=top_n,
    )
    lines = [
        (
            "hf_gpt2_ft_sweep "
            f"status={summary.get('status')} "
            f"runs={summary.get('completed_run_count')}/{summary.get('run_count')} "
            f"failed={summary.get('failed_run_count')} "
            f"reused={summary.get('reused_run_count')} "
            f"skipped={summary.get('skipped_run_count')}"
        ),
        (
            "hf_gpt2_ft_sweep_best "
            f"selected={summary.get('selected_run_label')} "
            f"reason={summary.get('selected_reason')} "
            f"best_eval_after={summary.get('best_eval_after_loss')} "
            f"best_delta={summary.get('best_eval_loss_delta')}"
        ),
    ]
    if summary.get("scale_up_candidate_label") is not None:
        lines.append(
            "hf_gpt2_ft_sweep_scale_up "
            f"candidate={summary.get('scale_up_candidate_label')} "
            f"reason={summary.get('scale_up_candidate_reason')} "
            f"status={summary.get('scale_up_candidate_status')} "
            "adjusted_eval="
            f"{summary.get('scale_up_candidate_distortion_adjusted_eval_loss')} "
            "pressure="
            f"{summary.get('scale_up_candidate_distortion_pressure_index')} "
            "eval_after="
            f"{summary.get('scale_up_candidate_effective_eval_after_loss')} "
            f"card={summary.get('scale_up_candidate_run_card')} "
            f"trace={summary.get('scale_up_candidate_trainer_trace_jsonl')} "
            f"dir={summary.get('scale_up_candidate_run_dir')}"
        )
    if summary.get("scale_up_command_status") is not None:
        lines.append(
            "hf_gpt2_ft_sweep_scale_up_command "
            f"status={summary.get('scale_up_command_status')} "
            f"overrides={summary.get('scale_up_command_applied_override_count')} "
            f"path={summary.get('scale_up_command_path')} "
            f"preview={summary.get('scale_up_command_preview')}"
        )
    if summary.get("inference_distortion_recommended_probe") is not None:
        lines.append(
            "hf_gpt2_ft_sweep_inference_handoff "
            f"status={summary.get('inference_distortion_handoff_status')} "
            f"probe={summary.get('inference_distortion_recommended_probe')} "
            f"effect={summary.get('inference_distortion_effect_score')} "
            f"risk={summary.get('inference_distortion_risk_score')} "
            f"api_compat={summary.get('inference_distortion_api_compatibility_score')} "
            f"desire={summary.get('inference_distortion_desire_pressure')} "
            f"psi={summary.get('inference_distortion_psi_total')} "
            f"api={summary.get('inference_distortion_api_provider')} "
            "api_dropped="
            f"{summary.get('inference_distortion_api_request_dropped_key_count')}"
            " api_retry_dropped="
            f"{summary.get('inference_distortion_api_request_retry_dropped_key_count')}"
        )
        bridge_args = list(summary.get("inference_distortion_bridge_cli_args") or [])
        if bridge_args:
            lines.append(
                "hf_gpt2_ft_sweep_inference_handoff_replay "
                f"arg_count={len(bridge_args)} "
                f"args={_cli_arg_preview(bridge_args)}"
            )
        generation_handoff_args = list(
            summary.get("inference_distortion_generation_handoff_cli_args") or []
        )
        if generation_handoff_args:
            lines.append(
                "hf_gpt2_ft_sweep_inference_handoff_generation "
                f"arg_count={len(generation_handoff_args)} "
                f"args={_cli_arg_preview(generation_handoff_args)}"
            )
    if summary.get("selected_run_card") or summary.get("selected_trainer_trace_jsonl"):
        lines.append(
            "hf_gpt2_ft_sweep_selected "
            f"run={summary.get('selected_run_label')} "
            f"status={summary.get('selected_run_status')} "
            f"card={summary.get('selected_run_card')} "
            f"trace={summary.get('selected_trainer_trace_jsonl')} "
            f"dir={summary.get('selected_run_dir')}"
        )
    if summary.get("generation_from_inference_distortion_plan_status") is not None:
        lines.append(
            "hf_gpt2_ft_sweep_generation_inference_plan "
            f"status={summary.get('generation_from_inference_distortion_plan_status')} "
            f"probe={summary.get('generation_from_inference_distortion_plan_probe')} "
            "repress="
            f"{summary.get('generation_from_inference_distortion_plan_repression_strength')} "
            "entropy="
            f"{summary.get('generation_from_inference_distortion_plan_entropy_target')} "
            "ngram_repress="
            f"{summary.get('generation_from_inference_distortion_plan_ngram_repression_strength')}"
        )
    if summary.get("trainer_telemetry_enabled") is not None:
        lines.append(
            "hf_gpt2_ft_sweep_trainer_telemetry "
            f"requested={summary.get('trainer_telemetry_requested')} "
            f"enabled={summary.get('trainer_telemetry_enabled')} "
            f"auto={summary.get('trainer_telemetry_auto_reason')}"
        )
    for row in summary.get("top_runs", []):
        if not isinstance(row, Mapping):
            continue
        inference_fragment = ""
        if row.get("inference_distortion_recommended_probe") is not None:
            inference_fragment = (
                f"infer_probe={row.get('inference_distortion_recommended_probe')} "
                f"infer_effect={row.get('inference_distortion_effect_score')} "
                f"infer_api_compat={row.get('inference_distortion_api_compatibility_score')} "
            )
        if row.get("trace_inference_distortion_telemetry_count") is not None:
            inference_fragment += (
                "infer_trace="
                f"{row.get('trace_inference_distortion_telemetry_count')} "
                "infer_trace_risk="
                f"{row.get('trace_last_inference_distortion_risk_score')} "
                "infer_trace_retry_drop="
                f"{row.get('trace_last_inference_distortion_api_request_retry_dropped_key_count')} "
                "infer_trace_repress="
                f"{row.get('trace_last_inference_distortion_logits_repression_strength')} "
            )
        generation_inference_fragment = ""
        if row.get("generation_from_inference_distortion_status") is not None:
            generation_inference_fragment = (
                "gen_infer="
                f"{row.get('generation_from_inference_distortion_status')} "
                "gen_infer_probe="
                f"{row.get('generation_from_inference_distortion_probe')} "
                "gen_repress="
                f"{row.get('generation_from_inference_distortion_repression_strength')} "
                "gen_entropy="
                f"{row.get('generation_from_inference_distortion_entropy_target')} "
            )
        lines.append(
            "hf_gpt2_ft_sweep_top "
            f"rank={row.get('rank')} "
            f"run={row.get('run_label')} "
            f"eval_after={row.get('effective_eval_after_loss')} "
            f"source={row.get('effective_eval_after_loss_source')} "
            f"delta={row.get('eval_loss_delta')} "
            f"adjusted={row.get('distortion_adjusted_eval_loss')} "
            f"pressure={row.get('distortion_pressure_index')} "
            f"trainer_sps={row.get('trainer_steps_per_second')} "
            f"trace_sps_mean={row.get('trace_log_steps_per_second_mean')} "
            f"eval_series={row.get('trace_eval_loss_series')} "
            f"psi={row.get('trace_last_psi_total')} "
            f"desire={row.get('trace_last_desire_pressure')} "
            f"telemetry={row.get('trainer_telemetry_enabled')} "
            f"telemetry_auto={row.get('trainer_telemetry_auto_reason')} "
            f"{inference_fragment}"
            f"{generation_inference_fragment}"
            f"changed={row.get('generation_continuation_changed')} "
            f"zcontrol_changed={row.get('generation_after_control_top_token_changed_count')} "
            f"zcontrol_backend={row.get('generation_after_control_backend')}"
        )
    return lines


def _curve_step_value(value: object) -> int | float | None:
    number = _safe_number(value)
    if number is None:
        return None
    finite = float(number)
    if not math.isfinite(finite):
        return None
    return int(finite) if finite.is_integer() else finite


def _curve_eval_loss_points(
    trace_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    raw_points = trace_summary.get("trace_eval_loss_points")
    if not isinstance(raw_points, Sequence) or isinstance(
        raw_points,
        (str, bytes, bytearray),
    ):
        return []
    points: list[dict[str, object]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, Mapping):
            continue
        step = _curve_step_value(raw_point.get("step"))
        loss = _safe_number(raw_point.get("eval_loss"))
        if step is None or loss is None:
            continue
        points.append(
            {
                "step": step,
                "eval_loss": float(loss),
                "eval_runtime": _safe_number(raw_point.get("eval_runtime")),
                "time_unix_s": _safe_number(raw_point.get("time_unix_s")),
            }
        )
    return sorted(points, key=lambda point: float(point["step"]))


def _curve_eval_loss_by_step(
    points: Sequence[Mapping[str, object]],
) -> dict[int | float, float]:
    losses: dict[int | float, float] = {}
    for point in points:
        step = _curve_step_value(point.get("step"))
        loss = _safe_number(point.get("eval_loss"))
        if step is not None and loss is not None:
            losses[step] = float(loss)
    return losses


def _curve_path_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).as_posix().rstrip("/")


def _curve_pathish_equal(left: object, right: object) -> bool:
    left_text = _curve_path_text(left)
    right_text = _curve_path_text(right)
    if not left_text or not right_text:
        return False
    return (
        left_text == right_text
        or left_text.endswith(f"/{right_text}")
        or right_text.endswith(f"/{left_text}")
    )


def _curve_run_dir_candidates(
    card: Mapping[str, object],
    source_path: str | None,
) -> list[str]:
    candidates: list[str] = []
    for key in ("output_dir", "run_dir", "scale_up_candidate_output_dir"):
        text = _curve_path_text(card.get(key))
        if text:
            candidates.append(text)
    trace_path = _curve_path_text(card.get("trainer_trace_jsonl"))
    if trace_path:
        candidates.append(Path(trace_path).parent.as_posix())
    if source_path:
        candidates.append(Path(source_path).parent.as_posix())
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _curve_checkpoint_step(model_name: object) -> int | None:
    text = _curve_path_text(model_name)
    if not text:
        return None
    for segment in reversed(text.split("/")):
        if not segment.startswith("checkpoint-"):
            continue
        raw_step = segment[len("checkpoint-") :]
        if raw_step.isdigit():
            return int(raw_step)
    return None


def _curve_explicit_step(
    model_name: object,
    step_by_model: Mapping[str, object] | None,
) -> int | float | None:
    if not step_by_model:
        return None
    name = str(model_name)
    basename = Path(name).name
    for key, value in step_by_model.items():
        key_text = str(key)
        if key_text == name or key_text == basename:
            return _curve_step_value(value)
    return None


def _curve_model_step(
    model_name: object,
    *,
    card: Mapping[str, object],
    source_path: str | None,
    step_by_model: Mapping[str, object] | None,
    eval_loss_by_step: Mapping[int | float, float],
) -> int | float | None:
    explicit = _curve_explicit_step(model_name, step_by_model)
    if explicit is not None:
        return explicit
    checkpoint_step = _curve_checkpoint_step(model_name)
    if checkpoint_step is not None:
        return checkpoint_step
    base_model = card.get("model_name")
    if base_model is not None and str(model_name) == str(base_model):
        return 0 if 0 in eval_loss_by_step else None
    max_step = max(eval_loss_by_step, key=float, default=None)
    if max_step is None:
        return None
    for run_dir in _curve_run_dir_candidates(card, source_path):
        if _curve_pathish_equal(model_name, run_dir):
            return max_step
    return None


def _curve_best_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None

    def sort_key(row: Mapping[str, object]) -> tuple[float, float, float, str]:
        best_loop = _safe_number(row.get("mean_best_loop_score"))
        reduction = _safe_number(row.get("mean_loop_score_reduction_ratio"))
        eval_loss = _safe_number(row.get("eval_loss"))
        return (
            math.inf if best_loop is None else float(best_loop),
            math.inf if reduction is None else -float(reduction),
            math.inf if eval_loss is None else float(eval_loss),
            str(row.get("model_name") or ""),
        )

    return dict(min(rows, key=sort_key))


def hf_gpt2_finetune_generation_curve_report(
    card_or_path: str | Path | Mapping[str, object],
    sweeps_or_paths: Mapping[str, object] | Sequence[object],
    *,
    labels: Sequence[str] | None = None,
    step_by_model: Mapping[str, object] | None = None,
    top_n: int = 5,
) -> dict[str, object]:
    """Join FT eval-loss trace points with generation-control sweep summaries."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    from .hf_generation import compare_zspace_generation_control_sweeps

    card, source_path = _run_card_payload(card_or_path)
    trace_summary = _trainer_trace_summary_for_card(card)
    eval_points = _curve_eval_loss_points(trace_summary)
    eval_by_step = _curve_eval_loss_by_step(eval_points)
    comparison = compare_zspace_generation_control_sweeps(
        sweeps_or_paths,
        labels=labels,
        top_n=top_n,
    )
    rows: list[dict[str, object]] = []
    for model_row in comparison.get("model_rows", []):
        if not isinstance(model_row, Mapping):
            continue
        model_name = str(model_row.get("model_name") or "unknown")
        step = _curve_model_step(
            model_name,
            card=card,
            source_path=source_path,
            step_by_model=step_by_model,
            eval_loss_by_step=eval_by_step,
        )
        eval_loss = eval_by_step.get(step) if step is not None else None
        rows.append(
            {
                "model_name": model_name,
                "step": step,
                "eval_loss": eval_loss,
                "eval_loss_source": (
                    None if step is None or eval_loss is None else f"trace_step_{step}"
                ),
                "is_checkpoint": _curve_checkpoint_step(model_name) is not None,
                "is_final_output": any(
                    _curve_pathish_equal(model_name, run_dir)
                    for run_dir in _curve_run_dir_candidates(card, source_path)
                ),
                "sweep_count": model_row.get("sweep_count"),
                "prompt_count": model_row.get("prompt_count"),
                "zspace_helped_count": model_row.get("zspace_helped_count"),
                "mean_baseline_loop_score": model_row.get(
                    "mean_baseline_loop_score"
                ),
                "mean_best_loop_score": model_row.get("mean_best_loop_score"),
                "mean_loop_score_delta_from_baseline": model_row.get(
                    "mean_loop_score_delta_from_baseline"
                ),
                "mean_loop_score_reduction_ratio": model_row.get(
                    "mean_loop_score_reduction_ratio"
                ),
                "max_top_token_changed_count": model_row.get(
                    "max_top_token_changed_count"
                ),
            }
        )
    rows = sorted(
        rows,
        key=lambda row: (
            math.inf if row.get("step") is None else float(row["step"]),
            str(row.get("model_name") or ""),
        ),
    )
    best_row = _curve_best_row(rows)
    return {
        "row_type": "hf_gpt2_finetune_generation_curve",
        "status": "complete",
        "run_card_path": source_path,
        "model_name": card.get("model_name"),
        "dataset_name": card.get("dataset_name"),
        "dataset_config": card.get("dataset_config"),
        "eval_loss_series": trace_summary.get("trace_eval_loss_series"),
        "eval_loss_points": eval_points,
        "comparison": comparison,
        "curve_rows": rows,
        "curve_model_count": len(rows),
        "sweep_count": comparison.get("sweep_count"),
        "completed_sweep_count": comparison.get("completed_sweep_count"),
        "prompt_count": comparison.get("prompt_count"),
        "recommended_model_name": (
            None if best_row is None else best_row.get("model_name")
        ),
        "recommended_step": None if best_row is None else best_row.get("step"),
        "recommended_eval_loss": (
            None if best_row is None else best_row.get("eval_loss")
        ),
        "recommended_mean_best_loop_score": (
            None if best_row is None else best_row.get("mean_best_loop_score")
        ),
        "recommended_reason": (
            None
            if best_row is None
            else "lowest_mean_best_loop_highest_reduction_eval_loss_tiebreak"
        ),
    }


def hf_gpt2_finetune_generation_curve_lines(
    report_or_card: str | Path | Mapping[str, object],
    sweeps_or_paths: Mapping[str, object] | Sequence[object] | None = None,
    *,
    labels: Sequence[str] | None = None,
    step_by_model: Mapping[str, object] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Render a compact checkpoint-generation curve for FT run artifacts."""

    from .hf_generation import summarize_zspace_generation_control_sweep_comparison_lines

    if (
        isinstance(report_or_card, Mapping)
        and report_or_card.get("row_type") == "hf_gpt2_finetune_generation_curve"
    ):
        report = dict(report_or_card)
    else:
        if sweeps_or_paths is None:
            raise TypeError(
                "sweeps_or_paths is required when report_or_card is not a curve"
            )
        report = hf_gpt2_finetune_generation_curve_report(
            report_or_card,
            sweeps_or_paths,
            labels=labels,
            step_by_model=step_by_model,
            top_n=top_n,
        )
    lines = [
        (
            "hf_gpt2_ft_generation_curve "
            f"status={report.get('status')} "
            f"prompts={report.get('prompt_count')} "
            f"models={report.get('curve_model_count')} "
            f"sweeps={report.get('completed_sweep_count')}/"
            f"{report.get('sweep_count')} "
            f"eval_loss_series={report.get('eval_loss_series')} "
            f"recommend={report.get('recommended_model_name')} "
            f"step={report.get('recommended_step')}"
        )
    ]
    for row in report.get("curve_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "hf_gpt2_ft_generation_curve_row "
            f"model={row.get('model_name')} "
            f"step={row.get('step')} "
            f"eval_loss={row.get('eval_loss')} "
            f"sweeps={row.get('sweep_count')} "
            f"helped={row.get('zspace_helped_count')} "
            f"mean_baseline_loop={row.get('mean_baseline_loop_score')} "
            f"mean_best_loop={row.get('mean_best_loop_score')} "
            f"mean_delta={row.get('mean_loop_score_delta_from_baseline')} "
            f"mean_reduction={row.get('mean_loop_score_reduction_ratio')} "
            f"top_changes={row.get('max_top_token_changed_count')}"
        )
    comparison = report.get("comparison")
    if isinstance(comparison, Mapping):
        lines.extend(
            summarize_zspace_generation_control_sweep_comparison_lines(
                comparison,
                top_n=top_n,
            )
        )
    return lines


def compare_hf_gpt2_finetune_run_cards(
    cards_or_paths: Sequence[str | Path | Mapping[str, object]],
    *,
    run_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compare GPT-2 FT run cards by eval, generation, and trainer signals."""

    labels = list(run_labels or [])
    summaries = [
        summarize_hf_gpt2_finetune_run_card(
            card,
            run_label=labels[index] if index < len(labels) else None,
        )
        for index, card in enumerate(cards_or_paths)
    ]
    best_after = _best_summary(summaries, "effective_eval_after_loss")
    best_delta = _best_summary(summaries, "eval_loss_delta")
    best_adjusted = _best_summary(summaries, "distortion_adjusted_eval_loss")
    run_label_values = [str(summary.get("run_label")) for summary in summaries]
    return {
        "row_type": "hf_gpt2_finetune_run_card_comparison",
        "run_count": len(summaries),
        "run_labels": csv_label(run_label_values),
        "successful_run_count": sum(
            1 for summary in summaries if not summary.get("failure_stage")
        ),
        "eval_after_ok_count": sum(
            1 for summary in summaries if summary.get("eval_after_status") == "ok"
        ),
        "eval_loss_improved_count": sum(
            1 for summary in summaries if summary.get("eval_loss_improved") is True
        ),
        "generation_changed_count": sum(
            1
            for summary in summaries
            if summary.get("generation_continuation_changed") is True
        ),
        "generation_from_inference_distortion_count": sum(
            1
            for summary in summaries
            if summary.get("generation_from_inference_distortion_status") == "ok"
        ),
        "best_eval_after_run_label": (
            None if best_after is None else best_after.get("run_label")
        ),
        "best_eval_after_loss": (
            None if best_after is None else best_after.get("effective_eval_after_loss")
        ),
        "best_eval_after_loss_source": (
            None
            if best_after is None
            else best_after.get("effective_eval_after_loss_source")
        ),
        "best_eval_loss_delta_run_label": (
            None if best_delta is None else best_delta.get("run_label")
        ),
        "best_eval_loss_delta": (
            None if best_delta is None else best_delta.get("eval_loss_delta")
        ),
        "best_distortion_adjusted_run_label": (
            None if best_adjusted is None else best_adjusted.get("run_label")
        ),
        "best_distortion_adjusted_eval_loss": (
            None
            if best_adjusted is None
            else best_adjusted.get("distortion_adjusted_eval_loss")
        ),
        "best_distortion_adjusted_pressure_index": (
            None
            if best_adjusted is None
            else best_adjusted.get("distortion_pressure_index")
        ),
        "summaries": summaries,
    }


def hf_gpt2_finetune_trainer_trace_callback(
    path: str | Path,
    *,
    run_id: str | None = None,
    reset: bool = True,
    zspace_probe_tokens: Sequence[int | float] | None = None,
    zspace_probe_kwargs: Mapping[str, object] | None = None,
    inference_distortion_handoff: Mapping[str, object] | None = None,
    training_telemetry: bool = False,
    telemetry_prefix: str = "hf_ft",
    desire_gain: float = 1.0,
    psi_gain: float = 1.0,
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
    inference_handoff = (
        _json_safe(inference_distortion_handoff)
        if isinstance(inference_distortion_handoff, Mapping)
        else None
    )
    desire_gain_value = _finite_non_negative(desire_gain, label="desire_gain")
    psi_gain_value = _finite_non_negative(psi_gain, label="psi_gain")

    class SpiralTorchHFTrainerTraceCallback(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.path = trace_path
            self.run_id = run_id
            self.event_count = 0
            self.last_telemetry_loss: float | None = None
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
            merged_extra = dict(extra or {})
            if training_telemetry:
                telemetry_frame = hf_gpt2_finetune_training_telemetry_frame(
                    event,
                    logs=logs,
                    metrics=metrics,
                    state=state,
                    previous_loss=self.last_telemetry_loss,
                    telemetry_prefix=telemetry_prefix,
                    desire_gain=desire_gain_value,
                    psi_gain=psi_gain_value,
                    inference_distortion_handoff=(
                        inference_handoff
                        if isinstance(inference_handoff, Mapping)
                        else None
                    ),
                )
                merged_extra["training_telemetry"] = telemetry_frame
                merged_extra["telemetry"] = telemetry_frame.get("telemetry")
                merged_extra["desire"] = telemetry_frame.get("desire")
                merged_extra["psi"] = telemetry_frame.get("psi")
                frame_loss = _safe_number(telemetry_frame.get("loss"))
                if frame_loss is not None:
                    self.last_telemetry_loss = float(frame_loss)
            row = hf_gpt2_finetune_trainer_trace_event(
                event,
                args=args,
                state=state,
                control=control,
                logs=logs,
                metrics=metrics,
                run_id=self.run_id,
                extra=merged_extra,
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
            if inference_handoff is not None:
                extra["inference_distortion_handoff"] = inference_handoff
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

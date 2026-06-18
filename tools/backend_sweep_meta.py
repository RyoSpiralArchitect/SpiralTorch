#!/usr/bin/env python3
"""Small helpers for surfacing backend runtime metadata in sweep tools."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


BACKEND_META_HEADERS = [
    "backend_status",
    "backend_kernels",
    "policy_matmul",
    "policy_softmax",
    "rt_wgpu_initialized",
    "rt_wgpu_error",
    "rt_wgpu_ctx",
    "rt_wgpu_ready",
    "rt_wgpu_statuses",
]

BACKEND_RESIDUAL_HEADERS = [
    "fallback_share",
    "util_route_status",
    "util_route_values",
    "util_route_threshold",
    "cpu_residual_ops",
    "cpu_residual_share",
    "cpu_residual_top",
    "cpu_threshold_ops",
    "cpu_threshold_share",
    "cpu_threshold_top",
    "cpu_trace_ops",
    "cpu_trace_share",
    "cpu_trace_top",
    "cpu_control_ops",
    "cpu_control_share",
    "cpu_control_top",
    "cpu_runtime_fallback_ops",
    "cpu_runtime_fallback_share",
    "cpu_runtime_fallback_top",
    "cpu_copy_ops",
    "cpu_copy_share",
    "cpu_copy_top",
    "cpu_debt_ops",
    "cpu_debt_share",
    "cpu_debt_top",
    "wgpu_kernel_ops",
    "wgpu_kernel_share",
    "wgpu_kernel_top",
]

_TENSOR_OP_PREFIX = "tensor_op_backend_"
_TENSOR_OP_WGPU_RUNTIME_FALLBACK_PREFIX = "tensor_op_backend_wgpu_runtime_fallback_"
_CPU_BACKEND_SUFFIXES = ("cpu_eigen", "cpu_simd", "f64_cpu", "cpu", "naive")
_WGPU_BACKEND_SUFFIXES = ("wgpu",)
_TENSOR_UTIL_ROUTE_STATUS_PREFIX = "backend_policy_status_tensor_util_route_"
_THRESHOLD_PROTECTED_OPS = {
    "add",
    "add_row_inplace",
    "add_scaled",
    "categorical_cross_entropy_backward",
    "categorical_cross_entropy_forward",
    "dynamic_field_hamilton_jacobi_backward",
    "dynamic_field_hamilton_jacobi_forward",
    "dynamic_field_klein_gordon_backward",
    "dynamic_field_klein_gordon_forward",
    "dynamic_field_stochastic_schrodinger_backward",
    "dynamic_field_stochastic_schrodinger_forward",
    "embedding_backward",
    "embedding_forward",
    "gelu_backward",
    "hadamard",
    "hyperbolic_cross_entropy_backward",
    "hyperbolic_cross_entropy_forward",
    "hypergrad_accumulate_wave",
    "avg_pool2d_backward",
    "avg_pool2d_forward",
    "max_pool2d_backward",
    "max_pool2d_forward",
    "mse_loss_backward",
    "mse_loss_forward",
    "mul_row",
    "row_affine",
    "project_to_poincare",
    "relu_backward",
    "relu_forward",
    "scale",
    "squared_l2_norm",
    "sub",
    "sum_axis0",
    "sum_axis0_scaled",
    "transpose",
    "wave_gate_backward",
    "wave_gate_project",
    "wave_scan_backward",
    "wave_scan_forward",
    "zspace_coherence_scan_backward",
    "zspace_coherence_scan_forward",
    "zspace_softmax_backward",
}
_TRACE_ONLY_OPS = {
    "graph_flow_drain",
    "graph_flow_elliptic_annotation",
    "graph_flow_layer_begin",
    "graph_flow_roundtable_annotation",
    "graph_flow_weight_update",
    "lawvere_guard_probability_slice_control",
    "psi_heatmap_distribution_summary",
    "spectral_lr_scale_optimizer_control",
    "tensor_biome_renormalise_weights_control",
    "warmup_cosine_lr_step_optimizer_control",
    "zspace_optimizer_lr_scale_optimizer_control",
    "zspace_semantic_window_semantic_control",
    "zspace_maxwell_pulse_summary_summary",
    "zrba_metric_weights_normalise_control",
    "zrba_workspace_softmax_summary_summary",
    "zspace_region_heatmap_report",
}
_CONTROL_PLANE_OPS = {
    "ameba_autograd_round",
    "distributed_topk_merge",
    "distributed_trainer_async_enqueue",
    "distributed_trainer_async_merge",
    "distributed_trainer_sync_step",
    "onebit_allreduce_hook",
    "zero_partition_hook",
}
_COPY_ONLY_OPS = {
    "cat_rows",
}
_ALIAS_ONLY_OPS = {
    "lstm_backward_bptt_scan",
    "lstm_backward_bias_gradient",
    "lstm_backward_input_gradient",
    "lstm_backward_parameter_gradient_reduction",
    "lstm_backward_parameter_gradient_scale",
    "lstm_backward_raw_parameter_gradient",
    "lstm_backward_recurrent",
    "lstm_forward_bias",
    "lstm_forward_input_projection",
    "lstm_forward_recurrent",
}


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def status_label(summary: dict[str, Any]) -> str:
    if summary.get("failed"):
        return "failed"
    if summary.get("skipped"):
        return "skipped"
    return "ok"


def failure_label(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    return str(value) if value else "-"


def returncode_label(value: Any) -> str:
    if not isinstance(value, int):
        return "-"
    if value < 0:
        return f"signal:{-value}"
    return str(value)


def first_log_line(text: str, *, limit: int = 160) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return "-"


def first_failure_line(text: str, *, limit: int = 160) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("$ "):
            continue
        if line.startswith("error") or "could not compile" in line:
            return line[:limit]
    return first_log_line(text, limit=limit)


def classify_failure(returncode: int, log_path: Path) -> tuple[str, str]:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    stripped = text.strip()
    if returncode < 0:
        detail = f"signal:{-returncode}"
        if not stripped:
            detail += ":empty_log"
        return "signal", detail
    if "could not compile" in text or "error[E" in text:
        return "compile", first_failure_line(stripped)
    return "exit", first_log_line(stripped) if stripped else f"exit:{returncode}:empty_log"


def run_logged_command(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def write_failure(run_dir: Path, failure: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def preflight_skipped_run_record(
    *,
    schema: str,
    backend: str,
    seed: int,
    run_dir: Path,
    log_path: Path,
    command: list[str],
    preflight_failure: dict[str, Any],
) -> dict[str, Any]:
    failure_kind = f"preflight_{preflight_failure.get('failure_kind', 'failed')}"
    failure_detail = str(preflight_failure.get("failure_detail") or "preflight failed")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "skipped after WGPU sweep preflight failure",
                f"preflight_log_path={preflight_failure.get('log_path')}",
                f"failure_kind={failure_kind}",
                f"failure_detail={failure_detail}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    failure = {
        "schema": schema,
        "backend": backend,
        "seed": seed,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": preflight_failure.get("returncode", 1),
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "command": command,
        "preflight_failure": preflight_failure,
    }
    write_failure(run_dir, failure)
    return {
        "backend": backend,
        "seed": seed,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": preflight_failure.get("returncode", 1),
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "skipped": True,
        "failed": True,
        "command": command,
    }


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def load_run_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {}
    return read_json_object(path)


def bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "-"


def short_label(value: Any, *, limit: int = 96) -> str:
    if not isinstance(value, str) or not value:
        return "-"
    return value.replace("\n", " ")[:limit]


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fmt_number(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _metric_number(summary: dict[str, Any], key: str, field: str = "last") -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    return _number(item.get(field))


def _metric_sum(summary: dict[str, Any], key: str) -> float | None:
    return _metric_number(summary, key, "sum")


def _metric_backend(metric_name: str, suffixes: tuple[str, ...]) -> tuple[str, str] | None:
    if not metric_name.startswith(_TENSOR_OP_PREFIX):
        return None
    body = metric_name[len(_TENSOR_OP_PREFIX) :]
    for suffix in sorted(suffixes, key=len, reverse=True):
        marker = f"_{suffix}"
        if body.endswith(marker):
            return body[: -len(marker)], suffix
    return None


def _tensor_op_counts(
    summary: dict[str, Any],
    suffixes: tuple[str, ...],
) -> dict[str, float]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    counts: dict[str, float] = {}
    for metric_name, item in metrics.items():
        if not isinstance(metric_name, str) or not isinstance(item, dict):
            continue
        if metric_name.startswith(_TENSOR_OP_WGPU_RUNTIME_FALLBACK_PREFIX):
            continue
        split = _metric_backend(metric_name, suffixes)
        if split is None:
            continue
        op_name, _backend = split
        value = _number(item.get("last"))
        if value is None or value <= 0.0:
            continue
        counts[op_name] = counts.get(op_name, 0.0) + value
    return counts


def _top_counts_label(counts: dict[str, float], limit: int) -> str:
    ranked = [
        (count, name)
        for name, count in counts.items()
        if count > 0.0
    ]
    if not ranked:
        return "-"
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ",".join(f"{name}:{_fmt_number(count, 0)}" for count, name in ranked[:limit])


def _dominant_metric_suffix_label(summary: dict[str, Any], prefix: str) -> str:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return "-"
    candidates: list[tuple[float, str]] = []
    for key in sorted(metrics):
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        total = _metric_sum(summary, key)
        if total is not None and total > 0.0:
            candidates.append((total, key[len(prefix) :]))
    if not candidates:
        return "-"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    count, label = candidates[0]
    return f"{label}:{_fmt_number(count, 0)}"


def _share(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _filtered_counts(counts: dict[str, float], names: set[str]) -> dict[str, float]:
    return {name: count for name, count in counts.items() if name in names}


def _without_counts(counts: dict[str, float], names: set[str]) -> dict[str, float]:
    return {name: count for name, count in counts.items() if name not in names}


def _subtract_counts(counts: dict[str, float], subtract: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, count in counts.items():
        remaining = count - subtract.get(name, 0.0)
        if remaining > 0.0:
            out[name] = remaining
    return out


def _has_tensor_util_cpu_threshold_route(summary: dict[str, Any]) -> bool:
    return (
        _metric_sum(
            summary,
            f"{_TENSOR_UTIL_ROUTE_STATUS_PREFIX}cpu_threshold",
        )
        or 0.0
    ) > 0.0


def _wgpu_runtime_fallback_counts(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    counts: dict[str, float] = {}
    for metric_name, item in metrics.items():
        if not isinstance(metric_name, str) or not isinstance(item, dict):
            continue
        if not metric_name.startswith(_TENSOR_OP_WGPU_RUNTIME_FALLBACK_PREFIX):
            continue
        body = metric_name[len(_TENSOR_OP_WGPU_RUNTIME_FALLBACK_PREFIX) :]
        for suffix in sorted(_CPU_BACKEND_SUFFIXES, key=len, reverse=True):
            marker = f"_{suffix}"
            if not body.endswith(marker):
                continue
            op_name = body[: -len(marker)]
            value = _number(item.get("last"))
            if value is not None and value > 0.0:
                counts[op_name] = counts.get(op_name, 0.0) + value
            break
    return counts


def backend_residual_columns(
    trainer_summary: dict[str, Any],
    *,
    top_n: int = 3,
) -> dict[str, str]:
    """Summarize remaining CPU/WGPU op mix from a trainer trace summary."""

    defaults = {key: "-" for key in BACKEND_RESIDUAL_HEADERS}
    tensor_ops = _metric_number(trainer_summary, "tensor_ops_total")
    fallbacks = _metric_number(trainer_summary, "tensor_backend_fallbacks")

    cpu_counts = _without_counts(
        _tensor_op_counts(trainer_summary, _CPU_BACKEND_SUFFIXES),
        _ALIAS_ONLY_OPS,
    )
    cpu_detail_total = sum(cpu_counts.values())
    cpu_backend_total = sum(
        value
        for value in (
            _metric_number(trainer_summary, "tensor_backend_cpu"),
            _metric_number(trainer_summary, "tensor_backend_cpu_simd"),
            _metric_number(trainer_summary, "tensor_backend_f64_cpu"),
            _metric_number(trainer_summary, "tensor_backend_naive"),
        )
        if value is not None
    )
    cpu_total = cpu_detail_total if cpu_detail_total > 0.0 else cpu_backend_total

    wgpu_counts = _without_counts(
        _tensor_op_counts(trainer_summary, _WGPU_BACKEND_SUFFIXES),
        _ALIAS_ONLY_OPS,
    )
    wgpu_detail_total = sum(wgpu_counts.values())
    wgpu_backend_total = _metric_number(trainer_summary, "tensor_backend_wgpu")
    wgpu_total = (
        wgpu_detail_total
        if wgpu_detail_total > 0.0
        else (wgpu_backend_total if wgpu_backend_total is not None else 0.0)
    )
    threshold_counts = (
        _filtered_counts(cpu_counts, _THRESHOLD_PROTECTED_OPS)
        if _has_tensor_util_cpu_threshold_route(trainer_summary)
        else {}
    )
    threshold_total = sum(threshold_counts.values())
    trace_counts = _filtered_counts(cpu_counts, _TRACE_ONLY_OPS)
    trace_total = sum(trace_counts.values())
    control_counts = _filtered_counts(cpu_counts, _CONTROL_PLANE_OPS)
    control_total = sum(control_counts.values())
    runtime_fallback_counts = _wgpu_runtime_fallback_counts(trainer_summary)
    runtime_fallback_total = sum(runtime_fallback_counts.values())
    copy_counts = _filtered_counts(cpu_counts, _COPY_ONLY_OPS)
    copy_total = sum(copy_counts.values())
    excluded_debt_ops = (
        set(threshold_counts) | set(trace_counts) | set(control_counts) | set(copy_counts)
    )
    debt_counts = _subtract_counts(
        _without_counts(cpu_counts, excluded_debt_ops),
        runtime_fallback_counts,
    )
    debt_total = sum(debt_counts.values())

    defaults["fallback_share"] = _fmt_number(_share(fallbacks, tensor_ops), digits=3)
    defaults["util_route_status"] = _dominant_metric_suffix_label(
        trainer_summary,
        _TENSOR_UTIL_ROUTE_STATUS_PREFIX,
    )
    defaults["util_route_values"] = _fmt_number(
        _metric_number(trainer_summary, "backend_policy_tensor_util_last_values"),
        digits=0,
    )
    defaults["util_route_threshold"] = _fmt_number(
        _metric_number(trainer_summary, "backend_policy_tensor_util_last_threshold"),
        digits=0,
    )
    defaults["cpu_residual_ops"] = _fmt_number(cpu_total, digits=0)
    defaults["cpu_residual_share"] = _fmt_number(_share(cpu_total, tensor_ops), digits=3)
    defaults["cpu_residual_top"] = _top_counts_label(cpu_counts, top_n)
    defaults["cpu_threshold_ops"] = _fmt_number(threshold_total, digits=0)
    defaults["cpu_threshold_share"] = _fmt_number(
        _share(threshold_total, tensor_ops),
        digits=3,
    )
    defaults["cpu_threshold_top"] = _top_counts_label(threshold_counts, top_n)
    defaults["cpu_trace_ops"] = _fmt_number(trace_total, digits=0)
    defaults["cpu_trace_share"] = _fmt_number(_share(trace_total, tensor_ops), digits=3)
    defaults["cpu_trace_top"] = _top_counts_label(trace_counts, top_n)
    defaults["cpu_control_ops"] = _fmt_number(control_total, digits=0)
    defaults["cpu_control_share"] = _fmt_number(_share(control_total, tensor_ops), digits=3)
    defaults["cpu_control_top"] = _top_counts_label(control_counts, top_n)
    defaults["cpu_runtime_fallback_ops"] = _fmt_number(runtime_fallback_total, digits=0)
    defaults["cpu_runtime_fallback_share"] = _fmt_number(
        _share(runtime_fallback_total, tensor_ops),
        digits=3,
    )
    defaults["cpu_runtime_fallback_top"] = _top_counts_label(runtime_fallback_counts, top_n)
    defaults["cpu_copy_ops"] = _fmt_number(copy_total, digits=0)
    defaults["cpu_copy_share"] = _fmt_number(_share(copy_total, tensor_ops), digits=3)
    defaults["cpu_copy_top"] = _top_counts_label(copy_counts, top_n)
    defaults["cpu_debt_ops"] = _fmt_number(debt_total, digits=0)
    defaults["cpu_debt_share"] = _fmt_number(_share(debt_total, tensor_ops), digits=3)
    defaults["cpu_debt_top"] = _top_counts_label(debt_counts, top_n)
    defaults["wgpu_kernel_ops"] = _fmt_number(wgpu_total, digits=0)
    defaults["wgpu_kernel_share"] = _fmt_number(_share(wgpu_total, tensor_ops), digits=3)
    defaults["wgpu_kernel_top"] = _top_counts_label(wgpu_counts, top_n)
    return defaults


def backend_residual_row(trainer_summary: dict[str, Any]) -> list[str]:
    columns = backend_residual_columns(trainer_summary)
    return [columns[header] for header in BACKEND_RESIDUAL_HEADERS]


def roundtable_wgpu_statuses(run_meta: dict[str, Any]) -> list[str]:
    audit = run_meta.get("roundtable_backend_audit")
    if not isinstance(audit, dict):
        return []
    bands = audit.get("bands")
    if not isinstance(bands, list):
        return []
    statuses: list[str] = []
    for band in bands:
        if not isinstance(band, dict):
            continue
        band_name = str(band.get("band", "?"))
        status = band.get("wgpu_exact_status")
        if isinstance(status, str) and status:
            statuses.append(f"{band_name}:{status}")
    return statuses


def roundtable_wgpu_summary(run_meta: dict[str, Any]) -> dict[str, Any]:
    audit = run_meta.get("roundtable_backend_audit")
    if not isinstance(audit, dict):
        return {}
    return {
        "requested_backend": audit.get("requested_backend"),
        "wgpu_runtime_compiled": audit.get("wgpu_runtime_compiled"),
        "wgpu_runtime_context_installed": audit.get("wgpu_runtime_context_installed"),
        "any_wgpu_exact_runtime_ready": audit.get("any_wgpu_exact_runtime_ready"),
        "statuses": roundtable_wgpu_statuses(run_meta),
    }


def backend_meta_row(run_meta: dict[str, Any]) -> list[str]:
    policy = run_meta.get("tensor_policy")
    runtime = run_meta.get("backend_runtime")
    roundtable = run_meta.get("roundtable_backend_audit")
    policy = policy if isinstance(policy, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    roundtable = roundtable if isinstance(roundtable, dict) else {}
    statuses = roundtable_wgpu_statuses(run_meta)
    return [
        str(runtime.get("requested_backend_status", "-")),
        bool_label(runtime.get("requested_backend_kernels_wired")),
        str(policy.get("matmul_backend", "-")),
        str(policy.get("softmax_backend", "-")),
        bool_label(runtime.get("wgpu_rank_runtime_initialized")),
        short_label(runtime.get("wgpu_rank_runtime_error")),
        bool_label(runtime.get("wgpu_rank_runtime_context_installed")),
        bool_label(roundtable.get("any_wgpu_exact_runtime_ready")),
        ",".join(statuses) if statuses else "-",
    ]


def backend_manifest_fields(run_meta: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    runtime = run_meta.get("backend_runtime")
    if isinstance(runtime, dict):
        fields["backend_runtime"] = runtime
    policy = run_meta.get("tensor_policy")
    if isinstance(policy, dict):
        fields["tensor_policy"] = policy
    roundtable_summary = roundtable_wgpu_summary(run_meta)
    if roundtable_summary:
        fields["roundtable_wgpu"] = roundtable_summary
    return fields

#!/usr/bin/env python3
"""Summarize learning backend debt from traces and source heuristics.

This is intentionally lightweight: it combines current trainer trace residuals
with static source patterns that often indicate older CPU-only learning
boundaries. The output is a review aid, not a verifier.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOTS = (
    Path("crates/st-nn/src"),
    Path("crates/st-tensor/src"),
    Path("crates/spiral-selfsup/src"),
    Path("crates/st-core/src/distributed"),
    Path("crates/st-core/src/engine"),
    Path("crates/st-core/src/backend"),
)


@dataclass(frozen=True)
class SourceCandidate:
    priority: str
    kind: str
    file: Path
    line: int
    evidence: str
    note: str


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER_TRACE = _load_module(
    "spiraltorch_trainer_trace", ROOT / "bindings/st-py/spiraltorch/trainer_trace.py"
)
BACKEND_META = _load_module("backend_sweep_meta", ROOT / "tools/backend_sweep_meta.py")


def _metric_last(summary: dict[str, Any], key: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("last")
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.{digits}f}"
    return str(value)


def _run_backend_label(run_dir: Path) -> str | None:
    meta = BACKEND_META.load_run_meta(run_dir)
    backend = meta.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    runtime = meta.get("backend_runtime")
    if isinstance(runtime, dict):
        requested = runtime.get("requested_backend")
        if isinstance(requested, str) and requested:
            return requested
    return None


def _trace_rows(run_root: Path, *, include_cpu_runs: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trace_path in sorted(run_root.glob("**/trainer_trace.jsonl")):
        run_dir = trace_path.parent
        backend = _run_backend_label(run_dir)
        if not include_cpu_runs and backend == "cpu":
            continue
        try:
            summary = TRAINER_TRACE.summarize_trainer_trace_events(trace_path)
            residual = BACKEND_META.backend_residual_columns(summary)
        except Exception as exc:  # pragma: no cover - best-effort audit helper
            rows.append(
                {
                    "run": str(run_dir.relative_to(run_root)),
                    "error": repr(exc),
                }
            )
            continue
        rows.append(
            {
                "run": str(run_dir.relative_to(run_root)),
                "tensor_ops": _fmt(_metric_last(summary, "tensor_ops_total"), 0),
                "wgpu_ops": _fmt(_metric_last(summary, "tensor_backend_wgpu"), 0),
                "cpu_ops": _fmt(_metric_last(summary, "tensor_backend_cpu"), 0),
                "fallbacks": _fmt(_metric_last(summary, "tensor_backend_fallbacks"), 0),
                "cpu_debt_ops": residual["cpu_debt_ops"],
                "lstm_est_cpu_debt_ops": _fmt(
                    _metric_last(summary, "lstm_estimated_cpu_debt_ops"), 0
                ),
                "lstm_est_gate_cpu_debt_ops": _fmt(
                    _metric_last(summary, "lstm_estimated_gate_activation_cpu_debt_ops"), 0
                ),
                "lstm_est_gate_wgpu_ops": _fmt(
                    _metric_last(summary, "lstm_estimated_gate_activation_wgpu_ops"), 0
                ),
                "lstm_est_bptt_cpu_debt_ops": _fmt(
                    _metric_last(summary, "lstm_estimated_bptt_cpu_debt_ops"), 0
                ),
                "lstm_est_bptt_wgpu_ops": _fmt(
                    _metric_last(summary, "lstm_estimated_bptt_wgpu_ops"), 0
                ),
                "lstm_scan_rt_req": _fmt(
                    _metric_last(summary, "lstm_backward_bptt_scan_runtime_requested"), 0
                ),
                "lstm_scan_rt_ok": _fmt(
                    _metric_last(summary, "lstm_backward_bptt_scan_runtime_available"), 0
                ),
                "lstm_scan_rt_miss": _fmt(
                    _metric_last(summary, "lstm_backward_bptt_scan_runtime_unavailable"), 0
                ),
                "cpu_debt_top": residual["cpu_debt_top"],
                "cpu_trace_top": residual["cpu_trace_top"],
                "cpu_control_ops": residual["cpu_control_ops"],
                "cpu_control_top": residual["cpu_control_top"],
                "cpu_runtime_fallback_ops": residual["cpu_runtime_fallback_ops"],
                "cpu_runtime_fallback_top": residual["cpu_runtime_fallback_top"],
                "cpu_copy_top": residual["cpu_copy_top"],
                "wgpu_top": residual["wgpu_kernel_top"],
            }
        )
    return rows


def _relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path)


def _failure_row_from_record(
    record: dict[str, Any],
    *,
    run_root: Path,
    fallback_run: str,
) -> dict[str, str] | None:
    failed = record.get("failed") is True or record.get("run_status") == "failed"
    failure_kind = record.get("failure_kind")
    if not failed and not failure_kind:
        return None
    run_label = _run_label_from_record(record, run_root=run_root, fallback_run=fallback_run)
    log_path = record.get("log_path")
    log_label = "-"
    if isinstance(log_path, str) and log_path:
        log_label = _relative_label(Path(log_path), run_root)
    return {
        "run": run_label,
        "backend": str(record.get("backend") or "-"),
        "status": str(record.get("run_status") or ("failed" if failed else "-")),
        "returncode": BACKEND_META.returncode_label(record.get("returncode")),
        "failure_kind": str(failure_kind or "-"),
        "failure_detail": str(record.get("failure_detail") or "-"),
        "log": log_label,
    }


def _run_label_from_record(
    record: dict[str, Any],
    *,
    run_root: Path,
    fallback_run: str,
) -> str:
    run_dir = record.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        return _relative_label(Path(run_dir), run_root)
    name = record.get("name")
    if isinstance(name, str) and name:
        return name
    return fallback_run


def _failed_run_rows(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    resolved_runs: set[str] = set()
    resolved_preflights: set[str] = set()

    for sweep_path in sorted(run_root.glob("**/sweep.json")):
        try:
            payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best-effort audit helper
            continue
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            continue
        sweep_label = _relative_label(sweep_path.parent, run_root)
        preflight_failures = payload.get("preflight_failures")
        backends = payload.get("backends")
        if isinstance(preflight_failures, dict) and isinstance(backends, list):
            failed_backends = {
                str(backend) for backend, failure in preflight_failures.items() if failure
            }
            for backend in backends:
                backend_label = str(backend)
                if backend_label not in failed_backends:
                    resolved_preflights.add(
                        f"{sweep_label}/_preflight/backend-{backend_label}"
                    )
        for idx, record in enumerate(runs):
            if not isinstance(record, dict):
                continue
            fallback = f"{sweep_label}#run-{idx}"
            run_label = _run_label_from_record(
                record,
                run_root=run_root,
                fallback_run=fallback,
            )
            row = _failure_row_from_record(record, run_root=run_root, fallback_run=fallback)
            if row is None:
                resolved_runs.add(run_label)
                continue
            key = row["run"]
            seen.add(key)
            rows.append(row)

    for failure_path in sorted(run_root.glob("**/failure.json")):
        try:
            record = json.loads(failure_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best-effort audit helper
            continue
        if not isinstance(record, dict):
            continue
        fallback = _relative_label(failure_path.parent, run_root)
        row = _failure_row_from_record(record, run_root=run_root, fallback_run=fallback)
        if (
            row is None
            or row["run"] in seen
            or row["run"] in resolved_runs
            or row["run"] in resolved_preflights
        ):
            continue
        seen.add(row["run"])
        rows.append(row)

    return rows


def _candidate_priority(path: Path, kind: str) -> str:
    path_str = str(path)
    if "crates/st-nn/src/layers" in path_str or "crates/st-nn/src/trainer" in path_str:
        if kind in {
            "plain_matmul",
            "plain_transpose",
            "plain_softmax",
            "cpu_auto_meta",
            "recurrent_cpu_meta",
            "normalization_cpu_meta",
        }:
            return "P1"
        return "P2"
    if "crates/st-nn/src/z_rba" in path_str or "crates/st-nn/src/zspace" in path_str:
        return "P2"
    if kind == "semantic_sparse_scan_cpu_meta" and path_str == "crates/st-nn/src/language/geometry.rs":
        return "P2"
    if path_str == "crates/st-core/src/distributed/prob_params.rs":
        return "P3"
    if "crates/st-core/src/distributed" in path_str:
        return "P2" if kind == "distributed_gradient_cpu_meta" else "P3"
    if "crates/st-core/src/engine/hook_points.rs" in path_str:
        return "P2" if kind == "engine_gradient_hook_cpu_meta" else "P3"
    if "crates/st-core/src/backend" in path_str:
        return "P3"
    if "crates/st-tensor/src/pure/topos.rs" in path_str:
        return "P2"
    if "crates/st-tensor/src" in path_str:
        return "P3"
    return "P3"


def _test_start(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "mod tests {" or stripped.startswith("mod tests {"):
            return idx
    return None


def _source_candidates(source_roots: Iterable[Path], *, include_tests: bool) -> list[SourceCandidate]:
    patterns = [
        (
            "plain_matmul",
            re.compile(r"\.matmul\("),
            "plain Tensor::matmul bypasses current_matmul_backend() policy",
        ),
        (
            "plain_transpose",
            re.compile(r"\.transpose\(\)"),
            "plain Tensor::transpose bypasses tensor-util routing",
        ),
        (
            "plain_softmax",
            re.compile(r"\.row_softmax\(\)"),
            "plain row_softmax bypasses current_softmax_backend() policy",
        ),
        (
            "explicit_auto_backend",
            re.compile(r"TensorUtilBackend::Auto|MatmulBackend::Auto|SoftmaxBackend::Auto"),
            "explicit Auto may select legacy backend outside trainer policy",
        ),
    ]
    candidates: list[SourceCandidate] = []
    for root in source_roots:
        abs_root = (ROOT / root).resolve()
        if not abs_root.exists():
            continue
        for path in sorted(abs_root.rglob("*.rs")):
            rel = path.relative_to(ROOT)
            lines = path.read_text(encoding="utf-8").splitlines()
            test_start = _test_start(lines)
            for idx, line in enumerate(lines):
                if not include_tests and test_start is not None and idx >= test_start:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                for kind, pattern, note in patterns:
                    if kind == "explicit_auto_backend" and rel == Path("crates/st-nn/src/execution.rs"):
                        # execution.rs owns the thread-local backend-policy fallback.
                        # Its Auto values are the policy default, not stale call sites.
                        continue
                    if kind == "explicit_auto_backend" and "=>" in stripped:
                        # Label helpers like `TensorUtilBackend::Auto => "auto"` are not
                        # backend decisions.
                        continue
                    if kind == "plain_transpose" and stripped.startswith(".transpose()"):
                        # Rust's Option/Result transpose commonly appears as a chained
                        # `.transpose()?`; this is not Tensor::transpose.
                        continue
                    if kind == "plain_transpose" and "eigen" in stripped:
                        # Nalgebra eigensolver reconstruction uses matrix/vector
                        # transpose internally; it is not Tensor::transpose debt.
                        continue
                    if kind == "plain_transpose" and str(rel).endswith(
                        "zspace_coherence/vae.rs"
                    ):
                        # The VAE path uses nalgebra vector/matrix transpose, not
                        # SpiralTorch Tensor::transpose.
                        continue
                    if pattern.search(stripped) and "_with_backend" not in stripped:
                        candidate_kind = kind
                        candidate_note = note
                        if kind == "explicit_auto_backend" and rel == Path(
                            "crates/st-tensor/src/pure.rs"
                        ):
                            candidate_kind = "standalone_auto_default"
                            candidate_note = (
                                "standalone tensor/LinearModel Auto default; learning paths can "
                                "opt into explicit *_with_backend APIs when policy context exists"
                            )
                        candidates.append(
                            SourceCandidate(
                                priority=_candidate_priority(rel, candidate_kind),
                                kind=candidate_kind,
                                file=rel,
                                line=idx + 1,
                                evidence=stripped,
                                note=candidate_note,
                            )
                        )
                is_recurrent_cpu_meta = (
                    '"backend": "recurrent_cpu"' in stripped
                    or '"recurrent_backend": "cpu"' in stripped
                    or '"bptt_backend": if backward { Some("cpu") } else { None }' in stripped
                    or '"bptt_backend": bptt_backend' in stripped
                )
                normalization_window = "\n".join(
                    lines[max(0, idx - 8) : min(idx + 8, len(lines))]
                )
                is_normalization_cpu_meta = (
                    '"input_gradient_backend": "cpu"' in stripped
                    or '"normalization_backend": "cpu"' in stripped
                    or (
                        'Some("cpu")' in stripped
                        and "emit_normalization_backward_meta" in normalization_window
                    )
                )
                if '"backend": "normalization_cpu"' in stripped:
                    is_normalization_cpu_meta = True
                is_psd_eigen_cpu_meta = '"psd_projection_backend": "cpu_eigen"' in stripped
                is_semantic_sparse_scan_cpu_meta = (
                    '"semantic_sparse_scan_backend": "semantic_cpu"' in stripped
                )
                is_semantic_cpu_meta = (
                    '"backend": "semantic_cpu"' in stripped
                    or '"semantic_inference_backend": "semantic_cpu"' in stripped
                    or is_semantic_sparse_scan_cpu_meta
                    or '"semantic_accumulation_backend": "semantic_cpu"' in stripped
                    or '"semantic_sanitize_backend": "semantic_cpu"' in stripped
                    or '"window_energy_backend": "semantic_cpu"' in stripped
                    or '"fusion_accumulation_backend": "semantic_cpu"' in stripped
                )
                is_probability_cpu_meta = (
                    '"backend": "probability_cpu"' in stripped
                    or '"exp_backend": "probability_cpu"' in stripped
                    or '"sanitize_backend": "probability_cpu"' in stripped
                    or '"marginal_scan_backend": "probability_cpu"' in stripped
                    or '"row_scan_backend": "probability_cpu"' in stripped
                )
                is_f64_precision_cpu_meta = (
                    '"backend": "f64_cpu"' in stripped
                    or '"precision_backend": "f64_cpu"' in stripped
                    or '"state_sum_backend": "f64_cpu"' in stripped
                    or '"distribution_scale_backend": "f64_cpu"' in stripped
                )
                is_topos_cpu_meta = '"backend": "topos_cpu"' in stripped
                is_topos_rewrite_cpu_meta = '"rewrite_backend": "topos_cpu"' in stripped
                is_plain_cpu_meta = '"backend": "cpu"' in stripped or 'json!("cpu")' in stripped
                if is_probability_cpu_meta:
                    candidates.append(
                        SourceCandidate(
                            priority=_candidate_priority(rel, "probability_cpu_meta"),
                            kind="probability_cpu_meta",
                            file=rel,
                            line=idx + 1,
                            evidence=stripped,
                            note="metadata reports probability-normalisation CPU work that may matter in larger language/Z-space distributions",
                        )
                    )
                if is_f64_precision_cpu_meta:
                    candidates.append(
                        SourceCandidate(
                            priority=_candidate_priority(rel, "f64_precision_cpu_meta"),
                            kind="f64_precision_cpu_meta",
                            file=rel,
                            line=idx + 1,
                            evidence=stripped,
                            note="metadata reports f64 probability state work that needs an f64 tensor/backend path before it can leave CPU",
                        )
                    )
                if is_semantic_cpu_meta:
                    semantic_kind = (
                        "semantic_sparse_scan_cpu_meta"
                        if is_semantic_sparse_scan_cpu_meta
                        else "semantic_cpu_meta"
                    )
                    semantic_note = (
                        "metadata reports remaining sparse semantic row scan CPU work after routeable dense accumulation or scaling"
                        if is_semantic_sparse_scan_cpu_meta
                        else "metadata reports remaining sparse semantic scan/window CPU work after routeable tensor accumulation or scaling"
                    )
                    candidates.append(
                        SourceCandidate(
                            priority=_candidate_priority(rel, semantic_kind),
                            kind=semantic_kind,
                            file=rel,
                            line=idx + 1,
                            evidence=stripped,
                            note=semantic_note,
                        )
                    )
                if is_topos_cpu_meta or is_topos_rewrite_cpu_meta:
                    candidates.append(
                        SourceCandidate(
                            priority=_candidate_priority(rel, "topos_cpu_meta"),
                            kind="topos_cpu_meta",
                            file=rel,
                            line=idx + 1,
                            evidence=stripped,
                            note="metadata reports guarded topos rewrite/control CPU work around tensor utility-routed biome operations",
                        )
                    )
                if is_psd_eigen_cpu_meta:
                    candidates.append(
                        SourceCandidate(
                            priority=_candidate_priority(rel, "psd_eigen_cpu_meta"),
                            kind="psd_eigen_cpu_meta",
                            file=rel,
                            line=idx + 1,
                            evidence=stripped,
                            note=(
                                "metadata reports a hybrid covariance head whose PSD/eigen "
                                "projection remains on nalgebra/f32 CPU eigensolver and dense "
                                "reconstruction"
                            ),
                        )
                    )
                if is_plain_cpu_meta or is_recurrent_cpu_meta or is_normalization_cpu_meta:
                    window_start = (
                        max(0, idx - 24)
                        if is_recurrent_cpu_meta or is_normalization_cpu_meta
                        else idx
                    )
                    window = "\n".join(lines[window_start : min(idx + 8, len(lines))])
                    has_requested_backend = is_normalization_cpu_meta or (
                        "requested_backend" in window
                        and (
                            '"auto"' in window
                            or '"host"' in window
                            or '"requested_backend": requested_backend' in window
                            or 'insert_meta!("requested_backend", requested_backend)' in window
                        )
                    )
                    if has_requested_backend:
                        kind = "cpu_auto_meta"
                        note = "metadata reports CPU with requested_backend=auto; confirm policy routing or classify as trace-only"
                        if str(rel).startswith("crates/st-core/src/distributed/"):
                            kind = "distributed_gradient_cpu_meta"
                            note = (
                                "st-core distributed gradient/autograd metadata reports "
                                "host Vec/queue/sort reductions with mode/blocker fields; "
                                "check hot learning traces before backendizing"
                            )
                        if str(rel) == "crates/st-core/src/distributed/prob_params.rs":
                            kind = "distributed_control_cpu_meta"
                            note = (
                                "st-core distributed lane consensus reports CPU control-plane "
                                "work; treat as routing/policy debt unless hot learning traces "
                                "show it on the critical path"
                            )
                        if rel == Path("crates/st-core/src/engine/hook_points.rs"):
                            kind = "engine_gradient_hook_cpu_meta"
                            note = (
                                "engine hook-point metadata reports opaque CPU callback dispatch "
                                "around allreduce/ZeRO gradients or parameters"
                            )
                        if str(rel).startswith("crates/st-core/src/backend/"):
                            kind = "backend_control_cpu_meta"
                            note = (
                                "backend/runtime heuristic metadata reports CPU control-plane "
                                "work; usually trace/control debt unless it appears in hot "
                                "learning traces"
                            )
                        if is_recurrent_cpu_meta:
                            kind = "recurrent_cpu_meta"
                            note = (
                                "metadata reports the remaining reverse-time BPTT scan work; "
                                "projection/reduction dispatch counts are already split and a "
                                "WGPU single-dispatch scan helper now exists with CPU fallback, "
                                "so the next step is runtime verification and a more parallel "
                                "scan over grad_h_next/grad_c_next recurrence"
                            )
                        if is_normalization_cpu_meta:
                            kind = "normalization_cpu_meta"
                            note = "metadata reports normalization analytic input-gradient CPU work while affine reducers may route through tensor utilities"
                        candidates.append(
                            SourceCandidate(
                                priority=_candidate_priority(rel, kind),
                                kind=kind,
                                file=rel,
                                line=idx + 1,
                                evidence=stripped,
                                note=note,
                            )
                        )
    priority_order = {"P1": 0, "P2": 1, "P3": 2}
    candidates.sort(
        key=lambda item: (
            priority_order.get(item.priority, 9),
            item.kind,
            str(item.file),
            item.line,
        )
    )
    return candidates


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_report(
    *,
    run_root: Path,
    source_roots: Iterable[Path],
    include_tests: bool,
    include_cpu_runs: bool,
    max_candidates: int,
) -> str:
    trace_rows = _trace_rows(run_root, include_cpu_runs=include_cpu_runs)
    failed_rows = _failed_run_rows(run_root)
    source_candidates = _source_candidates(source_roots, include_tests=include_tests)
    lines = [
        "# Learning Backend Backlog Audit",
        "",
        f"- run_root: `{run_root}`",
        f"- source_roots: {', '.join(f'`{root}`' for root in source_roots)}",
        "- note: static candidates are heuristics; dynamic `cpu_debt_ops` is stronger evidence.",
        f"- include_cpu_runs: `{str(include_cpu_runs).lower()}`",
        "",
        "## Dynamic Residuals",
        "",
    ]
    if trace_rows:
        rows = []
        for row in trace_rows:
            if "error" in row:
                rows.append(
                    [
                        row["run"],
                        "ERR",
                        row["error"],
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ]
                )
            else:
                rows.append(
                    [
                        row["run"],
                        row["tensor_ops"],
                        row["wgpu_ops"],
                        row["cpu_ops"],
                        row["fallbacks"],
                        row["cpu_debt_ops"],
                        row["lstm_est_cpu_debt_ops"],
                        row["lstm_est_gate_cpu_debt_ops"],
                        row["lstm_est_gate_wgpu_ops"],
                        row["lstm_est_bptt_cpu_debt_ops"],
                        row["lstm_est_bptt_wgpu_ops"],
                        row["lstm_scan_rt_req"],
                        row["lstm_scan_rt_ok"],
                        row["lstm_scan_rt_miss"],
                        row["cpu_debt_top"],
                        row["cpu_trace_top"],
                        row["cpu_control_ops"],
                        row["cpu_control_top"],
                        row["cpu_runtime_fallback_ops"],
                        row["cpu_runtime_fallback_top"],
                        row["cpu_copy_top"],
                    ]
                )
        lines.extend(
            _markdown_table(
                [
                    "run",
                    "tensor_ops",
                    "wgpu",
                    "cpu",
                    "fallbacks",
                    "cpu_debt",
                    "lstm_est_cpu_debt",
                    "lstm_est_gate_cpu_debt",
                    "lstm_est_gate_wgpu",
                    "lstm_est_bptt_cpu_debt",
                    "lstm_est_bptt_wgpu",
                    "lstm_scan_rt_req",
                    "lstm_scan_rt_ok",
                    "lstm_scan_rt_miss",
                    "debt_top",
                    "trace_top",
                    "control_ops",
                    "control_top",
                    "runtime_fallback_ops",
                    "runtime_fallback_top",
                    "copy_top",
                ],
                rows,
            )
        )
    else:
        lines.append("_No trainer_trace.jsonl files found._")

    lines.extend(["", "## Failed Runs", ""])
    if failed_rows:
        rows = [
            [
                item["run"],
                item["backend"],
                item["status"],
                item["returncode"],
                item["failure_kind"],
                item["failure_detail"].replace("|", "\\|"),
                f"`{item['log']}`",
            ]
            for item in failed_rows
        ]
        lines.extend(
            _markdown_table(
                ["run", "backend", "status", "returncode", "failure_kind", "failure_detail", "log"],
                rows,
            )
        )
    else:
        lines.append("_No failed sweep records found._")

    lines.extend(["", "## Static Source Candidates", ""])
    if source_candidates:
        rows = [
            [
                item.priority,
                item.kind,
                f"`{item.file}:{item.line}`",
                item.evidence.replace("|", "\\|"),
                item.note.replace("|", "\\|"),
            ]
            for item in source_candidates[:max_candidates]
        ]
        lines.extend(_markdown_table(["priority", "kind", "location", "evidence", "note"], rows))
        if len(source_candidates) > max_candidates:
            lines.append("")
            lines.append(f"_Truncated {len(source_candidates) - max_candidates} additional candidates._")
    else:
        lines.append("_No source candidates found._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/current_backend_audit"))
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        dest="source_roots",
        help="source root to scan; may be repeated",
    )
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--include-cpu-runs", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_roots = tuple(args.source_roots) if args.source_roots else DEFAULT_SOURCE_ROOTS
    report = render_report(
        run_root=args.run_root,
        source_roots=source_roots,
        include_tests=args.include_tests,
        include_cpu_runs=args.include_cpu_runs,
        max_candidates=args.max_candidates,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()

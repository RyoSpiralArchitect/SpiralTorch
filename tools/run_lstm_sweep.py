#!/usr/bin/env python3
"""Run focused LSTM sequence probes and compare backend debt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from backend_sweep_meta import (
    BACKEND_META_HEADERS,
    BACKEND_RESIDUAL_HEADERS,
    backend_manifest_fields,
    backend_meta_row,
    backend_residual_row,
    classify_failure,
    failure_label,
    load_run_meta,
    md_cell,
    preflight_skipped_run_record,
    returncode_label,
    run_logged_command,
    status_label,
    write_failure,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "modelzoo_lstm_sequence_probe"


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

LSTM_CPU_DEBT_TRACE_OPS = [
    ("lstm_fwd_gate_cpu", "tensor_op_backend_lstm_forward_gate_activation_cpu"),
    ("lstm_bwd_gate_cpu", "tensor_op_backend_lstm_backward_gate_activation_cpu"),
    ("lstm_bwd_bptt_cpu", "tensor_op_backend_lstm_backward_bptt_cpu"),
    (
        "lstm_bwd_bptt_gate_cpu",
        "tensor_op_backend_lstm_backward_bptt_gate_derivative_cpu",
    ),
    (
        "lstm_bwd_bptt_cell_cpu",
        "tensor_op_backend_lstm_backward_bptt_cell_recurrence_cpu",
    ),
    (
        "lstm_bwd_bptt_state_cpu",
        "tensor_op_backend_lstm_backward_bptt_state_carry_cpu",
    ),
]

LSTM_WGPU_TRACE_OPS = [
    ("lstm_fwd_proj_wgpu", "tensor_op_backend_lstm_forward_input_projection_wgpu"),
    ("lstm_fwd_bias_wgpu", "tensor_op_backend_lstm_forward_bias_wgpu"),
    ("lstm_fwd_recurrent_wgpu", "tensor_op_backend_lstm_forward_recurrent_wgpu"),
    ("lstm_fwd_gate_wgpu", "tensor_op_backend_lstm_forward_gate_activation_wgpu"),
    ("lstm_bwd_recurrent_wgpu", "tensor_op_backend_lstm_backward_recurrent_wgpu"),
    ("lstm_bwd_gate_wgpu", "tensor_op_backend_lstm_backward_gate_activation_wgpu"),
    ("lstm_bwd_bptt_wgpu", "tensor_op_backend_lstm_backward_bptt_wgpu"),
    ("lstm_bwd_bptt_scan_wgpu", "tensor_op_backend_lstm_backward_bptt_scan_wgpu"),
    ("lstm_bwd_bptt_gate_wgpu", "tensor_op_backend_lstm_backward_bptt_gate_derivative_wgpu"),
    ("lstm_bwd_bptt_cell_wgpu", "tensor_op_backend_lstm_backward_bptt_cell_recurrence_wgpu"),
    ("lstm_bwd_bptt_state_wgpu", "tensor_op_backend_lstm_backward_bptt_state_carry_wgpu"),
    ("lstm_bwd_input_wgpu", "tensor_op_backend_lstm_backward_input_gradient_wgpu"),
    (
        "lstm_bwd_param_reduce_wgpu",
        "tensor_op_backend_lstm_backward_parameter_gradient_reduction_wgpu",
    ),
    ("lstm_bwd_bias_wgpu", "tensor_op_backend_lstm_backward_bias_gradient_wgpu"),
    (
        "lstm_bwd_param_scale_wgpu",
        "tensor_op_backend_lstm_backward_parameter_gradient_scale_wgpu",
    ),
]

LSTM_ESTIMATED_WORK_OPS = [
    ("lstm_est_fwd_gate_ops", "lstm_forward_estimated_gate_activation_ops"),
    (
        "lstm_est_fwd_gate_cpu_debt_ops",
        "lstm_forward_estimated_gate_activation_cpu_debt_ops",
    ),
    ("lstm_est_fwd_gate_wgpu_ops", "lstm_forward_estimated_gate_activation_wgpu_ops"),
    ("lstm_est_bwd_gate_ops", "lstm_backward_estimated_gate_activation_ops"),
    (
        "lstm_est_bwd_gate_cpu_debt_ops",
        "lstm_backward_estimated_gate_activation_cpu_debt_ops",
    ),
    ("lstm_est_bwd_gate_wgpu_ops", "lstm_backward_estimated_gate_activation_wgpu_ops"),
    ("lstm_est_gate_cpu_debt_ops", "lstm_estimated_gate_activation_cpu_debt_ops"),
    ("lstm_est_gate_wgpu_ops", "lstm_estimated_gate_activation_wgpu_ops"),
    ("lstm_est_bptt_ops", "lstm_backward_estimated_bptt_ops"),
    (
        "lstm_est_bptt_gate_ops",
        "lstm_backward_estimated_bptt_gate_derivative_ops",
    ),
    (
        "lstm_est_bptt_cell_ops",
        "lstm_backward_estimated_bptt_cell_recurrence_ops",
    ),
    (
        "lstm_est_bptt_state_ops",
        "lstm_backward_estimated_bptt_state_carry_ops",
    ),
    ("lstm_est_bptt_scan_steps", "lstm_backward_estimated_bptt_scan_steps"),
]

LSTM_SCAN_RUNTIME_OPS = [
    ("lstm_bwd_scan_shape_ok", "lstm_backward_bptt_scan_shape_supported"),
    ("lstm_bwd_scan_rt_req", "lstm_backward_bptt_scan_runtime_requested"),
    ("lstm_bwd_scan_rt_ok", "lstm_backward_bptt_scan_runtime_available"),
    ("lstm_bwd_scan_rt_miss", "lstm_backward_bptt_scan_runtime_unavailable"),
]

LSTM_SCAN_PROFILE_OPS = [
    ("lstm_bwd_scan_us", "lstm_backward_bptt_scan_elapsed_us"),
    ("lstm_bwd_scan_hidden_values", "lstm_backward_bptt_scan_hidden_values"),
    ("lstm_bwd_scan_gate_values", "lstm_backward_bptt_scan_gate_values"),
    ("lstm_bwd_scan_cell_values", "lstm_backward_bptt_scan_cell_values"),
    (
        "lstm_bwd_scan_recurrent_weight_values",
        "lstm_backward_bptt_scan_recurrent_weight_values",
    ),
    ("lstm_bwd_scan_scratch_values", "lstm_backward_bptt_scan_scratch_values"),
    ("lstm_bwd_scan_dispatches", "lstm_backward_bptt_scan_kernel_dispatches"),
    ("lstm_bwd_scan_serial_steps", "lstm_backward_bptt_scan_serial_steps"),
    ("lstm_bwd_scan_workgroup", "lstm_backward_bptt_scan_workgroup_size"),
    ("lstm_bwd_scan_parallel_lanes", "lstm_backward_bptt_scan_parallel_lanes"),
    (
        "lstm_est_bptt_ops_per_scan_step",
        "lstm_backward_estimated_bptt_ops_per_scan_step",
    ),
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def metric(summary: dict[str, Any], key: str, field: str = "last") -> Any:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    return item.get(field)


def metric_number(summary: dict[str, Any], key: str, field: str = "last") -> float | None:
    value = metric(summary, key, field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_number_sum(
    summary: dict[str, Any], keys: Iterable[str], field: str = "last"
) -> float | None:
    values = [metric_number(summary, key, field) for key in keys]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def cargo_prefix(args: argparse.Namespace, backends: Iterable[str]) -> list[str]:
    features = args.cargo_features
    if not features and any(backend == "wgpu" for backend in backends):
        features = "wgpu"
    command = ["cargo", "run", "-q", "-p", "st-nn"]
    if args.no_default_features:
        command.append("--no-default-features")
    if features:
        command.extend(["--features", features])
    command.extend(["--example", EXAMPLE, "--"])
    return command


def run_command(command: list[str], log_path: Path, *, threshold: int | None) -> int:
    env_overrides = {}
    if threshold is not None:
        env_overrides["SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"] = str(threshold)
    return run_logged_command(
        command,
        log_path,
        cwd=ROOT,
        env_overrides=env_overrides or None,
    )


def example_command(
    args: argparse.Namespace,
    backend: str,
    seed: int,
    run_dir: Path,
    events_jsonl: Path,
    *,
    epochs: int | None = None,
    batches: int | None = None,
    steps: int | None = None,
    hidden: int | None = None,
) -> list[str]:
    # Pick implicit cargo features per backend so CPU baselines stay CPU-only
    # even when the same sweep also includes a WGPU run.
    command = cargo_prefix(args, [backend])
    command.extend(
        [
            "--run-dir",
            str(run_dir),
            "--events",
            str(events_jsonl),
            "--backend",
            backend,
            "--epochs",
            str(args.epochs if epochs is None else epochs),
            "--batches",
            str(args.batches if batches is None else batches),
            "--steps",
            str(args.steps if steps is None else steps),
            "--hidden",
            str(args.hidden if hidden is None else hidden),
            "--seed",
            str(seed),
            "--lr",
            str(args.lr),
            "--curvature",
            str(args.curvature),
        ]
    )
    return command


def run_wgpu_preflight(args: argparse.Namespace, backend: str) -> dict[str, Any] | None:
    if backend != "wgpu" or args.no_wgpu_preflight:
        return None

    run_dir = args.run_root / "_preflight" / f"backend-{backend}"
    events_jsonl = run_dir / "preflight_events.jsonl"
    log_path = run_dir / "process.log"
    command = example_command(
        args,
        backend,
        0,
        run_dir,
        events_jsonl,
        epochs=1,
        batches=1,
        steps=1,
        hidden=1,
    )
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    if returncode == 0:
        return None

    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure = {
        "schema": "st.sequence.lstm_sweep_preflight_failure.v1",
        "backend": backend,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": returncode,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "command": command,
    }
    write_failure(run_dir, failure)
    if not args.continue_on_error:
        raise subprocess.CalledProcessError(returncode, command)
    return failure


def preflight_skipped_run(
    args: argparse.Namespace, backend: str, seed: int, preflight_failure: dict[str, Any]
) -> dict[str, Any]:
    run_name = f"backend-{backend}__seed-{seed}"
    run_dir = args.run_root / run_name
    events_jsonl = run_dir / "trainer_trace.jsonl"
    log_path = run_dir / "process.log"
    command = example_command(args, backend, seed, run_dir, events_jsonl)
    return preflight_skipped_run_record(
        schema="st.sequence.lstm_sweep_failure.v1",
        backend=backend,
        seed=seed,
        run_dir=run_dir,
        log_path=log_path,
        command=command,
        preflight_failure=preflight_failure,
    )


def run_one(args: argparse.Namespace, backend: str, seed: int) -> dict[str, Any]:
    run_name = f"backend-{backend}__seed-{seed}"
    run_dir = args.run_root / run_name
    trace_json = run_dir / "lstm_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    if args.skip_existing and trace_json.exists() and events_jsonl.exists():
        return {
            "backend": backend,
            "seed": seed,
            "run_dir": str(run_dir),
            "log_path": str(run_dir / "process.log"),
            "returncode": 0,
            "skipped": True,
            "failed": False,
        }

    command = example_command(args, backend, seed, run_dir, events_jsonl)
    log_path = run_dir / "process.log"
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    failed = returncode != 0
    failure_kind = None
    failure_detail = None
    if failed:
        failure_kind, failure_detail = classify_failure(returncode, log_path)
        failure = {
            "schema": "st.sequence.lstm_sweep_failure.v1",
            "backend": backend,
            "seed": seed,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "returncode": returncode,
            "failure_kind": failure_kind,
            "failure_detail": failure_detail,
            "command": command,
        }
        write_failure(run_dir, failure)
        if not args.continue_on_error:
            raise subprocess.CalledProcessError(returncode, command)
    return {
        "backend": backend,
        "seed": seed,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": returncode,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "skipped": False,
        "failed": failed,
        "command": command,
    }


def read_trace(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "lstm_trace.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LSTM trace at {run_dir} is not a JSON object")
    return payload


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / run["run_dir"]
    if run.get("failed"):
        return {
            **run,
            "trace": {},
            "run_meta": load_run_meta(run_dir),
            "trainer_summary": {"metrics": {}},
        }
    return {
        **run,
        "trace": read_trace(run_dir),
        "run_meta": load_run_meta(run_dir),
        "trainer_summary": TRAINER_TRACE.summarize_trainer_trace_events(
            run_dir / "trainer_trace.jsonl"
        ),
    }


def run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    trace_summary = summary["trace"].get("summary")
    return trace_summary if isinstance(trace_summary, dict) else {}


def row_for(summary: dict[str, Any]) -> list[str]:
    prefix = [
        str(summary["backend"]),
        str(summary["seed"]),
        status_label(summary),
        returncode_label(summary.get("returncode", 0)),
        failure_label(summary, "failure_kind"),
        failure_label(summary, "failure_detail"),
        str(summary.get("log_path") or "-"),
    ]
    if summary.get("failed"):
        return prefix + ["-" for _ in data_column_headers()]
    trainer = summary["trainer_summary"]
    trace = run_summary(summary)
    return [
        *prefix,
        fmt(trace.get("pretrain_cpu_reference_loss"), 6),
        fmt(trace.get("pretrain_loss"), 6),
        fmt(trace.get("pretrain_backend_gap"), 6),
        fmt(trace.get("first_loss"), 6),
        fmt(trace.get("last_loss"), 6),
        fmt(trace.get("loss_delta"), 6),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_cpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_fallbacks", "last"), 0),
        *backend_meta_row(summary.get("run_meta", {})),
        *backend_residual_row(trainer),
        fmt(metric_number_sum(trainer, [key for _label, key in LSTM_CPU_DEBT_TRACE_OPS]), 0),
        fmt(metric(trainer, "lstm_estimated_cpu_debt_ops", "last"), 0),
        *[fmt(metric(trainer, key, "last"), 0) for _label, key in LSTM_CPU_DEBT_TRACE_OPS],
        *[fmt(metric(trainer, key, "last"), 0) for _label, key in LSTM_ESTIMATED_WORK_OPS],
        *[fmt(metric(trainer, key, "last"), 0) for _label, key in LSTM_SCAN_RUNTIME_OPS],
        *[fmt(metric(trainer, key, "last"), 0) for _label, key in LSTM_SCAN_PROFILE_OPS],
        fmt(metric_number_sum(trainer, [key for _label, key in LSTM_WGPU_TRACE_OPS]), 0),
        *[fmt(metric(trainer, key, "last"), 0) for _label, key in LSTM_WGPU_TRACE_OPS],
    ]


def average_row(backend: str, summaries: list[dict[str, Any]]) -> list[str]:
    trainer_summaries = [summary["trainer_summary"] for summary in summaries]
    run_summaries = [run_summary(summary) for summary in summaries]
    return [
        backend,
        str(len(summaries)),
        fmt(mean(number(summary.get("pretrain_backend_gap")) for summary in run_summaries), 6),
        fmt(mean(number(summary.get("last_loss")) for summary in run_summaries), 6),
        fmt(mean(number(summary.get("loss_delta")) for summary in run_summaries), 6),
        fmt(mean(metric_number(trainer, "tensor_ops_total", "last") for trainer in trainer_summaries), 0),
        fmt(mean(metric_number(trainer, "tensor_backend_wgpu", "last") for trainer in trainer_summaries), 0),
        fmt(mean(metric_number(trainer, "tensor_backend_fallbacks", "last") for trainer in trainer_summaries), 0),
        fmt(
            mean(
                metric_number_sum(trainer, [key for _label, key in LSTM_CPU_DEBT_TRACE_OPS])
                for trainer in trainer_summaries
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_estimated_cpu_debt_ops", "last")
                for trainer in trainer_summaries
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_estimated_gate_activation_cpu_debt_ops", "last")
                for trainer in trainer_summaries
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_estimated_gate_activation_wgpu_ops", "last")
                for trainer in trainer_summaries
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_estimated_bptt_ops", "last")
                for trainer in trainer_summaries
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_estimated_bptt_scan_steps", "last")
                for trainer in trainer_summaries
            ),
            0,
        ),
        *[
            fmt(mean(metric_number(trainer, key, "last") for trainer in trainer_summaries), 0)
            for _label, key in LSTM_SCAN_RUNTIME_OPS
        ],
        *[
            fmt(mean(metric_number(trainer, key, "last") for trainer in trainer_summaries), 0)
            for _label, key in LSTM_SCAN_PROFILE_OPS
        ],
        fmt(
            mean(
                metric_number_sum(trainer, [key for _label, key in LSTM_WGPU_TRACE_OPS])
                for trainer in trainer_summaries
            ),
            0,
        ),
    ]


def write_compare(run_root: Path, summaries: list[dict[str, Any]]) -> Path:
    headers = [
        "backend",
        "seed",
        "run_status",
        "returncode",
        "failure_kind",
        "failure_detail",
        "log_path",
        *data_column_headers(),
    ]
    lines = [
        "# LSTM Sequence Sweep",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row_for(summary)) + " |")

    successful_summaries = [summary for summary in summaries if not summary.get("failed")]
    average_headers = [
        "backend",
        "runs",
        "failed",
        "avg_backend_gap",
        "avg_last_loss",
        "avg_delta",
        "avg_tensor_ops",
        "avg_tensor_wgpu",
        "avg_fallbacks",
        "avg_lstm_cpu_debt_ops",
        "avg_lstm_est_cpu_debt_ops",
        "avg_lstm_est_gate_cpu_debt_ops",
        "avg_lstm_est_gate_wgpu_ops",
        "avg_lstm_est_bptt_ops",
        "avg_lstm_est_bptt_scan_steps",
        "avg_lstm_bwd_scan_shape_ok",
        "avg_lstm_bwd_scan_rt_req",
        "avg_lstm_bwd_scan_rt_ok",
        "avg_lstm_bwd_scan_rt_miss",
        *[f"avg_{label}" for label, _key in LSTM_SCAN_PROFILE_OPS],
        "avg_lstm_wgpu_ops",
    ]
    lines.extend(
        [
            "",
            "## Backend Averages",
            "",
            "| " + " | ".join(average_headers) + " |",
            "| " + " | ".join("---" for _ in average_headers) + " |",
        ]
    )
    for backend in sorted({str(summary["backend"]) for summary in summaries}):
        backend_summaries = [
            summary for summary in successful_summaries if str(summary["backend"]) == backend
        ]
        failed_count = sum(
            1
            for summary in summaries
            if str(summary["backend"]) == backend and summary.get("failed")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    backend,
                    str(len(backend_summaries)),
                    str(failed_count),
                    *average_row(backend, backend_summaries)[2:],
                ]
            )
            + " |"
        )

    path = run_root / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def data_column_headers() -> list[str]:
    return [
        "pretrain_cpu_ref",
        "pretrain_loss",
        "pretrain_backend_gap",
        "first_loss",
        "last_loss",
        "delta",
        "step_ms_last",
        "tensor_ops",
        "tensor_wgpu",
        "tensor_cpu",
        "fallbacks",
        *BACKEND_META_HEADERS,
        *BACKEND_RESIDUAL_HEADERS,
        "lstm_cpu_debt_ops",
        "lstm_est_cpu_debt_ops",
        *[label for label, _key in LSTM_CPU_DEBT_TRACE_OPS],
        *[label for label, _key in LSTM_ESTIMATED_WORK_OPS],
        *[label for label, _key in LSTM_SCAN_RUNTIME_OPS],
        *[label for label, _key in LSTM_SCAN_PROFILE_OPS],
        "lstm_wgpu_ops",
        *[label for label, _key in LSTM_WGPU_TRACE_OPS],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/lstm_sweep"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("cpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--wgpu-min-values", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-wgpu-preflight", action="store_true")
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    preflight_failures = {}
    for backend in dict.fromkeys(args.backends):
        failure = run_wgpu_preflight(args, backend)
        if failure is not None:
            preflight_failures[backend] = failure

    runs = []
    for backend in args.backends:
        for seed in args.seeds:
            preflight_failure = preflight_failures.get(backend)
            if preflight_failure is not None:
                runs.append(preflight_skipped_run(args, backend, seed, preflight_failure))
            else:
                runs.append(run_one(args, backend, seed))
    summaries = [summarize_run(run) for run in runs]
    compare_path = write_compare(args.run_root, summaries)
    sweep = {
        "schema": "st.sequence.lstm_sweep.v1",
        "run_root": str(args.run_root),
        "backends": args.backends,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "batches": args.batches,
            "steps": args.steps,
            "hidden": args.hidden,
            "lr": args.lr,
            "curvature": args.curvature,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
            "wgpu_min_values": args.wgpu_min_values,
            "wgpu_preflight": not args.no_wgpu_preflight,
        },
        "preflight_failures": preflight_failures,
        "runs": [
            {
                "backend": summary["backend"],
                "seed": summary["seed"],
                "run_dir": summary["run_dir"],
                "log_path": summary.get("log_path"),
                "returncode": summary.get("returncode", 0),
                "failed": bool(summary.get("failed")),
                "failure_kind": summary.get("failure_kind"),
                "failure_detail": summary.get("failure_detail"),
                "skipped": summary["skipped"],
                "command": summary.get("command"),
                **backend_manifest_fields(summary.get("run_meta", {})),
            }
            for summary in summaries
        ],
        "failed": any(bool(summary.get("failed")) for summary in summaries),
    }
    (args.run_root / "sweep.json").write_text(
        json.dumps(sweep, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"compare={compare_path}")
    if sweep["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

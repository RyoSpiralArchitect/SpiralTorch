#!/usr/bin/env python3
"""Run WaveRnn sequence probes and compare backend traces."""

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
EXAMPLE = "modelzoo_wave_rnn_sequence"


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


CPU_HEAVY_TRACE_OPS = [
    ("reshape_cpu", "tensor_op_backend_reshape_cpu"),
    ("add_row_cpu", "tensor_op_backend_add_row_inplace_cpu"),
    ("sum_axis0_cpu", "tensor_op_backend_sum_axis0_cpu"),
    ("sum_axis0_scaled_cpu", "tensor_op_backend_sum_axis0_scaled_cpu"),
    ("poincare_cpu", "tensor_op_backend_project_to_poincare_cpu"),
    ("scale_cpu", "tensor_op_backend_scale_cpu"),
    ("transpose_cpu", "tensor_op_backend_transpose_cpu"),
]

SHAPE_TRACE_OPS = [
    ("reshape_view", "tensor_op_backend_reshape_view"),
]

WGPU_UTILITY_TRACE_OPS = [
    ("add_row_wgpu", "tensor_op_backend_add_row_inplace_wgpu"),
    ("sum_axis0_wgpu", "tensor_op_backend_sum_axis0_wgpu"),
    ("sum_axis0_scaled_wgpu", "tensor_op_backend_sum_axis0_scaled_wgpu"),
    ("poincare_wgpu", "tensor_op_backend_project_to_poincare_wgpu"),
    ("wave_gate_project_wgpu", "tensor_op_backend_wave_gate_project_wgpu"),
    ("wave_gate_backward_wgpu", "tensor_op_backend_wave_gate_backward_wgpu"),
    ("scale_wgpu", "tensor_op_backend_scale_wgpu"),
]


def cpu_heavy_stats(summary: dict[str, Any]) -> tuple[list[float], float, float | None]:
    tensor_ops = metric_number(summary, "tensor_ops_total", "last")
    values = [
        metric_number(summary, metric_name, "last") or 0.0
        for _header, metric_name in CPU_HEAVY_TRACE_OPS
    ]
    total = sum(values)
    share = total / tensor_ops if tensor_ops is not None and tensor_ops > 0.0 else None
    return values, total, share


def wgpu_utility_stats(summary: dict[str, Any]) -> tuple[list[float], float, float | None]:
    tensor_ops = metric_number(summary, "tensor_ops_total", "last")
    values = [
        metric_number(summary, metric_name, "last") or 0.0
        for _header, metric_name in WGPU_UTILITY_TRACE_OPS
    ]
    total = sum(values)
    share = total / tensor_ops if tensor_ops is not None and tensor_ops > 0.0 else None
    return values, total, share


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


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


def example_command(
    args: argparse.Namespace,
    backend: str,
    seed: int,
    run_dir: Path,
    events_jsonl: Path,
    *,
    epochs: int | None = None,
    batches: int | None = None,
    batch: int | None = None,
    steps: int | None = None,
    hidden: int | None = None,
) -> list[str]:
    # Match LSTM sweep semantics: CPU baselines stay CPU-only in mixed sweeps.
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
            "--batch",
            str(args.batch if batch is None else batch),
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
            "--temperature",
            str(args.temperature),
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
        batch=1,
        steps=1,
        hidden=1,
    )
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    if returncode == 0:
        return None

    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure = {
        "schema": "st.sequence.wave_rnn_sweep_preflight_failure.v1",
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
        schema="st.sequence.wave_rnn_sweep_failure.v1",
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
    trace_json = run_dir / "sequence_trace.json"
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
            "schema": "st.sequence.wave_rnn_sweep_failure.v1",
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
    payload = json.loads((run_dir / "sequence_trace.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sequence trace at {run_dir} is not a JSON object")
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
    trace = read_trace(run_dir)
    run_meta = load_run_meta(run_dir)
    trainer_summary = TRAINER_TRACE.summarize_trainer_trace_events(
        run_dir / "trainer_trace.jsonl"
    )
    return {**run, "trace": trace, "run_meta": run_meta, "trainer_summary": trainer_summary}


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

    trace = summary["trace"]
    trainer = summary["trainer_summary"]
    run_summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    cpu_heavy_values, cpu_heavy_ops, cpu_heavy_share = cpu_heavy_stats(trainer)
    wgpu_utility_values, wgpu_utility_ops, wgpu_utility_share = wgpu_utility_stats(trainer)
    return [
        *prefix,
        fmt(run_summary.get("pretrain_cpu_reference_loss"), 6),
        fmt(run_summary.get("pretrain_loss"), 6),
        fmt(run_summary.get("pretrain_backend_gap"), 6),
        fmt(run_summary.get("pretrain_forward_gap"), 6),
        fmt(run_summary.get("pretrain_loss_gap"), 6),
        fmt(run_summary.get("first_loss"), 6),
        fmt(run_summary.get("last_loss"), 6),
        fmt(run_summary.get("loss_delta"), 6),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_cpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_naive", "last"), 0),
        *backend_meta_row(summary.get("run_meta", {})),
        *backend_residual_row(trainer),
        *[
            fmt(metric(trainer, metric_name, "last"), 0)
            for _header, metric_name in SHAPE_TRACE_OPS
        ],
        fmt(
            metric_number_sum(
                trainer,
                [
                    "tensor_op_backend_matmul_prepacked_wgpu",
                    "tensor_op_backend_matmul_prepacked_bias_wgpu",
                ],
            ),
            0,
        ),
        fmt(
            metric_number_sum(
                trainer,
                [
                    "tensor_op_backend_matmul_prepacked_naive",
                    "tensor_op_backend_matmul_prepacked_bias_naive",
                ],
            ),
            0,
        ),
        fmt(
            metric_number_sum(
                trainer,
                [
                    "tensor_op_backend_matmul_wgpu",
                    "tensor_op_backend_matmul_scaled_wgpu",
                    "tensor_op_backend_matmul_lhs_transpose_scaled_wgpu",
                ],
            ),
            0,
        ),
        fmt(
            metric_number_sum(
                trainer,
                [
                    "tensor_op_backend_matmul_naive",
                    "tensor_op_backend_matmul_scaled_naive",
                    "tensor_op_backend_matmul_lhs_transpose_scaled_naive",
                ],
            ),
            0,
        ),
        fmt(metric(trainer, "tensor_op_backend_row_softmax_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_row_softmax_cpu", "last"), 0),
        fmt(cpu_heavy_ops, 0),
        fmt(cpu_heavy_share, 3),
        *[fmt(value, 0) for value in cpu_heavy_values],
        fmt(wgpu_utility_ops, 0),
        fmt(wgpu_utility_share, 3),
        *[fmt(value, 0) for value in wgpu_utility_values],
        fmt(metric(trainer, "backend_policy_events", "sum"), 0),
    ]


def average_row(backend: str, summaries: list[dict[str, Any]]) -> list[str]:
    trainer_summaries = [summary["trainer_summary"] for summary in summaries]
    run_summaries = [
        summary["trace"].get("summary")
        if isinstance(summary["trace"].get("summary"), dict)
        else {}
        for summary in summaries
    ]
    cpu_heavy = [cpu_heavy_stats(trainer) for trainer in trainer_summaries]
    wgpu_utility = [wgpu_utility_stats(trainer) for trainer in trainer_summaries]
    return [
        backend,
        str(len(summaries)),
        fmt(mean(number(summary.get("pretrain_backend_gap")) for summary in run_summaries), 6),
        fmt(mean(number(summary.get("last_loss")) for summary in run_summaries), 6),
        fmt(mean(metric_number(trainer, "step_time_ms", "last") for trainer in trainer_summaries), 3),
        fmt(mean(metric_number(trainer, "tensor_ops_total", "last") for trainer in trainer_summaries), 0),
        fmt(mean(metric_number(trainer, "tensor_backend_wgpu", "last") for trainer in trainer_summaries), 0),
        fmt(mean(metric_number(trainer, "tensor_backend_cpu", "last") for trainer in trainer_summaries), 0),
        *[
            fmt(mean(metric_number(trainer, metric_name, "last") for trainer in trainer_summaries), 0)
            for _header, metric_name in SHAPE_TRACE_OPS
        ],
        fmt(mean(total for _values, total, _share in cpu_heavy), 0),
        fmt(mean(share for _values, _total, share in cpu_heavy), 3),
        fmt(mean(total for _values, total, _share in wgpu_utility), 0),
        fmt(mean(share for _values, _total, share in wgpu_utility), 3),
        fmt(mean(metric_number(trainer, "backend_policy_events", "sum") for trainer in trainer_summaries), 0),
    ]


def data_column_headers() -> list[str]:
    return [
        "pretrain_cpu_ref",
        "pretrain_loss",
        "pretrain_backend_gap",
        "pretrain_forward_gap",
        "pretrain_loss_gap",
        "first_loss",
        "last_loss",
        "delta",
        "step_ms_last",
        "tensor_ops",
        "tensor_wgpu",
        "tensor_cpu",
        "tensor_naive",
        *BACKEND_META_HEADERS,
        *BACKEND_RESIDUAL_HEADERS,
        *[header for header, _metric_name in SHAPE_TRACE_OPS],
        "prepacked_wgpu",
        "prepacked_naive",
        "matmul_wgpu",
        "matmul_naive",
        "softmax_wgpu",
        "softmax_cpu",
        "cpu_heavy_ops",
        "cpu_heavy_share",
        *[header for header, _metric_name in CPU_HEAVY_TRACE_OPS],
        "wgpu_utility_ops",
        "wgpu_utility_share",
        *[header for header, _metric_name in WGPU_UTILITY_TRACE_OPS],
        "policy_events",
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
        "# WaveRnn Sequence Sweep",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row_for(summary)) + " |")
    average_headers = [
        "backend",
        "runs",
        "failed",
        "avg_backend_gap",
        "avg_last_loss",
        "avg_step_ms_last",
        "avg_tensor_ops",
        "avg_tensor_wgpu",
        "avg_tensor_cpu",
        *[f"avg_{header}" for header, _metric_name in SHAPE_TRACE_OPS],
        "avg_cpu_heavy_ops",
        "avg_cpu_heavy_share",
        "avg_wgpu_utility_ops",
        "avg_wgpu_utility_share",
        "avg_policy_events",
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
    successful_summaries = [summary for summary in summaries if not summary.get("failed")]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/wave_rnn_sweep"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("cpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
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
        "schema": "st.sequence.wave_rnn_sweep.v1",
        "run_root": str(args.run_root),
        "backends": args.backends,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "batches": args.batches,
            "batch": args.batch,
            "steps": args.steps,
            "hidden": args.hidden,
            "lr": args.lr,
            "curvature": args.curvature,
            "temperature": args.temperature,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
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

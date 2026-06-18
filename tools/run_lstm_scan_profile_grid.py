#!/usr/bin/env python3
"""Run an LSTM backward-scan shape grid and summarize scan profile columns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from backend_sweep_meta import (
    backend_meta_row,
    backend_manifest_fields,
    failure_label,
    load_run_meta,
    md_cell,
    returncode_label,
    status_label,
)

ROOT = Path(__file__).resolve().parents[1]
LSTM_SWEEP = ROOT / "tools" / "run_lstm_sweep.py"


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


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_trainer_trace() -> Any:
    import importlib.util

    helper = ROOT / "bindings/st-py/spiraltorch/trainer_trace.py"
    spec = importlib.util.spec_from_file_location("spiraltorch_trainer_trace", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load trainer trace helper from {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER_TRACE = _load_trainer_trace()


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


def trace_summary(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "lstm_trace.json"
    if not trace_path.exists():
        return {}
    payload = read_json(trace_path)
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def trainer_summary(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "trainer_trace.jsonl"
    if not trace_path.exists():
        return {"metrics": {}}
    return TRAINER_TRACE.summarize_trainer_trace_events(trace_path)


def latest_lstm_scan_route(run_dir: Path) -> dict[str, str | None]:
    trace_path = run_dir / "trainer_trace.jsonl"
    route: dict[str, str | None] = {
        "backend": None,
        "kernel": None,
        "lowering": None,
        "fallback_reason": None,
    }
    if not trace_path.exists():
        return route

    for event in TRAINER_TRACE.load_trainer_trace_events(trace_path, event_type="TensorOpMeta"):
        if event.get("op_name") != "lstm_backward":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        for source, target in (
            ("bptt_scan_backend", "backend"),
            ("bptt_scan_kernel", "kernel"),
            ("bptt_scan_lowering", "lowering"),
            ("bptt_scan_fallback_reason", "fallback_reason"),
        ):
            value = data.get(source)
            if isinstance(value, str) and value:
                route[target] = value
    return route


def cell_name(steps: int, hidden: int) -> str:
    return f"steps-{steps}__hidden-{hidden}"


def sweep_command(args: argparse.Namespace, *, steps: int, hidden: int, cell_root: Path) -> list[str]:
    command = [
        sys.executable,
        "-S",
        "-s",
        str(LSTM_SWEEP),
        "--run-root",
        str(cell_root),
        "--backends",
        ",".join(args.backends),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--epochs",
        str(args.epochs),
        "--batches",
        str(args.batches),
        "--steps",
        str(steps),
        "--hidden",
        str(hidden),
        "--lr",
        str(args.lr),
        "--curvature",
        str(args.curvature),
    ]
    if args.cargo_features:
        command.extend(["--cargo-features", args.cargo_features])
    if args.no_default_features:
        command.append("--no-default-features")
    if args.wgpu_min_values is not None:
        command.extend(["--wgpu-min-values", str(args.wgpu_min_values)])
    if args.skip_existing:
        command.append("--skip-existing")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.no_wgpu_preflight:
        command.append("--no-wgpu-preflight")
    return command


def run_cell(args: argparse.Namespace, *, steps: int, hidden: int) -> dict[str, Any]:
    name = cell_name(steps, hidden)
    cell_root = args.run_root / name
    cell_root.mkdir(parents=True, exist_ok=True)
    command = sweep_command(args, steps=steps, hidden=hidden, cell_root=cell_root)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (cell_root / "grid_child.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 and not args.continue_on_error:
        raise subprocess.CalledProcessError(proc.returncode, command, output=proc.stdout)
    return {
        "steps": steps,
        "hidden": hidden,
        "cell_root": str(cell_root),
        "command": command,
        "returncode": proc.returncode,
        "failed": proc.returncode != 0,
    }


def summarize_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    cell_root = ROOT / str(cell["cell_root"])
    sweep_path = cell_root / "sweep.json"
    if not sweep_path.exists():
        return [
            {
                **cell,
                "backend": "-",
                "seed": "-",
                "run_dir": str(cell_root),
                "log_path": str(cell_root / "grid_child.log"),
                "failure_kind": "cell_sweep_missing",
                "failure_detail": "sweep.json missing",
                "skipped": False,
                "trainer_summary": {"metrics": {}},
                "trace_summary": {},
                "run_meta": {},
                "scan_route": latest_lstm_scan_route(cell_root),
            }
        ]

    sweep = read_json(sweep_path)
    rows: list[dict[str, Any]] = []
    for run in sweep.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_dir = ROOT / str(run.get("run_dir", ""))
        rows.append(
            {
                **cell,
                **run,
                "trainer_summary": trainer_summary(run_dir),
                "trace_summary": trace_summary(run_dir),
                "run_meta": load_run_meta(run_dir),
                "scan_route": latest_lstm_scan_route(run_dir),
            }
        )
    return rows


def ops_per_us(row: dict[str, Any]) -> float | None:
    trainer = row.get("trainer_summary", {})
    ops = metric_number(trainer, "lstm_backward_estimated_bptt_ops", "last")
    elapsed = metric_number(trainer, "lstm_backward_bptt_scan_elapsed_us", "last")
    if ops is None or elapsed is None or elapsed <= 0:
        return None
    return ops / elapsed


def scan_parallel_axis(row: dict[str, Any]) -> str:
    lanes = metric_number(
        row.get("trainer_summary", {}),
        "lstm_backward_bptt_scan_parallel_lanes",
        "last",
    )
    return "hidden" if lanes is not None and lanes > 1 else "none"


def count_labels(values: Iterable[Any], *, missing_label: str = "-") -> str:
    counts: dict[str, int] = {}
    for value in values:
        label = str(value) if value else missing_label
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "-"
    return ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def scan_profile_fields(row: dict[str, Any]) -> dict[str, Any]:
    trainer = row.get("trainer_summary", {})
    run_meta = row.get("run_meta", {})
    backend_meta = backend_meta_row(run_meta)
    scan_route = row.get("scan_route")
    route = scan_route if isinstance(scan_route, dict) else {}
    return {
        "last_loss": row.get("trace_summary", {}).get("last_loss"),
        "step_ms_last": metric_number(trainer, "step_time_ms", "last"),
        "tensor_ops": metric_number(trainer, "tensor_ops_total", "last"),
        "tensor_wgpu": metric_number(trainer, "tensor_backend_wgpu", "last"),
        "fallbacks": metric_number(trainer, "tensor_backend_fallbacks", "last"),
        "lstm_est_cpu_debt_ops": metric_number(trainer, "lstm_estimated_cpu_debt_ops", "last"),
        "lstm_est_bptt_wgpu_ops": metric_number(
            trainer, "lstm_estimated_bptt_wgpu_ops", "last"
        ),
        "lstm_scan_rt_req": metric_number(
            trainer, "lstm_backward_bptt_scan_runtime_requested", "last"
        ),
        "lstm_scan_rt_ok": metric_number(
            trainer, "lstm_backward_bptt_scan_runtime_available", "last"
        ),
        "lstm_scan_rt_miss": metric_number(
            trainer, "lstm_backward_bptt_scan_runtime_unavailable", "last"
        ),
        "lstm_scan_backend": route.get("backend"),
        "lstm_scan_kernel": route.get("kernel"),
        "lstm_scan_lowering": route.get("lowering"),
        "lstm_scan_fallback": route.get("fallback_reason"),
        "lstm_scan_us": metric_number(trainer, "lstm_backward_bptt_scan_elapsed_us", "last"),
        "lstm_scan_hidden_values": metric_number(
            trainer, "lstm_backward_bptt_scan_hidden_values", "last"
        ),
        "lstm_scan_gate_values": metric_number(
            trainer, "lstm_backward_bptt_scan_gate_values", "last"
        ),
        "lstm_scan_recurrent_weight_values": metric_number(
            trainer, "lstm_backward_bptt_scan_recurrent_weight_values", "last"
        ),
        "lstm_scan_dispatches": metric_number(
            trainer, "lstm_backward_bptt_scan_kernel_dispatches", "last"
        ),
        "lstm_scan_serial_steps": metric_number(
            trainer, "lstm_backward_bptt_scan_serial_steps", "last"
        ),
        "lstm_scan_workgroup": metric_number(
            trainer, "lstm_backward_bptt_scan_workgroup_size", "last"
        ),
        "lstm_scan_parallel_lanes": metric_number(
            trainer, "lstm_backward_bptt_scan_parallel_lanes", "last"
        ),
        "lstm_scan_parallel_axis": scan_parallel_axis(row),
        "lstm_est_bptt_ops_per_scan_step": metric_number(
            trainer, "lstm_backward_estimated_bptt_ops_per_scan_step", "last"
        ),
        "lstm_est_bptt_ops_per_us": ops_per_us(row),
        "backend_status": backend_meta[0],
        "backend_kernels": backend_meta[1],
    }


def row_values(row: dict[str, Any]) -> list[str]:
    trainer = row.get("trainer_summary", {})
    trace = row.get("trace_summary", {})
    run_meta = row.get("run_meta", {})
    backend_meta = backend_meta_row(run_meta)
    scan_route = row.get("scan_route")
    route = scan_route if isinstance(scan_route, dict) else {}
    return [
        str(row.get("steps", "-")),
        str(row.get("hidden", "-")),
        str(row.get("backend", "-")),
        str(row.get("seed", "-")),
        status_label(row),
        returncode_label(row.get("returncode", 0)),
        failure_label(row, "failure_kind"),
        failure_label(row, "failure_detail"),
        fmt(trace.get("pretrain_backend_gap"), 6),
        fmt(trace.get("last_loss"), 6),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_fallbacks", "last"), 0),
        fmt(metric(trainer, "lstm_estimated_cpu_debt_ops", "last"), 0),
        fmt(metric(trainer, "lstm_estimated_bptt_wgpu_ops", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_runtime_requested", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_runtime_available", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_runtime_unavailable", "last"), 0),
        str(route.get("backend") or "-"),
        str(route.get("kernel") or "-"),
        str(route.get("lowering") or "-"),
        str(route.get("fallback_reason") or "-"),
        fmt(metric(trainer, "lstm_backward_bptt_scan_elapsed_us", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_hidden_values", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_gate_values", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_recurrent_weight_values", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_kernel_dispatches", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_serial_steps", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_workgroup_size", "last"), 0),
        fmt(metric(trainer, "lstm_backward_bptt_scan_parallel_lanes", "last"), 0),
        scan_parallel_axis(row),
        fmt(metric(trainer, "lstm_backward_estimated_bptt_ops_per_scan_step", "last"), 0),
        fmt(ops_per_us(row), 3),
        backend_meta[0],
        backend_meta[1],
        str(row.get("run_dir") or "-"),
    ]


HEADERS = [
    "steps",
    "hidden",
    "backend",
    "seed",
    "run_status",
    "returncode",
    "failure_kind",
    "failure_detail",
    "pretrain_backend_gap",
    "last_loss",
    "step_ms_last",
    "tensor_ops",
    "tensor_wgpu",
    "fallbacks",
    "lstm_est_cpu_debt_ops",
    "lstm_est_bptt_wgpu_ops",
    "lstm_scan_rt_req",
    "lstm_scan_rt_ok",
    "lstm_scan_rt_miss",
    "lstm_scan_backend",
    "lstm_scan_kernel",
    "lstm_scan_lowering",
    "lstm_scan_fallback",
    "lstm_scan_us",
    "lstm_scan_hidden_values",
    "lstm_scan_gate_values",
    "lstm_scan_recurrent_weight_values",
    "lstm_scan_dispatches",
    "lstm_scan_serial_steps",
    "lstm_scan_workgroup",
    "lstm_scan_parallel_lanes",
    "lstm_scan_parallel_axis",
    "lstm_est_bptt_ops_per_scan_step",
    "lstm_est_bptt_ops_per_us",
    "backend_status",
    "backend_kernels",
    "run_dir",
]


AVERAGE_HEADERS = [
    "steps",
    "hidden",
    "backend",
    "runs",
    "failed",
    "avg_last_loss",
    "avg_step_ms",
    "avg_lstm_est_cpu_debt_ops",
    "avg_lstm_est_bptt_wgpu_ops",
    "avg_lstm_scan_rt_ok",
    "avg_lstm_scan_rt_miss",
    "avg_lstm_scan_us",
    "avg_lstm_scan_gate_values",
    "avg_lstm_scan_workgroup",
    "avg_lstm_scan_parallel_lanes",
    "lstm_scan_backend_counts",
    "lstm_scan_fallback_counts",
    "avg_lstm_est_bptt_ops_per_scan_step",
    "avg_lstm_est_bptt_ops_per_us",
]


def average_values(rows: list[dict[str, Any]], *, steps: int, hidden: int, backend: str) -> list[str]:
    successful = [row for row in rows if not row.get("failed")]
    trainers = [row.get("trainer_summary", {}) for row in successful]
    traces = [row.get("trace_summary", {}) for row in successful]
    return [
        str(steps),
        str(hidden),
        backend,
        str(len(successful)),
        str(len(rows) - len(successful)),
        fmt(mean(number(trace.get("last_loss")) for trace in traces), 6),
        fmt(mean(metric_number(trainer, "step_time_ms", "last") for trainer in trainers), 3),
        fmt(
            mean(metric_number(trainer, "lstm_estimated_cpu_debt_ops", "last") for trainer in trainers),
            0,
        ),
        fmt(
            mean(metric_number(trainer, "lstm_estimated_bptt_wgpu_ops", "last") for trainer in trainers),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_bptt_scan_runtime_available", "last")
                for trainer in trainers
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_bptt_scan_runtime_unavailable", "last")
                for trainer in trainers
            ),
            0,
        ),
        fmt(
            mean(metric_number(trainer, "lstm_backward_bptt_scan_elapsed_us", "last") for trainer in trainers),
            0,
        ),
        fmt(
            mean(metric_number(trainer, "lstm_backward_bptt_scan_gate_values", "last") for trainer in trainers),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_bptt_scan_workgroup_size", "last")
                for trainer in trainers
            ),
            0,
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_bptt_scan_parallel_lanes", "last")
                for trainer in trainers
            ),
            0,
        ),
        count_labels(
            [
                (row.get("scan_route") or {}).get("backend")
                for row in successful
                if isinstance(row.get("scan_route"), dict)
            ]
        ),
        count_labels(
            [
                (row.get("scan_route") or {}).get("fallback_reason")
                for row in successful
                if isinstance(row.get("scan_route"), dict)
            ],
            missing_label="none",
        ),
        fmt(
            mean(
                metric_number(trainer, "lstm_backward_estimated_bptt_ops_per_scan_step", "last")
                for trainer in trainers
            ),
            0,
        ),
        fmt(mean(ops_per_us(row) for row in successful), 3),
    ]


def write_grid(run_root: Path, rows: list[dict[str, Any]]) -> Path:
    lines = [
        "# LSTM Backward Scan Profile Grid",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(value) for value in row_values(row)) + " |")

    lines.extend(
        [
            "",
            "## Shape Averages",
            "",
            "| " + " | ".join(AVERAGE_HEADERS) + " |",
            "| " + " | ".join("---" for _ in AVERAGE_HEADERS) + " |",
        ]
    )
    groups = sorted(
        {
            (int(row["steps"]), int(row["hidden"]), str(row.get("backend", "-")))
            for row in rows
            if isinstance(row.get("steps"), int) and isinstance(row.get("hidden"), int)
        }
    )
    for steps, hidden, backend in groups:
        group_rows = [
            row
            for row in rows
            if row.get("steps") == steps
            and row.get("hidden") == hidden
            and str(row.get("backend", "-")) == backend
        ]
        lines.append(
            "| "
            + " | ".join(md_cell(value) for value in average_values(group_rows, steps=steps, hidden=hidden, backend=backend))
            + " |"
        )

    path = run_root / "grid.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/lstm_scan_profile_grid"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("wgpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--steps-list", type=parse_int_csv, default=parse_int_csv("3,6,12"))
    parser.add_argument("--hidden-list", type=parse_int_csv, default=parse_int_csv("2,4,8"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--wgpu-min-values", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-wgpu-preflight", action="store_true")
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    cells = [
        run_cell(args, steps=steps, hidden=hidden)
        for steps in args.steps_list
        for hidden in args.hidden_list
    ]
    rows = [row for cell in cells for row in summarize_cell(cell)]
    grid_path = write_grid(args.run_root, rows)
    grid = {
        "schema": "st.sequence.lstm_scan_profile_grid.v1",
        "run_root": str(args.run_root),
        "backends": args.backends,
        "seeds": args.seeds,
        "steps_list": args.steps_list,
        "hidden_list": args.hidden_list,
        "config": {
            "epochs": args.epochs,
            "batches": args.batches,
            "lr": args.lr,
            "curvature": args.curvature,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
            "wgpu_min_values": args.wgpu_min_values,
            "wgpu_preflight": not args.no_wgpu_preflight,
        },
        "cells": cells,
        "runs": [
            {
                "steps": row.get("steps"),
                "hidden": row.get("hidden"),
                "backend": row.get("backend"),
                "seed": row.get("seed"),
                "run_dir": row.get("run_dir"),
                "log_path": row.get("log_path"),
                "returncode": row.get("returncode", 0),
                "failed": bool(row.get("failed")),
                "failure_kind": row.get("failure_kind"),
                "failure_detail": row.get("failure_detail"),
                "scan_route": row.get("scan_route"),
                "scan_profile": scan_profile_fields(row),
                **backend_manifest_fields(row.get("run_meta", {})),
            }
            for row in rows
        ],
        "failed": any(bool(row.get("failed")) for row in rows),
    }
    (args.run_root / "grid.json").write_text(
        json.dumps(grid, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"grid={grid_path}")
    if grid["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

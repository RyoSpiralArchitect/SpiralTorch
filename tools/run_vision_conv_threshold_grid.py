#!/usr/bin/env python3
"""Run a Conv2d tensor-utility threshold grid and summarize routing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from backend_sweep_meta import (
    BACKEND_META_HEADERS,
    BACKEND_RESIDUAL_HEADERS,
    backend_manifest_fields,
    backend_meta_row,
    backend_residual_row,
    load_run_meta,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "modelzoo_vision_conv_pool_classification"


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


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


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


def metric_count(summary: dict[str, Any], key: str, field: str = "last") -> float:
    return metric_number(summary, key, field) or 0.0


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


def read_trace(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "vision_trace.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"vision trace at {run_dir} is not a JSON object")
    return payload


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


def run_command(command: list[str], log_path: Path, *, threshold: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"] = str(threshold)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command, output=proc.stdout)


def run_one(
    args: argparse.Namespace,
    *,
    threshold: int,
    batch: int,
    height: int,
    width: int,
    out_channels: int,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    run_name = (
        f"threshold-{threshold}__batch-{batch}__height-{height}__width-{width}"
        f"__out-{out_channels}__backend-{backend}__seed-{seed}"
    )
    run_dir = args.run_root / run_name
    trace_json = run_dir / "vision_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    if args.skip_existing and trace_json.exists() and events_jsonl.exists():
        return {
            "threshold": threshold,
            "batch": batch,
            "height": height,
            "width": width,
            "out_channels": out_channels,
            "backend": backend,
            "seed": seed,
            "run_dir": str(run_dir),
            "skipped": True,
        }

    command = cargo_prefix(args, args.backends)
    command.extend(
        [
            "--run-dir",
            str(run_dir),
            "--events",
            str(events_jsonl),
            "--backend",
            backend,
            "--epochs",
            str(args.epochs),
            "--batches",
            str(args.batches),
            "--batch",
            str(batch),
            "--seed",
            str(seed),
            "--lr",
            str(args.lr),
            "--curvature",
            str(args.curvature),
            "--height",
            str(height),
            "--width",
            str(width),
            "--out-channels",
            str(out_channels),
        ]
    )
    run_command(command, run_dir / "process.log", threshold=threshold)
    return {
        "threshold": threshold,
        "batch": batch,
        "height": height,
        "width": width,
        "out_channels": out_channels,
        "backend": backend,
        "seed": seed,
        "run_dir": str(run_dir),
        "skipped": False,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / run["run_dir"]
    return {
        **run,
        "trace": read_trace(run_dir),
        "run_meta": load_run_meta(run_dir),
        "trainer_summary": TRAINER_TRACE.summarize_trainer_trace_events(
            run_dir / "trainer_trace.jsonl"
        ),
    }


CPU_THRESHOLD_UTILITY_METRICS = [
    "tensor_op_backend_add_row_inplace_cpu",
    "tensor_op_backend_sum_axis0_cpu",
    "tensor_op_backend_sum_axis0_scaled_cpu",
    "tensor_op_backend_project_to_poincare_cpu",
    "tensor_op_backend_scale_cpu",
]


def cpu_threshold_utility_ops(summary: dict[str, Any], backend: str) -> float:
    if backend != "wgpu":
        return 0.0
    return metric_number_sum(summary, CPU_THRESHOLD_UTILITY_METRICS) or 0.0


def cpu_heavy_ops(summary: dict[str, Any]) -> float:
    return metric_number_sum(
        summary,
        [
            "tensor_op_backend_reshape_cpu",
            "tensor_op_backend_transpose_cpu",
        ],
    ) or 0.0


def wgpu_utility_ops(summary: dict[str, Any]) -> float:
    return metric_number_sum(
        summary,
        [
            "tensor_op_backend_add_row_inplace_wgpu",
            "tensor_op_backend_sum_axis0_wgpu",
            "tensor_op_backend_sum_axis0_scaled_wgpu",
            "tensor_op_backend_project_to_poincare_wgpu",
            "tensor_op_backend_wave_gate_project_wgpu",
            "tensor_op_backend_scale_wgpu",
        ],
    ) or 0.0


def route_label(values: int, threshold: int, backend: str) -> str:
    if backend != "wgpu":
        return "cpu-policy"
    return "wgpu" if values >= threshold else "cpu-threshold"


def conv_output_hw(height: int, width: int) -> tuple[int, int]:
    # The vision probe fixes Conv2d to 3x3, stride 1, padding 1.
    return height, width


def row_for(summary: dict[str, Any]) -> list[str]:
    trainer = summary["trainer_summary"]
    trace = summary["trace"]
    run_summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    threshold = int(summary["threshold"])
    backend = str(summary["backend"])
    batch = int(summary["batch"])
    height = int(summary["height"])
    width = int(summary["width"])
    out_channels = int(summary["out_channels"])
    conv_h, conv_w = conv_output_hw(height, width)
    conv_bias_values = batch * conv_h * conv_w * out_channels
    head_bias_values = batch
    return [
        str(threshold),
        backend,
        str(batch),
        str(height),
        str(width),
        str(out_channels),
        str(conv_bias_values),
        str(head_bias_values),
        route_label(conv_bias_values, threshold, backend),
        route_label(head_bias_values, threshold, backend),
        str(summary["seed"]),
        fmt(run_summary.get("pretrain_backend_gap"), 6),
        fmt(run_summary.get("last_loss"), 6),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_cpu", "last"), 0),
        fmt(metric(trainer, "tensor_kernel_backend_wgpu", "last"), 0),
        *backend_meta_row(summary.get("run_meta", {})),
        *backend_residual_row(trainer),
        fmt(metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_matmul_lhs_transpose_scaled_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_matmul_prepacked_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_matmul_prepacked_bias_wgpu"), 0),
        fmt(wgpu_utility_ops(trainer), 0),
        fmt(cpu_threshold_utility_ops(trainer, backend), 0),
        fmt(cpu_heavy_ops(trainer), 0),
    ]


def grouped_key(summary: dict[str, Any]) -> tuple[int, str, int, int, int, int]:
    return (
        int(summary["threshold"]),
        str(summary["backend"]),
        int(summary["batch"]),
        int(summary["height"]),
        int(summary["width"]),
        int(summary["out_channels"]),
    )


def average_row(
    group: tuple[int, str, int, int, int, int], summaries: list[dict[str, Any]]
) -> list[str]:
    threshold, backend, batch, height, width, out_channels = group
    conv_h, conv_w = conv_output_hw(height, width)
    conv_bias_values = batch * conv_h * conv_w * out_channels
    head_bias_values = batch
    trainers = [summary["trainer_summary"] for summary in summaries]
    traces = [
        summary["trace"].get("summary")
        if isinstance(summary["trace"].get("summary"), dict)
        else {}
        for summary in summaries
    ]
    return [
        str(threshold),
        backend,
        str(batch),
        str(height),
        str(width),
        str(out_channels),
        str(len(summaries)),
        str(conv_bias_values),
        str(head_bias_values),
        route_label(conv_bias_values, threshold, backend),
        route_label(head_bias_values, threshold, backend),
        fmt(mean(metric_number(trainer, "step_time_ms", "last") for trainer in trainers), 3),
        fmt(mean(number(trace.get("pretrain_backend_gap")) for trace in traces), 6),
        fmt(mean(number(trace.get("last_loss")) for trace in traces), 6),
        fmt(
            mean(
                metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_wgpu")
                for trainer in trainers
            ),
            0,
        ),
        fmt(
            mean(
                metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_cpu")
                for trainer in trainers
            ),
            0,
        ),
        fmt(mean(wgpu_utility_ops(trainer) for trainer in trainers), 0),
        fmt(
            mean(
                cpu_threshold_utility_ops(summary["trainer_summary"], backend)
                for summary in summaries
            ),
            0,
        ),
        fmt(mean(cpu_heavy_ops(trainer) for trainer in trainers), 0),
    ]


def write_compare(run_root: Path, summaries: list[dict[str, Any]]) -> Path:
    headers = [
        "threshold",
        "backend",
        "batch",
        "height",
        "width",
        "out_channels",
        "conv_bias_values",
        "head_bias_values",
        "conv_bias_route",
        "head_bias_route",
        "seed",
        "pretrain_backend_gap",
        "last_loss",
        "step_ms_last",
        "tensor_ops",
        "tensor_wgpu",
        "tensor_cpu",
        "kernel_wgpu",
        *BACKEND_META_HEADERS,
        *BACKEND_RESIDUAL_HEADERS,
        "sum_axis0_scaled_wgpu",
        "sum_axis0_scaled_cpu",
        "matmul_lhs_t_scaled_wgpu",
        "prepacked_wgpu",
        "prepacked_bias_wgpu",
        "wgpu_utility_ops",
        "cpu_threshold_utility_ops",
        "cpu_heavy_ops",
    ]
    lines = [
        "# Vision Conv Tensor Utility Threshold Grid",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(row_for(summary)) + " |")

    average_headers = [
        "threshold",
        "backend",
        "batch",
        "height",
        "width",
        "out_channels",
        "runs",
        "conv_bias_values",
        "head_bias_values",
        "conv_bias_route",
        "head_bias_route",
        "avg_step_ms_last",
        "avg_backend_gap",
        "avg_last_loss",
        "avg_sum_axis0_scaled_wgpu",
        "avg_sum_axis0_scaled_cpu",
        "avg_wgpu_utility_ops",
        "avg_cpu_threshold_utility_ops",
        "avg_cpu_heavy_ops",
    ]
    lines.extend(
        [
            "",
            "## Group Averages",
            "",
            "| " + " | ".join(average_headers) + " |",
            "| " + " | ".join("---" for _ in average_headers) + " |",
        ]
    )
    groups: dict[tuple[int, str, int, int, int, int], list[dict[str, Any]]] = {}
    for summary in summaries:
        groups.setdefault(grouped_key(summary), []).append(summary)
    for group in sorted(groups):
        lines.append("| " + " | ".join(average_row(group, groups[group])) + " |")

    path = run_root / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("models/runs/vision_conv_tensor_util_threshold_grid"),
    )
    parser.add_argument("--thresholds", type=parse_int_csv, default=parse_int_csv("1,1024"))
    parser.add_argument("--batch-sizes", type=parse_int_csv, default=parse_int_csv("2,4"))
    parser.add_argument("--heights", type=parse_int_csv, default=parse_int_csv("8"))
    parser.add_argument("--widths", type=parse_int_csv, default=parse_int_csv("8"))
    parser.add_argument("--out-channels", type=parse_int_csv, default=parse_int_csv("4"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("wgpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("41"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    runs = [
        run_one(
            args,
            threshold=threshold,
            batch=batch,
            height=height,
            width=width,
            out_channels=out_channels,
            backend=backend,
            seed=seed,
        )
        for threshold in args.thresholds
        for batch in args.batch_sizes
        for height in args.heights
        for width in args.widths
        for out_channels in args.out_channels
        for backend in args.backends
        for seed in args.seeds
    ]
    summaries = [summarize_run(run) for run in runs]
    compare_path = write_compare(args.run_root, summaries)
    grid = {
        "schema": "st.vision.conv_tensor_util_threshold_grid.v1",
        "run_root": str(args.run_root),
        "thresholds": args.thresholds,
        "batch_sizes": args.batch_sizes,
        "heights": args.heights,
        "widths": args.widths,
        "out_channels": args.out_channels,
        "backends": args.backends,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "batches": args.batches,
            "lr": args.lr,
            "curvature": args.curvature,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
        },
        "runs": [
            {
                "threshold": summary["threshold"],
                "batch": summary["batch"],
                "height": summary["height"],
                "width": summary["width"],
                "out_channels": summary["out_channels"],
                "backend": summary["backend"],
                "seed": summary["seed"],
                "run_dir": summary["run_dir"],
                "skipped": summary["skipped"],
                **backend_manifest_fields(summary.get("run_meta", {})),
            }
            for summary in summaries
        ],
    }
    (args.run_root / "grid.json").write_text(
        json.dumps(grid, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"compare={compare_path}")


if __name__ == "__main__":
    main()

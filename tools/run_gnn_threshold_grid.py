#!/usr/bin/env python3
"""Run a GNN tensor-utility threshold grid and summarize routing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from dataclasses import dataclass
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
EXAMPLE = "gnn_trainer_band_trace_demo"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GNN_TRACE = _load_module("spiraltorch_gnn_trace", ROOT / "bindings/st-py/spiraltorch/gnn_trace.py")
TRAINER_TRACE = _load_module(
    "spiraltorch_trainer_trace", ROOT / "bindings/st-py/spiraltorch/trainer_trace.py"
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def positive_ints(values: list[int], *, label: str) -> list[int]:
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"--{label} entries must be positive")
    return values


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


@dataclass(frozen=True)
class ThresholdRunAxes:
    threshold: int
    nodes: int
    features: int
    batch: int
    backend: str
    seed: int

    @property
    def input_rows(self) -> int:
        return self.nodes * self.batch

    @property
    def output_values(self) -> int:
        return self.input_rows * self.features

    @property
    def hidden_values(self) -> int:
        return self.output_values * 2

    def as_record(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "nodes": self.nodes,
            "features": self.features,
            "batch": self.batch,
            "backend": self.backend,
            "seed": self.seed,
            "input_rows": self.input_rows,
            "output_values": self.output_values,
            "hidden_values": self.hidden_values,
        }

    def run_name(self) -> str:
        return (
            f"threshold-{self.threshold}__nodes-{self.nodes}__features-{self.features}"
            f"__batch-{self.batch}__backend-{self.backend}__seed-{self.seed}"
        )


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


def band_max_delta(summary: dict[str, Any], band: str) -> Any:
    bands = summary.get("bands")
    if not isinstance(bands, dict):
        return None
    band_summary = bands.get(band)
    if not isinstance(band_summary, dict):
        return None
    scales = band_summary.get("band_pass_scales")
    if not isinstance(scales, dict):
        return None
    return scales.get("max_abs_delta")


def readout_mse(summary: dict[str, Any]) -> Any:
    readout = summary.get("readout")
    if not isinstance(readout, dict):
        return None
    error = readout.get("error")
    if not isinstance(error, dict):
        return None
    return error.get("mean_squared_error")


def readout_nmse(summary: dict[str, Any]) -> Any:
    readout = summary.get("readout")
    if not isinstance(readout, dict):
        return None
    error = readout.get("error")
    if not isinstance(error, dict):
        return None
    return error.get("normalized_mean_squared_error")


def readout_trace(summary: dict[str, Any]) -> dict[str, Any]:
    readout = summary.get("readout")
    if not isinstance(readout, dict):
        return {}
    trace = readout.get("trace")
    return trace if isinstance(trace, dict) else {}


def readout_graph_count(summary: dict[str, Any]) -> Any:
    return readout_trace(summary).get("graph_count")


def readout_total_rows(summary: dict[str, Any]) -> Any:
    return readout_trace(summary).get("total_rows")


def validation_readout(summary: dict[str, Any]) -> dict[str, Any]:
    readout = summary.get("validation_readout")
    return readout if isinstance(readout, dict) else {}


def validation_readout_mse(summary: dict[str, Any]) -> Any:
    return validation_readout(summary).get("mean_squared_error")


def validation_readout_nmse(summary: dict[str, Any]) -> Any:
    return validation_readout(summary).get("normalized_mean_squared_error")


def validation_readout_graph_count(summary: dict[str, Any]) -> Any:
    return validation_readout(summary).get("graph_count")


def validation_readout_total_rows(summary: dict[str, Any]) -> Any:
    return validation_readout(summary).get("total_rows")


def best_score(summary: dict[str, Any]) -> Any:
    trainer = summary.get("trainer")
    if not isinstance(trainer, dict):
        return None
    return trainer.get("best_score")


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


def run_command(command: list[str], log_path: Path, *, threshold: int) -> int:
    return run_logged_command(
        command,
        log_path,
        cwd=ROOT,
        env_overrides={"SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES": str(threshold)},
    )


def command_for(args: argparse.Namespace, axes: ThresholdRunAxes, run_dir: Path) -> list[str]:
    trace_json = run_dir / "gnn_band_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    command = cargo_prefix(args, [axes.backend])
    command.extend(
        [
            "--run-dir",
            str(run_dir),
            "--trace-json",
            str(trace_json),
            "--events",
            str(events_jsonl),
            "--backend",
            axes.backend,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--train-graphs",
            str(args.train_graphs),
            "--validation-graphs",
            str(args.validation_graphs),
            "--batch",
            str(axes.batch),
            "--nodes",
            str(axes.nodes),
            "--features",
            str(axes.features),
            "--seed",
            str(axes.seed),
            "--lr",
            str(args.lr),
            "--curvature",
            str(args.curvature),
        ]
    )
    return command


def planned_run(args: argparse.Namespace, axes: ThresholdRunAxes) -> dict[str, Any]:
    run_name = axes.run_name()
    run_dir = args.run_root / run_name
    return {
        "name": run_name,
        **axes.as_record(),
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "process.log"),
        "returncode": 0,
        "failure_kind": None,
        "failure_detail": None,
        "skipped": False,
        "failed": False,
        "command": command_for(args, axes, run_dir),
    }


def preflight_axes(args: argparse.Namespace, backend: str) -> ThresholdRunAxes:
    return ThresholdRunAxes(
        threshold=max(args.thresholds),
        nodes=3,
        features=1,
        batch=1,
        backend=backend,
        seed=0,
    )


def run_wgpu_preflight(args: argparse.Namespace, backend: str) -> dict[str, Any] | None:
    if backend != "wgpu" or args.no_wgpu_preflight:
        return None

    axes = preflight_axes(args, backend)
    run_dir = args.run_root / "_preflight" / f"backend-{backend}"
    command = command_for(args, axes, run_dir)
    log_path = run_dir / "process.log"
    returncode = run_command(command, log_path, threshold=axes.threshold)
    if returncode == 0:
        return None

    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure = {
        "schema": "st.gnn.tensor_util_threshold_grid_preflight_failure.v1",
        "backend": backend,
        "threshold": axes.threshold,
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
    args: argparse.Namespace,
    axes: ThresholdRunAxes,
    preflight_failure: dict[str, Any],
) -> dict[str, Any]:
    planned = planned_run(args, axes)
    run_dir = Path(planned["run_dir"])
    log_path = Path(planned["log_path"])
    record = preflight_skipped_run_record(
        schema="st.gnn.tensor_util_threshold_grid_failure.v1",
        backend=axes.backend,
        seed=axes.seed,
        run_dir=run_dir,
        log_path=log_path,
        command=planned["command"],
        preflight_failure=preflight_failure,
    )
    record.update(
        {
            "name": planned["name"],
            **axes.as_record(),
        }
    )
    return record


def run_one(args: argparse.Namespace, axes: ThresholdRunAxes) -> dict[str, Any]:
    planned = planned_run(args, axes)
    run_dir = Path(planned["run_dir"])
    trace_json = run_dir / "gnn_band_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    if args.skip_existing and trace_json.exists() and events_jsonl.exists():
        return {
            **planned,
            "returncode": 0,
            "skipped": True,
            "failed": False,
        }

    command = planned["command"]
    log_path = run_dir / "process.log"
    returncode = run_command(command, log_path, threshold=axes.threshold)
    failed = returncode != 0
    failure_kind = None
    failure_detail = None
    if failed:
        failure_kind, failure_detail = classify_failure(returncode, log_path)
        failure = {
            "schema": "st.gnn.tensor_util_threshold_grid_failure.v1",
            "threshold": axes.threshold,
            "nodes": axes.nodes,
            "features": axes.features,
            "batch": axes.batch,
            "backend": axes.backend,
            "seed": axes.seed,
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
        **planned,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": returncode,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "skipped": False,
        "failed": failed,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / run["run_dir"]
    if run.get("failed"):
        return {
            **run,
            "gnn_summary": {},
            "run_meta": load_run_meta(run_dir),
            "trainer_summary": {"metrics": {}},
        }
    return {
        **run,
        "gnn_summary": GNN_TRACE.summarize_gnn_band_replays(run_dir / "gnn_band_trace.json"),
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
            "tensor_op_backend_graph_readout_cpu",
            "tensor_op_backend_graph_readout_backward_cpu",
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


def row_for(summary: dict[str, Any]) -> list[str]:
    threshold = int(summary["threshold"])
    backend = str(summary["backend"])
    nodes = int(summary["nodes"])
    features = int(summary["features"])
    batch = int(summary["batch"])
    rows = nodes * batch
    output_values = rows * features
    hidden_values = rows * features * 2
    prefix = [
        str(threshold),
        backend,
        str(nodes),
        str(features),
        str(batch),
        str(rows),
        str(output_values),
        str(hidden_values),
        route_label(output_values, threshold, backend),
        route_label(hidden_values, threshold, backend),
        str(summary["seed"]),
        status_label(summary),
        returncode_label(summary.get("returncode", 0)),
        failure_label(summary, "failure_kind"),
        failure_label(summary, "failure_detail"),
        str(summary.get("log_path") or "-"),
    ]
    if summary.get("failed"):
        return prefix + ["-" for _ in data_column_headers()]

    gnn = summary["gnn_summary"]
    trainer = summary["trainer_summary"]
    return [
        *prefix,
        fmt(best_score(gnn), 6),
        fmt(readout_mse(gnn), 6),
        fmt(readout_nmse(gnn), 6),
        fmt(readout_graph_count(gnn), 0),
        fmt(readout_total_rows(gnn), 0),
        fmt(validation_readout_mse(gnn), 6),
        fmt(validation_readout_nmse(gnn), 6),
        fmt(validation_readout_graph_count(gnn), 0),
        fmt(validation_readout_total_rows(gnn), 0),
        fmt(band_max_delta(gnn, "above"), 4),
        fmt(band_max_delta(gnn, "here"), 4),
        fmt(band_max_delta(gnn, "beneath"), 4),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_cpu", "last"), 0),
        *backend_meta_row(summary.get("run_meta", {})),
        *backend_residual_row(trainer),
        fmt(metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_sum_axis0_scaled_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_add_row_inplace_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_add_row_inplace_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_scale_wgpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_scale_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_graph_readout_composite"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_graph_readout_backward_composite"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_graph_readout_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_graph_readout_backward_cpu"), 0),
        fmt(metric_count(trainer, "tensor_op_backend_matmul_lhs_transpose_scaled_wgpu"), 0),
        fmt(wgpu_utility_ops(trainer), 0),
        fmt(cpu_threshold_utility_ops(trainer, backend), 0),
        fmt(cpu_heavy_ops(trainer), 0),
    ]


def data_column_headers() -> list[str]:
    return [
        "best_score",
        "readout_mse",
        "readout_nmse",
        "readout_graphs",
        "readout_rows",
        "validation_readout_mse",
        "validation_readout_nmse",
        "validation_readout_graphs",
        "validation_readout_rows",
        "above_max_delta",
        "here_max_delta",
        "beneath_max_delta",
        "step_ms_last",
        "tensor_ops",
        "tensor_wgpu",
        "tensor_cpu",
        *BACKEND_META_HEADERS,
        *BACKEND_RESIDUAL_HEADERS,
        "sum_axis0_scaled_wgpu",
        "sum_axis0_scaled_cpu",
        "add_row_wgpu",
        "add_row_cpu",
        "scale_wgpu",
        "scale_cpu",
        "graph_readout_composite",
        "graph_readout_backward_composite",
        "graph_readout_cpu",
        "graph_readout_backward_cpu",
        "matmul_lhs_t_scaled_wgpu",
        "wgpu_utility_ops",
        "cpu_threshold_utility_ops",
        "cpu_heavy_ops",
    ]


def grouped_key(summary: dict[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(summary["threshold"]),
        str(summary["backend"]),
        int(summary["nodes"]),
        int(summary["features"]),
        int(summary["batch"]),
    )


def average_row(
    group: tuple[int, str, int, int, int], summaries: list[dict[str, Any]]
) -> list[str]:
    threshold, backend, nodes, features, batch = group
    rows = nodes * batch
    output_values = rows * features
    hidden_values = rows * features * 2
    successful = [summary for summary in summaries if not summary.get("failed")]
    trainers = [summary["trainer_summary"] for summary in successful]
    gnns = [summary["gnn_summary"] for summary in successful]
    return [
        str(threshold),
        backend,
        str(nodes),
        str(features),
        str(batch),
        str(len(successful)),
        str(output_values),
        str(hidden_values),
        route_label(output_values, threshold, backend),
        route_label(hidden_values, threshold, backend),
        fmt(mean(metric_number(trainer, "step_time_ms", "last") for trainer in trainers), 3),
        fmt(mean(best_score(gnn) for gnn in gnns), 6),
        fmt(mean(readout_mse(gnn) for gnn in gnns), 6),
        fmt(mean(readout_nmse(gnn) for gnn in gnns), 6),
        fmt(mean(readout_graph_count(gnn) for gnn in gnns), 0),
        fmt(mean(readout_total_rows(gnn) for gnn in gnns), 0),
        fmt(mean(validation_readout_mse(gnn) for gnn in gnns), 6),
        fmt(mean(validation_readout_nmse(gnn) for gnn in gnns), 6),
        fmt(mean(validation_readout_graph_count(gnn) for gnn in gnns), 0),
        fmt(mean(validation_readout_total_rows(gnn) for gnn in gnns), 0),
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
        fmt(
            mean(
                metric_count(trainer, "tensor_op_backend_add_row_inplace_wgpu")
                for trainer in trainers
            ),
            0,
        ),
        fmt(
            mean(
                metric_count(trainer, "tensor_op_backend_add_row_inplace_cpu")
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
        "nodes",
        "features",
        "batch",
        "rows",
        "output_bias_values",
        "hidden_bias_values",
        "output_bias_route",
        "hidden_bias_route",
        "seed",
        "run_status",
        "returncode",
        "failure_kind",
        "failure_detail",
        "log_path",
        *data_column_headers(),
    ]
    lines = [
        "# GNN Tensor Utility Threshold Grid",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row_for(summary)) + " |")

    average_headers = [
        "threshold",
        "backend",
        "nodes",
        "features",
        "batch",
        "runs",
        "output_bias_values",
        "hidden_bias_values",
        "output_bias_route",
        "hidden_bias_route",
        "avg_step_ms_last",
        "avg_best_score",
        "avg_readout_mse",
        "avg_readout_nmse",
        "avg_readout_graphs",
        "avg_readout_rows",
        "avg_validation_readout_mse",
        "avg_validation_readout_nmse",
        "avg_validation_readout_graphs",
        "avg_validation_readout_rows",
        "avg_sum_axis0_scaled_wgpu",
        "avg_sum_axis0_scaled_cpu",
        "avg_add_row_wgpu",
        "avg_add_row_cpu",
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
    groups: dict[tuple[int, str, int, int, int], list[dict[str, Any]]] = {}
    for summary in summaries:
        if summary.get("failed"):
            continue
        groups.setdefault(grouped_key(summary), []).append(summary)
    for group in sorted(groups):
        lines.append("| " + " | ".join(average_row(group, groups[group])) + " |")

    path = run_root / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("models/runs/gnn_tensor_util_threshold_grid"),
    )
    parser.add_argument("--thresholds", type=parse_int_csv, default=parse_int_csv("1,1024"))
    parser.add_argument("--nodes", type=parse_int_csv, default=parse_int_csv("8,128"))
    parser.add_argument("--features", type=parse_int_csv, default=parse_int_csv("4"))
    parser.add_argument("--batches", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("wgpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("41"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--train-graphs", type=int, default=2)
    parser.add_argument("--validation-graphs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-wgpu-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.thresholds = positive_ints(args.thresholds, label="thresholds")
    args.nodes = positive_ints(args.nodes, label="nodes")
    args.features = positive_ints(args.features, label="features")
    args.batches = positive_ints(args.batches, label="batches")
    args.seeds = positive_ints(args.seeds, label="seeds")
    return args


def iter_axes(args: argparse.Namespace) -> Iterable[ThresholdRunAxes]:
    for threshold in args.thresholds:
        for nodes in args.nodes:
            for features in args.features:
                for batch in args.batches:
                    for backend in args.backends:
                        for seed in args.seeds:
                            yield ThresholdRunAxes(
                                threshold=threshold,
                                nodes=nodes,
                                features=features,
                                batch=batch,
                                backend=backend,
                                seed=seed,
                            )


def grid_manifest(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    preflight_failures: dict[str, Any],
    *,
    summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary_rows = summaries if summaries is not None else runs
    return {
        "schema": "st.gnn.tensor_util_threshold_grid.v2",
        "run_root": str(args.run_root),
        "thresholds": args.thresholds,
        "nodes": args.nodes,
        "features": args.features,
        "batches": args.batches,
        "backends": args.backends,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "train_graphs": args.train_graphs,
            "validation_graphs": args.validation_graphs,
            "lr": args.lr,
            "curvature": args.curvature,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
            "wgpu_preflight": not args.no_wgpu_preflight,
            "dry_run": args.dry_run,
        },
        "preflight_failures": preflight_failures,
        "runs": [
            {
                "name": summary.get("name"),
                "threshold": summary["threshold"],
                "nodes": summary["nodes"],
                "features": summary["features"],
                "batch": summary["batch"],
                "backend": summary["backend"],
                "seed": summary["seed"],
                "input_rows": summary["input_rows"],
                "output_values": summary["output_values"],
                "hidden_values": summary["hidden_values"],
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
            for summary in summary_rows
        ],
        "failed": any(bool(summary.get("failed")) for summary in summary_rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        runs = [planned_run(args, axes) for axes in iter_axes(args)]
        grid = grid_manifest(args, runs, {})
        (args.run_root / "grid.json").write_text(
            json.dumps(grid, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"planned={len(runs)}")
        return 0

    preflight_failures = {}
    for backend in dict.fromkeys(args.backends):
        failure = run_wgpu_preflight(args, backend)
        if failure is not None:
            preflight_failures[backend] = failure

    runs = []
    for axes in iter_axes(args):
        preflight_failure = preflight_failures.get(axes.backend)
        if preflight_failure is not None:
            runs.append(preflight_skipped_run(args, axes, preflight_failure))
        else:
            runs.append(run_one(args, axes))
    summaries = [summarize_run(run) for run in runs]
    compare_path = write_compare(args.run_root, summaries)
    grid = grid_manifest(args, runs, preflight_failures, summaries=summaries)
    grid["compare_path"] = str(compare_path)
    (args.run_root / "grid.json").write_text(
        json.dumps(grid, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"compare={compare_path}")
    return 1 if grid["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

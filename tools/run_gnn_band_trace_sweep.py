#!/usr/bin/env python3
"""Run small GNN band-trace learning probes and compare their artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from backend_sweep_meta import (
    BACKEND_META_HEADERS,
    BACKEND_RESIDUAL_HEADERS,
    backend_manifest_fields,
    backend_meta_row,
    backend_residual_columns,
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
GROUP_FIELDS = (
    "backend",
    "epochs",
    "train_graphs",
    "validation_graphs",
    "batch",
    "nodes",
    "features",
    "lr",
    "top_k",
    "mid_k",
    "bottom_k",
    "here_tolerance",
)
ROUNDTABLE_AXES = ("top_k", "mid_k", "bottom_k", "here_tolerance")
ROUNDTABLE_AXIS_INDICES = {
    "top_k": GROUP_FIELDS.index("top_k"),
    "mid_k": GROUP_FIELDS.index("mid_k"),
    "bottom_k": GROUP_FIELDS.index("bottom_k"),
    "here_tolerance": GROUP_FIELDS.index("here_tolerance"),
}
FOLLOW_UP_NEIGHBORHOOD_AXES = ("lr", "top_k", "mid_k", "bottom_k", "here_tolerance")
FOLLOW_UP_VERDICTS = ("improved", "matched", "regressed", "unknown")
FOLLOW_UP_SOURCES = ("auto", "promotion", "top-candidate")
FOLLOW_UP_CHAIN_MAX_ANCESTORS = 16
FOLLOW_UP_COMMAND_PLACEHOLDERS = ("NEXT_RUN_ROOT", "NEW_SEEDS")
FOLLOW_UP_NEXT_COMMAND_SCRIPT = "next_follow_up_command.sh"
FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS = ("single_seed_probe", "volatile")
FOLLOW_UP_STABILITY_ACCEPTABLE = ("multi_seed_stable", "watch_spread")


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


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_follow_up_verdict_csv(value: str) -> list[str]:
    verdicts = parse_csv(value)
    unknown = [verdict for verdict in verdicts if verdict not in FOLLOW_UP_VERDICTS]
    if unknown:
        allowed = ", ".join(FOLLOW_UP_VERDICTS)
        raise argparse.ArgumentTypeError(
            "invalid follow-up verdicts: " + ", ".join(unknown) + f" (allowed: {allowed})"
        )
    return verdicts


def parse_follow_up_source(value: str) -> str:
    if value not in FOLLOW_UP_SOURCES:
        allowed = ", ".join(FOLLOW_UP_SOURCES)
        raise argparse.ArgumentTypeError(
            f"invalid follow-up source: {value} (allowed: {allowed})"
        )
    return value


def option_present(argv: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv for name in names)


def csv_number_values(values: Iterable[float | int]) -> str:
    return ",".join(f"{value:.12g}" if isinstance(value, float) else str(value) for value in values)


def stable_float(value: float) -> float:
    return float(f"{value:.12g}")


def positive_int_grid(raw: str | None, fallback: int, *, label: str) -> list[int]:
    values = [fallback] if raw is None else parse_int_csv(raw)
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"--{label} entries must be positive")
    return values


def positive_float_grid(raw: str | None, fallback: float, *, label: str) -> list[float]:
    values = [fallback] if raw is None else parse_float_csv(raw)
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"--{label} entries must be positive")
    return values


def nonnegative_float_grid(raw: str | None, fallback: float, *, label: str) -> list[float]:
    values = [fallback] if raw is None else parse_float_csv(raw)
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    if any(value < 0.0 for value in values):
        raise ValueError(f"--{label} entries must be non-negative")
    return values


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def axis_label(value: Any) -> str:
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if _number(value) is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _numbers(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if _number(value) is not None]


def _population_stddev(values: Iterable[Any]) -> float | None:
    present = _numbers(values)
    if not present:
        return None
    mean = sum(present) / len(present)
    return (sum((value - mean) ** 2 for value in present) / len(present)) ** 0.5


def _sort_number(value: Any) -> float:
    number = _number(value)
    return number if number is not None else float("inf")


def _delta_number(value: Any, baseline: Any) -> float | None:
    lhs = _number(value)
    rhs = _number(baseline)
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _center_and_wider_int_values(value: Any) -> list[int]:
    center = _int_value(value)
    if center is None:
        return []
    wider = max(center + 1, center * 2)
    return [center, wider] if wider != center else [center]


@dataclass(frozen=True)
class GnnRunAxes:
    backend: str
    seed: int
    epochs: int
    train_graphs: int
    validation_graphs: int
    batch: int
    nodes: int
    features: int
    lr: float
    top_k: int
    mid_k: int
    bottom_k: int
    here_tolerance: float

    @property
    def input_rows(self) -> int:
        return self.batch * self.nodes

    def as_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "seed": self.seed,
            "epochs": self.epochs,
            "train_graphs": self.train_graphs,
            "validation_graphs": self.validation_graphs,
            "batch": self.batch,
            "nodes": self.nodes,
            "features": self.features,
            "lr": self.lr,
            "top_k": self.top_k,
            "mid_k": self.mid_k,
            "bottom_k": self.bottom_k,
            "here_tolerance": self.here_tolerance,
            "input_rows": self.input_rows,
        }

    def run_name(self) -> str:
        return "__".join(
            [
                f"backend-{self.backend}",
                f"epochs-{self.epochs}",
                f"train-{self.train_graphs}",
                f"val-{self.validation_graphs}",
                f"batch-{self.batch}",
                f"nodes-{self.nodes}",
                f"features-{self.features}",
                f"lr-{axis_label(self.lr)}",
                f"top-{self.top_k}",
                f"mid-{self.mid_k}",
                f"bottom-{self.bottom_k}",
                f"tol-{axis_label(self.here_tolerance)}",
                f"seed-{self.seed}",
            ]
        )


def metric(summary: dict[str, Any], key: str, field: str = "last") -> Any:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    return item.get(field)


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


def best_score(summary: dict[str, Any]) -> Any:
    trainer = summary.get("trainer")
    if not isinstance(trainer, dict):
        return None
    return trainer.get("best_score")


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


def cpu_debt_ops(trainer_summary: dict[str, Any]) -> float | None:
    columns = backend_residual_columns(trainer_summary)
    return _number_from_string(columns.get("cpu_debt_ops"))


def _number_from_string(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
    axes: GnnRunAxes,
    run_dir: Path,
    trace_json: Path,
    events_jsonl: Path,
) -> list[str]:
    # Keep CPU baselines CPU-only even when WGPU rows are in the same sweep.
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
            str(axes.epochs),
            "--patience",
            str(args.patience),
            "--train-graphs",
            str(axes.train_graphs),
            "--validation-graphs",
            str(axes.validation_graphs),
            "--batch",
            str(axes.batch),
            "--nodes",
            str(axes.nodes),
            "--features",
            str(axes.features),
            "--seed",
            str(axes.seed),
            "--lr",
            str(axes.lr),
            "--curvature",
            str(args.curvature),
            "--top-k",
            str(axes.top_k),
            "--mid-k",
            str(axes.mid_k),
            "--bottom-k",
            str(axes.bottom_k),
            "--here-tolerance",
            str(axes.here_tolerance),
        ]
    )
    return command


def preflight_axes(args: argparse.Namespace, backend: str) -> GnnRunAxes:
    return GnnRunAxes(
        backend=backend,
        seed=0,
        epochs=1,
        train_graphs=1,
        validation_graphs=1,
        batch=1,
        nodes=3,
        features=1,
        lr=args.lr,
        top_k=args.top_k,
        mid_k=args.mid_k,
        bottom_k=args.bottom_k,
        here_tolerance=args.here_tolerance,
    )


def candidate_axis(candidate: dict[str, Any], key: str) -> Any:
    if key not in candidate:
        raise ValueError(f"follow-up candidate is missing {key!r}")
    return candidate[key]


def follow_up_sweep_json(path: Path) -> Path:
    return path / "sweep.json" if path.is_dir() else path


def load_top_candidate(comparison: dict[str, Any], rank: int, *, sweep_path: Path) -> dict[str, Any]:
    if rank <= 0:
        raise ValueError("--follow-up-rank must be positive")
    candidates = comparison.get("top_validation_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{sweep_path} does not contain top_validation_candidates")
    try:
        candidate = candidates[rank - 1]
    except IndexError as exc:
        raise ValueError(
            f"--follow-up-rank {rank} is out of range for {len(candidates)} candidates"
        ) from exc
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate rank {rank} is not an object")
    return candidate


def load_promotion_candidate(comparison: dict[str, Any], *, sweep_path: Path) -> dict[str, Any] | None:
    promotion = comparison.get("follow_up_promotion")
    if not isinstance(promotion, dict):
        return None
    candidate = promotion.get("selected_candidate")
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise ValueError(f"{sweep_path} follow_up_promotion.selected_candidate is not an object")
    return candidate


def follow_up_lineage_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    config = payload.get("config")
    follow_up = config.get("follow_up") if isinstance(config, dict) else None
    lineage = follow_up.get("lineage") if isinstance(follow_up, dict) else None
    if not isinstance(lineage, dict):
        lineage = payload.get("follow_up_lineage")
    return lineage if isinstance(lineage, dict) else None


def follow_up_parent_generation(payload: dict[str, Any]) -> int:
    lineage = follow_up_lineage_from_payload(payload)
    generation = lineage.get("generation") if isinstance(lineage, dict) else None
    return generation if isinstance(generation, int) and generation >= 0 else 0


def follow_up_parent_run_root(payload: dict[str, Any], sweep_path: Path) -> str:
    run_root = payload.get("run_root")
    return str(run_root) if run_root else str(sweep_path.parent)


def load_follow_up_candidate(
    path: Path,
    rank: int,
    source: str,
) -> tuple[dict[str, Any], str, Path, dict[str, Any]]:
    sweep_path = follow_up_sweep_json(path)
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"{sweep_path} does not contain a comparison payload")

    if source in ("auto", "promotion"):
        promoted = load_promotion_candidate(comparison, sweep_path=sweep_path)
        if promoted is not None:
            return promoted, "promotion", sweep_path, payload
        if source == "promotion":
            raise ValueError(f"{sweep_path} does not contain follow_up_promotion.selected_candidate")

    return load_top_candidate(comparison, rank, sweep_path=sweep_path), "top-candidate", sweep_path, payload


def apply_follow_up_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if args.follow_up_from is None:
        return
    candidate, source, sweep_path, payload = load_follow_up_candidate(
        args.follow_up_from,
        args.follow_up_rank,
        args.follow_up_source,
    )

    def set_if_absent(attr: str, value: Any, *options: str) -> None:
        if not option_present(argv, *options):
            setattr(args, attr, value)

    set_if_absent("backends", [str(candidate_axis(candidate, "backend"))], "--backends")
    seeds = candidate.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        seeds = [candidate_axis(candidate, "seed")]
    set_if_absent("seeds", [int(seed) for seed in seeds], "--seeds")
    set_if_absent(
        "epochs",
        int(candidate_axis(candidate, "epochs")),
        "--epochs",
        "--epoch-values",
    )
    set_if_absent(
        "train_graphs",
        int(candidate_axis(candidate, "train_graphs")),
        "--train-graphs",
        "--train-graph-values",
    )
    set_if_absent(
        "validation_graphs",
        int(candidate_axis(candidate, "validation_graphs")),
        "--validation-graphs",
        "--validation-graph-values",
    )
    set_if_absent("batch", int(candidate_axis(candidate, "batch")), "--batch", "--batch-values")
    set_if_absent("nodes", int(candidate_axis(candidate, "nodes")), "--nodes", "--node-values")
    set_if_absent(
        "features",
        int(candidate_axis(candidate, "features")),
        "--features",
        "--feature-values",
    )
    set_if_absent("lr", float(candidate_axis(candidate, "lr")), "--lr", "--lr-values")
    set_if_absent("top_k", int(candidate_axis(candidate, "top_k")), "--top-k", "--top-k-values")
    set_if_absent("mid_k", int(candidate_axis(candidate, "mid_k")), "--mid-k", "--mid-k-values")
    set_if_absent(
        "bottom_k",
        int(candidate_axis(candidate, "bottom_k")),
        "--bottom-k",
        "--bottom-k-values",
    )
    set_if_absent(
        "here_tolerance",
        float(candidate_axis(candidate, "here_tolerance")),
        "--here-tolerance",
        "--here-tolerance-values",
    )
    args.follow_up_candidate = candidate
    args.follow_up_candidate_source = source
    args.follow_up_parent_sweep_path = sweep_path
    args.follow_up_parent_run_root = follow_up_parent_run_root(payload, sweep_path)
    args.follow_up_parent_generation = follow_up_parent_generation(payload)
    apply_follow_up_neighborhood(args, argv)


def validate_follow_up_neighborhood_axes(axes: Iterable[str]) -> list[str]:
    requested = list(axes)
    unknown = [axis for axis in requested if axis not in FOLLOW_UP_NEIGHBORHOOD_AXES]
    if unknown:
        allowed = ", ".join(FOLLOW_UP_NEIGHBORHOOD_AXES)
        raise ValueError(
            "invalid --follow-up-neighborhood-axes entries: "
            + ", ".join(unknown)
            + f" (allowed: {allowed})"
        )
    return requested


def local_k_values(center: int, radius: int) -> list[int]:
    if radius < 0:
        raise ValueError("--follow-up-neighborhood-k-radius must be non-negative")
    return list(range(max(1, center - radius), center + radius + 1))


def local_lr_values(center: float, factors: Iterable[float]) -> list[float]:
    values = sorted({stable_float(center * factor) for factor in factors if factor > 0.0})
    if not values:
        raise ValueError("--follow-up-neighborhood-lr-factors must include a positive value")
    return values


def local_tolerance_values(center: float, factors: Iterable[float]) -> list[float]:
    if center <= 0.0:
        return [0.0, 1e-6]
    values = sorted({stable_float(center * factor) for factor in factors if factor >= 0.0})
    if not values:
        raise ValueError("--follow-up-neighborhood-tolerance-factors must not be empty")
    return values


def apply_follow_up_neighborhood(args: argparse.Namespace, argv: list[str]) -> None:
    if not args.follow_up_neighborhood:
        args.follow_up_neighborhood_expanded = {}
        return
    axes = validate_follow_up_neighborhood_axes(args.follow_up_neighborhood_axes)
    expanded: dict[str, list[float | int]] = {}

    if "lr" in axes and not option_present(argv, "--lr-values"):
        values = local_lr_values(float(args.lr), args.follow_up_neighborhood_lr_factors)
        args.lr_values = csv_number_values(values)
        expanded["learning_rates"] = values

    if "top_k" in axes and not option_present(argv, "--top-k-values"):
        values = local_k_values(int(args.top_k), args.follow_up_neighborhood_k_radius)
        args.top_k_values = csv_number_values(values)
        expanded["top_k"] = values

    if "mid_k" in axes and not option_present(argv, "--mid-k-values"):
        values = local_k_values(int(args.mid_k), args.follow_up_neighborhood_k_radius)
        args.mid_k_values = csv_number_values(values)
        expanded["mid_k"] = values

    if "bottom_k" in axes and not option_present(argv, "--bottom-k-values"):
        values = local_k_values(int(args.bottom_k), args.follow_up_neighborhood_k_radius)
        args.bottom_k_values = csv_number_values(values)
        expanded["bottom_k"] = values

    if "here_tolerance" in axes and not option_present(argv, "--here-tolerance-values"):
        values = local_tolerance_values(
            float(args.here_tolerance),
            args.follow_up_neighborhood_tolerance_factors,
        )
        args.here_tolerance_values = csv_number_values(values)
        expanded["here_tolerance"] = values

    args.follow_up_neighborhood_expanded = expanded


def follow_up_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.follow_up_from is None:
        return None
    parent_generation = getattr(args, "follow_up_parent_generation", 0)
    candidate_source = getattr(args, "follow_up_candidate_source", None)
    return {
        "source": str(args.follow_up_from),
        "rank": args.follow_up_rank,
        "source_mode": args.follow_up_source,
        "candidate_source": candidate_source,
        "lineage": {
            "schema": "st.gnn.band_trace_follow_up_lineage.v1",
            "generation": parent_generation + 1,
            "parent_generation": parent_generation,
            "parent_sweep_path": str(
                getattr(args, "follow_up_parent_sweep_path", args.follow_up_from)
            ),
            "parent_run_root": getattr(args, "follow_up_parent_run_root", None),
            "source_mode": args.follow_up_source,
            "candidate_source": candidate_source,
        },
        "neighborhood": {
            "enabled": args.follow_up_neighborhood,
            "axes": args.follow_up_neighborhood_axes,
            "expanded": getattr(args, "follow_up_neighborhood_expanded", {}),
        },
        "candidate": getattr(args, "follow_up_candidate", None),
    }


def run_wgpu_preflight(args: argparse.Namespace, backend: str) -> dict[str, Any] | None:
    if backend != "wgpu" or args.no_wgpu_preflight:
        return None

    axes = preflight_axes(args, backend)
    run_dir = args.run_root / "_preflight" / f"backend-{backend}"
    trace_json = run_dir / "preflight_gnn_band_trace.json"
    events_jsonl = run_dir / "preflight_events.jsonl"
    log_path = run_dir / "process.log"
    command = example_command(
        args,
        axes,
        run_dir,
        trace_json,
        events_jsonl,
    )
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    if returncode == 0:
        return None

    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure = {
        "schema": "st.gnn.band_trace_sweep_preflight_failure.v1",
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
    args: argparse.Namespace,
    axes: GnnRunAxes,
    preflight_failure: dict[str, Any],
) -> dict[str, Any]:
    run_name = axes.run_name()
    run_dir = args.run_root / run_name
    trace_json = run_dir / "gnn_band_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    log_path = run_dir / "process.log"
    command = example_command(args, axes, run_dir, trace_json, events_jsonl)
    record = preflight_skipped_run_record(
        schema="st.gnn.band_trace_sweep_failure.v1",
        backend=axes.backend,
        seed=axes.seed,
        run_dir=run_dir,
        log_path=log_path,
        command=command,
        preflight_failure=preflight_failure,
    )
    record.update({"name": run_name, **axes.as_record()})
    return record


def planned_run(args: argparse.Namespace, axes: GnnRunAxes) -> dict[str, Any]:
    run_name = axes.run_name()
    run_dir = args.run_root / run_name
    trace_json = run_dir / "gnn_band_trace.json"
    events_jsonl = run_dir / "trainer_trace.jsonl"
    command = example_command(args, axes, run_dir, trace_json, events_jsonl)
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
        "command": command,
    }


def run_one(args: argparse.Namespace, axes: GnnRunAxes) -> dict[str, Any]:
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
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    failed = returncode != 0
    failure_kind = None
    failure_detail = None
    if failed:
        failure_kind, failure_detail = classify_failure(returncode, log_path)
        failure = {
            "schema": "st.gnn.band_trace_sweep_failure.v1",
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
    gnn_summary = GNN_TRACE.summarize_gnn_band_replays(run_dir / "gnn_band_trace.json")
    run_meta = load_run_meta(run_dir)
    trainer_summary = TRAINER_TRACE.summarize_trainer_trace_events(
        run_dir / "trainer_trace.jsonl"
    )
    return {
        **run,
        "gnn_summary": gnn_summary,
        "run_meta": run_meta,
        "trainer_summary": trainer_summary,
    }


def row_for(summary: dict[str, Any]) -> list[str]:
    prefix = [
        str(summary["backend"]),
        str(summary["seed"]),
        str(summary["epochs"]),
        str(summary["train_graphs"]),
        str(summary["validation_graphs"]),
        str(summary["batch"]),
        str(summary["nodes"]),
        str(summary["features"]),
        fmt(summary["lr"], 6),
        str(summary["top_k"]),
        str(summary["mid_k"]),
        str(summary["bottom_k"]),
        fmt(summary["here_tolerance"], 6),
        str(summary["input_rows"]),
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
        fmt(metric(trainer, "backend_policy_events", "sum"), 0),
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
        "policy_events",
    ]


def group_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(summary[field] for field in GROUP_FIELDS)


def group_record(group: tuple[Any, ...]) -> dict[str, Any]:
    record = dict(zip(GROUP_FIELDS, group, strict=True))
    record["input_rows"] = int(record["batch"]) * int(record["nodes"])
    return record


def seeds_for(summaries: list[dict[str, Any]]) -> list[int]:
    return [int(summary["seed"]) for summary in sorted(summaries, key=lambda row: row["seed"])]


def average_row(group: tuple[Any, ...], summaries: list[dict[str, Any]]) -> list[str]:
    (
        backend,
        epochs,
        train_graphs,
        validation_graphs,
        batch,
        nodes,
        features,
        lr,
        top_k,
        mid_k,
        bottom_k,
        here_tolerance,
    ) = group
    gnns = [summary["gnn_summary"] for summary in summaries]
    trainers = [summary["trainer_summary"] for summary in summaries]
    seeds = ",".join(str(summary["seed"]) for summary in sorted(summaries, key=lambda row: row["seed"]))
    return [
        str(backend),
        str(epochs),
        str(train_graphs),
        str(validation_graphs),
        str(batch),
        str(nodes),
        str(features),
        fmt(lr, 6),
        str(top_k),
        str(mid_k),
        str(bottom_k),
        fmt(here_tolerance, 6),
        str(int(batch) * int(nodes)),
        str(len(summaries)),
        seeds,
        fmt(_mean(best_score(gnn) for gnn in gnns), 6),
        fmt(_mean(readout_mse(gnn) for gnn in gnns), 6),
        fmt(_mean(readout_nmse(gnn) for gnn in gnns), 6),
        fmt(_mean(readout_graph_count(gnn) for gnn in gnns), 0),
        fmt(_mean(readout_total_rows(gnn) for gnn in gnns), 0),
        fmt(_mean(validation_readout_mse(gnn) for gnn in gnns), 6),
        fmt(_mean(validation_readout_nmse(gnn) for gnn in gnns), 6),
        fmt(_mean(validation_readout_graph_count(gnn) for gnn in gnns), 0),
        fmt(_mean(validation_readout_total_rows(gnn) for gnn in gnns), 0),
        fmt(_mean(band_max_delta(gnn, "above") for gnn in gnns), 4),
        fmt(_mean(band_max_delta(gnn, "here") for gnn in gnns), 4),
        fmt(_mean(band_max_delta(gnn, "beneath") for gnn in gnns), 4),
        fmt(_mean(metric(trainer, "step_time_ms", "last") for trainer in trainers), 3),
        fmt(_mean(metric(trainer, "tensor_ops_total", "last") for trainer in trainers), 0),
        fmt(_mean(metric(trainer, "tensor_backend_wgpu", "last") for trainer in trainers), 0),
        fmt(_mean(metric(trainer, "tensor_backend_cpu", "last") for trainer in trainers), 0),
        fmt(_mean(cpu_debt_ops(trainer) for trainer in trainers), 0),
        fmt(_mean(metric(trainer, "backend_policy_events", "sum") for trainer in trainers), 0),
    ]


def group_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    gnns = [summary["gnn_summary"] for summary in summaries]
    trainers = [summary["trainer_summary"] for summary in summaries]
    validation_mses = _numbers(validation_readout_mse(gnn) for gnn in gnns)
    validation_nmses = _numbers(validation_readout_nmse(gnn) for gnn in gnns)
    validation_mse_min = min(validation_mses) if validation_mses else None
    validation_mse_max = max(validation_mses) if validation_mses else None
    validation_nmse_min = min(validation_nmses) if validation_nmses else None
    validation_nmse_max = max(validation_nmses) if validation_nmses else None
    validation_mse_spread = (
        validation_mse_max - validation_mse_min
        if validation_mse_min is not None and validation_mse_max is not None
        else None
    )
    validation_nmse_spread = (
        validation_nmse_max - validation_nmse_min
        if validation_nmse_min is not None and validation_nmse_max is not None
        else None
    )
    return {
        "best_score": _mean(best_score(gnn) for gnn in gnns),
        "validation_readout_mse": _mean(validation_mses),
        "validation_readout_mse_values": validation_mses,
        "validation_readout_mse_min": validation_mse_min,
        "validation_readout_mse_max": validation_mse_max,
        "validation_readout_mse_spread": validation_mse_spread,
        "validation_readout_mse_stddev": _population_stddev(validation_mses),
        "validation_readout_nmse": _mean(validation_nmses),
        "validation_readout_nmse_values": validation_nmses,
        "validation_readout_nmse_min": validation_nmse_min,
        "validation_readout_nmse_max": validation_nmse_max,
        "validation_readout_nmse_spread": validation_nmse_spread,
        "validation_readout_nmse_stddev": _population_stddev(validation_nmses),
        "above_max_delta": _mean(band_max_delta(gnn, "above") for gnn in gnns),
        "here_max_delta": _mean(band_max_delta(gnn, "here") for gnn in gnns),
        "beneath_max_delta": _mean(band_max_delta(gnn, "beneath") for gnn in gnns),
        "step_ms_last": _mean(metric(trainer, "step_time_ms", "last") for trainer in trainers),
        "cpu_debt_ops": _mean(cpu_debt_ops(trainer) for trainer in trainers),
    }


def validation_stability_score(metrics: dict[str, Any]) -> float | None:
    mean = _number(metrics.get("validation_readout_mse"))
    stddev = _number(metrics.get("validation_readout_mse_stddev"))
    if mean is None:
        return None
    return mean + (stddev or 0.0)


def validation_stability_status(metrics: dict[str, Any], *, runs: int) -> str:
    if runs < 2:
        return "single_seed_probe"
    mean = _number(metrics.get("validation_readout_mse"))
    spread = _number(metrics.get("validation_readout_mse_spread"))
    if mean is None or spread is None:
        return "unknown"
    stable_tolerance = max(abs(mean) * 0.05, 1e-6)
    watch_tolerance = max(abs(mean) * 0.20, 1e-5)
    if spread <= stable_tolerance:
        return "multi_seed_stable"
    if spread <= watch_tolerance:
        return "watch_spread"
    return "volatile"


def top_validation_candidate_records(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    candidates = sorted(
        groups.items(),
        key=lambda item: (
            _sort_number(group_metrics(item[1])["validation_readout_mse"]),
            _sort_number(group_metrics(item[1])["cpu_debt_ops"]),
            _sort_number(group_metrics(item[1])["step_ms_last"]),
        ),
    )
    records = []
    for rank, (group, summaries) in enumerate(candidates[:limit], start=1):
        metrics = group_metrics(summaries)
        stability_score = validation_stability_score(metrics)
        records.append(
            {
                "rank": rank,
                **group_record(group),
                "runs": len(summaries),
                "seeds": seeds_for(summaries),
                "avg_validation_readout_mse": metrics["validation_readout_mse"],
                "validation_readout_mse_values": metrics["validation_readout_mse_values"],
                "validation_readout_mse_min": metrics["validation_readout_mse_min"],
                "validation_readout_mse_max": metrics["validation_readout_mse_max"],
                "validation_readout_mse_spread": metrics["validation_readout_mse_spread"],
                "validation_readout_mse_stddev": metrics["validation_readout_mse_stddev"],
                "avg_validation_readout_nmse": metrics["validation_readout_nmse"],
                "validation_readout_nmse_values": metrics["validation_readout_nmse_values"],
                "validation_readout_nmse_min": metrics["validation_readout_nmse_min"],
                "validation_readout_nmse_max": metrics["validation_readout_nmse_max"],
                "validation_readout_nmse_spread": metrics["validation_readout_nmse_spread"],
                "validation_readout_nmse_stddev": metrics["validation_readout_nmse_stddev"],
                "validation_stability_score": stability_score,
                "validation_stability_status": validation_stability_status(
                    metrics,
                    runs=len(summaries),
                ),
                "avg_best_score": metrics["best_score"],
                "avg_above_max_delta": metrics["above_max_delta"],
                "avg_here_max_delta": metrics["here_max_delta"],
                "avg_beneath_max_delta": metrics["beneath_max_delta"],
                "avg_step_ms_last": metrics["step_ms_last"],
                "avg_cpu_debt_ops": metrics["cpu_debt_ops"],
            }
        )
    return records


def stable_validation_candidate_records(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    validation_candidates = top_validation_candidate_records(groups, limit=len(groups))
    stable_candidates = sorted(
        validation_candidates,
        key=lambda record: (
            _sort_number(record.get("validation_stability_score")),
            _sort_number(record.get("validation_readout_mse_stddev")),
            _sort_number(record.get("avg_validation_readout_mse")),
            _sort_number(record.get("avg_cpu_debt_ops")),
            _sort_number(record.get("avg_step_ms_last")),
        ),
    )
    records = []
    for rank, candidate in enumerate(stable_candidates[:limit], start=1):
        records.append(
            {
                **candidate,
                "stability_rank": rank,
                "validation_rank": candidate.get("rank"),
            }
        )
    return records


def schedule_text(candidate: dict[str, Any]) -> str:
    return ",".join(
        [
            f"lr={fmt(candidate.get('lr'), 6)}",
            f"top={candidate.get('top_k', '-')}",
            f"mid={candidate.get('mid_k', '-')}",
            f"bottom={candidate.get('bottom_k', '-')}",
            f"tol={fmt(candidate.get('here_tolerance'), 6)}",
        ]
    )


def schedule_match_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    values = []
    for field in GROUP_FIELDS:
        value = candidate.get(field)
        if field in ("lr", "here_tolerance"):
            number = _number(value)
            values.append(round(number, 12) if number is not None else value)
        else:
            values.append(value)
    return tuple(values)


def source_replay_candidate(
    source_candidate: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    source_key = schedule_match_key(source_candidate)
    for candidate in candidates:
        if schedule_match_key(candidate) == source_key:
            return candidate
    return None


def follow_up_verdict(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta < 0.0:
        return "improved"
    if delta > 0.0:
        return "regressed"
    return "matched"


def follow_up_result_record(
    source_candidate: dict[str, Any] | None,
    top_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if source_candidate is None or not top_candidates:
        return None
    best = top_candidates[0]
    replay = source_replay_candidate(source_candidate, top_candidates)
    validation_delta = _delta_number(
        best.get("avg_validation_readout_mse"),
        source_candidate.get("avg_validation_readout_mse"),
    )
    replay_delta = _delta_number(
        replay.get("avg_validation_readout_mse") if replay is not None else None,
        source_candidate.get("avg_validation_readout_mse"),
    )
    best_vs_replay_delta = _delta_number(
        best.get("avg_validation_readout_mse"),
        replay.get("avg_validation_readout_mse") if replay is not None else None,
    )
    validation_nmse_delta = _delta_number(
        best.get("avg_validation_readout_nmse"),
        source_candidate.get("avg_validation_readout_nmse"),
    )
    replay_nmse_delta = _delta_number(
        replay.get("avg_validation_readout_nmse") if replay is not None else None,
        source_candidate.get("avg_validation_readout_nmse"),
    )
    best_vs_replay_nmse_delta = _delta_number(
        best.get("avg_validation_readout_nmse"),
        replay.get("avg_validation_readout_nmse") if replay is not None else None,
    )
    return {
        "schema": "st.gnn.band_trace_follow_up_result.v1",
        "source_rank": source_candidate.get("rank"),
        "source_replay_rank": replay.get("rank") if replay is not None else None,
        "best_rank": best.get("rank"),
        "verdict": follow_up_verdict(validation_delta),
        "source_schedule": schedule_text(source_candidate),
        "source_replay_schedule": schedule_text(replay) if replay is not None else None,
        "best_schedule": schedule_text(best),
        "source_avg_validation_readout_mse": source_candidate.get("avg_validation_readout_mse"),
        "source_replay_avg_validation_readout_mse": (
            replay.get("avg_validation_readout_mse") if replay is not None else None
        ),
        "best_avg_validation_readout_mse": best.get("avg_validation_readout_mse"),
        "source_avg_validation_readout_nmse": source_candidate.get(
            "avg_validation_readout_nmse"
        ),
        "source_replay_avg_validation_readout_nmse": (
            replay.get("avg_validation_readout_nmse") if replay is not None else None
        ),
        "best_avg_validation_readout_nmse": best.get("avg_validation_readout_nmse"),
        "validation_mse_delta": validation_delta,
        "source_replay_validation_mse_delta": replay_delta,
        "best_vs_source_replay_validation_mse_delta": best_vs_replay_delta,
        "validation_nmse_delta": validation_nmse_delta,
        "source_replay_validation_nmse_delta": replay_nmse_delta,
        "best_vs_source_replay_validation_nmse_delta": best_vs_replay_nmse_delta,
        "source_validation_stability_status": source_candidate.get(
            "validation_stability_status"
        ),
        "source_replay_validation_stability_status": (
            replay.get("validation_stability_status") if replay is not None else None
        ),
        "best_validation_stability_status": best.get("validation_stability_status"),
        "source_validation_stability_score": source_candidate.get(
            "validation_stability_score"
        ),
        "source_replay_validation_stability_score": (
            replay.get("validation_stability_score") if replay is not None else None
        ),
        "best_validation_stability_score": best.get("validation_stability_score"),
        "source_validation_mse_stddev": source_candidate.get(
            "validation_readout_mse_stddev"
        ),
        "source_replay_validation_mse_stddev": (
            replay.get("validation_readout_mse_stddev") if replay is not None else None
        ),
        "best_validation_mse_stddev": best.get("validation_readout_mse_stddev"),
        "source_validation_mse_spread": source_candidate.get(
            "validation_readout_mse_spread"
        ),
        "source_replay_validation_mse_spread": (
            replay.get("validation_readout_mse_spread") if replay is not None else None
        ),
        "best_validation_mse_spread": best.get("validation_readout_mse_spread"),
        "source_validation_nmse_stddev": source_candidate.get(
            "validation_readout_nmse_stddev"
        ),
        "source_replay_validation_nmse_stddev": (
            replay.get("validation_readout_nmse_stddev") if replay is not None else None
        ),
        "best_validation_nmse_stddev": best.get("validation_readout_nmse_stddev"),
        "source_validation_nmse_spread": source_candidate.get(
            "validation_readout_nmse_spread"
        ),
        "source_replay_validation_nmse_spread": (
            replay.get("validation_readout_nmse_spread") if replay is not None else None
        ),
        "best_validation_nmse_spread": best.get("validation_readout_nmse_spread"),
        "source_avg_cpu_debt_ops": source_candidate.get("avg_cpu_debt_ops"),
        "source_replay_avg_cpu_debt_ops": (
            replay.get("avg_cpu_debt_ops") if replay is not None else None
        ),
        "best_avg_cpu_debt_ops": best.get("avg_cpu_debt_ops"),
        "cpu_debt_delta": _delta_number(
            best.get("avg_cpu_debt_ops"),
            source_candidate.get("avg_cpu_debt_ops"),
        ),
        "best_vs_source_replay_cpu_debt_delta": _delta_number(
            best.get("avg_cpu_debt_ops"),
            replay.get("avg_cpu_debt_ops") if replay is not None else None,
        ),
        "source_avg_step_ms_last": source_candidate.get("avg_step_ms_last"),
        "source_replay_avg_step_ms_last": (
            replay.get("avg_step_ms_last") if replay is not None else None
        ),
        "best_avg_step_ms_last": best.get("avg_step_ms_last"),
        "step_ms_delta": _delta_number(
            best.get("avg_step_ms_last"),
            source_candidate.get("avg_step_ms_last"),
        ),
        "best_vs_source_replay_step_ms_delta": _delta_number(
            best.get("avg_step_ms_last"),
            replay.get("avg_step_ms_last") if replay is not None else None,
        ),
        "source_candidate": source_candidate,
        "source_replay_candidate": replay,
        "best_candidate": best,
    }


def follow_up_result_row(result: dict[str, Any]) -> list[str]:
    return [
        str(result.get("source_rank") or "-"),
        str(result.get("source_replay_rank") or "-"),
        str(result.get("best_rank") or "-"),
        str(result.get("verdict") or "-"),
        str(result.get("source_schedule") or "-"),
        str(result.get("source_replay_schedule") or "-"),
        str(result.get("best_schedule") or "-"),
        fmt(result.get("source_avg_validation_readout_mse"), 6),
        fmt(result.get("source_replay_avg_validation_readout_mse"), 6),
        fmt(result.get("best_avg_validation_readout_mse"), 6),
        fmt(result.get("source_avg_validation_readout_nmse"), 6),
        fmt(result.get("source_replay_avg_validation_readout_nmse"), 6),
        fmt(result.get("best_avg_validation_readout_nmse"), 6),
        fmt(result.get("validation_mse_delta"), 6),
        fmt(result.get("source_replay_validation_mse_delta"), 6),
        fmt(result.get("best_vs_source_replay_validation_mse_delta"), 6),
        fmt(result.get("validation_nmse_delta"), 6),
        fmt(result.get("source_replay_validation_nmse_delta"), 6),
        fmt(result.get("best_vs_source_replay_validation_nmse_delta"), 6),
        str(result.get("source_validation_stability_status") or "-"),
        str(result.get("source_replay_validation_stability_status") or "-"),
        str(result.get("best_validation_stability_status") or "-"),
        fmt(result.get("source_validation_stability_score"), 6),
        fmt(result.get("source_replay_validation_stability_score"), 6),
        fmt(result.get("best_validation_stability_score"), 6),
        fmt(result.get("source_validation_mse_stddev"), 6),
        fmt(result.get("source_replay_validation_mse_stddev"), 6),
        fmt(result.get("best_validation_mse_stddev"), 6),
        fmt(result.get("source_validation_mse_spread"), 6),
        fmt(result.get("source_replay_validation_mse_spread"), 6),
        fmt(result.get("best_validation_mse_spread"), 6),
        fmt(result.get("source_validation_nmse_stddev"), 6),
        fmt(result.get("source_replay_validation_nmse_stddev"), 6),
        fmt(result.get("best_validation_nmse_stddev"), 6),
        fmt(result.get("source_validation_nmse_spread"), 6),
        fmt(result.get("source_replay_validation_nmse_spread"), 6),
        fmt(result.get("best_validation_nmse_spread"), 6),
        fmt(result.get("source_avg_cpu_debt_ops"), 0),
        fmt(result.get("source_replay_avg_cpu_debt_ops"), 0),
        fmt(result.get("best_avg_cpu_debt_ops"), 0),
        fmt(result.get("cpu_debt_delta"), 0),
        fmt(result.get("best_vs_source_replay_cpu_debt_delta"), 0),
        fmt(result.get("source_avg_step_ms_last"), 3),
        fmt(result.get("source_replay_avg_step_ms_last"), 3),
        fmt(result.get("best_avg_step_ms_last"), 3),
        fmt(result.get("step_ms_delta"), 3),
        fmt(result.get("best_vs_source_replay_step_ms_delta"), 3),
    ]


def follow_up_gate_record(
    fail_on_verdicts: Iterable[str],
    follow_up_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    fail_on = list(fail_on_verdicts)
    if not fail_on:
        return None
    verdict = "unknown"
    if isinstance(follow_up_result, dict):
        raw_verdict = follow_up_result.get("verdict")
        if isinstance(raw_verdict, str) and raw_verdict in FOLLOW_UP_VERDICTS:
            verdict = raw_verdict
    failed = verdict in set(fail_on)
    return {
        "schema": "st.gnn.band_trace_follow_up_gate.v1",
        "fail_on_verdicts": fail_on,
        "verdict": verdict,
        "failed": failed,
    }


def follow_up_gate_failed(comparison: dict[str, Any] | None) -> bool:
    if not isinstance(comparison, dict):
        return False
    gate = comparison.get("follow_up_gate")
    return isinstance(gate, dict) and bool(gate.get("failed"))


def follow_up_gate_row(gate: dict[str, Any]) -> list[str]:
    fail_on = gate.get("fail_on_verdicts")
    return [
        str(gate.get("verdict") or "-"),
        "yes" if gate.get("failed") else "no",
        ",".join(str(verdict) for verdict in fail_on) if isinstance(fail_on, list) else "-",
    ]


def follow_up_stability_guarded_source(
    follow_up_result: dict[str, Any] | None,
) -> bool:
    if not isinstance(follow_up_result, dict):
        return False
    if follow_up_result.get("verdict") != "improved":
        return False
    source_status = follow_up_result.get("source_validation_stability_status")
    best_status = follow_up_result.get("best_validation_stability_status")
    if (
        source_status not in FOLLOW_UP_STABILITY_ACCEPTABLE
        or best_status not in FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS
    ):
        return False
    source_score = _number(follow_up_result.get("source_validation_stability_score"))
    best_score = _number(follow_up_result.get("best_validation_stability_score"))
    if source_score is None or best_score is None:
        return True
    return source_score <= best_score


def follow_up_promotion_record(
    follow_up_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(follow_up_result, dict):
        return None
    verdict = follow_up_result.get("verdict")
    stability_guard = follow_up_stability_guarded_source(follow_up_result)
    if verdict == "improved" and not stability_guard:
        action = "promote_best"
        selected_origin = "best"
        selected_candidate = follow_up_result.get("best_candidate")
    elif stability_guard:
        action = "keep_source_stability_guard"
        selected_origin = "source"
        selected_candidate = follow_up_result.get("source_candidate")
    else:
        action = "keep_source"
        selected_origin = "source"
        selected_candidate = follow_up_result.get("source_candidate")
    if not isinstance(selected_candidate, dict):
        return None
    return {
        "schema": "st.gnn.band_trace_follow_up_promotion.v1",
        "action": action,
        "verdict": verdict or "unknown",
        "selected_origin": selected_origin,
        "selected_schedule": schedule_text(selected_candidate),
        "selected_avg_validation_readout_mse": selected_candidate.get(
            "avg_validation_readout_mse"
        ),
        "selected_avg_validation_readout_nmse": selected_candidate.get(
            "avg_validation_readout_nmse"
        ),
        "validation_mse_delta": follow_up_result.get("validation_mse_delta"),
        "validation_nmse_delta": follow_up_result.get("validation_nmse_delta"),
        "stability_guard": stability_guard,
        "source_validation_stability_status": follow_up_result.get(
            "source_validation_stability_status"
        ),
        "best_validation_stability_status": follow_up_result.get(
            "best_validation_stability_status"
        ),
        "source_validation_stability_score": follow_up_result.get(
            "source_validation_stability_score"
        ),
        "best_validation_stability_score": follow_up_result.get(
            "best_validation_stability_score"
        ),
        "selected_candidate": selected_candidate,
    }


def follow_up_promotion_row(promotion: dict[str, Any]) -> list[str]:
    return [
        str(promotion.get("action") or "-"),
        str(promotion.get("verdict") or "-"),
        str(promotion.get("selected_origin") or "-"),
        str(promotion.get("selected_schedule") or "-"),
        fmt(promotion.get("selected_avg_validation_readout_mse"), 6),
        fmt(promotion.get("selected_avg_validation_readout_nmse"), 6),
        fmt(promotion.get("validation_mse_delta"), 6),
        fmt(promotion.get("validation_nmse_delta"), 6),
        "yes" if promotion.get("stability_guard") else "no",
        str(promotion.get("source_validation_stability_status") or "-"),
        str(promotion.get("best_validation_stability_status") or "-"),
        fmt(promotion.get("source_validation_stability_score"), 6),
        fmt(promotion.get("best_validation_stability_score"), 6),
    ]


def follow_up_next_command_record(
    run_root: Path,
    promotion: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(promotion, dict):
        return None
    command = [
        "python3",
        "-S",
        "-s",
        "tools/run_gnn_band_trace_sweep.py",
        "--run-root",
        "NEXT_RUN_ROOT",
        "--follow-up-from",
        str(run_root),
        "--follow-up-source",
        "auto",
    ]
    return {
        "schema": "st.gnn.band_trace_follow_up_next_command.v1",
        "source_run_root": str(run_root),
        "uses_promotion": True,
        "promotion_action": promotion.get("action"),
        "selected_origin": promotion.get("selected_origin"),
        "command": command,
        "shell": " ".join(command),
    }


def follow_up_next_command_row(record: dict[str, Any]) -> list[str]:
    return [
        "yes" if record.get("uses_promotion") else "no",
        str(record.get("promotion_action") or "-"),
        str(record.get("selected_origin") or "-"),
        str(record.get("source_run_root") or "-"),
        str(record.get("shell") or "-"),
    ]


def _optional_record_value(record: dict[str, Any] | None, key: str) -> Any:
    return record.get(key) if isinstance(record, dict) else None


def follow_up_chain_ancestor_record(sweep_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    follow_up = config.get("follow_up") if isinstance(config, dict) else None
    lineage = follow_up_lineage_from_payload(payload)
    comparison = payload.get("comparison")
    result = comparison.get("follow_up_result") if isinstance(comparison, dict) else None
    promotion = comparison.get("follow_up_promotion") if isinstance(comparison, dict) else None
    return {
        "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
        "sweep_path": str(sweep_path),
        "run_root": follow_up_parent_run_root(payload, sweep_path),
        "generation": _optional_record_value(lineage, "generation"),
        "parent_generation": _optional_record_value(lineage, "parent_generation"),
        "source_mode": (
            _optional_record_value(follow_up, "source_mode")
            or _optional_record_value(lineage, "source_mode")
        ),
        "candidate_source": (
            _optional_record_value(follow_up, "candidate_source")
            or _optional_record_value(lineage, "candidate_source")
        ),
        "verdict": _optional_record_value(result, "verdict"),
        "promotion_action": _optional_record_value(promotion, "action"),
        "selected_origin": _optional_record_value(promotion, "selected_origin"),
        "selected_schedule": _optional_record_value(promotion, "selected_schedule"),
        "selected_avg_validation_readout_mse": _optional_record_value(
            promotion,
            "selected_avg_validation_readout_mse",
        ),
        "selected_avg_validation_readout_nmse": _optional_record_value(
            promotion,
            "selected_avg_validation_readout_nmse",
        ),
    }


def follow_up_chain_missing_ancestor_record(sweep_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
        "sweep_path": str(sweep_path),
        "missing": True,
        "error": str(error),
    }


def follow_up_chain_ancestor_records(
    lineage: dict[str, Any],
    *,
    max_ancestors: int = FOLLOW_UP_CHAIN_MAX_ANCESTORS,
) -> list[dict[str, Any]]:
    ancestors = []
    next_path = lineage.get("parent_sweep_path")
    seen: set[str] = set()
    for _ in range(max_ancestors):
        if not next_path:
            break
        sweep_path = Path(str(next_path))
        sweep_key = str(sweep_path)
        if sweep_key in seen:
            ancestors.append(
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "sweep_path": sweep_key,
                    "cycle_detected": True,
                }
            )
            break
        seen.add(sweep_key)
        try:
            payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ancestors.append(follow_up_chain_missing_ancestor_record(sweep_path, exc))
            break
        ancestors.append(follow_up_chain_ancestor_record(sweep_path, payload))
        parent_lineage = follow_up_lineage_from_payload(payload)
        next_path = parent_lineage.get("parent_sweep_path") if parent_lineage else None
    return ancestors


def follow_up_chain_record(
    run_root: Path,
    follow_up: dict[str, Any] | None,
    follow_up_result: dict[str, Any] | None,
    follow_up_promotion: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(follow_up, dict):
        return None
    lineage = follow_up.get("lineage")
    if not isinstance(lineage, dict):
        return None
    ancestors = follow_up_chain_ancestor_records(lineage)
    neighborhood = follow_up.get("neighborhood")
    neighborhood_enabled = (
        bool(neighborhood.get("enabled")) if isinstance(neighborhood, dict) else False
    )
    return {
        "schema": "st.gnn.band_trace_follow_up_chain.v1",
        "run_root": str(run_root),
        "generation": lineage.get("generation"),
        "parent_generation": lineage.get("parent_generation"),
        "parent_sweep_path": lineage.get("parent_sweep_path"),
        "parent_run_root": lineage.get("parent_run_root"),
        "source_mode": follow_up.get("source_mode") or lineage.get("source_mode"),
        "candidate_source": follow_up.get("candidate_source")
        or lineage.get("candidate_source"),
        "verdict": _optional_record_value(follow_up_result, "verdict"),
        "promotion_action": _optional_record_value(follow_up_promotion, "action"),
        "selected_origin": _optional_record_value(follow_up_promotion, "selected_origin"),
        "neighborhood_enabled": neighborhood_enabled,
        "chain_depth": len(ancestors) + 1,
        "ancestor_count": len(ancestors),
        "ancestors": ancestors,
    }


def follow_up_chain_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get("generation") or "-"),
        str(record.get("chain_depth") or "-"),
        str(record.get("source_mode") or "-"),
        str(record.get("candidate_source") or "-"),
        str(record.get("parent_generation") or "-"),
        str(record.get("parent_run_root") or "-"),
        str(record.get("parent_sweep_path") or "-"),
        str(record.get("verdict") or "-"),
        str(record.get("promotion_action") or "-"),
        str(record.get("selected_origin") or "-"),
        "yes" if record.get("neighborhood_enabled") else "no",
    ]


def follow_up_chain_ancestor_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get("generation") or "-"),
        str(record.get("run_root") or "-"),
        str(record.get("sweep_path") or "-"),
        str(record.get("source_mode") or "-"),
        str(record.get("candidate_source") or "-"),
        str(record.get("verdict") or "-"),
        str(record.get("promotion_action") or "-"),
        str(record.get("selected_origin") or "-"),
        str(record.get("selected_schedule") or "-"),
        fmt(record.get("selected_avg_validation_readout_mse"), 6),
        fmt(record.get("selected_avg_validation_readout_nmse"), 6),
        "yes" if record.get("missing") else "no",
        "yes" if record.get("cycle_detected") else "no",
        str(record.get("error") or "-"),
    ]


def _leading_count(values: Iterable[str], predicate: Any) -> int:
    count = 0
    for value in values:
        if not predicate(value):
            break
        count += 1
    return count


def follow_up_chain_verdicts(
    chain: dict[str, Any],
    follow_up_result: dict[str, Any] | None,
) -> list[str]:
    verdicts = []
    current = _optional_record_value(follow_up_result, "verdict")
    if isinstance(current, str):
        verdicts.append(current)
    ancestors = chain.get("ancestors")
    if isinstance(ancestors, list):
        for ancestor in ancestors:
            if isinstance(ancestor, dict):
                verdict = ancestor.get("verdict")
                if isinstance(verdict, str):
                    verdicts.append(verdict)
    return verdicts


def follow_up_sample_budget_flags(candidate: dict[str, Any] | None) -> str:
    flags = ["--follow-up-source auto", "--seeds NEW_SEEDS"]
    if not isinstance(candidate, dict):
        return " ".join(flags)

    for field, flag in (
        ("epochs", "--epoch-values"),
        ("train_graphs", "--train-graph-values"),
        ("validation_graphs", "--validation-graph-values"),
    ):
        values = _center_and_wider_int_values(candidate.get(field))
        if values:
            flags.extend([flag, csv_number_values(values)])
    return " ".join(flags)


def follow_up_validation_budget_flags(candidate: dict[str, Any] | None) -> str:
    flags = ["--follow-up-source auto", "--seeds NEW_SEEDS"]
    if not isinstance(candidate, dict):
        return " ".join(flags)
    values = _center_and_wider_int_values(candidate.get("validation_graphs"))
    if values:
        flags.extend(["--validation-graph-values", csv_number_values(values)])
    return " ".join(flags)


def follow_up_has_stability_tradeoff(follow_up_result: dict[str, Any] | None) -> bool:
    if not isinstance(follow_up_result, dict):
        return False
    if follow_up_result.get("verdict") != "regressed":
        return False
    source_status = follow_up_result.get("source_validation_stability_status")
    best_status = follow_up_result.get("best_validation_stability_status")
    return (
        source_status in FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS
        and best_status in FOLLOW_UP_STABILITY_ACCEPTABLE
    )


def follow_up_has_source_replay_seed_shift(follow_up_result: dict[str, Any] | None) -> bool:
    if not isinstance(follow_up_result, dict):
        return False
    if follow_up_result.get("verdict") != "regressed":
        return False
    replay_delta = _number(follow_up_result.get("source_replay_validation_mse_delta"))
    best_vs_replay_delta = _number(
        follow_up_result.get("best_vs_source_replay_validation_mse_delta")
    )
    if replay_delta is None or best_vs_replay_delta is None:
        return False
    return replay_delta > 0.0 and best_vs_replay_delta <= 0.0


def follow_up_seed_shift_needs_validation_budget(
    follow_up_result: dict[str, Any] | None,
) -> bool:
    if not follow_up_has_source_replay_seed_shift(follow_up_result):
        return False
    if not isinstance(follow_up_result, dict):
        return False
    return (
        follow_up_result.get("source_replay_validation_stability_status")
        in FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS
        or follow_up_result.get("best_validation_stability_status")
        in FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS
    )


def follow_up_has_target_scale_seed_shift(follow_up_result: dict[str, Any] | None) -> bool:
    if not follow_up_has_source_replay_seed_shift(follow_up_result):
        return False
    if not isinstance(follow_up_result, dict):
        return False
    replay_nmse_delta = _number(
        follow_up_result.get("source_replay_validation_nmse_delta")
    )
    best_vs_replay_nmse_delta = _number(
        follow_up_result.get("best_vs_source_replay_validation_nmse_delta")
    )
    if replay_nmse_delta is None or best_vs_replay_nmse_delta is None:
        return False
    return replay_nmse_delta <= 0.0 and best_vs_replay_nmse_delta <= 0.0


def follow_up_has_normalized_regression(follow_up_result: dict[str, Any] | None) -> bool:
    if not isinstance(follow_up_result, dict):
        return False
    if follow_up_result.get("verdict") != "improved":
        return False
    nmse_delta = _number(follow_up_result.get("validation_nmse_delta"))
    return nmse_delta is not None and nmse_delta > 0.0


def follow_up_chain_guidance_action(
    current_verdict: str,
    *,
    improved_streak: int,
    regressed_streak: int,
    non_improving_streak: int,
    promotion_action: str | None = None,
    candidate_stability_status: str | None = None,
    neighborhood_enabled: bool = False,
    sample_budget_flags: str | None = None,
    validation_budget_flags: str | None = None,
    stability_tradeoff: bool = False,
    source_replay_seed_shift: bool = False,
    seed_shift_needs_validation_budget: bool = False,
    target_scale_seed_shift: bool = False,
    normalized_regression: bool = False,
) -> tuple[str, str, str]:
    if current_verdict == "improved" and normalized_regression:
        return (
            "review_normalized_tradeoff",
            "Raw validation MSE improved, but target-normalized NMSE worsened; rerun the promoted schedule with fresh seeds and a wider validation graph budget before broadening.",
            validation_budget_flags or "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if current_verdict == "improved" and promotion_action == "keep_source_stability_guard":
        return (
            "keep_stable_source",
            "The stability guard kept the seed-stable source over a volatile average-only gain; rerun the guarded source with fresh seeds before accepting the volatile candidate.",
            "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "improved"
        and candidate_stability_status in FOLLOW_UP_STABILITY_ACCEPTABLE
        and neighborhood_enabled
    ):
        return (
            "confirm_stable_promotion",
            "A local neighborhood search found a seed-stable improvement; confirm the promoted schedule with fresh seeds before broadening again.",
            "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "improved"
        and candidate_stability_status in FOLLOW_UP_STABILITY_ACCEPTABLE
        and improved_streak >= 2
        and not neighborhood_enabled
    ):
        return (
            "explore_stable_neighborhood",
            "Repeated improvements are now seed-stable; use the promoted candidate as a fresh-seed anchor for a local schedule neighborhood search.",
            "--follow-up-source auto --follow-up-neighborhood --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "improved"
        and candidate_stability_status == "volatile"
        and improved_streak >= 2
        and neighborhood_enabled
    ):
        return (
            "increase_sample_budget",
            "Repeated improvements remain seed-volatile after a local schedule search; increase train/validation sample budget before another promotion.",
            sample_budget_flags or "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "improved"
        and candidate_stability_status == "volatile"
        and improved_streak >= 2
    ):
        return (
            "widen_stability_search",
            "Repeated improvements remain seed-volatile; search the local schedule neighborhood with fresh seeds.",
            "--follow-up-source auto --follow-up-neighborhood --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "improved"
        and candidate_stability_status in FOLLOW_UP_STABILITY_NEEDS_FRESH_SEEDS
    ):
        return (
            "repeat_with_fresh_seeds",
            "The improved candidate is not seed-stable yet; validate it with fresh seeds before continuing promotion.",
            "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if current_verdict == "improved":
        return (
            "continue_promotion",
            "Use the promoted candidate as the next source.",
            "--follow-up-source auto",
        )
    if current_verdict == "matched":
        return (
            "repeat_with_fresh_seeds",
            "Keep the source candidate and validate with fresh seeds.",
            "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if current_verdict == "regressed" and stability_tradeoff:
        return (
            "review_stability_tradeoff",
            "The new best regressed on average but is more seed-stable than the source; rerun the top candidate with fresh seeds before discarding it.",
            "--follow-up-source top-candidate --seeds NEW_SEEDS",
        )
    if current_verdict == "regressed" and target_scale_seed_shift:
        return (
            "review_target_scale_shift",
            "Raw validation MSE regressed, but target-normalized NMSE did not; rerun the source with fresh seeds and a wider validation graph budget before treating this as a schedule loss.",
            validation_budget_flags or "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "regressed"
        and seed_shift_needs_validation_budget
    ):
        return (
            "increase_seed_shift_validation_budget",
            "The source replay shifted upward and the replay/best evidence is volatile; increase the validation graph budget with fresh seeds before making another promotion decision.",
            validation_budget_flags or "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "regressed"
        and source_replay_seed_shift
        and regressed_streak >= 3
    ):
        return (
            "audit_seed_sensitivity",
            "Seed-shift regressions have persisted across multiple follow-ups; pause schedule widening and audit the source anchor with a broader fresh-seed list and validation budget.",
            validation_budget_flags or "--follow-up-source auto --seeds NEW_SEEDS",
        )
    if (
        current_verdict == "regressed"
        and source_replay_seed_shift
        and regressed_streak >= 2
    ):
        return (
            "widen_seed_shift_neighborhood",
            "Repeated regressions still show source-replay seed shift; widen or shift the local neighborhood with fresh seeds rather than replaying an old seed surface.",
            "--follow-up-source auto --follow-up-neighborhood --seeds NEW_SEEDS",
        )
    if current_verdict == "regressed" and neighborhood_enabled and source_replay_seed_shift:
        return (
            "review_seed_shift_neighborhood",
            "The neighborhood regressed against the previous source, but the source replay also worsened on the fresh seeds while the best neighborhood candidate matched or beat that replay; rerun the source-anchored neighborhood with fresh seeds before locking the source.",
            "--follow-up-source auto --follow-up-neighborhood --seeds NEW_SEEDS",
        )
    if current_verdict == "regressed" and regressed_streak >= 2:
        return (
            "widen_neighborhood",
            "Repeated regressions suggest widening or shifting the local schedule search.",
            "--follow-up-source auto --follow-up-neighborhood",
        )
    if current_verdict == "regressed":
        return (
            "keep_source",
            "Do not promote the regressed run; replay the source/promotion candidate next.",
            "--follow-up-source auto",
        )
    if non_improving_streak >= 2:
        return (
            "inspect_trace",
            "Repeated non-improving verdicts need trace inspection before another narrow replay.",
            "--follow-up-source auto --follow-up-neighborhood",
        )
    return (
        "inspect_trace",
        "The follow-up verdict is unknown; inspect logs or rerun before promotion.",
        "--follow-up-source auto",
    )


def follow_up_guidance_candidate(follow_up_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(follow_up_result, dict):
        return None
    verdict = follow_up_result.get("verdict")
    if verdict == "improved":
        candidate = follow_up_result.get("best_candidate")
    else:
        candidate = follow_up_result.get("source_candidate")
    return candidate if isinstance(candidate, dict) else None


def follow_up_chain_guidance_record(
    chain: dict[str, Any] | None,
    follow_up_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(chain, dict):
        return None
    verdicts = follow_up_chain_verdicts(chain, follow_up_result)
    current_verdict = verdicts[0] if verdicts else "unknown"
    regressed_streak = _leading_count(verdicts, lambda verdict: verdict == "regressed")
    improved_streak = _leading_count(verdicts, lambda verdict: verdict == "improved")
    non_improving_streak = _leading_count(
        verdicts,
        lambda verdict: verdict in ("matched", "regressed", "unknown"),
    )
    stability_tradeoff = follow_up_has_stability_tradeoff(follow_up_result)
    source_replay_seed_shift = follow_up_has_source_replay_seed_shift(follow_up_result)
    seed_shift_needs_validation_budget = follow_up_seed_shift_needs_validation_budget(
        follow_up_result
    )
    target_scale_seed_shift = follow_up_has_target_scale_seed_shift(follow_up_result)
    normalized_regression = follow_up_has_normalized_regression(follow_up_result)
    guidance_candidate = follow_up_guidance_candidate(follow_up_result)
    candidate_stability_status = (
        guidance_candidate.get("validation_stability_status")
        if isinstance(guidance_candidate, dict)
        else None
    )
    neighborhood_enabled = bool(chain.get("neighborhood_enabled"))
    action, rationale, suggested_flags = follow_up_chain_guidance_action(
        current_verdict,
        improved_streak=improved_streak,
        regressed_streak=regressed_streak,
        non_improving_streak=non_improving_streak,
        promotion_action=_optional_record_value(chain, "promotion_action"),
        candidate_stability_status=(
            str(candidate_stability_status) if candidate_stability_status is not None else None
        ),
        neighborhood_enabled=neighborhood_enabled,
        sample_budget_flags=follow_up_sample_budget_flags(guidance_candidate),
        validation_budget_flags=follow_up_validation_budget_flags(guidance_candidate),
        stability_tradeoff=stability_tradeoff,
        source_replay_seed_shift=source_replay_seed_shift,
        seed_shift_needs_validation_budget=seed_shift_needs_validation_budget,
        target_scale_seed_shift=target_scale_seed_shift,
        normalized_regression=normalized_regression,
    )
    ancestors = chain.get("ancestors")
    missing_ancestors = (
        sum(1 for ancestor in ancestors if isinstance(ancestor, dict) and ancestor.get("missing"))
        if isinstance(ancestors, list)
        else 0
    )
    return {
        "schema": "st.gnn.band_trace_follow_up_chain_guidance.v1",
        "action": action,
        "current_verdict": current_verdict,
        "recent_verdicts": verdicts[:6],
        "improved_streak": improved_streak,
        "regressed_streak": regressed_streak,
        "non_improving_streak": non_improving_streak,
        "missing_ancestors": missing_ancestors,
        "neighborhood_enabled": neighborhood_enabled,
        "stability_tradeoff": stability_tradeoff,
        "source_replay_seed_shift": source_replay_seed_shift,
        "seed_shift_needs_validation_budget": seed_shift_needs_validation_budget,
        "target_scale_seed_shift": target_scale_seed_shift,
        "normalized_regression": normalized_regression,
        "promotion_action": _optional_record_value(chain, "promotion_action"),
        "source_replay_validation_stability_status": _optional_record_value(
            follow_up_result,
            "source_replay_validation_stability_status",
        ),
        "best_validation_stability_status": _optional_record_value(
            follow_up_result,
            "best_validation_stability_status",
        ),
        "source_replay_validation_mse_delta": _optional_record_value(
            follow_up_result,
            "source_replay_validation_mse_delta",
        ),
        "best_vs_source_replay_validation_mse_delta": _optional_record_value(
            follow_up_result,
            "best_vs_source_replay_validation_mse_delta",
        ),
        "source_replay_validation_nmse_delta": _optional_record_value(
            follow_up_result,
            "source_replay_validation_nmse_delta",
        ),
        "best_vs_source_replay_validation_nmse_delta": _optional_record_value(
            follow_up_result,
            "best_vs_source_replay_validation_nmse_delta",
        ),
        "candidate_validation_stability_status": candidate_stability_status,
        "candidate_validation_stability_score": _optional_record_value(
            guidance_candidate,
            "validation_stability_score",
        ),
        "candidate_validation_mse_stddev": _optional_record_value(
            guidance_candidate,
            "validation_readout_mse_stddev",
        ),
        "candidate_validation_mse_spread": _optional_record_value(
            guidance_candidate,
            "validation_readout_mse_spread",
        ),
        "candidate_validation_nmse_stddev": _optional_record_value(
            guidance_candidate,
            "validation_readout_nmse_stddev",
        ),
        "candidate_validation_nmse_spread": _optional_record_value(
            guidance_candidate,
            "validation_readout_nmse_spread",
        ),
        "suggested_flags": suggested_flags,
        "rationale": rationale,
    }


def follow_up_chain_guidance_row(record: dict[str, Any]) -> list[str]:
    verdicts = record.get("recent_verdicts")
    return [
        str(record.get("action") or "-"),
        str(record.get("current_verdict") or "-"),
        ",".join(str(verdict) for verdict in verdicts) if isinstance(verdicts, list) else "-",
        str(record.get("improved_streak") or 0),
        str(record.get("regressed_streak") or 0),
        str(record.get("non_improving_streak") or 0),
        str(record.get("missing_ancestors") or 0),
        "yes" if record.get("neighborhood_enabled") else "no",
        "yes" if record.get("stability_tradeoff") else "no",
        "yes" if record.get("source_replay_seed_shift") else "no",
        "yes" if record.get("target_scale_seed_shift") else "no",
        "yes" if record.get("normalized_regression") else "no",
        str(record.get("promotion_action") or "-"),
        fmt(record.get("source_replay_validation_mse_delta"), 6),
        fmt(record.get("best_vs_source_replay_validation_mse_delta"), 6),
        fmt(record.get("source_replay_validation_nmse_delta"), 6),
        fmt(record.get("best_vs_source_replay_validation_nmse_delta"), 6),
        str(record.get("candidate_validation_stability_status") or "-"),
        fmt(record.get("candidate_validation_stability_score"), 6),
        fmt(record.get("candidate_validation_mse_stddev"), 6),
        fmt(record.get("candidate_validation_mse_spread"), 6),
        fmt(record.get("candidate_validation_nmse_stddev"), 6),
        fmt(record.get("candidate_validation_nmse_spread"), 6),
        str(record.get("suggested_flags") or "-"),
        str(record.get("rationale") or "-"),
    ]


def follow_up_command_placeholders(command: Iterable[str]) -> list[str]:
    return [token for token in command if token in FOLLOW_UP_COMMAND_PLACEHOLDERS]


def follow_up_guided_script_usage(path: Path, placeholders: Iterable[str]) -> str:
    env = [f"{placeholder}=..." for placeholder in placeholders]
    script = shlex.quote(str(path))
    return " ".join([*env, "bash", script]) if env else f"bash {script}"


def follow_up_guided_next_command_record(
    run_root: Path,
    guidance: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(guidance, dict):
        return None
    suggested_flags = str(guidance.get("suggested_flags") or "--follow-up-source auto")
    command = [
        "python3",
        "-S",
        "-s",
        "tools/run_gnn_band_trace_sweep.py",
        "--run-root",
        "NEXT_RUN_ROOT",
        "--follow-up-from",
        str(run_root),
        *shlex.split(suggested_flags),
    ]
    placeholders = follow_up_command_placeholders(command)
    script_path = run_root / FOLLOW_UP_NEXT_COMMAND_SCRIPT
    return {
        "schema": "st.gnn.band_trace_follow_up_guided_next_command.v1",
        "source_run_root": str(run_root),
        "guidance_action": guidance.get("action"),
        "current_verdict": guidance.get("current_verdict"),
        "suggested_flags": suggested_flags,
        "placeholders": placeholders,
        "requires_user_input": bool(placeholders),
        "script_path": str(script_path),
        "script_usage": follow_up_guided_script_usage(script_path, placeholders),
        "command": command,
        "shell": " ".join(command),
    }


def shell_array_arg(token: str) -> str:
    if token == "NEXT_RUN_ROOT":
        return '"${NEXT_RUN_ROOT:?Set NEXT_RUN_ROOT to the next run directory}"'
    if token == "NEW_SEEDS":
        return '"${NEW_SEEDS:?Set NEW_SEEDS to a comma-separated fresh seed list}"'
    return shlex.quote(token)


def write_follow_up_guided_next_command_script(record: dict[str, Any]) -> Path:
    path = Path(str(record["script_path"]))
    command = record.get("command")
    if not isinstance(command, list):
        raise ValueError("guided next command record is missing command")
    placeholders = record.get("placeholders")
    placeholder_text = (
        ", ".join(str(item) for item in placeholders)
        if isinstance(placeholders, list) and placeholders
        else "none"
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by tools/run_gnn_band_trace_sweep.py.",
        f"# Guidance action: {record.get('guidance_action') or '-'}",
        f"# Current verdict: {record.get('current_verdict') or '-'}",
        f"# Required placeholders: {placeholder_text}",
        "# Example:",
        f"#   {record.get('script_usage') or f'bash {path}'}",
        "",
        "cmd=(",
        *[f"  {shell_array_arg(str(token))}" for token in command],
        ")",
        "",
        "printf 'running:'",
        "printf ' %q' \"${cmd[@]}\"",
        "printf '\\n'",
        "exec \"${cmd[@]}\"",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def follow_up_guided_next_command_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get("guidance_action") or "-"),
        str(record.get("current_verdict") or "-"),
        str(record.get("source_run_root") or "-"),
        str(record.get("suggested_flags") or "-"),
        ",".join(str(item) for item in record.get("placeholders", [])) or "-",
        "yes" if record.get("requires_user_input") else "no",
        str(record.get("script_path") or "-"),
        str(record.get("script_usage") or "-"),
        str(record.get("shell") or "-"),
    ]


def recommendation_row(
    rank: int,
    group: tuple[Any, ...],
    summaries: list[dict[str, Any]],
) -> list[str]:
    (
        backend,
        epochs,
        train_graphs,
        validation_graphs,
        batch,
        nodes,
        features,
        lr,
        top_k,
        mid_k,
        bottom_k,
        here_tolerance,
    ) = group
    metrics = group_metrics(summaries)
    seeds = ",".join(str(seed) for seed in seeds_for(summaries))
    return [
        str(rank),
        str(backend),
        str(epochs),
        str(train_graphs),
        str(validation_graphs),
        str(batch),
        str(nodes),
        str(features),
        fmt(lr, 6),
        str(top_k),
        str(mid_k),
        str(bottom_k),
        fmt(here_tolerance, 6),
        str(len(summaries)),
        seeds,
        fmt(metrics["validation_readout_mse"], 6),
        fmt(metrics["validation_readout_nmse"], 6),
        fmt(metrics["best_score"], 6),
        fmt(metrics["above_max_delta"], 4),
        fmt(metrics["here_max_delta"], 4),
        fmt(metrics["beneath_max_delta"], 4),
        fmt(metrics["step_ms_last"], 3),
        fmt(metrics["cpu_debt_ops"], 0),
        fmt(metrics["validation_readout_mse_stddev"], 6),
        fmt(metrics["validation_readout_mse_min"], 6),
        fmt(metrics["validation_readout_mse_max"], 6),
        fmt(metrics["validation_readout_mse_spread"], 6),
        fmt(metrics["validation_readout_nmse_stddev"], 6),
        fmt(metrics["validation_readout_nmse_spread"], 6),
        fmt(validation_stability_score(metrics), 6),
        validation_stability_status(metrics, runs=len(summaries)),
    ]


def stable_candidate_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get("stability_rank") or "-"),
        str(record.get("validation_rank") or "-"),
        str(record["backend"]),
        str(record["epochs"]),
        str(record["train_graphs"]),
        str(record["validation_graphs"]),
        str(record["batch"]),
        str(record["nodes"]),
        str(record["features"]),
        fmt(record["lr"], 6),
        str(record["top_k"]),
        str(record["mid_k"]),
        str(record["bottom_k"]),
        fmt(record["here_tolerance"], 6),
        str(record["runs"]),
        ",".join(str(seed) for seed in record.get("seeds", [])),
        fmt(record.get("avg_validation_readout_mse"), 6),
        fmt(record.get("avg_validation_readout_nmse"), 6),
        fmt(record.get("validation_readout_mse_stddev"), 6),
        fmt(record.get("validation_readout_mse_spread"), 6),
        fmt(record.get("validation_readout_nmse_stddev"), 6),
        fmt(record.get("validation_readout_nmse_spread"), 6),
        fmt(record.get("validation_stability_score"), 6),
        str(record.get("validation_stability_status") or "-"),
        fmt(record.get("avg_step_ms_last"), 3),
        fmt(record.get("avg_cpu_debt_ops"), 0),
    ]


def top_k_delta_rows(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> list[list[str]]:
    return roundtable_axis_delta_rows(groups, axis="top_k")


def roundtable_axis_delta_records(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    axis_index = ROUNDTABLE_AXIS_INDICES[axis]
    bucket_indices = tuple(idx for idx in range(12) if idx != axis_index)
    buckets: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], list[dict[str, Any]]]]] = {}
    for group, summaries in groups.items():
        bucket_key = tuple(group[idx] for idx in bucket_indices)
        buckets.setdefault(bucket_key, []).append((group, summaries))

    records: list[dict[str, Any]] = []
    for bucket_groups in buckets.values():
        if len(bucket_groups) < 2:
            continue
        bucket_groups = sorted(bucket_groups, key=lambda item: item[0][axis_index])
        baseline_group, baseline_summaries = bucket_groups[0]
        baseline_metrics = group_metrics(baseline_summaries)
        baseline_value = baseline_group[axis_index]
        for group, summaries in bucket_groups[1:]:
            metrics = group_metrics(summaries)
            value = group[axis_index]
            records.append(
                {
                    "axis": axis,
                    "baseline_value": baseline_value,
                    "value": value,
                    **group_record(group),
                    "avg_validation_readout_mse": metrics["validation_readout_mse"],
                    "avg_validation_readout_nmse": metrics["validation_readout_nmse"],
                    "validation_mse_delta": _delta_number(
                        metrics["validation_readout_mse"],
                        baseline_metrics["validation_readout_mse"],
                    ),
                    "validation_nmse_delta": _delta_number(
                        metrics["validation_readout_nmse"],
                        baseline_metrics["validation_readout_nmse"],
                    ),
                    "here_delta_delta": _delta_number(
                        metrics["here_max_delta"],
                        baseline_metrics["here_max_delta"],
                    ),
                    "above_delta_delta": _delta_number(
                        metrics["above_max_delta"],
                        baseline_metrics["above_max_delta"],
                    ),
                    "beneath_delta_delta": _delta_number(
                        metrics["beneath_max_delta"],
                        baseline_metrics["beneath_max_delta"],
                    ),
                }
            )
    axis_order = {"top_k": 0, "mid_k": 1, "bottom_k": 2, "here_tolerance": 3}
    return sorted(
        records,
        key=lambda row: (
            axis_order.get(row["axis"], 99),
            row["backend"],
            row["lr"],
            row["baseline_value"],
            row["value"],
        ),
    )


def roundtable_axis_value_text(axis: str, value: Any) -> str:
    return fmt(value, 6) if axis == "here_tolerance" else str(value)


def roundtable_axis_delta_row(record: dict[str, Any]) -> list[str]:
    axis = str(record["axis"])
    return [
        axis,
        str(record["backend"]),
        str(record["epochs"]),
        str(record["train_graphs"]),
        str(record["validation_graphs"]),
        str(record["batch"]),
        str(record["nodes"]),
        str(record["features"]),
        fmt(record["lr"], 6),
        roundtable_axis_value_text(axis, record["baseline_value"]),
        roundtable_axis_value_text(axis, record["value"]),
        str(record["top_k"]),
        str(record["mid_k"]),
        str(record["bottom_k"]),
        fmt(record["here_tolerance"], 6),
        fmt(record["avg_validation_readout_mse"], 6),
        fmt(record["avg_validation_readout_nmse"], 6),
        fmt(record["validation_mse_delta"], 6),
        fmt(record["validation_nmse_delta"], 6),
        fmt(record["here_delta_delta"], 4),
        fmt(record["above_delta_delta"], 4),
        fmt(record["beneath_delta_delta"], 4),
    ]


def roundtable_axis_delta_rows(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    axis: str,
) -> list[list[str]]:
    return [
        roundtable_axis_delta_row(record)
        for record in roundtable_axis_delta_records(groups, axis=axis)
    ]


def comparison_summary(
    summaries: list[dict[str, Any]],
    *,
    run_root: Path | None = None,
    follow_up: dict[str, Any] | None = None,
    follow_up_candidate: dict[str, Any] | None = None,
    follow_up_fail_on_verdict: Iterable[str] = (),
) -> dict[str, Any]:
    successful = [summary for summary in summaries if not summary.get("failed")]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for summary in successful:
        groups.setdefault(group_key(summary), []).append(summary)
    deltas = []
    for axis in ROUNDTABLE_AXES:
        deltas.extend(roundtable_axis_delta_records(groups, axis=axis))
    top_candidates = top_validation_candidate_records(groups)
    all_candidates = top_validation_candidate_records(groups, limit=len(groups))
    stable_candidates = stable_validation_candidate_records(groups)
    follow_up_result = follow_up_result_record(follow_up_candidate, all_candidates)
    follow_up_gate = follow_up_gate_record(follow_up_fail_on_verdict, follow_up_result)
    follow_up_promotion = follow_up_promotion_record(follow_up_result)
    follow_up_next_command = (
        follow_up_next_command_record(run_root, follow_up_promotion)
        if run_root is not None
        else None
    )
    follow_up_chain = (
        follow_up_chain_record(run_root, follow_up, follow_up_result, follow_up_promotion)
        if run_root is not None
        else None
    )
    follow_up_chain_guidance = follow_up_chain_guidance_record(
        follow_up_chain,
        follow_up_result,
    )
    follow_up_guided_next_command = (
        follow_up_guided_next_command_record(run_root, follow_up_chain_guidance)
        if run_root is not None
        else None
    )
    return {
        "schema": "st.gnn.band_trace_compare.v1",
        "successful_runs": len(successful),
        "group_count": len(groups),
        "top_validation_candidates": top_candidates,
        "stable_validation_candidates": stable_candidates,
        "roundtable_axis_deltas": deltas,
        "follow_up_result": follow_up_result,
        "follow_up_gate": follow_up_gate,
        "follow_up_promotion": follow_up_promotion,
        "follow_up_next_command": follow_up_next_command,
        "follow_up_chain": follow_up_chain,
        "follow_up_chain_guidance": follow_up_chain_guidance,
        "follow_up_guided_next_command": follow_up_guided_next_command,
    }


def write_compare(
    run_root: Path,
    summaries: list[dict[str, Any]],
    *,
    follow_up: dict[str, Any] | None = None,
    follow_up_candidate: dict[str, Any] | None = None,
    follow_up_fail_on_verdict: Iterable[str] = (),
) -> Path:
    headers = [
        "backend",
        "seed",
        "epochs",
        "train_graphs",
        "validation_graphs",
        "batch",
        "nodes",
        "features",
        "lr",
        "top_k",
        "mid_k",
        "bottom_k",
        "here_tolerance",
        "input_rows",
        "run_status",
        "returncode",
        "failure_kind",
        "failure_detail",
        "log_path",
        *data_column_headers(),
    ]
    lines = [
        "# GNN Band Trace Sweep",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row_for(summary)) + " |")

    successful = [summary for summary in summaries if not summary.get("failed")]
    aggregate_headers = [
        "backend",
        "epochs",
        "train_graphs",
        "validation_graphs",
        "batch",
        "nodes",
        "features",
        "lr",
        "top_k",
        "mid_k",
        "bottom_k",
        "here_tolerance",
        "input_rows",
        "runs",
        "seeds",
        "avg_best_score",
        "avg_readout_mse",
        "avg_readout_nmse",
        "avg_readout_graphs",
        "avg_readout_rows",
        "avg_validation_readout_mse",
        "avg_validation_readout_nmse",
        "avg_validation_readout_graphs",
        "avg_validation_readout_rows",
        "avg_above_max_delta",
        "avg_here_max_delta",
        "avg_beneath_max_delta",
        "avg_step_ms_last",
        "avg_tensor_ops",
        "avg_tensor_wgpu",
        "avg_tensor_cpu",
        "avg_cpu_debt_ops",
        "avg_policy_events",
    ]
    lines.extend(
        [
            "",
            "## Group Averages",
            "",
            "| " + " | ".join(aggregate_headers) + " |",
            "| " + " | ".join("---" for _ in aggregate_headers) + " |",
        ]
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for summary in successful:
        groups.setdefault(group_key(summary), []).append(summary)
    for group in sorted(groups):
        lines.append(
            "| "
            + " | ".join(md_cell(cell) for cell in average_row(group, groups[group]))
            + " |"
        )

    candidate_headers = [
        "rank",
        "backend",
        "epochs",
        "train_graphs",
        "validation_graphs",
        "batch",
        "nodes",
        "features",
        "lr",
        "top_k",
        "mid_k",
        "bottom_k",
        "here_tolerance",
        "runs",
        "seeds",
        "avg_validation_readout_mse",
        "avg_validation_readout_nmse",
        "avg_best_score",
        "avg_above_max_delta",
        "avg_here_max_delta",
        "avg_beneath_max_delta",
        "avg_step_ms_last",
        "avg_cpu_debt_ops",
        "validation_mse_stddev",
        "validation_mse_min",
        "validation_mse_max",
        "validation_mse_spread",
        "validation_nmse_stddev",
        "validation_nmse_spread",
        "validation_stability_score",
        "validation_stability_status",
    ]
    candidates = top_validation_candidate_records(groups)
    all_candidates = top_validation_candidate_records(groups, limit=len(groups))
    lines.extend(
        [
            "",
            "## Top Validation Candidates",
            "",
            "| " + " | ".join(candidate_headers) + " |",
            "| " + " | ".join("---" for _ in candidate_headers) + " |",
        ]
    )
    for candidate in candidates:
        group = tuple(candidate[field] for field in GROUP_FIELDS)
        grouped = groups[group]
        lines.append(
            "| "
            + " | ".join(
                md_cell(cell) for cell in recommendation_row(candidate["rank"], group, grouped)
            )
            + " |"
        )

    stable_candidate_headers = [
        "stability_rank",
        "validation_rank",
        "backend",
        "epochs",
        "train_graphs",
        "validation_graphs",
        "batch",
        "nodes",
        "features",
        "lr",
        "top_k",
        "mid_k",
        "bottom_k",
        "here_tolerance",
        "runs",
        "seeds",
        "avg_validation_readout_mse",
        "avg_validation_readout_nmse",
        "validation_mse_stddev",
        "validation_mse_spread",
        "validation_nmse_stddev",
        "validation_nmse_spread",
        "validation_stability_score",
        "validation_stability_status",
        "avg_step_ms_last",
        "avg_cpu_debt_ops",
    ]
    stable_candidates = stable_validation_candidate_records(groups)
    lines.extend(
        [
            "",
            "## Stable Validation Candidates",
            "",
            "| " + " | ".join(stable_candidate_headers) + " |",
            "| " + " | ".join("---" for _ in stable_candidate_headers) + " |",
        ]
    )
    for candidate in stable_candidates:
        lines.append(
            "| " + " | ".join(md_cell(cell) for cell in stable_candidate_row(candidate)) + " |"
        )

    follow_up_result = follow_up_result_record(follow_up_candidate, all_candidates)
    if follow_up_result is not None:
        follow_up_headers = [
            "source_rank",
            "source_replay_rank",
            "best_rank",
            "verdict",
            "source_schedule",
            "source_replay_schedule",
            "best_schedule",
            "source_validation_mse",
            "source_replay_validation_mse",
            "best_validation_mse",
            "source_validation_nmse",
            "source_replay_validation_nmse",
            "best_validation_nmse",
            "validation_mse_delta",
            "source_replay_mse_delta",
            "best_vs_source_replay_mse_delta",
            "validation_nmse_delta",
            "source_replay_nmse_delta",
            "best_vs_source_replay_nmse_delta",
            "source_stability_status",
            "source_replay_stability_status",
            "best_stability_status",
            "source_stability_score",
            "source_replay_stability_score",
            "best_stability_score",
            "source_validation_mse_stddev",
            "source_replay_validation_mse_stddev",
            "best_validation_mse_stddev",
            "source_validation_mse_spread",
            "source_replay_validation_mse_spread",
            "best_validation_mse_spread",
            "source_validation_nmse_stddev",
            "source_replay_validation_nmse_stddev",
            "best_validation_nmse_stddev",
            "source_validation_nmse_spread",
            "source_replay_validation_nmse_spread",
            "best_validation_nmse_spread",
            "source_cpu_debt_ops",
            "source_replay_cpu_debt_ops",
            "best_cpu_debt_ops",
            "cpu_debt_delta",
            "best_vs_source_replay_cpu_debt_delta",
            "source_step_ms",
            "source_replay_step_ms",
            "best_step_ms",
            "step_ms_delta",
            "best_vs_source_replay_step_ms_delta",
        ]
        lines.extend(
            [
                "",
                "## Follow-Up Result",
                "",
                "| " + " | ".join(follow_up_headers) + " |",
                "| " + " | ".join("---" for _ in follow_up_headers) + " |",
                "| "
                + " | ".join(md_cell(cell) for cell in follow_up_result_row(follow_up_result))
                + " |",
            ]
        )
    follow_up_gate = follow_up_gate_record(follow_up_fail_on_verdict, follow_up_result)
    if follow_up_gate is not None:
        gate_headers = ["verdict", "failed", "fail_on_verdicts"]
        lines.extend(
            [
                "",
                "## Follow-Up Gate",
                "",
                "| " + " | ".join(gate_headers) + " |",
                "| " + " | ".join("---" for _ in gate_headers) + " |",
                "| "
                + " | ".join(md_cell(cell) for cell in follow_up_gate_row(follow_up_gate))
                + " |",
            ]
        )
    follow_up_promotion = follow_up_promotion_record(follow_up_result)
    if follow_up_promotion is not None:
        promotion_headers = [
            "action",
            "verdict",
            "selected_origin",
            "selected_schedule",
            "selected_validation_mse",
            "selected_validation_nmse",
            "validation_mse_delta",
            "validation_nmse_delta",
            "stability_guard",
            "source_stability_status",
            "best_stability_status",
            "source_stability_score",
            "best_stability_score",
        ]
        lines.extend(
            [
                "",
                "## Follow-Up Promotion",
                "",
                "| " + " | ".join(promotion_headers) + " |",
                "| " + " | ".join("---" for _ in promotion_headers) + " |",
                "| "
                + " | ".join(
                    md_cell(cell) for cell in follow_up_promotion_row(follow_up_promotion)
                )
                + " |",
            ]
        )
    follow_up_next_command = follow_up_next_command_record(run_root, follow_up_promotion)
    if follow_up_next_command is not None:
        command_headers = [
            "uses_promotion",
            "promotion_action",
            "selected_origin",
            "source_run_root",
            "shell",
        ]
        lines.extend(
            [
                "",
                "## Next Follow-Up Command",
                "",
                "| " + " | ".join(command_headers) + " |",
                "| " + " | ".join("---" for _ in command_headers) + " |",
                "| "
                + " | ".join(
                    md_cell(cell) for cell in follow_up_next_command_row(follow_up_next_command)
                )
                + " |",
            ]
        )
    follow_up_chain = follow_up_chain_record(
        run_root,
        follow_up,
        follow_up_result,
        follow_up_promotion,
    )
    if follow_up_chain is not None:
        chain_headers = [
            "generation",
            "chain_depth",
            "source_mode",
            "candidate_source",
            "parent_generation",
            "parent_run_root",
            "parent_sweep_path",
            "verdict",
            "promotion_action",
            "selected_origin",
            "neighborhood_enabled",
        ]
        lines.extend(
            [
                "",
                "## Follow-Up Chain",
                "",
                "| " + " | ".join(chain_headers) + " |",
                "| " + " | ".join("---" for _ in chain_headers) + " |",
                "| "
                + " | ".join(md_cell(cell) for cell in follow_up_chain_row(follow_up_chain))
                + " |",
            ]
        )
        ancestors = follow_up_chain.get("ancestors")
        if isinstance(ancestors, list) and ancestors:
            ancestor_headers = [
                "generation",
                "run_root",
                "sweep_path",
                "source_mode",
                "candidate_source",
                "verdict",
                "promotion_action",
                "selected_origin",
                "selected_schedule",
                "selected_validation_mse",
                "selected_validation_nmse",
                "missing",
                "cycle_detected",
                "error",
            ]
            lines.extend(
                [
                    "",
                    "## Follow-Up Ancestors",
                    "",
                    "| " + " | ".join(ancestor_headers) + " |",
                    "| " + " | ".join("---" for _ in ancestor_headers) + " |",
                ]
            )
            for ancestor in ancestors:
                lines.append(
                    "| "
                    + " | ".join(
                        md_cell(cell) for cell in follow_up_chain_ancestor_row(ancestor)
                    )
                    + " |"
                )
        follow_up_chain_guidance = follow_up_chain_guidance_record(
            follow_up_chain,
            follow_up_result,
        )
        if follow_up_chain_guidance is not None:
            guidance_headers = [
                "action",
                "current_verdict",
                "recent_verdicts",
                "improved_streak",
                "regressed_streak",
                "non_improving_streak",
                "missing_ancestors",
                "neighborhood_enabled",
                "stability_tradeoff",
                "source_replay_seed_shift",
                "target_scale_seed_shift",
                "normalized_regression",
                "promotion_action",
                "source_replay_mse_delta",
                "best_vs_source_replay_mse_delta",
                "source_replay_nmse_delta",
                "best_vs_source_replay_nmse_delta",
                "candidate_stability_status",
                "candidate_stability_score",
                "candidate_validation_mse_stddev",
                "candidate_validation_mse_spread",
                "candidate_validation_nmse_stddev",
                "candidate_validation_nmse_spread",
                "suggested_flags",
                "rationale",
            ]
            lines.extend(
                [
                    "",
                    "## Follow-Up Chain Guidance",
                    "",
                    "| " + " | ".join(guidance_headers) + " |",
                    "| " + " | ".join("---" for _ in guidance_headers) + " |",
                    "| "
                    + " | ".join(
                        md_cell(cell)
                        for cell in follow_up_chain_guidance_row(follow_up_chain_guidance)
                    )
                    + " |",
                ]
            )
            follow_up_guided_next_command = follow_up_guided_next_command_record(
                run_root,
                follow_up_chain_guidance,
            )
            if follow_up_guided_next_command is not None:
                guided_command_headers = [
                    "guidance_action",
                    "current_verdict",
                    "source_run_root",
                    "suggested_flags",
                    "placeholders",
                    "requires_user_input",
                    "script_path",
                    "script_usage",
                    "shell",
                ]
                write_follow_up_guided_next_command_script(follow_up_guided_next_command)
                lines.extend(
                    [
                        "",
                        "## Guided Next Follow-Up Command",
                        "",
                        "| " + " | ".join(guided_command_headers) + " |",
                        "| " + " | ".join("---" for _ in guided_command_headers) + " |",
                        "| "
                        + " | ".join(
                            md_cell(cell)
                            for cell in follow_up_guided_next_command_row(
                                follow_up_guided_next_command
                            )
                        )
                        + " |",
                    ]
                )

    delta_headers = [
        "axis",
        "backend",
        "epochs",
        "train_graphs",
        "validation_graphs",
        "batch",
        "nodes",
        "features",
        "lr",
        "baseline_value",
        "value",
        "top_k",
        "mid_k",
        "bottom_k",
        "here_tolerance",
        "avg_validation_readout_mse",
        "avg_validation_readout_nmse",
        "validation_mse_delta",
        "validation_nmse_delta",
        "here_delta_delta",
        "above_delta_delta",
        "beneath_delta_delta",
    ]
    delta_rows = []
    for axis in ROUNDTABLE_AXES:
        delta_rows.extend(roundtable_axis_delta_rows(groups, axis=axis))
    if delta_rows:
        lines.extend(
            [
                "",
                "## Roundtable Axis Deltas",
                "",
                "| " + " | ".join(delta_headers) + " |",
                "| " + " | ".join("---" for _ in delta_headers) + " |",
            ]
        )
        for row in delta_rows:
            lines.append("| " + " | ".join(md_cell(cell) for cell in row) + " |")
    path = run_root / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/gnn_band_trace_sweep"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("cpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--train-graphs", type=int, default=4)
    parser.add_argument("--validation-graphs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--features", type=int, default=2)
    parser.add_argument("--epoch-values", default=None)
    parser.add_argument("--train-graph-values", default=None)
    parser.add_argument("--validation-graph-values", default=None)
    parser.add_argument("--batch-values", default=None)
    parser.add_argument("--node-values", default=None)
    parser.add_argument("--feature-values", default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-values", default=None)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--mid-k", type=int, default=1)
    parser.add_argument("--bottom-k", type=int, default=1)
    parser.add_argument("--here-tolerance", type=float, default=1e-5)
    parser.add_argument("--top-k-values", default=None)
    parser.add_argument("--mid-k-values", default=None)
    parser.add_argument("--bottom-k-values", default=None)
    parser.add_argument("--here-tolerance-values", default=None)
    parser.add_argument("--wgpu-min-values", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-wgpu-preflight", action="store_true")
    parser.add_argument(
        "--follow-up-from",
        type=Path,
        default=None,
        help="load defaults from a previous sweep.json comparison top candidate",
    )
    parser.add_argument(
        "--follow-up-rank",
        type=int,
        default=1,
        help="1-based top_validation_candidates rank to replay from --follow-up-from",
    )
    parser.add_argument(
        "--follow-up-source",
        type=parse_follow_up_source,
        default="auto",
        help=(
            "candidate source for --follow-up-from: auto uses follow_up_promotion "
            "when present, otherwise top-candidate"
        ),
    )
    parser.add_argument(
        "--follow-up-neighborhood",
        action="store_true",
        help="expand local schedule values around the selected follow-up candidate",
    )
    parser.add_argument(
        "--follow-up-neighborhood-axes",
        type=parse_csv,
        default=parse_csv("lr,top_k,mid_k,bottom_k,here_tolerance"),
        help="comma-separated follow-up axes to expand when --follow-up-neighborhood is set",
    )
    parser.add_argument(
        "--follow-up-neighborhood-lr-factors",
        type=parse_float_csv,
        default=parse_float_csv("0.75,1,1.25"),
        help="multipliers for the local learning-rate neighborhood",
    )
    parser.add_argument(
        "--follow-up-neighborhood-k-radius",
        type=int,
        default=1,
        help="integer radius for local top/mid/bottom-k neighborhoods",
    )
    parser.add_argument(
        "--follow-up-neighborhood-tolerance-factors",
        type=parse_float_csv,
        default=parse_float_csv("0.1,1,10"),
        help="multipliers for the local here-tolerance neighborhood",
    )
    parser.add_argument(
        "--follow-up-fail-on-verdict",
        type=parse_follow_up_verdict_csv,
        default=[],
        help=(
            "comma-separated follow-up verdicts that should make the sweep exit non-zero "
            "(for example: regressed,unknown)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def grid_manifest(args: argparse.Namespace) -> dict[str, list[Any]]:
    return {
        "epochs": positive_int_grid(args.epoch_values, args.epochs, label="epoch-values"),
        "train_graphs": positive_int_grid(
            args.train_graph_values,
            args.train_graphs,
            label="train-graph-values",
        ),
        "validation_graphs": positive_int_grid(
            args.validation_graph_values,
            args.validation_graphs,
            label="validation-graph-values",
        ),
        "batch": positive_int_grid(args.batch_values, args.batch, label="batch-values"),
        "nodes": positive_int_grid(args.node_values, args.nodes, label="node-values"),
        "features": positive_int_grid(
            args.feature_values,
            args.features,
            label="feature-values",
        ),
        "learning_rates": positive_float_grid(args.lr_values, args.lr, label="lr-values"),
        "top_k": positive_int_grid(args.top_k_values, args.top_k, label="top-k-values"),
        "mid_k": positive_int_grid(args.mid_k_values, args.mid_k, label="mid-k-values"),
        "bottom_k": positive_int_grid(
            args.bottom_k_values,
            args.bottom_k,
            label="bottom-k-values",
        ),
        "here_tolerance": nonnegative_float_grid(
            args.here_tolerance_values,
            args.here_tolerance,
            label="here-tolerance-values",
        ),
    }


def iter_axes(args: argparse.Namespace, grid: dict[str, list[Any]]) -> Iterable[GnnRunAxes]:
    for (
        backend,
        epochs,
        train_graphs,
        validation_graphs,
        batch,
        nodes,
        features,
        lr,
        top_k,
        mid_k,
        bottom_k,
        here_tolerance,
        seed,
    ) in product(
        args.backends,
        grid["epochs"],
        grid["train_graphs"],
        grid["validation_graphs"],
        grid["batch"],
        grid["nodes"],
        grid["features"],
        grid["learning_rates"],
        grid["top_k"],
        grid["mid_k"],
        grid["bottom_k"],
        grid["here_tolerance"],
        args.seeds,
    ):
        yield GnnRunAxes(
            backend=backend,
            seed=seed,
            epochs=epochs,
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            batch=batch,
            nodes=nodes,
            features=features,
            lr=lr,
            top_k=top_k,
            mid_k=mid_k,
            bottom_k=bottom_k,
            here_tolerance=here_tolerance,
        )


def sweep_manifest(
    args: argparse.Namespace,
    grid: dict[str, list[Any]],
    runs: list[dict[str, Any]],
    preflight_failures: dict[str, Any],
    *,
    summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary_rows = summaries if summaries is not None else runs
    follow_up = follow_up_manifest(args)
    manifest = {
        "schema": "st.gnn.band_trace_sweep.v2",
        "run_root": str(args.run_root),
        "backends": args.backends,
        "seeds": args.seeds,
        "grid": grid,
        "config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "train_graphs": args.train_graphs,
            "validation_graphs": args.validation_graphs,
            "batch": args.batch,
            "nodes": args.nodes,
            "features": args.features,
            "lr": args.lr,
            "curvature": args.curvature,
            "top_k": args.top_k,
            "mid_k": args.mid_k,
            "bottom_k": args.bottom_k,
            "here_tolerance": args.here_tolerance,
            "cargo_features": args.cargo_features,
            "no_default_features": args.no_default_features,
            "wgpu_preflight": not args.no_wgpu_preflight,
            "dry_run": args.dry_run,
            "follow_up_fail_on_verdict": args.follow_up_fail_on_verdict,
            "follow_up": follow_up,
        },
        "preflight_failures": preflight_failures,
        "runs": [
            {
                "name": summary.get("name"),
                "backend": summary["backend"],
                "seed": summary["seed"],
                "epochs": summary["epochs"],
                "train_graphs": summary["train_graphs"],
                "validation_graphs": summary["validation_graphs"],
                "batch": summary["batch"],
                "nodes": summary["nodes"],
                "features": summary["features"],
                "lr": summary["lr"],
                "top_k": summary["top_k"],
                "mid_k": summary["mid_k"],
                "bottom_k": summary["bottom_k"],
                "here_tolerance": summary["here_tolerance"],
                "input_rows": summary["input_rows"],
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
    if summaries is not None:
        manifest["comparison"] = comparison_summary(
            summaries,
            run_root=args.run_root,
            follow_up=follow_up,
            follow_up_candidate=getattr(args, "follow_up_candidate", None),
            follow_up_fail_on_verdict=args.follow_up_fail_on_verdict,
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)
    apply_follow_up_defaults(args, raw_argv)
    grid = grid_manifest(args)

    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        runs = [planned_run(args, axes) for axes in iter_axes(args, grid)]
        sweep = sweep_manifest(args, grid, runs, {})
        (args.run_root / "sweep.json").write_text(
            json.dumps(sweep, indent=2, sort_keys=True),
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
    for axes in iter_axes(args, grid):
        preflight_failure = preflight_failures.get(axes.backend)
        if preflight_failure is not None:
            runs.append(preflight_skipped_run(args, axes, preflight_failure))
        else:
            runs.append(run_one(args, axes))
    summaries = [summarize_run(run) for run in runs]
    compare_path = write_compare(
        args.run_root,
        summaries,
        follow_up=follow_up_manifest(args),
        follow_up_candidate=getattr(args, "follow_up_candidate", None),
        follow_up_fail_on_verdict=args.follow_up_fail_on_verdict,
    )
    sweep = sweep_manifest(args, grid, runs, preflight_failures, summaries=summaries)
    sweep["compare_path"] = str(compare_path)
    (args.run_root / "sweep.json").write_text(
        json.dumps(sweep, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"compare={compare_path}")
    if sweep["failed"]:
        return 1
    if follow_up_gate_failed(sweep.get("comparison")):
        gate = sweep.get("comparison", {}).get("follow_up_gate", {})
        print(f"follow_up_gate_failed={gate.get('verdict', 'unknown')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

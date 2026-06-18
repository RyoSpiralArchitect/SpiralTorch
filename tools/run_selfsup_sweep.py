#!/usr/bin/env python3
"""Run SpiralLightning self-supervised InfoNCE probes and compare backend traces."""

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
EXAMPLE = "modelzoo_lightning_selfsup_minimal"


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


def info_nce_losses(trace: dict[str, Any]) -> list[float]:
    losses: list[float] = []
    stages = trace.get("stages")
    if not isinstance(stages, list):
        return losses
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        epochs = stage.get("epochs")
        if not isinstance(epochs, list):
            continue
        for epoch in epochs:
            if not isinstance(epoch, dict):
                continue
            info_nce = epoch.get("info_nce")
            if not isinstance(info_nce, dict):
                continue
            value = info_nce.get("mean_loss")
            if isinstance(value, (int, float)):
                losses.append(float(value))
    return losses


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
    pair_batch: int | None = None,
    input_dim: int | None = None,
    embed_dim: int | None = None,
) -> list[str]:
    # Keep CPU baselines CPU-only even when WGPU rows are in the same sweep.
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
            "--pair-batch",
            str(args.pair_batch if pair_batch is None else pair_batch),
            "--input-dim",
            str(args.input_dim if input_dim is None else input_dim),
            "--embed-dim",
            str(args.embed_dim if embed_dim is None else embed_dim),
            "--seed",
            str(seed),
            "--lr",
            str(args.lr),
            "--curvature",
            str(args.curvature),
            "--temperature",
            str(args.temperature),
            "--normalize",
            "true" if args.normalize else "false",
            "--accumulator-sync",
            args.accumulator_sync,
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
        pair_batch=2,
        input_dim=2,
        embed_dim=2,
    )
    returncode = run_command(command, log_path, threshold=args.wgpu_min_values)
    if returncode == 0:
        return None

    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure = {
        "schema": "st.selfsup.sweep_preflight_failure.v1",
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
        schema="st.selfsup.sweep_failure.v1",
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
    trace_json = run_dir / "selfsup_trace.json"
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
            "schema": "st.selfsup.sweep_failure.v1",
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


def read_selfsup_trace(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "selfsup_trace.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"selfsup trace at {run_dir} is not a JSON object")
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
    trace = read_selfsup_trace(run_dir)
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
    losses = info_nce_losses(trace)
    best_info_nce = min(losses) if losses else None
    best_epoch = losses.index(best_info_nce) + 1 if best_info_nce is not None else None
    pretrain_info_nce = run_summary.get("pretrain_info_nce")
    pretrain_cpu_reference = run_summary.get("pretrain_cpu_reference_info_nce")
    pretrain_backend_gap = run_summary.get("pretrain_backend_gap")
    pretrain_forward_gap = run_summary.get("pretrain_forward_gap")
    pretrain_loss_gap = run_summary.get("pretrain_loss_gap")
    last_info_nce = run_summary.get("last_info_nce")
    pretrain_to_best = (
        best_info_nce - float(pretrain_info_nce)
        if isinstance(pretrain_info_nce, (int, float)) and best_info_nce is not None
        else None
    )
    pretrain_to_final = (
        float(last_info_nce) - float(pretrain_info_nce)
        if isinstance(pretrain_info_nce, (int, float))
        and isinstance(last_info_nce, (int, float))
        else None
    )
    final_minus_best = (
        float(last_info_nce) - best_info_nce
        if isinstance(last_info_nce, (int, float)) and best_info_nce is not None
        else None
    )
    return [
        *prefix,
        fmt(pretrain_cpu_reference, 6),
        fmt(pretrain_info_nce, 6),
        fmt(pretrain_backend_gap, 6),
        fmt(pretrain_forward_gap, 6),
        fmt(pretrain_loss_gap, 6),
        fmt(run_summary.get("first_info_nce"), 6),
        fmt(run_summary.get("last_info_nce"), 6),
        fmt(best_info_nce, 6),
        fmt(best_epoch, 0),
        fmt(pretrain_to_best, 6),
        fmt(pretrain_to_final, 6),
        fmt(run_summary.get("info_nce_delta"), 6),
        fmt(final_minus_best, 6),
        fmt(metric(trainer, "step_time_ms", "last"), 3),
        fmt(metric(trainer, "tensor_ops_total", "last"), 0),
        fmt(metric(trainer, "tensor_backend_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_cpu", "last"), 0),
        fmt(metric(trainer, "tensor_backend_naive", "last"), 0),
        *backend_meta_row(summary.get("run_meta", {})),
        *backend_residual_row(trainer),
        fmt(metric(trainer, "tensor_op_backend_matmul_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_matmul_naive", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_row_softmax_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_row_softmax_cpu", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_transpose_wgpu", "last"), 0),
        fmt(metric(trainer, "tensor_op_backend_transpose_cpu", "last"), 0),
        fmt(metric(trainer, "optim_accumulator_sync_enabled", "last"), 0),
        fmt(metric(trainer, "optim_accumulator_sync_world_size", "last"), 0),
        fmt(metric(trainer, "optim_accumulator_sync_buffers", "last"), 0),
        fmt(metric(trainer, "optim_accumulator_sync_values", "last"), 0),
        fmt(metric(trainer, "backend_policy_events", "sum"), 0),
    ]


def data_column_headers() -> list[str]:
    return [
        "pretrain_cpu_ref",
        "pretrain_info_nce",
        "pretrain_backend_gap",
        "pretrain_forward_gap",
        "pretrain_loss_gap",
        "first_info_nce",
        "last_info_nce",
        "best_info_nce",
        "best_epoch",
        "pretrain_to_best",
        "pretrain_to_final",
        "delta",
        "final_minus_best",
        "step_ms_last",
        "tensor_ops",
        "tensor_wgpu",
        "tensor_cpu",
        "tensor_naive",
        *BACKEND_META_HEADERS,
        *BACKEND_RESIDUAL_HEADERS,
        "matmul_wgpu",
        "matmul_naive",
        "softmax_wgpu",
        "softmax_cpu",
        "transpose_wgpu",
        "transpose_cpu",
        "sync_enabled",
        "sync_world",
        "sync_buffers",
        "sync_values",
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
        "# Self-Supervised InfoNCE Sweep",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(md_cell(cell) for cell in row_for(summary)) + " |")
    path = run_root / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("models/runs/selfsup_sweep"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("cpu"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1"))
    parser.add_argument("--cargo-features", default="")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--pair-batch", type=int, default=3)
    parser.add_argument("--input-dim", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--wgpu-min-values", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accumulator-sync", choices=["none", "local"], default="none")
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
        "schema": "st.selfsup.sweep.v1",
        "run_root": str(args.run_root),
        "backends": args.backends,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "batches": args.batches,
            "pair_batch": args.pair_batch,
            "input_dim": args.input_dim,
            "embed_dim": args.embed_dim,
            "lr": args.lr,
            "curvature": args.curvature,
            "temperature": args.temperature,
            "normalize": args.normalize,
            "accumulator_sync": args.accumulator_sync,
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

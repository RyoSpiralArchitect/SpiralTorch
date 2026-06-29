#!/usr/bin/env python3
"""Run a scratch -> reload Python char-LM finetune pair.

This is the smallest first-class harness for checking whether non-scratch
tokenizerless char-LM finetuning is reload-safe and compare-ready.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_char_lm_sweep as sweep


SCHEMA = "st.llm_char_finetune.reload_pair.v1"
PREFLIGHT_SCHEMA = "st.llm_char_finetune.reload_pair.preflight.v1"
OUTCOME_SCHEMA = "st.llm_char_finetune.reload_pair.outcome.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_SCRIPT = REPO_ROOT / "models" / "python" / "llm_char_finetune.py"
COMPARE_SCRIPT = REPO_ROOT / "tools" / "compare_char_lm_runs.py"
REQUIRED_NN_SYMBOLS = [
    "Sequential",
    "Embedding",
    "SpiralRnn",
    "Linear",
    "ZSpaceSoftmax",
    "ModuleTrainer",
    "CategoricalCrossEntropy",
    "save",
    "load",
]
BACKEND_FEATURE_REQUIREMENTS = {
    "wgpu": ["wgpu", "wgpu-rt"],
    "webgpu": ["wgpu", "wgpu-rt"],
    "auto": ["wgpu", "wgpu-rt"],
    "cuda": ["cuda"],
    "hip": ["hip", "hip-real"],
    "rocm": ["hip", "hip-real"],
}


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "models" / "runs" / f"llm_char_finetune_reload_pair_{stamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def metric_value(payload: dict[str, Any], section: str, metric: str) -> float | None:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return None
    return finite_float(section_payload.get(metric))


def nll_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def nll_status(delta: float | None, *, tolerance: float = 1.0e-9) -> str:
    if delta is None:
        return "unknown"
    if delta < -tolerance:
        return "improved"
    if delta > tolerance:
        return "regressed"
    return "tied"


def backend_readiness(backend: str, features: dict[str, bool]) -> dict[str, Any]:
    normalized = str(backend).strip().lower()
    required_any = BACKEND_FEATURE_REQUIREMENTS.get(normalized, [])
    if normalized == "cpu":
        ready = True
        status = "available"
    elif required_any:
        ready = any(bool(features.get(feature)) for feature in required_any)
        status = "available" if ready else "backend_unavailable"
    else:
        ready = False
        status = "unknown_backend"
    return {
        "backend_normalized": normalized,
        "backend_ready": ready,
        "backend_status": status,
        "backend_required_any_features": required_any,
    }


def preflight_command(
    data_paths: list[Path],
    *,
    run_dir: Path,
    backend: str,
) -> list[str]:
    return [
        sys.executable,
        "-S",
        "-s",
        str(FINETUNE_SCRIPT),
        *[str(path) for path in data_paths],
        "--run-dir",
        str(run_dir),
        "--backend",
        backend,
        "--preflight-only",
    ]


def run_finetune_preflight(
    data_paths: list[Path],
    *,
    run_root: Path,
    backend: str,
) -> dict[str, Any]:
    preflight_run_dir = run_root / "_preflight"
    child_preflight_path = preflight_run_dir / "preflight.json"
    child_summary_path = preflight_run_dir / "summary.json"
    command = preflight_command(data_paths, run_dir=preflight_run_dir, backend=backend)
    returncode, elapsed = run_command(
        command,
        preflight_run_dir / "process.log",
        dry_run=False,
    )
    try:
        child_preflight = (
            read_json(child_preflight_path) if child_preflight_path.exists() else {}
        )
    except Exception as exc:
        child_preflight = {"error": str(exc)}
    child_ready = child_preflight.get("ready") is True and returncode == 0
    issues = child_preflight.get("issues")
    if not isinstance(issues, list):
        issues = []
    reason = str(child_preflight.get("reason") or "")
    if returncode != 0 and not reason:
        reason = "preflight_command_failed"
    if not child_preflight_path.exists():
        reason = "preflight_artifact_missing"
        issues = [reason]

    payload: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "python_executable": sys.executable,
        "backend": backend,
        "ready": child_ready,
        "reason": "ready" if child_ready else reason,
        "issues": [] if child_ready else issues,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "run_dir": str(preflight_run_dir),
        "log_path": str(preflight_run_dir / "process.log"),
        "command": command,
        "shell_command": shell_command(command),
        "child_preflight_path": str(child_preflight_path),
        "child_summary_path": str(child_summary_path),
        "child_preflight": child_preflight,
    }
    for key in (
        "backend_normalized",
        "backend_ready",
        "backend_status",
        "backend_required_any_features",
        "native_features",
        "required_nn_symbols",
        "missing_nn_symbols",
        "build_info_error",
        "spiraltorch_module",
    ):
        if key in child_preflight:
            payload[key] = child_preflight[key]
    return payload


def shell_command(command: list[str]) -> str:
    return "PYTHONNOUSERSITE=1 " + shlex.join(command)


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shell_command(command) + "\n")
        log_file.flush()
        if dry_run:
            log_file.write("[dry-run] command not executed\n")
            return 0, 0.0
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode, time.time() - started


def finetune_command(
    data_paths: list[Path],
    *,
    run_dir: Path,
    epochs: int,
    batches: int,
    batch: int,
    steps: int,
    embed_dim: int,
    hidden: int,
    lr: float,
    eval_samples: int,
    val_split: float,
    gen: int,
    topk: int,
    seed: int,
    eval_seed: int,
    backend: str,
    early_stop_patience: int,
    restore_best_at_end: bool,
    load_run: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-S",
        "-s",
        str(FINETUNE_SCRIPT),
        *[str(path) for path in data_paths],
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(epochs),
        "--batches",
        str(batches),
        "--batch",
        str(batch),
        "--steps",
        str(steps),
        "--embed-dim",
        str(embed_dim),
        "--hidden",
        str(hidden),
        "--lr",
        str(lr),
        "--eval-samples",
        str(eval_samples),
        "--val-split",
        str(val_split),
        "--gen",
        str(gen),
        "--topk",
        str(topk),
        "--seed",
        str(seed),
        "--eval-seed",
        str(eval_seed),
        "--backend",
        backend,
    ]
    if early_stop_patience > 0:
        command.extend(["--early-stop-patience", str(early_stop_patience)])
    if restore_best_at_end:
        command.append("--restore-best-at-end")
    if load_run is not None:
        command.extend(["--load-run", str(load_run)])
    return command


def compare_command(
    base_run_dir: Path,
    reload_run_dir: Path,
    *,
    run_root: Path,
    curves: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-S",
        "-s",
        str(COMPARE_SCRIPT),
        "--aggregate",
        "--json-out",
        str(run_root / "compare.json"),
    ]
    if curves:
        command.append("--curves")
    command.extend([str(base_run_dir), str(reload_run_dir)])
    return command


def run_item(
    *,
    name: str,
    command: list[str],
    run_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path = run_dir / "process.log"
    returncode, elapsed = run_command(command, log_path, dry_run=dry_run)
    return {
        "name": name,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
        "shell_command": shell_command(command),
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "status": "dry_run" if dry_run else ("ok" if returncode == 0 else "failed"),
        "summary_path": str(run_dir / "summary.json"),
        "run_json_path": str(run_dir / "run.json"),
    }


def run_summary_contract(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    run_json_path = run_dir / "run.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    run_json = read_json(run_json_path) if run_json_path.exists() else {}
    final_nll = metric_value(summary, "final_validation", "mean_nll")
    best_nll = finite_float(summary.get("best_validation_mean_nll"))
    initial_nll = metric_value(summary, "initial_validation", "mean_nll")
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "run_json_path": str(run_json_path),
        "run_json_exists": run_json_path.exists(),
        "seed": run_json.get("seed"),
        "eval_seed": run_json.get("eval_seed"),
        "validation_sample_seed": run_json.get("validation_sample_seed"),
        "data_paths": run_json.get("data_paths"),
        "validation_tokens": run_json.get("validation_tokens"),
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "best_nll": best_nll,
        "validation_nll_delta": finite_float(summary.get("validation_nll_delta")),
        "final_minus_best_nll": finite_float(
            summary.get("final_minus_best_validation_nll")
        ),
        "best_epoch": summary.get("best_validation_epoch"),
        "early_stopped_epoch": summary.get("early_stopped_epoch"),
        "epochs_completed": summary.get("epochs_completed"),
        "restore_best_at_end": summary.get("restore_best_at_end") is True,
        "restored_best_at_end": summary.get("restored_best_at_end") is True,
        "best_checkpoint_exists": summary.get("best_checkpoint_exists") is True,
    }


def reload_pair_outcome(base_run_dir: Path, reload_run_dir: Path) -> dict[str, Any]:
    base = run_summary_contract(base_run_dir)
    reload = run_summary_contract(reload_run_dir)
    reload_best_minus_base_best = nll_delta(reload["best_nll"], base["best_nll"])
    reload_final_minus_base_final = nll_delta(reload["final_nll"], base["final_nll"])
    reload_best_minus_reload_initial = nll_delta(
        reload["best_nll"], reload["initial_nll"]
    )
    reload_final_minus_reload_initial = nll_delta(
        reload["final_nll"], reload["initial_nll"]
    )
    status = nll_status(reload_best_minus_base_best)
    issues: list[str] = []
    if base["summary_exists"] is not True:
        issues.append("missing_base_summary")
    if reload["summary_exists"] is not True:
        issues.append("missing_reload_summary")
    if reload_best_minus_base_best is None:
        issues.append("missing_best_nll_pair")
    comparison_issues: list[str] = []
    if (
        base.get("eval_seed") is not None
        and reload.get("eval_seed") is not None
        and base.get("eval_seed") != reload.get("eval_seed")
    ):
        comparison_issues.append("eval_seed_mismatch")
    if (
        base.get("data_paths") is not None
        and reload.get("data_paths") is not None
        and base.get("data_paths") != reload.get("data_paths")
    ):
        comparison_issues.append("data_paths_mismatch")
    issues.extend(comparison_issues)
    evaluation_comparable = not comparison_issues
    if not evaluation_comparable:
        status = "unknown"

    return {
        "schema": OUTCOME_SCHEMA,
        "base": base,
        "reload": reload,
        "status": status,
        "issues": issues,
        "ready": not issues,
        "evaluation_comparable": evaluation_comparable,
        "comparison_issues": comparison_issues,
        "reload_improved_best": status == "improved",
        "reload_regressed_best": status == "regressed",
        "reload_best_minus_base_best_nll": reload_best_minus_base_best,
        "reload_final_minus_base_final_nll": reload_final_minus_base_final,
        "reload_best_minus_reload_initial_nll": reload_best_minus_reload_initial,
        "reload_final_minus_reload_initial_nll": reload_final_minus_reload_initial,
    }


def positive_int(value: int, *, label: str, allow_zero: bool = False) -> int:
    if value < 0 or (value == 0 and not allow_zero):
        suffix = "non-negative" if allow_zero else "positive"
        raise ValueError(f"--{label} must be {suffix}")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Python char-LM scratch base and reload finetune pair."
    )
    parser.add_argument("data_paths", nargs="+", type=Path)
    parser.add_argument(
        "--reload-data",
        dest="reload_data_paths",
        action="append",
        type=Path,
        default=[],
        help="text file or corpus directory for the reload FT stage; repeatable",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reload-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=2)
    parser.add_argument("--reload-epochs", type=int, default=2)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--reload-lr", type=float, default=None)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gen", type=int, default=120)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--restore-best-at-end", action="store_true")
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=8)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="write preflight.json/reload_pair.json and exit before launching training",
    )
    parser.add_argument(
        "--ignore-preflight",
        action="store_true",
        help="launch training even when the readiness preflight reports missing symbols",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_int(args.base_epochs, label="base-epochs", allow_zero=True)
    positive_int(args.reload_epochs, label="reload-epochs", allow_zero=True)
    positive_int(args.batches, label="batches")
    positive_int(args.batch, label="batch")
    positive_int(args.steps, label="steps")
    positive_int(args.embed_dim, label="embed-dim")
    positive_int(args.hidden, label="hidden")
    positive_int(args.eval_samples, label="eval-samples", allow_zero=True)
    positive_int(args.gen, label="gen", allow_zero=True)
    positive_int(args.topk, label="topk")
    positive_int(args.early_stop_patience, label="early-stop-patience", allow_zero=True)
    positive_int(args.summary_limit, label="summary-limit", allow_zero=True)
    if not (0.0 <= float(args.val_split) < 0.95):
        raise ValueError("--val-split must be within [0, 0.95)")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive")
    if args.reload_lr is not None and float(args.reload_lr) <= 0.0:
        raise ValueError("--reload-lr must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_root = args.run_root.resolve() if args.run_root else default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    base_run_dir = run_root / "base_scratch"
    reload_run_dir = run_root / "reload_finetune"
    reload_data_paths = list(args.reload_data_paths) or list(args.data_paths)
    reload_seed = args.reload_seed if args.reload_seed is not None else args.seed + 1
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    reload_lr = args.reload_lr if args.reload_lr is not None else args.lr

    base_command = finetune_command(
        list(args.data_paths),
        run_dir=base_run_dir,
        epochs=args.base_epochs,
        batches=args.batches,
        batch=args.batch,
        steps=args.steps,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        lr=args.lr,
        eval_samples=args.eval_samples,
        val_split=args.val_split,
        gen=args.gen,
        topk=args.topk,
        seed=args.seed,
        eval_seed=eval_seed,
        backend=args.backend,
        early_stop_patience=args.early_stop_patience,
        restore_best_at_end=args.restore_best_at_end,
    )
    reload_command = finetune_command(
        reload_data_paths,
        run_dir=reload_run_dir,
        epochs=args.reload_epochs,
        batches=args.batches,
        batch=args.batch,
        steps=args.steps,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        lr=reload_lr,
        eval_samples=args.eval_samples,
        val_split=args.val_split,
        gen=args.gen,
        topk=args.topk,
        seed=reload_seed,
        eval_seed=eval_seed,
        backend=args.backend,
        early_stop_patience=args.early_stop_patience,
        restore_best_at_end=args.restore_best_at_end,
        load_run=base_run_dir,
    )
    planned_compare_command = compare_command(
        base_run_dir,
        reload_run_dir,
        run_root=run_root,
        curves=args.curves,
    )
    manifest_path = run_root / "reload_pair.json"
    preflight_path = run_root / "preflight.json"
    started = time.time()
    preflight = run_finetune_preflight(
        list(args.data_paths),
        run_root=run_root,
        backend=args.backend,
    )
    write_json(preflight_path, preflight)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "started_at_unix": started,
        "run_root": str(run_root),
        "dry_run": bool(args.dry_run),
        "base_run_dir": str(base_run_dir),
        "reload_run_dir": str(reload_run_dir),
        "preflight_path": str(preflight_path),
        "preflight": preflight,
        "data_paths": [str(path) for path in args.data_paths],
        "reload_data_paths": [str(path) for path in reload_data_paths],
        "base_seed": int(args.seed),
        "reload_seed": int(reload_seed),
        "eval_seed": int(eval_seed),
        "settings": {
            "backend": args.backend,
            "base_epochs": args.base_epochs,
            "reload_epochs": args.reload_epochs,
            "batches": args.batches,
            "batch": args.batch,
            "steps": args.steps,
            "embed_dim": args.embed_dim,
            "hidden": args.hidden,
            "lr": args.lr,
            "reload_lr": reload_lr,
            "eval_samples": args.eval_samples,
            "val_split": args.val_split,
            "gen": args.gen,
            "topk": args.topk,
            "eval_seed": int(eval_seed),
            "early_stop_patience": int(args.early_stop_patience),
            "restore_best_at_end": bool(args.restore_best_at_end),
        },
        "runs": [],
        "compare_command": planned_compare_command,
        "compare_shell_command": shell_command(planned_compare_command),
        "compare_path": None,
        "compare_json_path": None,
        "compare_summary_path": None,
        "compare_summary_json_path": None,
        "outcome_path": None,
        "outcome": None,
        "failed": False,
        "preflight_only": bool(args.preflight_only),
        "ignore_preflight": bool(args.ignore_preflight),
        "preflight_blocked": False,
    }
    write_json(manifest_path, manifest)
    if args.preflight_only:
        if preflight.get("ready") is not True:
            manifest["failed"] = True
        manifest["finished_at_unix"] = time.time()
        manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
        write_json(manifest_path, manifest)
        print(f"preflight: {preflight_path}")
        print(f"manifest: {manifest_path}")
        return 0 if preflight.get("ready") is True else 1
    if not args.dry_run and not args.ignore_preflight and preflight.get("ready") is not True:
        manifest["failed"] = True
        manifest["preflight_blocked"] = True
        manifest["finished_at_unix"] = time.time()
        manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
        write_json(manifest_path, manifest)
        print(f"preflight failed: {preflight_path}", file=sys.stderr)
        print(f"manifest: {manifest_path}")
        return 1

    successful_run_dirs: list[Path] = []
    for name, command, run_dir in (
        ("base_scratch", base_command, base_run_dir),
        ("reload_finetune", reload_command, reload_run_dir),
    ):
        item = run_item(name=name, command=command, run_dir=run_dir, dry_run=args.dry_run)
        manifest["runs"].append(item)
        if item["returncode"] == 0:
            successful_run_dirs.append(run_dir)
        else:
            manifest["failed"] = True
            break
        write_json(manifest_path, manifest)

    compare_output = None
    compare_summary_output = None
    if not args.dry_run and not manifest["failed"] and len(successful_run_dirs) == 2:
        compare_output = sweep.render_compare(
            successful_run_dirs,
            run_root,
            curves=args.curves,
        )
        if compare_output is not None:
            compare_summary_output = sweep.render_compare_summary(
                run_root,
                options=sweep.CompareSummaryOptions(
                    limit=args.summary_limit,
                    route_clean_only=False,
                    prefer_clean_route=False,
                    sort_metric="final_nll",
                ),
            )
            if compare_summary_output is None:
                manifest["failed"] = True
        outcome = reload_pair_outcome(base_run_dir, reload_run_dir)
        outcome_path = run_root / "outcome.json"
        write_json(outcome_path, outcome)
        manifest["outcome_path"] = str(outcome_path)
        manifest["outcome"] = outcome

    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
    manifest["compare_path"] = (
        str(run_root / "compare.md") if compare_output is not None else None
    )
    manifest["compare_json_path"] = (
        str(run_root / "compare.json") if compare_output is not None else None
    )
    manifest["compare_summary_path"] = (
        str(run_root / "compare_summary.md")
        if (run_root / "compare_summary.md").exists()
        else None
    )
    compare_summary_json = run_root / "compare_summary.json"
    manifest["compare_summary_json_path"] = (
        str(compare_summary_json) if compare_summary_json.exists() else None
    )
    write_json(manifest_path, manifest)

    print(f"manifest: {manifest_path}")
    if manifest["compare_path"] is not None:
        print(f"compare: {manifest['compare_path']}")
    if manifest["compare_summary_path"] is not None:
        print(f"compare_summary: {manifest['compare_summary_path']}")
    if manifest["outcome_path"] is not None:
        print(f"outcome: {manifest['outcome_path']}")
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

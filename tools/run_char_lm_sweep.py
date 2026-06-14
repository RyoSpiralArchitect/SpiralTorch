#!/usr/bin/env python3
"""Run reproducible Rust char-LM model-zoo sweeps.

The script intentionally stays dependency-free: it shells out to the existing
Rust examples, captures per-run logs, writes a machine-readable sweep manifest,
and renders a Markdown comparison table with tools/compare_char_lm_runs.py.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = {
    "finetune": "modelzoo_llm_char_finetune",
    "scan": "modelzoo_llm_char_coherence_scan",
    "wave": "modelzoo_llm_char_coherence_wave",
}

PRESETS = {
    "smoke": {
        "epochs": 1,
        "batches": 2,
        "batch": 4,
        "eval_samples": 16,
        "gen": 16,
        "early_stop_patience": 0,
    },
    "small": {
        "epochs": 3,
        "batches": 8,
        "batch": 4,
        "eval_samples": 64,
        "gen": 64,
        "early_stop_patience": 2,
    },
    "base": {
        "epochs": 6,
        "batches": 24,
        "batch": 8,
        "eval_samples": 256,
        "gen": 200,
        "early_stop_patience": 3,
    },
}


@dataclass(frozen=True)
class SweepSettings:
    epochs: int
    batches: int
    batch: int
    eval_samples: int
    gen: int
    early_stop_patience: int
    steps: int | None
    embed_dim: int | None
    hidden: int | None
    memory: int | None
    lr: float | None
    curvature: float | None
    temperature: float | None
    backend: str


def parse_csv(raw: str, *, label: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    return values


def parse_csv_int(raw: str, *, label: str) -> list[int]:
    values = []
    for part in parse_csv(raw, label=label):
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid --{label} entry: {part}") from exc
    return values


def slug(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("-")
    compact = "".join(safe).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "value"


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return REPO_ROOT / "models" / "runs" / f"char_lm_sweep_{stamp}"


def settings_from_args(args: argparse.Namespace) -> SweepSettings:
    preset = PRESETS[args.preset]
    settings = SweepSettings(
        epochs=preset["epochs"],
        batches=preset["batches"],
        batch=preset["batch"],
        eval_samples=preset["eval_samples"],
        gen=preset["gen"],
        early_stop_patience=preset["early_stop_patience"],
        steps=args.steps,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        memory=args.memory,
        lr=args.lr,
        curvature=args.curvature,
        temperature=args.temperature,
        backend=args.backend,
    )
    for field in ("epochs", "batches", "batch", "eval_samples", "gen", "early_stop_patience"):
        value = getattr(args, field)
        if value is not None:
            settings = replace(settings, **{field: value})
    return settings


def add_optional_int(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_optional_float(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, f"{value:g}"])


def build_command(
    *,
    cargo_bin: str,
    architecture: str,
    data_paths: list[Path],
    run_dir: Path,
    char_feature: str,
    head_prior: str,
    seed: int,
    settings: SweepSettings,
    extra_args: list[str],
) -> list[str]:
    command = [
        cargo_bin,
        "run",
        "-p",
        "st-nn",
        "--example",
        EXAMPLES[architecture],
        "--",
    ]
    command.extend(str(path) for path in data_paths)
    command.extend(
        [
            "--backend",
            settings.backend,
            "--run-dir",
            str(run_dir),
            "--head-prior",
            head_prior,
            "--char-feature",
            char_feature,
            "--epochs",
            str(settings.epochs),
            "--batches",
            str(settings.batches),
            "--batch",
            str(settings.batch),
            "--eval-samples",
            str(settings.eval_samples),
            "--early-stop-patience",
            str(settings.early_stop_patience),
            "--gen",
            str(settings.gen),
            "--seed",
            str(seed),
        ]
    )
    add_optional_int(command, "--steps", settings.steps)
    add_optional_int(command, "--embed-dim", settings.embed_dim)
    add_optional_int(command, "--hidden", settings.hidden)
    if architecture in {"scan", "wave"}:
        add_optional_int(command, "--memory", settings.memory)
    add_optional_float(command, "--lr", settings.lr)
    add_optional_float(command, "--curvature", settings.curvature)
    add_optional_float(command, "--temperature", settings.temperature)
    command.extend(extra_args)
    return command


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shlex.join(command) + "\n")
        log_file.flush()
        if dry_run:
            log_file.write("[dry-run] command not executed\n")
            return 0, 0.0
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode, time.time() - started


def read_json(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if isinstance(value, dict):
        return value
    return None


def render_compare(run_dirs: Iterable[Path], run_root: Path, *, curves: bool) -> str | None:
    run_dirs = list(run_dirs)
    if not run_dirs:
        return None
    compare_script = REPO_ROOT / "tools" / "compare_char_lm_runs.py"
    command = [sys.executable, "-S", "-s", str(compare_script)]
    if curves:
        command.append("--curves")
    command.extend(str(path) for path in run_dirs)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    compare_path = run_root / "compare.md"
    compare_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        error_path = run_root / "compare.error.log"
        error_path.write_text(result.stderr, encoding="utf-8")
        return None
    return result.stdout


def write_sweep_manifest(run_root: Path, payload: dict[str, object]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_choices(values: list[str], allowed: set[str], *, label: str) -> None:
    invalid = [value for value in values if value not in allowed]
    if invalid:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"invalid --{label}: {', '.join(invalid)} (expected {expected})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Rust char-LM model-zoo sweeps and compare summary artifacts."
    )
    parser.add_argument("data_paths", nargs="+", type=Path, help="text files or corpus directories")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="sweep output directory (default: models/runs/char_lm_sweep_<timestamp>)",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument(
        "--architectures",
        default="finetune,scan,wave",
        help="comma-separated: finetune,scan,wave",
    )
    parser.add_argument(
        "--features",
        default="token-bigram",
        help="comma-separated: token,token-bigram",
    )
    parser.add_argument(
        "--head-priors",
        default="learned-unigram",
        help="comma-separated: none,unigram,learned-unigram",
    )
    parser.add_argument("--seeds", default="42", help="comma-separated integer seeds")
    parser.add_argument("--backend", default="cpu", help="auto|wgpu|cuda|hip|cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--cargo-bin", default="cargo")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra arg passed to every Rust example")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--curves", action="store_true", help="include epoch curves in compare.md")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        architectures = parse_csv(args.architectures, label="architectures")
        features = parse_csv(args.features, label="features")
        head_priors = parse_csv(args.head_priors, label="head-priors")
        seeds = parse_csv_int(args.seeds, label="seeds")
        validate_choices(architectures, set(EXAMPLES), label="architectures")
        validate_choices(features, {"token", "token-bigram"}, label="features")
        validate_choices(head_priors, {"none", "unigram", "learned-unigram"}, label="head-priors")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    settings = settings_from_args(args)
    run_root = args.run_root.resolve() if args.run_root else default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    runs: list[dict[str, object]] = []
    successful_run_dirs: list[Path] = []
    failed = False

    manifest: dict[str, object] = {
        "schema": "st.char_lm_sweep.v1",
        "started_at_unix": started,
        "run_root": str(run_root),
        "preset": args.preset,
        "settings": settings.__dict__,
        "data_paths": [str(path) for path in args.data_paths],
        "architectures": architectures,
        "features": features,
        "head_priors": head_priors,
        "seeds": seeds,
        "dry_run": args.dry_run,
        "runs": runs,
    }
    write_sweep_manifest(run_root, manifest)

    total = len(architectures) * len(features) * len(head_priors) * len(seeds)
    index = 0
    for architecture in architectures:
        for feature in features:
            for head_prior in head_priors:
                for seed in seeds:
                    index += 1
                    run_name = (
                        f"{slug(architecture)}__feature-{slug(feature)}__"
                        f"head-{slug(head_prior)}__seed-{seed}"
                    )
                    run_dir = run_root / run_name
                    log_path = run_dir / "process.log"
                    command = build_command(
                        cargo_bin=args.cargo_bin,
                        architecture=architecture,
                        data_paths=args.data_paths,
                        run_dir=run_dir,
                        char_feature=feature,
                        head_prior=head_prior,
                        seed=seed,
                        settings=settings,
                        extra_args=args.extra_arg,
                    )
                    skipped = False
                    if args.skip_existing and (run_dir / "summary.json").exists():
                        returncode = 0
                        elapsed = 0.0
                        skipped = True
                    else:
                        print(f"[{index}/{total}] {run_name}")
                        print("  " + shlex.join(command))
                        returncode, elapsed = run_command(command, log_path, dry_run=args.dry_run)
                    summary = read_json(run_dir / "summary.json")
                    run_record: dict[str, object] = {
                        "name": run_name,
                        "architecture": architecture,
                        "example": EXAMPLES[architecture],
                        "char_feature": feature,
                        "head_prior": head_prior,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "log_path": str(log_path),
                        "command": command,
                        "returncode": returncode,
                        "elapsed_seconds": elapsed,
                        "skipped": skipped,
                        "summary_path": str(run_dir / "summary.json"),
                        "has_summary": summary is not None,
                    }
                    if isinstance(summary, dict):
                        run_record["best_validation_mean_nll"] = summary.get(
                            "best_validation_mean_nll"
                        )
                        final = summary.get("final_validation")
                        if isinstance(final, dict):
                            run_record["final_validation_mean_nll"] = final.get("mean_nll")
                    runs.append(run_record)
                    missing_summary = returncode == 0 and summary is None and not args.dry_run
                    if returncode == 0 and summary is not None:
                        successful_run_dirs.append(run_dir)
                    elif returncode != 0 or missing_summary:
                        failed = True
                        if not args.continue_on_error:
                            write_sweep_manifest(run_root, manifest)
                            if missing_summary:
                                print(
                                    f"run produced no summary.json: {run_name}; see {log_path}",
                                    file=sys.stderr,
                                )
                            else:
                                print(f"run failed: {run_name}; see {log_path}", file=sys.stderr)
                            return returncode or 1
                    write_sweep_manifest(run_root, manifest)

    compare_output = None
    if not args.dry_run:
        compare_output = render_compare(successful_run_dirs, run_root, curves=args.curves)
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
    manifest["compare_path"] = str(run_root / "compare.md") if compare_output is not None else None
    manifest["failed"] = failed
    write_sweep_manifest(run_root, manifest)

    print(f"sweep: {run_root}")
    print(f"manifest: {run_root / 'sweep.json'}")
    if compare_output is not None:
        print(f"compare: {run_root / 'compare.md'}")
        print(compare_output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

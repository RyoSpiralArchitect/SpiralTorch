#!/usr/bin/env python3
"""Run Z-Space generation-control sweeps for FT checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

EXAMPLES_ROOT = Path(__file__).resolve().parent
SWEEP_SCRIPT = EXAMPLES_ROOT / "hf_gpt2_zspace_generation_control_sweep.py"
COMPARE_SCRIPT = EXAMPLES_ROOT / "hf_gpt2_zspace_generation_control_compare.py"

DEFAULT_PROMPTS: tuple[tuple[str, str, str], ...] = (
    (
        "spiral",
        "SpiralTorch is a geometry-aware learning system that",
        "",
    ),
    (
        "desire-coherence",
        "In Z-space, desire and coherence shape language by",
        "prompt-desire-coherence-",
    ),
    (
        "tokenless-ft",
        "A tokenless fine-tuning stack should preserve meaning while",
        "prompt-tokenless-ft-",
    ),
)


@dataclass(frozen=True)
class PromptSpec:
    label: str
    prompt: str
    filename_prefix: str


@dataclass(frozen=True)
class SweepJob:
    checkpoint: str
    prompt: PromptSpec
    model_dir: Path
    out: Path
    label: str


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in slug.split("-") if part) or "prompt"


def _checkpoint_slug(checkpoint: str) -> str:
    return _slugify(checkpoint.replace("checkpoint-", "ckpt-"))


def _checkpoint_token(checkpoint: str) -> str:
    if checkpoint.startswith("checkpoint-"):
        return _slugify(checkpoint[len("checkpoint-") :])
    return _checkpoint_slug(checkpoint)


def _prompt_spec(value: str) -> PromptSpec:
    if "::" not in value:
        raise argparse.ArgumentTypeError(
            "--prompt must use LABEL::TEXT so output labels stay stable"
        )
    label, prompt = value.split("::", 1)
    label = _slugify(label)
    prompt = prompt.strip()
    if not prompt:
        raise argparse.ArgumentTypeError("prompt text must not be empty")
    return PromptSpec(
        label=label,
        prompt=prompt,
        filename_prefix=f"prompt-{label}-",
    )


def default_prompt_specs() -> list[PromptSpec]:
    return [
        PromptSpec(label=label, prompt=prompt, filename_prefix=filename_prefix)
        for label, prompt, filename_prefix in DEFAULT_PROMPTS
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint directory name under --run-dir, e.g. checkpoint-2048.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        type=_prompt_spec,
        default=None,
        help="Prompt spec as LABEL::TEXT. Defaults to SpiralTorch's 3 prompt set.",
    )
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sweep-script", type=Path, default=SWEEP_SCRIPT)
    parser.add_argument("--compare-script", type=Path, default=COMPARE_SCRIPT)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sample-temperature", type=float, default=None)
    parser.add_argument("--sample-top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--compare-out", type=Path, default=None)
    parser.add_argument("--compare-lines-out", type=Path, default=None)
    parser.add_argument(
        "--compare-with-sweep",
        action="append",
        type=Path,
        default=None,
        help="Existing sweep JSON to include in the comparison.",
    )
    parser.add_argument(
        "--compare-with-label",
        action="append",
        default=None,
        help="Label for the matching --compare-with-sweep path.",
    )
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--ready-file",
        action="append",
        default=None,
        help=(
            "Relative file that must exist under the checkpoint before sweeping. "
            "Defaults to model.safetensors. May be repeated."
        ),
    )
    parser.add_argument("--no-ready-file-check", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum wait time when --wait is set. 0 means wait forever.",
    )
    args = parser.parse_args(argv)
    args.checkpoint = list(args.checkpoint or [])
    args.prompt = list(args.prompt or [])
    args.compare_with_sweep = list(args.compare_with_sweep or [])
    args.compare_with_label = list(args.compare_with_label or [])
    args.ready_file = [] if args.no_ready_file_check else list(args.ready_file or ["model.safetensors"])
    if not args.checkpoint:
        parser.error("--checkpoint must be provided at least once")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.sample_temperature is not None and args.sample_temperature <= 0.0:
        parser.error("--sample-temperature must be positive")
    if args.sample_top_k is not None and args.sample_top_k < 0:
        parser.error("--sample-top-k must be non-negative")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.timeout_seconds < 0.0:
        parser.error("--timeout-seconds must be non-negative")
    if args.compare_with_label and len(args.compare_with_label) != len(
        args.compare_with_sweep
    ):
        parser.error("--compare-with-label must match --compare-with-sweep count")
    return args


def prompt_specs(args: argparse.Namespace) -> list[PromptSpec]:
    return list(args.prompt) if args.prompt else default_prompt_specs()


def _sweep_out_path(run_dir: Path, checkpoint: str, prompt: PromptSpec) -> Path:
    return run_dir / f"{prompt.filename_prefix}{checkpoint}-generation-control-sweep.json"


def _job_label(args: argparse.Namespace, checkpoint: str, prompt: PromptSpec) -> str:
    parts = [part for part in (args.label_prefix, prompt.label, checkpoint) if part]
    return "-".join(_slugify(part) for part in parts)


def build_sweep_jobs(args: argparse.Namespace) -> list[SweepJob]:
    run_dir = args.run_dir
    jobs: list[SweepJob] = []
    for checkpoint in args.checkpoint:
        model_dir = run_dir / checkpoint
        for prompt in prompt_specs(args):
            jobs.append(
                SweepJob(
                    checkpoint=checkpoint,
                    prompt=prompt,
                    model_dir=model_dir,
                    out=_sweep_out_path(run_dir, checkpoint, prompt),
                    label=_job_label(args, checkpoint, prompt),
                )
            )
    return jobs


def build_sweep_command(args: argparse.Namespace, job: SweepJob) -> list[str]:
    command = [
        str(args.python),
        str(args.sweep_script),
        "--model-name",
        str(job.model_dir),
        "--prompt",
        job.prompt.prompt,
        "--out",
        str(job.out),
    ]
    if args.allow_remote:
        command.append("--allow-remote")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.do_sample:
        command.append("--do-sample")
    if args.sample_temperature is not None:
        command.extend(["--sample-temperature", str(args.sample_temperature)])
    if args.sample_top_k is not None:
        command.extend(["--sample-top-k", str(args.sample_top_k)])
    return command


def _default_compare_stem(checkpoints: Sequence[str]) -> str:
    if len(checkpoints) == 1:
        return f"generation-control-compare-3prompt-{_checkpoint_token(checkpoints[0])}"
    first = _checkpoint_token(checkpoints[0])
    last = _checkpoint_token(checkpoints[-1])
    return f"generation-control-compare-3prompt-{first}-to-{last}"


def compare_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = _default_compare_stem(args.checkpoint)
    out = args.compare_out or args.run_dir / f"{stem}.json"
    lines_out = args.compare_lines_out or args.run_dir / f"{stem}.txt"
    return out, lines_out


def build_compare_command(
    args: argparse.Namespace,
    jobs: Sequence[SweepJob],
) -> list[str]:
    compare_out, compare_lines_out = compare_output_paths(args)
    paths = [str(path) for path in args.compare_with_sweep]
    labels = list(args.compare_with_label)
    for job in jobs:
        paths.append(str(job.out))
        labels.append(job.label)
    command = [
        str(args.python),
        str(args.compare_script),
        *paths,
    ]
    for label in labels:
        command.extend(["--label", label])
    command.extend(
        [
            "--out",
            str(compare_out),
            "--lines-out",
            str(compare_lines_out),
            "--top-n",
            str(args.top_n),
        ]
    )
    return command


def _command_row(command: Sequence[str]) -> str:
    return " ".join(command)


def _wait_for_model_dir(args: argparse.Namespace, job: SweepJob) -> None:
    if args.dry_run or _checkpoint_ready(args, job):
        return
    if not args.wait:
        raise FileNotFoundError(_checkpoint_not_ready_message(args, job))
    deadline = None
    if args.timeout_seconds > 0.0:
        deadline = time.monotonic() + float(args.timeout_seconds)
    while not _checkpoint_ready(args, job):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(_checkpoint_not_ready_message(args, job))
        time.sleep(float(args.poll_seconds))


def _missing_ready_files(args: argparse.Namespace, job: SweepJob) -> list[Path]:
    if not job.model_dir.is_dir():
        return [job.model_dir]
    return [
        job.model_dir / str(ready_file)
        for ready_file in args.ready_file
        if not (job.model_dir / str(ready_file)).is_file()
    ]


def _checkpoint_ready(args: argparse.Namespace, job: SweepJob) -> bool:
    return not _missing_ready_files(args, job)


def _checkpoint_not_ready_message(args: argparse.Namespace, job: SweepJob) -> str:
    missing = ", ".join(str(path) for path in _missing_ready_files(args, job))
    return f"checkpoint is not ready: {job.model_dir}; missing {missing}"


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str] | None]


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=True, text=True)


def run_checkpoint_generation_control(
    args: argparse.Namespace,
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    runner = runner or _run_command
    jobs = build_sweep_jobs(args)
    rows: list[dict[str, Any]] = []
    runnable_compare_jobs: list[SweepJob] = []
    for job in jobs:
        command = build_sweep_command(args, job)
        row: dict[str, Any] = {
            "checkpoint": job.checkpoint,
            "label": job.label,
            "prompt_label": job.prompt.label,
            "prompt": job.prompt.prompt,
            "model_name": str(job.model_dir),
            "out": str(job.out),
            "command": list(command),
        }
        if job.out.is_file() and not args.overwrite:
            row["status"] = "skipped_existing"
            runnable_compare_jobs.append(job)
        elif args.dry_run:
            row["status"] = "planned"
            runnable_compare_jobs.append(job)
        else:
            _wait_for_model_dir(args, job)
            job.out.parent.mkdir(parents=True, exist_ok=True)
            print("checkpoint_generation_control_sweep", _command_row(command))
            runner(command)
            row["status"] = "complete"
            runnable_compare_jobs.append(job)
        rows.append(row)

    compare_row: dict[str, Any] | None = None
    if not args.no_compare:
        compare_command = build_compare_command(args, runnable_compare_jobs)
        compare_out, compare_lines_out = compare_output_paths(args)
        compare_row = {
            "out": str(compare_out),
            "lines_out": str(compare_lines_out),
            "command": list(compare_command),
        }
        if args.dry_run:
            compare_row["status"] = "planned"
        else:
            print("checkpoint_generation_control_compare", _command_row(compare_command))
            runner(compare_command)
            compare_row["status"] = "complete"

    status = "planned" if args.dry_run else "complete"
    if (
        not args.dry_run
        and rows
        and all(row["status"] == "skipped_existing" for row in rows)
    ):
        status = "complete_with_existing_sweeps" if compare_row else "skipped_existing"
    report: dict[str, Any] = {
        "row_type": "hf_gpt2_ft_checkpoint_generation_control",
        "status": status,
        "dry_run": bool(args.dry_run),
        "run_dir": str(args.run_dir),
        "checkpoint_count": len(args.checkpoint),
        "prompt_count": len(prompt_specs(args)),
        "sweep_count": len(rows),
        "sweeps": rows,
    }
    if compare_row is not None:
        report["compare"] = compare_row
    if args.run_card is not None:
        args.run_card.parent.mkdir(parents=True, exist_ok=True)
        args.run_card.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"checkpoint_generation_control_run_card {args.run_card}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checkpoint_generation_control(args)
    if args.dry_run and args.run_card is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

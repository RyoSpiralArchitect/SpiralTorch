#!/usr/bin/env python3
"""Run Z-Space generation-control sweeps for FT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
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
CURVE_SCRIPT = EXAMPLES_ROOT / "hf_gpt2_ft_generation_curve.py"

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
    parser.add_argument("--curve-script", type=Path, default=CURVE_SCRIPT)
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
    parser.add_argument("--curve-out", type=Path, default=None)
    parser.add_argument("--curve-lines-out", type=Path, default=None)
    parser.add_argument("--curve-run-card", type=Path, default=None)
    parser.add_argument("--curve-trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--curve-model-name", default=None)
    parser.add_argument("--curve-dataset-name", default=None)
    parser.add_argument("--curve-dataset-config", default=None)
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
    parser.add_argument(
        "--wait-for-process-pid-file",
        type=Path,
        default=None,
        help="Wait for the PID in this file to exit before checking checkpoints.",
    )
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
    parser.add_argument("--process-poll-seconds", type=float, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum wait time when --wait is set. 0 means wait forever.",
    )
    parser.add_argument(
        "--process-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Maximum wait time for --wait-for-process-pid-file. "
            "0 means wait forever."
        ),
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
    if args.process_poll_seconds is not None and args.process_poll_seconds <= 0.0:
        parser.error("--process-poll-seconds must be positive")
    if args.timeout_seconds < 0.0:
        parser.error("--timeout-seconds must be non-negative")
    if args.process_timeout_seconds < 0.0:
        parser.error("--process-timeout-seconds must be non-negative")
    if args.compare_with_label and len(args.compare_with_label) != len(
        args.compare_with_sweep
    ):
        parser.error("--compare-with-label must match --compare-with-sweep count")
    if (
        (args.curve_run_card is not None or args.curve_trainer_trace_jsonl is not None)
        and args.curve_out is None
        and args.curve_lines_out is None
    ):
        parser.error("curve source options require --curve-out or --curve-lines-out")
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


def _curve_requested(args: argparse.Namespace) -> bool:
    return args.curve_out is not None or args.curve_lines_out is not None


def _default_curve_run_card(args: argparse.Namespace) -> Path | None:
    candidates = [
        args.curve_run_card,
        args.run_dir / "spiraltorch-hf-gpt2-ft-run-card.json",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _default_curve_trainer_trace(args: argparse.Namespace) -> Path | None:
    candidates = [
        args.curve_trainer_trace_jsonl,
        args.run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def build_curve_command(
    args: argparse.Namespace,
    jobs: Sequence[SweepJob],
) -> list[str]:
    paths = [str(path) for path in args.compare_with_sweep]
    labels = list(args.compare_with_label)
    for job in jobs:
        paths.append(str(job.out))
        labels.append(job.label)
    command = [
        str(args.python),
        str(args.curve_script),
        *paths,
    ]
    for label in labels:
        command.extend(["--label", label])
    run_card = _default_curve_run_card(args)
    trainer_trace = _default_curve_trainer_trace(args)
    if run_card is not None:
        command.extend(["--run-card", str(run_card)])
    if trainer_trace is not None:
        command.extend(["--trainer-trace-jsonl", str(trainer_trace)])
    command.extend(["--run-dir", str(args.run_dir)])
    if args.curve_model_name is not None:
        command.extend(["--model-name", str(args.curve_model_name)])
    if args.curve_dataset_name is not None:
        command.extend(["--dataset-name", str(args.curve_dataset_name)])
    if args.curve_dataset_config is not None:
        command.extend(["--dataset-config", str(args.curve_dataset_config)])
    if args.curve_out is not None:
        command.extend(["--out", str(args.curve_out)])
    if args.curve_lines_out is not None:
        command.extend(["--lines-out", str(args.curve_lines_out)])
    command.extend(["--top-n", str(args.top_n)])
    return command


def _command_row(command: Sequence[str]) -> str:
    return " ".join(command)


def _write_status_card(
    args: argparse.Namespace,
    status: str,
    **extra: Any,
) -> None:
    if args.run_card is None:
        return
    report: dict[str, Any] = {
        "row_type": "hf_gpt2_ft_checkpoint_generation_control",
        "status": status,
        "dry_run": bool(args.dry_run),
        "run_dir": str(args.run_dir),
        "checkpoint_count": len(args.checkpoint),
        "prompt_count": len(prompt_specs(args)),
        "time_unix_s": time.time(),
    }
    report.update(extra)
    args.run_card.parent.mkdir(parents=True, exist_ok=True)
    args.run_card.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_pid_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    try:
        pid = int(text)
    except ValueError as exc:
        raise ValueError(f"PID file does not contain an integer: {path}") from exc
    if pid <= 0:
        raise ValueError(f"PID file must contain a positive PID: {path}")
    return pid


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_exit(args: argparse.Namespace) -> dict[str, Any] | None:
    pid_file: Path | None = args.wait_for_process_pid_file
    if pid_file is None:
        return None
    row: dict[str, Any] = {"pid_file": str(pid_file)}
    if args.dry_run:
        row["status"] = "planned"
        return row
    pid = _read_pid_file(pid_file)
    row["pid"] = pid
    if not _process_alive(pid):
        row["status"] = "already_exited"
        _write_status_card(args, "process_already_exited", process_wait=row)
        return row
    started = time.monotonic()
    row["status"] = "waiting"
    row["started_unix_s"] = time.time()
    deadline = None
    if args.process_timeout_seconds > 0.0:
        deadline = time.monotonic() + float(args.process_timeout_seconds)
    poll_seconds = (
        float(args.process_poll_seconds)
        if args.process_poll_seconds is not None
        else float(args.poll_seconds)
    )
    print(f"checkpoint_generation_control_wait_process pid={pid} pid_file={pid_file}")
    _write_status_card(args, "waiting_for_process", process_wait=row)
    while _process_alive(pid):
        row["waited_seconds"] = time.monotonic() - started
        row["last_heartbeat_unix_s"] = time.time()
        _write_status_card(args, "waiting_for_process", process_wait=row)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for process {pid} from {pid_file}")
        time.sleep(poll_seconds)
    row["status"] = "complete"
    row["waited_seconds"] = time.monotonic() - started
    row["completed_unix_s"] = time.time()
    _write_status_card(args, "process_exited", process_wait=row)
    return row


def _wait_for_model_dir(args: argparse.Namespace, job: SweepJob) -> None:
    if args.dry_run or _checkpoint_ready(args, job):
        return
    if not args.wait:
        raise FileNotFoundError(_checkpoint_not_ready_message(args, job))
    deadline = None
    if args.timeout_seconds > 0.0:
        deadline = time.monotonic() + float(args.timeout_seconds)
    started = time.monotonic()
    while not _checkpoint_ready(args, job):
        wait_row = {
            "checkpoint": job.checkpoint,
            "model_name": str(job.model_dir),
            "missing": [str(path) for path in _missing_ready_files(args, job)],
            "waited_seconds": time.monotonic() - started,
            "last_heartbeat_unix_s": time.time(),
        }
        _write_status_card(
            args,
            "waiting_for_checkpoint",
            checkpoint_wait=wait_row,
        )
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(_checkpoint_not_ready_message(args, job))
        time.sleep(float(args.poll_seconds))
    _write_status_card(
        args,
        "checkpoint_ready",
        checkpoint_wait={
            "checkpoint": job.checkpoint,
            "model_name": str(job.model_dir),
            "waited_seconds": time.monotonic() - started,
            "completed_unix_s": time.time(),
        },
    )


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
    process_wait = _wait_for_process_exit(args)
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

    curve_row: dict[str, Any] | None = None
    if _curve_requested(args):
        curve_command = build_curve_command(args, runnable_compare_jobs)
        curve_row = {
            "out": None if args.curve_out is None else str(args.curve_out),
            "lines_out": (
                None if args.curve_lines_out is None else str(args.curve_lines_out)
            ),
            "command": list(curve_command),
        }
        if args.dry_run:
            curve_row["status"] = "planned"
        else:
            print("checkpoint_generation_control_curve", _command_row(curve_command))
            runner(curve_command)
            curve_row["status"] = "complete"

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
    if curve_row is not None:
        report["curve"] = curve_row
    if process_wait is not None:
        report["process_wait"] = process_wait
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

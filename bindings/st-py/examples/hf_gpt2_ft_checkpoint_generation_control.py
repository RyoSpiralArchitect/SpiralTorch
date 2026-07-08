#!/usr/bin/env python3
"""Run Z-Space generation-control sweeps for FT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st  # noqa: E402
from spiraltorch.hf_generation import (  # noqa: E402
    default_zspace_checkpoint_generation_prompts,
    zspace_checkpoint_generation_control_compare_command,
    zspace_checkpoint_generation_control_compare_output_paths,
    zspace_checkpoint_generation_control_curve_command,
    zspace_checkpoint_generation_control_jobs,
    zspace_checkpoint_generation_control_report,
    zspace_checkpoint_generation_control_sweep_command,
)

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
        PromptSpec(
            label=spec.label,
            prompt=spec.prompt,
            filename_prefix=spec.filename_prefix,
        )
        for spec in default_zspace_checkpoint_generation_prompts()
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument(
        "--model-configs",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with Hugging Face model profiles. The selected "
            "profile can provide tokenizer and generation defaults."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Model profile id to use for tokenizer and generation defaults.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help=(
            "Tokenizer id/path passed to checkpoint generation sweeps. Defaults "
            "to the selected model profile tokenizer when available."
        ),
    )
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
    _apply_model_profile_defaults(args, parser=parser, raw_argv=raw_argv)
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


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _profile_value(profile: Mapping[str, Any], section: str, key: str) -> Any:
    payload = profile.get(section)
    if isinstance(payload, Mapping):
        return payload.get(key)
    return None


def _set_profile_default(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    attr: str,
    value: Any,
    *flags: str,
) -> None:
    if value is None or _argv_has_option(raw_argv, *flags):
        return
    setattr(args, attr, value)


def _apply_model_profile_defaults(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    raw_argv: Sequence[str],
) -> None:
    args._hf_finetune_model_profile = None
    if args.model_configs is None and args.model_profile is None:
        return
    try:
        profile = st.resolve_hf_finetune_model_profile(
            args.model_configs,
            profile=args.model_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"failed to resolve model profile: {exc}")
    args._hf_finetune_model_profile = profile
    _set_profile_default(
        args,
        raw_argv,
        "tokenizer_name",
        profile.get("tokenizer_name"),
        "--tokenizer-name",
    )
    for attr, key, flag in (
        ("max_new_tokens", "max_new_tokens", "--max-new-tokens"),
        ("sample_temperature", "temperature", "--sample-temperature"),
        ("sample_top_k", "top_k", "--sample-top-k"),
    ):
        _set_profile_default(
            args,
            raw_argv,
            attr,
            _profile_value(profile, "generation", key),
            flag,
        )
    if (
        _profile_value(profile, "generation", "do_sample") is True
        and not _argv_has_option(raw_argv, "--do-sample")
    ):
        args.do_sample = True
    if (
        _profile_value(profile, "runtime", "allow_remote") is True
        and not _argv_has_option(raw_argv, "--allow-remote")
    ):
        args.allow_remote = True
    if (
        _profile_value(profile, "runtime", "trust_remote_code") is True
        and not _argv_has_option(raw_argv, "--trust-remote-code")
    ):
        args.trust_remote_code = True
    _set_profile_default(
        args,
        raw_argv,
        "curve_model_name",
        profile.get("model_name"),
        "--curve-model-name",
    )


def _resolved_model_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    profile = getattr(args, "_hf_finetune_model_profile", None)
    return dict(profile) if isinstance(profile, Mapping) else None


def prompt_specs(args: argparse.Namespace) -> list[PromptSpec]:
    return list(args.prompt) if args.prompt else default_prompt_specs()


def _sweep_out_path(run_dir: Path, checkpoint: str, prompt: PromptSpec) -> Path:
    return run_dir / f"{prompt.filename_prefix}{checkpoint}-generation-control-sweep.json"


def _job_label(args: argparse.Namespace, checkpoint: str, prompt: PromptSpec) -> str:
    parts = [part for part in (args.label_prefix, prompt.label, checkpoint) if part]
    return "-".join(_slugify(part) for part in parts)


def build_sweep_jobs(args: argparse.Namespace) -> list[SweepJob]:
    return zspace_checkpoint_generation_control_jobs(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        prompt=prompt_specs(args),
        label_prefix=args.label_prefix,
    )


def build_sweep_command(args: argparse.Namespace, job: SweepJob) -> list[str]:
    return zspace_checkpoint_generation_control_sweep_command(
        job,
        python=args.python,
        sweep_script=args.sweep_script,
        tokenizer_name=args.tokenizer_name,
        model_configs=args.model_configs,
        model_profile=args.model_profile,
        allow_remote=args.allow_remote,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        sample_temperature=args.sample_temperature,
        sample_top_k=args.sample_top_k,
    )


def _default_compare_stem(checkpoints: Sequence[str]) -> str:
    if len(checkpoints) == 1:
        return f"generation-control-compare-3prompt-{_checkpoint_token(checkpoints[0])}"
    first = _checkpoint_token(checkpoints[0])
    last = _checkpoint_token(checkpoints[-1])
    return f"generation-control-compare-3prompt-{first}-to-{last}"


def compare_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return zspace_checkpoint_generation_control_compare_output_paths(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        compare_out=args.compare_out,
        compare_lines_out=args.compare_lines_out,
    )


def build_compare_command(
    args: argparse.Namespace,
    jobs: Sequence[SweepJob],
) -> list[str]:
    return zspace_checkpoint_generation_control_compare_command(
        jobs,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        python=args.python,
        compare_script=args.compare_script,
        compare_with_sweep=args.compare_with_sweep,
        compare_with_label=args.compare_with_label,
        compare_out=args.compare_out,
        compare_lines_out=args.compare_lines_out,
        top_n=args.top_n,
    )


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
    return zspace_checkpoint_generation_control_curve_command(
        jobs,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        python=args.python,
        curve_script=args.curve_script,
        curve_out=args.curve_out,
        curve_lines_out=args.curve_lines_out,
        curve_run_card=args.curve_run_card,
        curve_trainer_trace_jsonl=args.curve_trainer_trace_jsonl,
        curve_model_name=args.curve_model_name,
        curve_dataset_name=args.curve_dataset_name,
        curve_dataset_config=args.curve_dataset_config,
        compare_with_sweep=args.compare_with_sweep,
        compare_with_label=args.compare_with_label,
        top_n=args.top_n,
    )


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
        "tokenizer_name": args.tokenizer_name,
        "model_configs": (
            None if args.model_configs is None else str(args.model_configs)
        ),
        "model_profile": _resolved_model_profile(args),
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
    return zspace_checkpoint_generation_control_report(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        prompt=prompt_specs(args),
        label_prefix=args.label_prefix,
        python=args.python,
        sweep_script=args.sweep_script,
        compare_script=args.compare_script,
        curve_script=args.curve_script,
        tokenizer_name=args.tokenizer_name,
        model_configs=args.model_configs,
        model_profile=args.model_profile,
        allow_remote=args.allow_remote,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        sample_temperature=args.sample_temperature,
        sample_top_k=args.sample_top_k,
        overwrite=args.overwrite,
        no_compare=args.no_compare,
        top_n=args.top_n,
        compare_out=args.compare_out,
        compare_lines_out=args.compare_lines_out,
        curve_out=args.curve_out,
        curve_lines_out=args.curve_lines_out,
        curve_run_card=args.curve_run_card,
        curve_trainer_trace_jsonl=args.curve_trainer_trace_jsonl,
        curve_model_name=args.curve_model_name,
        curve_dataset_name=args.curve_dataset_name,
        curve_dataset_config=args.curve_dataset_config,
        compare_with_sweep=args.compare_with_sweep,
        compare_with_label=args.compare_with_label,
        run_card=args.run_card,
        dry_run=args.dry_run,
        wait_for_process_pid_file=args.wait_for_process_pid_file,
        wait=args.wait,
        ready_file=args.ready_file,
        no_ready_file_check=args.no_ready_file_check,
        poll_seconds=args.poll_seconds,
        process_poll_seconds=args.process_poll_seconds,
        timeout_seconds=args.timeout_seconds,
        process_timeout_seconds=args.process_timeout_seconds,
        runner=runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checkpoint_generation_control(args)
    if args.dry_run and args.run_card is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay or execute a Hugging Face FT scale-up command from a sweep artifact."""

from __future__ import annotations

import sys
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import spiraltorch as st  # noqa: E402
import hf_gpt2_finetune_scale_up as _legacy  # noqa: E402
from hf_gpt2_finetune_scale_up import *  # noqa: F401,F403,E402


DEFAULT_WAIT_LAUNCH_SCRIPT = Path(__file__).resolve().with_name(
    "hf_finetune_wait_launch.py"
)
SCALE_UP_ROW_TYPES = {
    "hf_finetune_scale_up_command",
    "hf_gpt2_finetune_scale_up_command",
}


def _argv_has_option(raw_argv: list[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def parse_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy.parse_args(argv)
    if not _argv_has_option(raw_argv, "--wait-launch-script"):
        args.wait_launch_script = DEFAULT_WAIT_LAUNCH_SCRIPT
    return args


def _scale_up_command_from_source(args) -> dict[str, Any]:
    source = _legacy._read_json(args.source)
    source_is_scale_up_artifact = source.get("row_type") in SCALE_UP_ROW_TYPES
    if source_is_scale_up_artifact:
        source_payload: str | Path | Mapping[str, Any] = (
            _legacy._summary_from_scale_up_artifact(source)
        )
    else:
        source_payload = source
    max_steps_multiplier = args.max_steps_multiplier
    max_train_samples_multiplier = args.max_train_samples_multiplier
    if not source_is_scale_up_artifact:
        max_steps_multiplier = (
            2.0 if max_steps_multiplier is None else max_steps_multiplier
        )
        max_train_samples_multiplier = (
            2.0
            if max_train_samples_multiplier is None
            else max_train_samples_multiplier
        )
    output_dir = args.output_dir
    run_card = args.run_card
    trainer_trace_jsonl = args.trainer_trace_jsonl
    output_suffix = args.output_suffix
    if source_is_scale_up_artifact and isinstance(source_payload, Mapping):
        source_command = source_payload.get("scale_up_candidate_command")
        if (
            output_dir is None
            and output_suffix is None
            and isinstance(source_command, Sequence)
            and not isinstance(source_command, (str, bytes))
        ):
            output_dir = _legacy._flag_value(source_command, "--output-dir")
            run_card = run_card or _legacy._flag_value(source_command, "--run-card")
            trainer_trace_jsonl = trainer_trace_jsonl or _legacy._flag_value(
                source_command,
                "--trainer-trace-jsonl",
            )
    else:
        output_suffix = "scaleup" if output_suffix is None else output_suffix
    command = st.hf_finetune_scale_up_command(
        source_payload,
        model_name=args.model_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_steps=args.max_steps,
        max_steps_multiplier=max_steps_multiplier,
        max_train_samples=args.max_train_samples,
        max_train_samples_multiplier=max_train_samples_multiplier,
        max_eval_samples=args.max_eval_samples,
        max_eval_blocks=args.max_eval_blocks,
        streaming_validation_samples=args.streaming_validation_samples,
        output_dir=output_dir,
        output_suffix=output_suffix or "scaleup",
        run_card=run_card,
        trainer_trace_jsonl=trainer_trace_jsonl,
        trainer_trace_run_id=args.trainer_trace_run_id,
    )
    command["source_path"] = str(args.source)
    if args.write_command is not None:
        command["artifact_path"] = str(args.write_command)
    return command


def _build_wait_launch_command(
    args,
    command_values: Sequence[object],
) -> list[str] | None:
    return _legacy._build_wait_launch_command(args, command_values)


def _attach_wait_launch_command(
    args,
    command: dict[str, Any],
) -> None:
    command_values = command.get("command")
    if not isinstance(command_values, Sequence) or isinstance(
        command_values,
        (str, bytes),
    ):
        return
    wait_command = _build_wait_launch_command(args, command_values)
    if wait_command is None:
        return
    command["wait_launch_command"] = wait_command
    command["wait_launch_command_display"] = _legacy.shlex.join(
        [str(item) for item in wait_command]
    )
    command["wait_launch_manifest"] = str(args.wait_launch_manifest)
    command["wait_launch_jsonl_out"] = (
        None if args.wait_launch_jsonl_out is None else str(args.wait_launch_jsonl_out)
    )
    command["wait_launch_checkpoint"] = (
        None
        if args.wait_launch_checkpoint is None
        else str(args.wait_launch_checkpoint)
    )
    command["wait_launch_detach"] = bool(args.wait_launch_detach)


def run_scale_up(args) -> dict[str, Any]:
    command = _scale_up_command_from_source(args)
    preflight = st.hf_finetune_scale_up_preflight_report(command)
    command["preflight"] = preflight
    command["preflight_status"] = preflight.get("status")
    command["preflight_error_count"] = preflight.get("error_count")
    command["preflight_warning_count"] = preflight.get("warning_count")
    _attach_wait_launch_command(args, command)
    if command.get("status") != "ok":
        if args.run or args.require_ready:
            command["run_returncode"] = 2
        _legacy._write_command_artifact(args, command)
        return command
    if (args.run or args.require_ready) and not preflight.get("ready"):
        command["run_returncode"] = 2
        _legacy._write_command_artifact(args, command)
        return command
    if not args.run:
        _legacy._write_command_artifact(args, command)
        return command
    command_values = command.get("wait_launch_command") or command.get("command")
    if not isinstance(command_values, Sequence) or isinstance(
        command_values,
        (str, bytes),
    ):
        command["run_returncode"] = 2
        _legacy._write_command_artifact(args, command)
        return command
    result = subprocess.run([str(item) for item in command_values], check=False)
    command["run_returncode"] = int(result.returncode)
    _legacy._write_command_artifact(args, command)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = run_scale_up(args)
    print(f"hf_ft_scale_up_command status={command.get('status')}")
    if command.get("artifact_path"):
        print(f"hf_ft_scale_up_artifact {command.get('artifact_path')}")
    print(
        "hf_ft_scale_up_preflight "
        f"status={command.get('preflight_status')} "
        f"errors={command.get('preflight_error_count')} "
        f"warnings={command.get('preflight_warning_count')}"
    )
    preflight = command.get("preflight")
    if isinstance(preflight, Mapping):
        for issue in preflight.get("issues", []):
            if not isinstance(issue, Mapping):
                continue
            print(
                "hf_ft_scale_up_preflight_issue "
                f"severity={issue.get('severity')} "
                f"field={issue.get('field')} "
                f"path={issue.get('path')} "
                f"message={issue.get('message')}"
            )
    display = command.get("command_display")
    if display:
        print(f"hf_ft_scale_up_replay {display}")
    elif command.get("command"):
        print(
            "hf_ft_scale_up_replay "
            f"{_legacy.shlex.join([str(item) for item in command['command']])}"
        )
    wait_display = command.get("wait_launch_command_display")
    if wait_display:
        print(f"hf_ft_scale_up_wait_launch {wait_display}")
    if command.get("run_returncode") is not None:
        print(f"hf_ft_scale_up_run returncode={command.get('run_returncode')}")
        return int(command.get("run_returncode") or 0)
    if args.require_ready and command.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

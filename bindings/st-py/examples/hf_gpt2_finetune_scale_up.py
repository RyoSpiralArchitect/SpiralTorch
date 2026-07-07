#!/usr/bin/env python3
"""Replay or execute a GPT-2 FT scale-up command from a sweep artifact."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import spiraltorch as st


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="A sweep-report.json or scale-up-command.json artifact.",
    )
    parser.add_argument(
        "--write-command",
        type=Path,
        default=None,
        help="Write the resolved scale-up command artifact to this path.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the resolved command. The default only prints it.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the source does not contain a runnable command.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-steps-multiplier", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-train-samples-multiplier", type=float, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.source.is_file():
        parser.error(f"source artifact does not exist: {args.source}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_steps_multiplier is not None and args.max_steps_multiplier <= 0.0:
        parser.error("--max-steps-multiplier must be positive")
    if args.max_train_samples is not None and args.max_train_samples < 0:
        parser.error("--max-train-samples must be non-negative")
    if (
        args.max_train_samples_multiplier is not None
        and args.max_train_samples_multiplier <= 0.0
    ):
        parser.error("--max-train-samples-multiplier must be positive")
    if args.max_eval_samples is not None and args.max_eval_samples < 0:
        parser.error("--max-eval-samples must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _flag_value(command: Sequence[object], flag: str) -> str | None:
    values = [str(item) for item in command]
    for index, item in enumerate(values):
        if item == flag and index + 1 < len(values):
            return values[index + 1]
    return None


def _summary_from_scale_up_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    command_value = artifact.get("command")
    if not isinstance(command_value, Sequence) or isinstance(
        command_value,
        (str, bytes),
    ):
        command_value = artifact.get("base_command")
    if not isinstance(command_value, Sequence) or isinstance(
        command_value,
        (str, bytes),
    ):
        command_value = []
    command = [str(item) for item in command_value]
    return {
        "row_type": "hf_gpt2_finetune_sweep_report_summary",
        "scale_up_candidate_label": artifact.get("scale_up_candidate_label"),
        "scale_up_candidate_reason": artifact.get("scale_up_candidate_reason"),
        "scale_up_candidate_distortion_adjusted_eval_loss": artifact.get(
            "scale_up_candidate_distortion_adjusted_eval_loss"
        ),
        "scale_up_candidate_distortion_pressure_index": artifact.get(
            "scale_up_candidate_distortion_pressure_index"
        ),
        "scale_up_candidate_command": command,
        "scale_up_candidate_output_dir": _flag_value(command, "--output-dir"),
        "scale_up_candidate_run_card": _flag_value(command, "--run-card"),
        "scale_up_candidate_trainer_trace_jsonl": _flag_value(
            command,
            "--trainer-trace-jsonl",
        ),
    }


def _scale_up_command_from_source(args: argparse.Namespace) -> dict[str, Any]:
    source = _read_json(args.source)
    source_is_scale_up_artifact = (
        source.get("row_type") == "hf_gpt2_finetune_scale_up_command"
    )
    if source_is_scale_up_artifact:
        source_payload: str | Path | Mapping[str, Any] = _summary_from_scale_up_artifact(
            source
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
            output_dir = _flag_value(source_command, "--output-dir")
            run_card = run_card or _flag_value(source_command, "--run-card")
            trainer_trace_jsonl = trainer_trace_jsonl or _flag_value(
                source_command,
                "--trainer-trace-jsonl",
            )
    else:
        output_suffix = "scaleup" if output_suffix is None else output_suffix
    command = st.hf_gpt2_finetune_scale_up_command(
        source_payload,
        max_steps=args.max_steps,
        max_steps_multiplier=max_steps_multiplier,
        max_train_samples=args.max_train_samples,
        max_train_samples_multiplier=max_train_samples_multiplier,
        max_eval_samples=args.max_eval_samples,
        output_dir=output_dir,
        output_suffix=output_suffix or "scaleup",
        run_card=run_card,
        trainer_trace_jsonl=trainer_trace_jsonl,
    )
    command["source_path"] = str(args.source)
    if args.write_command is not None:
        command["artifact_path"] = str(args.write_command)
    return command


def run_scale_up(args: argparse.Namespace) -> dict[str, Any]:
    command = _scale_up_command_from_source(args)
    if args.write_command is not None:
        _write_json(args.write_command, command)
    if command.get("status") != "ok":
        if args.run or args.require_ready:
            command["run_returncode"] = 2
        return command
    if not args.run:
        return command
    command_values = command.get("command")
    if not isinstance(command_values, Sequence) or isinstance(
        command_values,
        (str, bytes),
    ):
        command["run_returncode"] = 2
        return command
    result = subprocess.run([str(item) for item in command_values], check=False)
    command["run_returncode"] = int(result.returncode)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = run_scale_up(args)
    print(f"scale_up_command status={command.get('status')}")
    if command.get("artifact_path"):
        print(f"scale_up_artifact {command.get('artifact_path')}")
    display = command.get("command_display")
    if display:
        print(f"scale_up_replay {display}")
    elif command.get("command"):
        print(f"scale_up_replay {shlex.join([str(item) for item in command['command']])}")
    if command.get("run_returncode") is not None:
        print(f"scale_up_run returncode={command.get('run_returncode')}")
        return int(command.get("run_returncode") or 0)
    if args.require_ready and command.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

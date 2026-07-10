#!/usr/bin/env python3
"""Replay or execute a GPT-2 FT scale-up command from a sweep artifact."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st  # noqa: E402


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
    parser.add_argument("--max-eval-blocks", type=int, default=None)
    parser.add_argument("--streaming-validation-samples", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--adapter-continuation",
        choices=("auto", "replay", "continue"),
        default=None,
        help=(
            "Choose whether scale-up replays the selected configuration or "
            "continues from its saved PEFT adapter. Defaults to auto for sweep "
            "reports and replay for an already-resolved command artifact."
        ),
    )
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-resume-checkpoint",
        action="store_true",
        help=(
            "Allow writing a future-checkpoint handoff command before the "
            "--resume-from-checkpoint directory exists. Preflight still reports "
            "the missing checkpoint until it is ready."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--trainer-trace-run-id", default=None)
    parser.add_argument(
        "--wait-launch-manifest",
        type=Path,
        default=None,
        help=(
            "Also attach a wait-launch wrapper command that launches the "
            "resolved scale-up command after a process/checkpoint handoff."
        ),
    )
    parser.add_argument(
        "--wait-launch-script",
        type=Path,
        default=None,
        help=(
            "Override the wait-launch helper script embedded in the wrapper "
            "command. Defaults to the GPT-2-compatible helper."
        ),
    )
    parser.add_argument("--wait-launch-jsonl-out", type=Path, default=None)
    parser.add_argument("--wait-launch-pid", type=int, default=None)
    parser.add_argument("--wait-launch-pid-file", type=Path, default=None)
    parser.add_argument("--wait-launch-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--wait-launch-checkpoint-ready-file",
        default="model.safetensors",
    )
    parser.add_argument("--wait-launch-poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--wait-launch-checkpoint-timeout-seconds",
        type=float,
        default=1800.0,
    )
    parser.add_argument("--wait-launch-launched-pid-file", type=Path, default=None)
    parser.add_argument("--wait-launch-launched-log-file", type=Path, default=None)
    parser.add_argument(
        "--wait-launch-launched-log-mode",
        choices=("append", "write"),
        default="append",
    )
    parser.add_argument("--wait-launch-detach", action="store_true")
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
    if args.max_eval_blocks is not None and args.max_eval_blocks < 0:
        parser.error("--max-eval-blocks must be non-negative")
    if (
        args.streaming_validation_samples is not None
        and args.streaming_validation_samples < 0
    ):
        parser.error("--streaming-validation-samples must be non-negative")
    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.is_dir()
        and not args.allow_missing_resume_checkpoint
    ):
        parser.error(
            "--resume-from-checkpoint does not exist or is not a directory: "
            f"{args.resume_from_checkpoint}"
        )
    if args.wait_launch_poll_seconds <= 0.0:
        parser.error("--wait-launch-poll-seconds must be positive")
    if args.wait_launch_checkpoint_timeout_seconds < 0.0:
        parser.error("--wait-launch-checkpoint-timeout-seconds must be non-negative")
    wait_launch_inputs = [
        args.wait_launch_script,
        args.wait_launch_jsonl_out,
        args.wait_launch_pid,
        args.wait_launch_pid_file,
        args.wait_launch_checkpoint,
        args.wait_launch_launched_pid_file,
        args.wait_launch_launched_log_file,
    ]
    if args.wait_launch_detach and args.wait_launch_manifest is None:
        parser.error("--wait-launch-detach requires --wait-launch-manifest")
    if any(value is not None for value in wait_launch_inputs) and (
        args.wait_launch_manifest is None
    ):
        parser.error("wait-launch options require --wait-launch-manifest")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_command_artifact(args: argparse.Namespace, command: Mapping[str, Any]) -> None:
    if args.write_command is not None:
        _write_json(args.write_command, command)


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
    run_card = _flag_value(command, "--run-card")
    summary = {
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
        "scale_up_candidate_run_card": run_card,
        "scale_up_candidate_trainer_trace_jsonl": _flag_value(
            command,
            "--trainer-trace-jsonl",
        ),
        "scale_up_candidate_finetune_mode": _flag_value(
            command,
            "--finetune-mode",
        ),
        "scale_up_candidate_model_artifact_kind": _flag_value(
            command,
            "--model-artifact-kind",
        ),
        "scale_up_candidate_adapter_lineage_depth": artifact.get(
            "adapter_continuation_expected_child_lineage_depth"
        ),
    }
    if run_card is not None and Path(run_card).is_file():
        card_summary = st.summarize_hf_gpt2_finetune_run_card(run_card)
        summary.update(
            {
                "adapter_promotion_required": card_summary.get(
                    "adapter_promotion_gate_requested"
                )
                is True,
                "scale_up_candidate_adapter_saved": card_summary.get(
                    "adapter_saved"
                ),
                "scale_up_candidate_finetune_mode": card_summary.get(
                    "finetune_mode"
                ),
                "scale_up_candidate_model_artifact_kind": card_summary.get(
                    "model_artifact_kind"
                ),
                "scale_up_candidate_adapter_promotion_status": card_summary.get(
                    "adapter_promotion_status"
                ),
                "scale_up_candidate_adapter_promotion_ready": card_summary.get(
                    "adapter_promotion_ready"
                ),
                "scale_up_candidate_adapter_promotion_report_path": card_summary.get(
                    "adapter_promotion_report_path"
                ),
                "scale_up_candidate_adapter_artifact_probe_status": card_summary.get(
                    "adapter_artifact_probe_status"
                ),
                "scale_up_candidate_adapter_artifact_probe_report_path": (
                    card_summary.get("adapter_artifact_probe_report_path")
                ),
                "scale_up_candidate_adapter_artifact_probe_device": card_summary.get(
                    "adapter_artifact_probe_device"
                ),
                "scale_up_candidate_adapter_artifact_probe_new_token_count": (
                    card_summary.get("adapter_artifact_probe_new_token_count")
                ),
                "scale_up_candidate_adapter_lineage_status": card_summary.get(
                    "adapter_lineage_status"
                ),
                "scale_up_candidate_adapter_lineage_adapter_id": card_summary.get(
                    "adapter_lineage_adapter_id"
                ),
                "scale_up_candidate_adapter_lineage_parent_adapter_id": (
                    card_summary.get("adapter_lineage_parent_adapter_id")
                ),
                "scale_up_candidate_adapter_lineage_root_adapter_id": card_summary.get(
                    "adapter_lineage_root_adapter_id"
                ),
                "scale_up_candidate_adapter_lineage_depth": card_summary.get(
                    "adapter_lineage_depth"
                ),
                "scale_up_candidate_adapter_lineage_manifest_path": card_summary.get(
                    "adapter_lineage_manifest_path"
                ),
            }
        )
    return summary


def _preserve_adapter_continuation_provenance(
    command: dict[str, Any],
    source: Mapping[str, Any],
) -> None:
    preserved = False
    for key, value in source.items():
        if not str(key).startswith("adapter_continuation_"):
            continue
        command[str(key)] = value
        preserved = True
    if preserved:
        command["adapter_continuation_artifact_replay"] = True


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
    adapter_continuation = args.adapter_continuation or (
        "replay" if source_is_scale_up_artifact else "auto"
    )
    command = st.hf_gpt2_finetune_scale_up_command(
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
        adapter_continuation=adapter_continuation,
    )
    if (
        source_is_scale_up_artifact
        and adapter_continuation == "replay"
        and args.model_name is None
        and args.resume_from_checkpoint is None
    ):
        _preserve_adapter_continuation_provenance(command, source)
    command["source_path"] = str(args.source)
    if args.write_command is not None:
        command["artifact_path"] = str(args.write_command)
    return command


def _build_wait_launch_command(
    args: argparse.Namespace,
    command_values: Sequence[object],
) -> list[str] | None:
    if args.wait_launch_manifest is None:
        return None
    script = (
        args.wait_launch_script
        or "bindings/st-py/examples/hf_gpt2_finetune_wait_launch.py"
    )
    wait_command = [
        sys.executable,
        str(script),
        "--manifest",
        str(args.wait_launch_manifest),
        "--poll-seconds",
        str(args.wait_launch_poll_seconds),
        "--checkpoint-timeout-seconds",
        str(args.wait_launch_checkpoint_timeout_seconds),
    ]
    for flag, value in [
        ("--pid", args.wait_launch_pid),
        ("--pid-file", args.wait_launch_pid_file),
        ("--checkpoint", args.wait_launch_checkpoint),
        ("--jsonl-out", args.wait_launch_jsonl_out),
        ("--launched-pid-file", args.wait_launch_launched_pid_file),
        ("--launched-log-file", args.wait_launch_launched_log_file),
    ]:
        if value is not None:
            wait_command.extend([flag, str(value)])
    if args.wait_launch_checkpoint_ready_file != "model.safetensors":
        wait_command.extend(
            [
                "--checkpoint-ready-file",
                str(args.wait_launch_checkpoint_ready_file),
            ]
        )
    if args.wait_launch_launched_log_mode != "append":
        wait_command.extend(
            [
                "--launched-log-mode",
                str(args.wait_launch_launched_log_mode),
            ]
        )
    if args.wait_launch_detach:
        wait_command.append("--detach")
    wait_command.append("--")
    wait_command.extend(str(item) for item in command_values)
    return wait_command


def _attach_wait_launch_command(
    args: argparse.Namespace,
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
    command["wait_launch_command_display"] = shlex.join(
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


def run_scale_up(args: argparse.Namespace) -> dict[str, Any]:
    command = _scale_up_command_from_source(args)
    preflight = st.hf_gpt2_finetune_scale_up_preflight_report(command)
    command["preflight"] = preflight
    command["preflight_status"] = preflight.get("status")
    command["preflight_error_count"] = preflight.get("error_count")
    command["preflight_warning_count"] = preflight.get("warning_count")
    _attach_wait_launch_command(args, command)
    if command.get("status") != "ok":
        if args.run or args.require_ready:
            command["run_returncode"] = 2
        _write_command_artifact(args, command)
        return command
    if (args.run or args.require_ready) and not preflight.get("ready"):
        command["run_returncode"] = 2
        _write_command_artifact(args, command)
        return command
    if not args.run:
        _write_command_artifact(args, command)
        return command
    command_values = command.get("wait_launch_command") or command.get("command")
    if not isinstance(command_values, Sequence) or isinstance(
        command_values,
        (str, bytes),
    ):
        command["run_returncode"] = 2
        _write_command_artifact(args, command)
        return command
    result = subprocess.run([str(item) for item in command_values], check=False)
    command["run_returncode"] = int(result.returncode)
    _write_command_artifact(args, command)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = run_scale_up(args)
    print(f"scale_up_command status={command.get('status')}")
    if command.get("artifact_path"):
        print(f"scale_up_artifact {command.get('artifact_path')}")
    if command.get("adapter_continuation_policy") is not None:
        print(
            "scale_up_adapter_continuation "
            f"policy={command.get('adapter_continuation_policy')} "
            f"status={command.get('adapter_continuation_status')} "
            f"applied={command.get('adapter_continuation_applied')} "
            f"source={command.get('adapter_continuation_source')} "
            "source_adapter_id="
            f"{command.get('adapter_continuation_source_adapter_id')} "
            "expected_child_depth="
            f"{command.get('adapter_continuation_expected_child_lineage_depth')}"
        )
    print(
        "scale_up_preflight "
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
                "scale_up_preflight_issue "
                f"severity={issue.get('severity')} "
                f"field={issue.get('field')} "
                f"path={issue.get('path')} "
                f"message={issue.get('message')}"
            )
    display = command.get("command_display")
    if display:
        print(f"scale_up_replay {display}")
    elif command.get("command"):
        print(f"scale_up_replay {shlex.join([str(item) for item in command['command']])}")
    wait_display = command.get("wait_launch_command_display")
    if wait_display:
        print(f"scale_up_wait_launch {wait_display}")
    if command.get("run_returncode") is not None:
        print(f"scale_up_run returncode={command.get('run_returncode')}")
        return int(command.get("run_returncode") or 0)
    if args.require_ready and command.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

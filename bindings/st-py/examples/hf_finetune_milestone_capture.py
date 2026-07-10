#!/usr/bin/env python3
"""Refresh and capture a SpiralTorch Hugging Face FT milestone snapshot."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import spiraltorch as st  # noqa: E402
import hf_gpt2_finetune_milestone_capture as _legacy  # noqa: E402
from hf_gpt2_finetune_milestone_capture import *  # noqa: F401,F403,E402


ROW_TYPE_MAP = {
    "hf_gpt2_finetune_milestone_capture": "hf_finetune_milestone_capture",
    "hf_gpt2_finetune_monitor_report": "hf_finetune_monitor_report",
    "hf_gpt2_ft_monitor_snapshot": "hf_finetune_monitor_snapshot",
}
LEGACY_ROW_TYPE_MAP = {value: key for key, value in ROW_TYPE_MAP.items()}
LINE_PREFIX_MAP = {
    "hf_gpt2_ft_milestone_capture": "hf_ft_milestone_capture",
}


def parse_args(argv: list[str] | None = None):
    return _legacy.parse_args(argv)


def _repo_root() -> Path:
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "bindings" / "st-py" / "examples").is_dir():
        return source_root
    return Path.cwd().resolve()


def _path_arg(path: Path, *, repo: Path) -> str:
    return _legacy._path_arg(path, repo=repo)


def _map_row_types(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        payload = {key: _map_row_types(item, mapping) for key, item in value.items()}
        row_type = payload.get("row_type")
        if isinstance(row_type, str) and row_type in mapping:
            payload["row_type"] = mapping[row_type]
        return payload
    if isinstance(value, list):
        return [_map_row_types(item, mapping) for item in value]
    return value


def _genericize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _map_row_types(payload, ROW_TYPE_MAP)


def _legacyize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _map_row_types(payload, LEGACY_ROW_TYPE_MAP)


def _genericize_lines(lines: list[str]) -> list[str]:
    generic_lines = []
    for line in lines:
        for before, after in LINE_PREFIX_MAP.items():
            if line.startswith(before):
                line = line.replace(before, after, 1)
                break
        generic_lines.append(line)
    return generic_lines


def build_run_status_command(args, *, repo: Path) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES_ROOT / "hf_finetune_run_status.py"),
        _path_arg(args.run_dir, repo=repo),
    ]
    for flag, value in [
        ("--max-steps", args.max_steps),
        ("--eval-steps", args.eval_steps),
        ("--save-steps", args.save_steps),
        ("--min-free-disk-gb", args.min_free_disk_gb),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    if args.final_checkpoint is not None:
        command.extend(["--final-checkpoint", args.final_checkpoint])
    command.extend(
        [
            "--out",
            _path_arg(args.status_out, repo=repo),
            "--lines-out",
            _path_arg(args.status_lines_out, repo=repo),
            "--jsonl-out",
            _path_arg(args.status_jsonl_out, repo=repo),
            "--quiet",
        ]
    )
    return command


def build_monitor_command(args, *, repo: Path) -> list[str]:
    label = args.label or f"milestone-{args.milestone_step}"
    command = [
        sys.executable,
        str(EXAMPLES_ROOT / "hf_finetune_monitor_snapshot.py"),
        _path_arg(args.run_dir, repo=repo),
        "--label",
        label,
        "--run-status-history-jsonl",
        _path_arg(args.status_jsonl_out, repo=repo),
        "--milestone-step",
        str(args.milestone_step),
        "--out",
        _path_arg(args.out, repo=repo),
        "--lines-out",
        _path_arg(args.lines_out, repo=repo),
    ]
    if args.next_run_dir is not None:
        command.extend(["--next-run-dir", _path_arg(args.next_run_dir, repo=repo)])
    return command


def capture_once(
    args,
    *,
    repo: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo = repo or _repo_root()
    env = dict(env or os.environ)
    package_root = str(EXAMPLES_ROOT.parent)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        package_root
        if not existing_pythonpath
        else os.pathsep.join((package_root, existing_pythonpath))
    )
    result: dict[str, Any] = {"time_unix_s": time.time()}
    commands = [
        build_run_status_command(args, repo=repo),
        build_monitor_command(args, repo=repo),
    ]
    command_results = []
    for command in commands:
        command_result = _legacy._run_command(command, repo=repo, env=env)
        command_results.append(command_result)
        if command_result["returncode"] != 0:
            result.update({"status": "command_failed", "commands": command_results})
            return _genericize_payload(result)
    snapshot = json.loads(args.out.read_text(encoding="utf-8"))
    legacy_snapshot = _legacyize_payload(snapshot)
    result.update(
        st.hf_finetune_milestone_capture_report(
            legacy_snapshot,
            milestone_step=args.milestone_step,
            label=args.label or f"milestone-{args.milestone_step}",
            commands=command_results,
        )
    )
    return _genericize_payload(result)


def _write_state(args, state: dict[str, Any]) -> None:
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.history_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.history_jsonl_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n")


def state_line(state: dict[str, Any]) -> str:
    legacy_state = _legacyize_payload(state)
    return _genericize_lines(st.hf_finetune_milestone_capture_lines(legacy_state))[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    last_state: dict[str, Any] = {}
    for iteration in range(1, args.watch_count + 1):
        state = capture_once(args)
        state["iteration"] = iteration
        _write_state(args, state)
        last_state = state
        if not args.quiet:
            print(state_line(state))
        if state.get("milestone_ready") or state.get("process_status") not in {
            None,
            "alive",
        }:
            break
        if iteration < args.watch_count:
            time.sleep(args.watch_interval_seconds)
    if args.require_milestone_ready and not last_state.get("milestone_ready"):
        print("milestone is not ready yet", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

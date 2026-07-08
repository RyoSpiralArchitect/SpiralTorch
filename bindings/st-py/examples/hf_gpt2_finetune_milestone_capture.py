#!/usr/bin/env python3
"""Refresh and capture a SpiralTorch GPT-2 FT milestone snapshot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import spiraltorch as st


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--next-run-dir", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--milestone-step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=None)
    parser.add_argument("--final-checkpoint", default=None)
    parser.add_argument("--status-out", type=Path, default=None)
    parser.add_argument("--status-lines-out", type=Path, default=None)
    parser.add_argument("--status-jsonl-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--history-jsonl-out", type=Path, default=None)
    parser.add_argument("--watch-interval-seconds", type=float, default=120.0)
    parser.add_argument("--watch-count", type=int, default=1)
    parser.add_argument("--require-milestone-ready", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        parser.error(f"run_dir does not exist: {args.run_dir}")
    if args.next_run_dir is not None and not args.next_run_dir.is_dir():
        parser.error(f"next_run_dir does not exist: {args.next_run_dir}")
    if args.milestone_step < 0:
        parser.error("--milestone-step must be non-negative")
    if args.watch_count <= 0:
        parser.error("--watch-count must be positive")
    if args.watch_interval_seconds < 0.0:
        parser.error("--watch-interval-seconds must be non-negative")
    args.status_out = args.status_out or args.run_dir / "direct-run-status.json"
    args.status_lines_out = (
        args.status_lines_out or args.run_dir / "direct-run-status.txt"
    )
    args.status_jsonl_out = (
        args.status_jsonl_out or args.run_dir / "direct-run-status-history.jsonl"
    )
    args.out = args.out or args.run_dir / f"milestone-{args.milestone_step}-capture.json"
    args.lines_out = (
        args.lines_out or args.run_dir / f"milestone-{args.milestone_step}-capture.txt"
    )
    args.state_out = (
        args.state_out
        or args.run_dir / f"milestone-{args.milestone_step}-capture-watch-state.json"
    )
    args.history_jsonl_out = (
        args.history_jsonl_out
        or args.run_dir / f"milestone-{args.milestone_step}-capture-history.jsonl"
    )
    return args


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_arg(path: Path, *, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def build_run_status_command(args: argparse.Namespace, *, repo: Path) -> list[str]:
    command = [
        "python3",
        "bindings/st-py/examples/hf_gpt2_finetune_run_status.py",
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


def build_monitor_command(args: argparse.Namespace, *, repo: Path) -> list[str]:
    label = args.label or f"milestone-{args.milestone_step}"
    command = [
        "python3",
        "bindings/st-py/examples/hf_gpt2_finetune_monitor_snapshot.py",
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


def _run_command(
    command: list[str], *, repo: Path, env: dict[str, str]
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "args": command[:3],
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def capture_once(
    args: argparse.Namespace,
    *,
    repo: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo = repo or _repo_root()
    env = dict(env or os.environ)
    env["PYTHONPATH"] = "bindings/st-py"
    result: dict[str, Any] = {"time_unix_s": time.time()}
    commands = [
        build_run_status_command(args, repo=repo),
        build_monitor_command(args, repo=repo),
    ]
    command_results = []
    for command in commands:
        command_result = _run_command(command, repo=repo, env=env)
        command_results.append(command_result)
        if command_result["returncode"] != 0:
            result.update({"status": "command_failed", "commands": command_results})
            return result
    snapshot = json.loads(args.out.read_text(encoding="utf-8"))
    result.update(
        st.hf_gpt2_finetune_milestone_capture_report(
            snapshot,
            milestone_step=args.milestone_step,
            label=args.label or f"milestone-{args.milestone_step}",
            commands=command_results,
        )
    )
    return result


def _write_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.history_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.history_jsonl_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n")


def state_line(state: dict[str, Any]) -> str:
    return st.hf_gpt2_finetune_milestone_capture_lines(state)[0]


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

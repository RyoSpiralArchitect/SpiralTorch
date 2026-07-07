#!/usr/bin/env python3
"""Summarize a live SpiralTorch GPT-2 fine-tuning run directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st

DEFAULT_TRACE_NAME = "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
DEFAULT_RUN_CARD_NAME = "spiraltorch-hf-gpt2-ft-run-card.json"
DEFAULT_LOG_NAME = "ft.log"
PROGRESS_RE = re.compile(r"\b(?P<step>\d+)/(?:\s*)?(?P<max_steps>\d+)\s*\[")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-tail-bytes", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-card", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--final-checkpoint", default=None)
    parser.add_argument("--tail-evals", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.run_dir.exists():
        parser.error(f"run_dir does not exist: {args.run_dir}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.tail_evals < 0:
        parser.error("--tail-evals must be non-negative")
    if args.log_tail_bytes <= 0:
        parser.error("--log-tail-bytes must be positive")
    if args.trace_jsonl is None:
        args.trace_jsonl = args.run_dir / DEFAULT_TRACE_NAME
    if args.run_card is None:
        args.run_card = args.run_dir / DEFAULT_RUN_CARD_NAME
    if args.pid_file is None:
        args.pid_file = args.run_dir / "ft.pid"
    if args.log_file is None:
        args.log_file = args.run_dir / DEFAULT_LOG_NAME
    if args.final_checkpoint is None and args.max_steps is not None:
        args.final_checkpoint = f"checkpoint-{args.max_steps}"
    return args


def _number_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_pid(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _process_status(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "alive"
    return "alive"


def _read_tail(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="ignore")


def _log_progress(log_file: Path, max_bytes: int) -> dict[str, Any]:
    text = _read_tail(log_file, max_bytes)
    latest_step = None
    latest_max_steps = None
    for match in PROGRESS_RE.finditer(text):
        latest_step = int(match.group("step"))
        latest_max_steps = int(match.group("max_steps"))
    progress = None
    if latest_step is not None and latest_max_steps:
        progress = min(max(float(latest_step) / float(latest_max_steps), 0.0), 1.0)
    return {
        "log_file": str(log_file),
        "log_status": "ok" if log_file.is_file() else "missing",
        "log_latest_step": latest_step,
        "log_max_steps": latest_max_steps,
        "log_progress": progress,
    }


def _trace_summary(trace_jsonl: Path, max_steps: int | None) -> dict[str, Any]:
    if not trace_jsonl.is_file():
        return {
            "trace_jsonl": str(trace_jsonl),
            "trace_status": "missing",
            "trace_event_count": 0,
            "trace_max_global_step": None,
            "progress": None,
            "training_loss_guard_count": 0,
        }
    rows = st.load_hf_gpt2_finetune_trainer_trace(trace_jsonl)
    summary = st.summarize_hf_gpt2_finetune_trainer_trace(rows)
    step = summary.get("trace_max_global_step")
    progress = None
    if max_steps is not None and isinstance(step, (int, float)):
        progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
    summary.update(
        {
            "trace_jsonl": str(trace_jsonl),
            "trace_status": "ok",
            "max_steps": max_steps,
            "progress": progress,
            "training_loss_guard_count": sum(
                1 for row in rows if row.get("training_loss_guard")
            ),
        }
    )
    return summary


def _checkpoint_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        token = path.name.removeprefix("checkpoint-")
        try:
            step: int | None = int(token)
        except ValueError:
            step = None
        ready = (path / "model.safetensors").is_file()
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "step": step,
                "model_safetensors_ready": ready,
            }
        )
    return rows


def summarize_run(args: argparse.Namespace) -> dict[str, Any]:
    trace = _trace_summary(args.trace_jsonl, args.max_steps)
    log_progress = _log_progress(args.log_file, int(args.log_tail_bytes))
    run_card = _load_json(args.run_card)
    checkpoint_card = _load_json(args.checkpoint_card)
    pid = _read_pid(args.pid_file)
    checkpoints = _checkpoint_rows(args.run_dir)
    final_checkpoint_path = None
    final_checkpoint_ready = None
    if args.final_checkpoint:
        final_checkpoint_path = args.run_dir / str(args.final_checkpoint)
        final_checkpoint_ready = (final_checkpoint_path / "model.safetensors").is_file()
    disk = shutil.disk_usage(args.run_dir)
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    return {
        "row_type": "hf_gpt2_finetune_run_status",
        "run_dir": str(args.run_dir),
        "trace": trace,
        "log_progress": log_progress,
        "run_card_path": str(args.run_card),
        "run_card_status": (run_card or {}).get("status"),
        "pid_file": str(args.pid_file),
        "pid": pid,
        "process_status": _process_status(pid),
        "checkpoint_card_path": str(args.checkpoint_card)
        if args.checkpoint_card is not None
        else None,
        "checkpoint_card_status": (checkpoint_card or {}).get("status"),
        "checkpoint_card_process_wait": (checkpoint_card or {}).get("process_wait"),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "latest_checkpoint": latest_checkpoint,
        "final_checkpoint": args.final_checkpoint,
        "final_checkpoint_path": str(final_checkpoint_path)
        if final_checkpoint_path is not None
        else None,
        "final_checkpoint_ready": final_checkpoint_ready,
        "disk_free_gb": disk.free / (1024.0**3),
        "disk_total_gb": disk.total / (1024.0**3),
    }


def status_lines(status: dict[str, Any], *, tail_evals: int) -> list[str]:
    trace = status.get("trace") if isinstance(status.get("trace"), dict) else {}
    latest_checkpoint = status.get("latest_checkpoint")
    latest_checkpoint_name = (
        latest_checkpoint.get("name") if isinstance(latest_checkpoint, dict) else None
    )
    lines = [
        (
            "hf_gpt2_ft_run_status "
            f"run_dir={status.get('run_dir')} "
            f"process={_number_text(status.get('process_status'))} "
            f"pid={_number_text(status.get('pid'))} "
            f"latest_step={_number_text(trace.get('trace_max_global_step'))} "
            f"max_steps={_number_text(trace.get('max_steps'))} "
            f"progress={_number_text(trace.get('progress'))} "
            f"log_latest_step={_number_text((status.get('log_progress') or {}).get('log_latest_step'))} "
            f"log_progress={_number_text((status.get('log_progress') or {}).get('log_progress'))} "
            f"last_loss={_number_text(trace.get('trace_last_loss'))} "
            f"last_eval_loss={_number_text(trace.get('trace_last_eval_loss'))} "
            f"min_eval_loss={_number_text(trace.get('trace_min_eval_loss'))} "
            f"guard_count={_number_text(trace.get('training_loss_guard_count'))} "
            f"checkpoint_card={_number_text(status.get('checkpoint_card_status'))} "
            f"checkpoints={_number_text(status.get('checkpoint_count'))} "
            f"latest_checkpoint={_number_text(latest_checkpoint_name)} "
            f"final_ready={_number_text(status.get('final_checkpoint_ready'))} "
            f"disk_free_gb={_number_text(status.get('disk_free_gb'))}"
        )
    ]
    eval_points = trace.get("trace_eval_loss_points")
    if isinstance(eval_points, list) and tail_evals > 0:
        for point in eval_points[-tail_evals:]:
            if not isinstance(point, dict):
                continue
            lines.append(
                (
                    "hf_gpt2_ft_run_eval "
                    f"step={_number_text(point.get('step'))} "
                    f"eval_loss={_number_text(point.get('eval_loss'))} "
                    f"eval_runtime={_number_text(point.get('eval_runtime'))}"
                )
            )
    process_wait = status.get("checkpoint_card_process_wait")
    if isinstance(process_wait, dict):
        lines.append(
            (
                "hf_gpt2_ft_run_wait "
                f"status={_number_text(process_wait.get('status'))} "
                f"pid={_number_text(process_wait.get('pid'))} "
                f"waited_seconds={_number_text(process_wait.get('waited_seconds'))}"
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status = summarize_run(args)
    lines = status_lines(status, tail_evals=args.tail_evals)
    payload = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_gpt2_ft_run_status_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_gpt2_ft_run_status_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

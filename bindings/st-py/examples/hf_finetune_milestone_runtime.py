#!/usr/bin/env python3
"""Resolve, execute, and archive a Hugging Face FT milestone runtime handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import spiraltorch as st  # noqa: E402
import hf_gpt2_finetune_milestone_runtime as _legacy  # noqa: E402
from hf_gpt2_finetune_milestone_runtime import *  # noqa: F401,F403,E402


ROW_TYPE_MAP = {
    "hf_gpt2_finetune_milestone_capture": "hf_finetune_milestone_capture",
    "hf_gpt2_finetune_milestone_handoff": "hf_finetune_milestone_handoff",
    "hf_gpt2_finetune_milestone_handoff_execution": (
        "hf_finetune_milestone_handoff_execution"
    ),
    "hf_gpt2_finetune_milestone_runtime": "hf_finetune_milestone_runtime",
}
LEGACY_ROW_TYPE_MAP = {value: key for key, value in ROW_TYPE_MAP.items()}
LINE_PREFIX_MAP = {
    "hf_gpt2_ft_monitor": "hf_ft_monitor",
    "hf_gpt2_ft_milestone_capture": "hf_ft_milestone_capture",
    "hf_gpt2_ft_milestone_handoff": "hf_ft_milestone_handoff",
    "hf_gpt2_ft_milestone_handoff_execution": (
        "hf_ft_milestone_handoff_execution"
    ),
    "hf_gpt2_ft_milestone_runtime": "hf_ft_milestone_runtime",
}


def parse_args(argv: list[str] | None = None):
    return _legacy.parse_args(argv)


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


def build_report(
    args,
    *,
    package_runner: Callable[..., dict[str, Any]] | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    report = st.hf_finetune_milestone_runtime_from_run_dir_report(
        args.run_dir,
        next_run_dir=args.next_run_dir,
        direct=args.direct,
        eval_watch=args.eval_watch,
        checkpoint_watch=args.checkpoint_watch,
        final_watch=args.final_watch,
        wait_launch=args.wait_launch,
        milestone_step=args.milestone_step,
        label=args.label,
        label_prefix=args.label_prefix,
        checkpoint=args.checkpoint,
        compare_with_sweep=args.compare_with_sweep,
        compare_with_label=args.compare_with_label,
        curve_out=args.curve_out,
        curve_lines_out=args.curve_lines_out,
        trainer_trace_jsonl=args.trainer_trace_jsonl,
        run_card=args.run_card,
        top_n=args.top_n,
        wait=args.wait,
        dry_run=args.dry_run,
        execute=args.execute,
        use_package_api=args.command_backend == "package",
        cwd=args.cwd,
        timeout=args.timeout,
        package_runner=package_runner,
        runner=runner,
    )
    return _genericize_payload(report)


def _runtime_lines(report: dict[str, Any]) -> list[str]:
    legacy_report = _legacyize_payload(report)
    return _genericize_lines(st.hf_finetune_milestone_runtime_lines(legacy_report))


def write_report(
    report: dict[str, Any],
    args,
) -> tuple[Path, Path]:
    archived = dict(report)
    root = Path(args.run_dir or archived.get("run_dir") or ".")
    token = archived.get("milestone_step")
    token = "unknown" if token is None else str(token)
    out_path = Path(args.out or root / f"milestone-{token}-runtime.json")
    lines_path = Path(args.lines_out or root / f"milestone-{token}-runtime.txt")
    archived["out"] = str(out_path)
    archived["lines_out"] = str(lines_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(archived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines_path.write_text("\n".join(_runtime_lines(archived)) + "\n", encoding="utf-8")
    report.clear()
    report.update(archived)
    return out_path, lines_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_path, lines_path = write_report(report, args)
    lines = _runtime_lines(report)
    if not args.quiet:
        print("\n".join(lines))
    print(f"hf_ft_milestone_runtime_json {out_path}")
    print(f"hf_ft_milestone_runtime_lines {lines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

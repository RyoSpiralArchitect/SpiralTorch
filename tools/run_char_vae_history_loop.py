#!/usr/bin/env python3
"""Run a char VAE command bundle through bounded history-guided steps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.command_bundle_history_loop.v1"
DEFAULT_FAIL_ON_FINAL_ACTIONS = ("review_before_continuing", "inspect_history")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _load_runner() -> Any:
    module_path = Path(__file__).resolve().with_name(
        "run_char_vae_command_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_spiraltorch_char_vae_command_bundle_runner",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load command bundle runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def _csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def _token_matches_path(token: str, expected_path: Path) -> bool:
    token_path = Path(token)
    if not token_path.is_absolute():
        return False
    try:
        return token_path.resolve() == expected_path.resolve()
    except OSError:
        return token_path.absolute() == expected_path.resolve()


def _continuation_argv_from_report(command_dir: Path, report_path: Path) -> list[str]:
    payload = _read_json(report_path)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{report_path} is not a char VAE history loop report")
    report_command_dir = payload.get("command_dir")
    if not isinstance(report_command_dir, str) or not _token_matches_path(
        report_command_dir,
        command_dir,
    ):
        raise ValueError("run_loop report belongs to a different command bundle")
    handoff_status = payload.get("handoff_status")
    if handoff_status not in {None, "continuation_ready"}:
        raise ValueError(f"run_loop report is not continuation-ready: {handoff_status}")
    if payload.get("final_next_action_runnable") is not True:
        raise ValueError("run_loop report does not expose a runnable final action")
    continuation_command = payload.get("continuation_command")
    if not isinstance(continuation_command, str) or not continuation_command:
        raise ValueError("run_loop report does not contain a continuation command")
    tokens = shlex.split(continuation_command)
    if "--resume-from-report" in tokens:
        raise ValueError("run_loop continuation command cannot recursively resume")
    script_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if Path(token).name == Path(__file__).name
        ),
        None,
    )
    if script_index is None:
        raise ValueError("run_loop continuation command does not call this loop runner")
    argv = tokens[script_index + 1 :]
    if not argv:
        raise ValueError("run_loop continuation command is missing command_dir")
    if not _token_matches_path(argv[0], command_dir):
        raise ValueError("run_loop continuation command targets a different bundle")
    if "--max-steps" not in argv:
        raise ValueError("run_loop continuation command is missing --max-steps")
    return argv


def _remove_cli_option(argv: list[str], option: str, *, takes_value: bool) -> list[str]:
    cleaned: list[str] = []
    index = 0
    prefix = f"{option}="
    while index < len(argv):
        item = argv[index]
        if item == option:
            index += 2 if takes_value else 1
            continue
        if takes_value and item.startswith(prefix):
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return cleaned


def _cli_option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    prefix = f"{option}="
    while index < len(argv):
        item = argv[index]
        if item == option:
            if index + 1 < len(argv):
                values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith(prefix):
            values.append(item[len(prefix) :])
        index += 1
    return values


def _cli_flag_present(argv: list[str], flag: str) -> bool:
    return flag in argv


def _apply_resume_overrides(
    continuation_argv: list[str],
    raw_argv: list[str],
) -> list[str]:
    merged = list(continuation_argv)
    for option in ("--max-steps", "--json-out", "--markdown-out"):
        values = _cli_option_values(raw_argv, option)
        if not values:
            continue
        merged = _remove_cli_option(merged, option, takes_value=True)
        merged.extend([option, values[-1]])
    fail_actions = _cli_option_values(raw_argv, "--fail-on-final-action")
    if fail_actions:
        merged = _remove_cli_option(
            merged,
            "--fail-on-final-action",
            takes_value=True,
        )
        for value in fail_actions:
            merged.extend(["--fail-on-final-action", value])
    for flag in (
        "--no-strict",
        "--dry-run",
        "--json",
        "--write-loop-report",
        "--no-write-inspection-report",
        "--no-write-run-report",
        "--no-write-run-history-report",
        "--fail-on-max-steps-continuation",
    ):
        if _cli_flag_present(raw_argv, flag) and flag not in merged:
            merged.append(flag)
    return merged


def _loop_command_line(
    command_dir: Path,
    *,
    max_steps: int,
    strict: bool,
    dry_run: bool,
    write_inspection_report: bool,
    write_run_report: bool,
    write_run_history_report: bool,
    fail_on_final_actions: list[str] | tuple[str, ...],
    fail_on_max_steps_continuation: bool,
    json_out: Path | None,
    markdown_out: Path | None,
) -> str:
    script_path = Path(__file__).resolve()
    parts = [
        "env",
        "PYTHONNOUSERSITE=1",
        "python3",
        "-P",
        shlex.quote(str(script_path)),
        shlex.quote(str(command_dir)),
        "--max-steps",
        str(max_steps),
    ]
    if not strict:
        parts.append("--no-strict")
    if dry_run:
        parts.append("--dry-run")
    if not write_inspection_report:
        parts.append("--no-write-inspection-report")
    if not write_run_report:
        parts.append("--no-write-run-report")
    if not write_run_history_report:
        parts.append("--no-write-run-history-report")
    if fail_on_final_actions:
        parts.extend(
            [
                "--fail-on-final-action",
                shlex.quote(",".join(fail_on_final_actions)),
            ]
        )
    if fail_on_max_steps_continuation:
        parts.append("--fail-on-max-steps-continuation")
    default_json_out = command_dir / "run_loop.json"
    default_markdown_out = command_dir / "run_loop.md"
    if json_out == default_json_out and markdown_out == default_markdown_out:
        parts.append("--write-loop-report")
    else:
        if json_out is not None:
            parts.extend(["--json-out", shlex.quote(str(json_out))])
        if markdown_out is not None:
            parts.extend(["--markdown-out", shlex.quote(str(markdown_out))])
    return " ".join(parts)


def _resume_report_path(command_dir: Path, value: str | None) -> Path:
    if value is None or value == "":
        return command_dir / "run_loop.json"
    return Path(value).expanduser().resolve()


def _resume_from_report_command_line(command_dir: Path, report_path: Path) -> str:
    script_path = Path(__file__).resolve()
    default_report_path = command_dir / "run_loop.json"
    parts = [
        "env",
        "PYTHONNOUSERSITE=1",
        "python3",
        "-P",
        shlex.quote(str(script_path)),
        shlex.quote(str(command_dir)),
        "--resume-from-report",
    ]
    if not _token_matches_path(report_path, default_report_path):
        parts.append(shlex.quote(str(report_path.resolve())))
    return " ".join(parts)


def _md_cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _run_history_next_action(summary: dict[str, Any]) -> dict[str, Any]:
    history_summary = _mapping(summary.get("run_history_summary"))
    next_action = _mapping(history_summary.get("next_action"))
    if next_action:
        return next_action
    return _mapping(summary.get("history_next_action"))


def _step_record(
    index: int,
    *,
    returncode: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    execution_summary = _mapping(summary.get("execution_summary"))
    selected_next = _mapping(summary.get("selected_execution_next_command"))
    next_action = _run_history_next_action(summary)
    return {
        "index": index,
        "returncode": returncode,
        "target": summary.get("target"),
        "requested_target": summary.get("requested_target"),
        "target_kind": summary.get("target_kind"),
        "script_path": summary.get("script_path"),
        "dry_run": summary.get("dry_run"),
        "executed": summary.get("executed"),
        "error": summary.get("error"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "duration_seconds": summary.get("duration_seconds"),
        "history_next_action": summary.get("history_next_action"),
        "history_next_action_error": summary.get("history_next_action_error"),
        "run_history_next_action": next_action if next_action else None,
        "execution_evidence_status": summary.get("execution_evidence_status"),
        "execution_evidence_reason": summary.get("execution_evidence_reason"),
        "execution_evidence_should_continue": summary.get(
            "execution_evidence_should_continue"
        ),
        "execution_evidence_mixed_signal": summary.get(
            "execution_evidence_mixed_signal"
        ),
        "execution_summary_path": execution_summary.get("summary_path"),
        "execution_best_config": execution_summary.get("best_config_label"),
        "execution_verdict": execution_summary.get("follow_up_verdict"),
        "execution_guidance_action": execution_summary.get("guidance_action"),
        "selected_execution_next_source": selected_next.get("source"),
        "selected_execution_next_seeds": selected_next.get("default_new_seeds"),
        "selected_execution_next_script": selected_next.get("script_path"),
    }


def _stop_reason_for_step(
    runner: Any,
    *,
    returncode: int,
    summary: dict[str, Any],
    dry_run: bool,
) -> str | None:
    if dry_run:
        return "dry_run"
    if returncode != 0:
        if summary.get("history_next_action_error"):
            return "history_next_action_blocked"
        return "step_failed"
    next_action = _run_history_next_action(summary)
    if next_action.get("should_continue") is not True:
        return "history_next_action_stopped"
    if next_action.get("target") not in runner.TARGET_KEYS:
        return "history_next_action_not_runnable"
    return None


def _handoff_status(
    *,
    returncode: int,
    stop_reason: str | None,
    final_next_action: dict[str, Any],
    final_next_action_runnable: bool,
    final_action_failed: bool,
    max_steps_continuation_failed: bool,
) -> tuple[str, str]:
    action = final_next_action.get("action")
    action_reason = final_next_action.get("reason")
    if final_action_failed:
        if action == "review_before_continuing":
            return (
                "needs_review",
                action_reason or "final next action requested review",
            )
        if action == "inspect_history":
            return (
                "needs_inspection",
                action_reason or "final next action requested inspection",
            )
        return (
            "final_action_failed",
            action_reason or f"final next action requested failure: {action}",
        )
    if stop_reason == "dry_run":
        return ("dry_run", "dry-run resolved the next history-guided step")
    if max_steps_continuation_failed:
        return (
            "continuation_ready",
            action_reason or "max steps reached with runnable final next action",
        )
    if final_next_action_runnable:
        return (
            "continuation_ready",
            action_reason or "final next action is runnable",
        )
    if returncode != 0:
        if stop_reason == "history_next_action_blocked":
            return (
                "blocked",
                action_reason or "history next action blocked execution",
            )
        return ("failed", action_reason or "history loop step failed")
    if action == "collect_next_command":
        return (
            "awaiting_next_command",
            action_reason or "latest execution can continue but has no next script",
        )
    if action == "review_before_continuing":
        return (
            "needs_review",
            action_reason or "final next action requested review",
        )
    if action == "inspect_history":
        return (
            "needs_inspection",
            action_reason or "final next action requested inspection",
        )
    if action == "repair_blocker":
        return ("blocked", action_reason or "final next action requests repair")
    if stop_reason == "history_next_action_not_runnable":
        return (
            "not_runnable",
            action_reason or "final next action target is not runnable",
        )
    if stop_reason == "max_steps_reached":
        return ("max_steps_reached", "max steps reached")
    return ("stopped", action_reason or stop_reason or "history loop stopped")


def render_markdown(summary: dict[str, Any]) -> str:
    next_action = _mapping(summary.get("final_next_action"))
    lines = [
        "# Char VAE History Loop Runner",
        "",
        f"- command_dir: {_fmt(summary.get('command_dir'))}",
        f"- handoff_status: {_fmt(summary.get('handoff_status'))}",
        f"- handoff_reason: {_fmt(summary.get('handoff_reason'))}",
        f"- max_steps: {_fmt(summary.get('max_steps'))}",
        f"- strict: {_fmt(summary.get('strict'))}",
        f"- dry_run: {_fmt(summary.get('dry_run'))}",
        f"- started_at: {_fmt(summary.get('started_at'))}",
        f"- finished_at: {_fmt(summary.get('finished_at'))}",
        f"- duration_seconds: {_fmt(summary.get('duration_seconds'))}",
        f"- step_count: {_fmt(summary.get('step_count'))}",
        f"- executed_count: {_fmt(summary.get('executed_count'))}",
        f"- success_count: {_fmt(summary.get('success_count'))}",
        f"- failure_count: {_fmt(summary.get('failure_count'))}",
        f"- stop_reason: {_fmt(summary.get('stop_reason'))}",
        f"- fail_on_final_actions: {_fmt(summary.get('fail_on_final_actions'))}",
        f"- final_action_failed: {_fmt(summary.get('final_action_failed'))}",
        (
            "- fail_on_max_steps_continuation: "
            f"{_fmt(summary.get('fail_on_max_steps_continuation'))}"
        ),
        (
            "- max_steps_continuation_failed: "
            f"{_fmt(summary.get('max_steps_continuation_failed'))}"
        ),
        f"- returncode: {_fmt(summary.get('returncode'))}",
        f"- error: {_fmt(summary.get('error'))}",
        f"- final_next_action: {_fmt(next_action.get('action'))}",
        f"- final_next_action_reason: {_fmt(next_action.get('reason'))}",
        f"- final_next_action_target: {_fmt(next_action.get('target'))}",
        f"- final_next_action_command_source: {_fmt(next_action.get('command_source'))}",
        f"- final_next_action_script_path: {_fmt(next_action.get('script_path'))}",
        f"- final_next_action_default_new_seeds: {_fmt(next_action.get('default_new_seeds'))}",
        f"- final_next_action_should_continue: {_fmt(next_action.get('should_continue'))}",
        f"- final_next_action_runnable: {_fmt(summary.get('final_next_action_runnable'))}",
        f"- continuation_command: {_fmt(summary.get('continuation_command'))}",
        f"- resume_from_report_command: {_fmt(summary.get('resume_from_report_command'))}",
        f"- json_path: {_fmt(summary.get('json_path'))}",
        f"- markdown_path: {_fmt(summary.get('markdown_path'))}",
        "",
        "## Steps",
        "",
        "| # | target | kind | executed | returncode | evidence | next_action | next_target | error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for step in summary.get("steps") or []:
        if not isinstance(step, dict):
            continue
        next_step_action = _mapping(step.get("run_history_next_action"))
        row = [
            step.get("index"),
            step.get("target"),
            step.get("target_kind"),
            step.get("executed"),
            step.get("returncode"),
            step.get("execution_evidence_status"),
            next_step_action.get("action"),
            next_step_action.get("target"),
            step.get("error"),
        ]
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    lines.append("")
    return "\n".join(lines)


def write_loop_artifacts(
    summary: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
) -> dict[str, Any]:
    summary = dict(summary)
    summary["json_path"] = str(json_out) if json_out is not None else None
    summary["markdown_path"] = str(markdown_out) if markdown_out is not None else None
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _failure_summary(
    command_dir: Path,
    *,
    max_steps: int,
    strict: bool,
    dry_run: bool,
    fail_on_final_actions: list[str],
    fail_on_max_steps_continuation: bool,
    error: str,
    stop_reason: str = "loop_setup_failed",
) -> dict[str, Any]:
    finished_at = _utc_now()
    return {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "handoff_status": "failed",
        "handoff_reason": error,
        "max_steps": max_steps,
        "strict": strict,
        "dry_run": dry_run,
        "started_at": finished_at,
        "finished_at": finished_at,
        "duration_seconds": 0.0,
        "step_count": 0,
        "executed_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "stop_reason": stop_reason,
        "fail_on_final_actions": fail_on_final_actions,
        "final_action_failed": False,
        "fail_on_max_steps_continuation": fail_on_max_steps_continuation,
        "max_steps_continuation_failed": False,
        "returncode": 1,
        "error": error,
        "final_next_action": None,
        "final_next_action_runnable": False,
        "continuation_command": None,
        "resume_from_report_command": None,
        "latest_run_history_summary": None,
        "latest_run_json_path": None,
        "latest_run_markdown_path": None,
        "latest_run_history_jsonl_path": None,
        "latest_run_history_markdown_path": None,
        "latest_run_history_summary_path": None,
        "steps": [],
    }


def run_loop(
    command_dir: Path,
    *,
    max_steps: int,
    strict: bool,
    dry_run: bool,
    write_inspection_report: bool,
    write_run_report: bool,
    write_run_history_report: bool,
    fail_on_final_actions: list[str] | tuple[str, ...],
    fail_on_max_steps_continuation: bool,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    if max_steps < 1:
        raise ValueError("--max-steps must be at least 1")

    runner = _load_runner()
    command_dir = command_dir.resolve()
    started_at = _utc_now()
    started_perf = time.perf_counter()
    steps: list[dict[str, Any]] = []
    latest_summary: dict[str, Any] = {}
    stop_reason: str | None = None
    returncode = 0

    for index in range(1, max_steps + 1):
        step_returncode, step_summary = runner.run_bundle(
            command_dir,
            target="next",
            strict=strict,
            dry_run=dry_run,
            json_mode=True,
            write_inspection_report=write_inspection_report,
            write_run_report=write_run_report,
            append_run_history=not dry_run,
            write_run_history_report=write_run_history_report,
            use_history_next_action=True,
        )
        latest_summary = step_summary
        steps.append(_step_record(index, returncode=step_returncode, summary=step_summary))
        stop_reason = _stop_reason_for_step(
            runner,
            returncode=step_returncode,
            summary=step_summary,
            dry_run=dry_run,
        )
        if stop_reason is not None:
            returncode = step_returncode
            break
    else:
        stop_reason = "max_steps_reached"

    final_next_action = _run_history_next_action(latest_summary)
    fail_on_final_actions = list(fail_on_final_actions)
    final_action = final_next_action.get("action")
    final_target = final_next_action.get("target")
    final_next_action_runnable = (
        final_next_action.get("should_continue") is True
        and final_target in runner.TARGET_KEYS
    )
    final_action_failed = (
        returncode == 0
        and isinstance(final_action, str)
        and final_action in set(fail_on_final_actions)
    )
    max_steps_continuation_failed = (
        returncode == 0
        and fail_on_max_steps_continuation
        and stop_reason == "max_steps_reached"
        and final_next_action_runnable
    )
    continuation_command = (
        _loop_command_line(
            command_dir,
            max_steps=max_steps,
            strict=strict,
            dry_run=dry_run,
            write_inspection_report=write_inspection_report,
            write_run_report=write_run_report,
            write_run_history_report=write_run_history_report,
            fail_on_final_actions=fail_on_final_actions,
            fail_on_max_steps_continuation=fail_on_max_steps_continuation,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        if final_next_action_runnable
        else None
    )
    resume_from_report_command = (
        _resume_from_report_command_line(command_dir, json_out)
        if final_next_action_runnable and json_out is not None
        else None
    )
    error = latest_summary.get("error") if returncode != 0 else None
    if final_action_failed:
        returncode = 1
        error = f"final next action requested failure: {final_action}"
    elif max_steps_continuation_failed:
        returncode = 1
        error = f"max steps reached with runnable final next action: {final_action}"
    handoff_status, handoff_reason = _handoff_status(
        returncode=returncode,
        stop_reason=stop_reason,
        final_next_action=final_next_action,
        final_next_action_runnable=final_next_action_runnable,
        final_action_failed=final_action_failed,
        max_steps_continuation_failed=max_steps_continuation_failed,
    )
    summary = {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "handoff_status": handoff_status,
        "handoff_reason": handoff_reason,
        "max_steps": max_steps,
        "strict": strict,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started_perf, 6),
        "step_count": len(steps),
        "executed_count": sum(1 for step in steps if step.get("executed")),
        "success_count": sum(1 for step in steps if step.get("returncode") == 0),
        "failure_count": sum(1 for step in steps if step.get("returncode") != 0),
        "stop_reason": stop_reason,
        "fail_on_final_actions": fail_on_final_actions,
        "final_action_failed": final_action_failed,
        "fail_on_max_steps_continuation": fail_on_max_steps_continuation,
        "max_steps_continuation_failed": max_steps_continuation_failed,
        "returncode": returncode,
        "error": error,
        "final_next_action": final_next_action if final_next_action else None,
        "final_next_action_runnable": final_next_action_runnable,
        "continuation_command": continuation_command,
        "resume_from_report_command": resume_from_report_command,
        "latest_run_history_summary": latest_summary.get("run_history_summary"),
        "latest_run_json_path": latest_summary.get("run_json_path"),
        "latest_run_markdown_path": latest_summary.get("run_markdown_path"),
        "latest_run_history_jsonl_path": latest_summary.get("run_history_jsonl_path"),
        "latest_run_history_markdown_path": latest_summary.get(
            "run_history_markdown_path"
        ),
        "latest_run_history_summary_path": latest_summary.get(
            "run_history_summary_path"
        ),
        "steps": steps,
    }
    summary = write_loop_artifacts(summary, json_out=json_out, markdown_out=markdown_out)
    return returncode, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_dir", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="maximum history-guided steps to execute",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="require only bundle_ready instead of strict_ready before each step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the next history-guided step without executing or appending history",
    )
    parser.add_argument(
        "--resume-from-report",
        nargs="?",
        const="",
        default=None,
        metavar="REPORT_JSON",
        help=(
            "read run_loop.json, or REPORT_JSON when provided, and run its "
            "validated continuation_command when the previous handoff is "
            "continuation-ready"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON loop summary",
    )
    parser.add_argument(
        "--write-loop-report",
        action="store_true",
        help="write run_loop.json and run_loop.md into the command bundle",
    )
    parser.add_argument(
        "--no-write-inspection-report",
        action="store_true",
        help="skip rewriting inspection.json and inspection.md before each step",
    )
    parser.add_argument(
        "--no-write-run-report",
        action="store_true",
        help="skip rewriting run.json and run.md for each step",
    )
    parser.add_argument(
        "--no-write-run-history-report",
        action="store_true",
        help="skip rewriting run_history.md and run_history_summary.json",
    )
    parser.add_argument(
        "--fail-on-final-action",
        action="append",
        default=None,
        metavar="ACTION[,ACTION...]",
        help=(
            "return non-zero when the final run_history next_action has one "
            "of these action names; defaults to "
            f"{','.join(DEFAULT_FAIL_ON_FINAL_ACTIONS)}"
        ),
    )
    parser.add_argument(
        "--fail-on-max-steps-continuation",
        action="store_true",
        help=(
            "return non-zero when --max-steps is reached while the final "
            "run_history next_action is still runnable"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    command_dir = args.command_dir.resolve()
    json_out = args.json_out
    markdown_out = args.markdown_out
    if args.write_loop_report or json_out is not None or markdown_out is not None:
        json_out = json_out or command_dir / "run_loop.json"
        markdown_out = markdown_out or command_dir / "run_loop.md"
    fail_on_final_actions = (
        list(DEFAULT_FAIL_ON_FINAL_ACTIONS)
        if args.fail_on_final_action is None
        else _csv_values(args.fail_on_final_action)
    )
    if args.resume_from_report is not None:
        resume_report_path = _resume_report_path(command_dir, args.resume_from_report)
        try:
            continuation_argv = _continuation_argv_from_report(
                command_dir,
                resume_report_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            summary = _failure_summary(
                command_dir,
                max_steps=args.max_steps,
                strict=not args.no_strict,
                dry_run=bool(args.dry_run),
                fail_on_final_actions=fail_on_final_actions,
                fail_on_max_steps_continuation=bool(
                    args.fail_on_max_steps_continuation
                ),
                error=str(exc),
                stop_reason="loop_resume_failed",
            )
            summary = write_loop_artifacts(
                summary,
                json_out=json_out,
                markdown_out=markdown_out,
            )
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(render_markdown(summary), file=sys.stderr)
            return 1
        continuation_argv = _apply_resume_overrides(continuation_argv, raw_argv)
        return main(continuation_argv)
    try:
        returncode, summary = run_loop(
            command_dir,
            max_steps=args.max_steps,
            strict=not args.no_strict,
            dry_run=bool(args.dry_run),
            write_inspection_report=not args.no_write_inspection_report,
            write_run_report=not args.no_write_run_report,
            write_run_history_report=not args.no_write_run_history_report,
            fail_on_final_actions=fail_on_final_actions,
            fail_on_max_steps_continuation=bool(
                args.fail_on_max_steps_continuation
            ),
            json_out=json_out,
            markdown_out=markdown_out,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        summary = _failure_summary(
            command_dir,
            max_steps=args.max_steps,
            strict=not args.no_strict,
            dry_run=bool(args.dry_run),
            fail_on_final_actions=fail_on_final_actions,
            fail_on_max_steps_continuation=bool(
                args.fail_on_max_steps_continuation
            ),
            error=str(exc),
        )
        summary = write_loop_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        returncode = 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.dry_run or returncode != 0:
        stream = sys.stderr if returncode != 0 else sys.stdout
        print(render_markdown(summary), file=stream)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

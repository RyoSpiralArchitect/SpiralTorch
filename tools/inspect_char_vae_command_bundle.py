#!/usr/bin/env python3
"""Inspect a generated char VAE command bundle."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.command_bundle_inspection.v1"
RUN_HISTORY_SUMMARY_SCHEMA = (
    "st.llm_char_vae_context.command_bundle_run_history_summary.v1"
)
RUN_HISTORY_NEXT_ACTION_SCHEMA = (
    "st.llm_char_vae_context.command_bundle_history_next_action.v1"
)
RUN_LOOP_SCHEMA = "st.llm_char_vae_context.command_bundle_history_loop.v1"
RUN_LOOP_RUNNABLE_TARGETS = {"next", "follow-up", "review", "execution-next"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path_from(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _is_executable(path: Path | None) -> bool:
    return path is not None and path.is_file() and os.access(path, os.X_OK)


def _runner_wrapper_status(
    path: Path | None,
    *,
    runner_command: str | None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "present": path is not None,
        "exists": None if path is None else path.is_file(),
        "executable": None if path is None else _is_executable(path),
        "readable": None,
        "contains_runner_command": None,
        "executes_runner_command": None,
        "forwards_arguments": None,
        "error": None,
        "ok": path is None,
    }
    if path is None:
        return status
    if not path.is_file():
        status["ok"] = False
        return status
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        status["readable"] = False
        status["error"] = str(exc)
        status["ok"] = False
        return status
    status["readable"] = True
    status["contains_runner_command"] = (
        runner_command is not None and runner_command in text
    )
    expected_exec = (
        f'exec {runner_command} "$@"' if runner_command is not None else None
    )
    status["executes_runner_command"] = (
        expected_exec is not None and expected_exec in text
    )
    status["forwards_arguments"] = '"$@"' in text or "'$@'" in text
    status["ok"] = bool(
        status["executable"]
        and status["executes_runner_command"]
        and status["forwards_arguments"]
    )
    return status


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "-"
    return str(value)


def _fmt_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    return ", ".join(str(item) for item in value)


def _value(payload: dict[str, Any], *keys: str) -> Any:
    item: Any = payload
    for key in keys:
        if not isinstance(item, dict):
            return None
        item = item.get(key)
    return item


def _check(
    label: str,
    *,
    path: Path | None,
    ok: bool,
    required: bool,
) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path) if path is not None else None,
        "ok": bool(ok),
        "required": bool(required),
    }


def _declared_output(
    label: str,
    path: Path | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path) if path is not None else None,
        "exists": _exists(path),
    }


def _command_from(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _command_tokens(command: str | None) -> tuple[list[str] | None, str | None]:
    if command is None:
        return None, None
    try:
        return shlex.split(command), None
    except ValueError as exc:
        return None, str(exc)


def _has_required_command_token(tokens: list[str] | None, required: str) -> bool:
    if tokens is None:
        return False
    if required.endswith(".py"):
        return any(Path(token).name == required for token in tokens)
    return required in tokens


def _token_matches_path(token: str, expected_path: Path) -> bool:
    token_path = Path(token)
    if not token_path.is_absolute():
        return False
    try:
        return token_path.resolve() == expected_path.resolve()
    except OSError:
        return token_path.absolute() == expected_path.resolve()


def _has_command_dir_token(
    tokens: list[str] | None,
    command_dir: Path | None,
) -> bool | None:
    if command_dir is None:
        return None
    if tokens is None:
        return False
    return any(_token_matches_path(token, command_dir) for token in tokens)


def _declared_command(
    label: str,
    command: str | None,
    *,
    required_flags: tuple[str, ...] = (),
    forbidden_flags: tuple[str, ...] = (),
    command_dir: Path | None = None,
) -> dict[str, Any]:
    tokens, parse_error = _command_tokens(command)
    target_command_dir_ok = _has_command_dir_token(tokens, command_dir)
    missing_required_flags = [
        flag for flag in required_flags if not _has_required_command_token(tokens, flag)
    ]
    forbidden_flags_present = [
        flag for flag in forbidden_flags if tokens is not None and flag in tokens
    ]
    return {
        "label": label,
        "command": command,
        "present": command is not None,
        "tokens": tokens,
        "parse_error": parse_error,
        "target_command_dir": str(command_dir) if command_dir is not None else None,
        "target_command_dir_ok": target_command_dir_ok,
        "required_flags": list(required_flags),
        "missing_required_flags": missing_required_flags,
        "forbidden_flags": list(forbidden_flags),
        "forbidden_flags_present": forbidden_flags_present,
        "ok": command is not None
        and parse_error is None
        and target_command_dir_ok is not False
        and not missing_required_flags
        and not forbidden_flags_present,
    }


def _jsonl_event_count(path: Path | None) -> tuple[int | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        count = 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                return None, f"{path}:{lineno} did not contain a JSON object"
            count += 1
        return count, None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _run_history_summary_status(
    *,
    summary_path: Path | None,
    history_path: Path | None,
) -> dict[str, Any]:
    history_event_count, history_error = _jsonl_event_count(history_path)
    exists = _exists(summary_path)
    status: dict[str, Any] = {
        "path": str(summary_path) if summary_path is not None else None,
        "exists": exists,
        "valid_json": None,
        "schema": None,
        "schema_ok": None,
        "total_runs": None,
        "next_action": None,
        "next_action_reason": None,
        "next_action_target": None,
        "next_action_command_source": None,
        "next_action_script_path": None,
        "next_action_default_new_seeds": None,
        "next_action_should_continue": None,
        "next_action_schema_ok": None,
        "history_event_count": history_event_count,
        "matches_history_event_count": None,
        "error": history_error,
    }
    if summary_path is None or not exists:
        return status
    try:
        payload = _read_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        status["valid_json"] = False
        status["error"] = str(exc)
        return status
    schema = payload.get("schema")
    total_runs = payload.get("total_runs")
    next_action = payload.get("next_action")
    next_action = next_action if isinstance(next_action, dict) else {}
    status.update(
        {
            "valid_json": True,
            "schema": schema,
            "schema_ok": schema == RUN_HISTORY_SUMMARY_SCHEMA,
            "total_runs": total_runs,
            "next_action": next_action.get("action"),
            "next_action_reason": next_action.get("reason"),
            "next_action_target": next_action.get("target"),
            "next_action_command_source": next_action.get("command_source"),
            "next_action_script_path": next_action.get("script_path"),
            "next_action_default_new_seeds": next_action.get("default_new_seeds"),
            "next_action_should_continue": next_action.get("should_continue"),
            "next_action_schema_ok": (
                next_action.get("schema") == RUN_HISTORY_NEXT_ACTION_SCHEMA
                if next_action
                else None
            ),
        }
    )
    if isinstance(total_runs, int) and history_event_count is not None:
        status["matches_history_event_count"] = total_runs == history_event_count
    return status


def _derived_run_loop_handoff(
    payload: dict[str, Any],
    final_next_action: dict[str, Any],
) -> tuple[str | None, str | None]:
    explicit_status = payload.get("handoff_status")
    explicit_reason = payload.get("handoff_reason")
    if explicit_status is not None or explicit_reason is not None:
        return (
            explicit_status if isinstance(explicit_status, str) else None,
            explicit_reason if isinstance(explicit_reason, str) else None,
        )
    action = final_next_action.get("action")
    action_reason = final_next_action.get("reason")
    stop_reason = payload.get("stop_reason")
    returncode = payload.get("returncode")
    final_action_failed = payload.get("final_action_failed") is True
    max_steps_continuation_failed = (
        payload.get("max_steps_continuation_failed") is True
    )
    final_next_action_runnable = payload.get("final_next_action_runnable")
    if not isinstance(final_next_action_runnable, bool):
        final_next_action_runnable = (
            final_next_action.get("should_continue") is True
            and final_next_action.get("target") in RUN_LOOP_RUNNABLE_TARGETS
        )
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
        return "dry_run", "dry-run resolved the next history-guided step"
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
        return "failed", action_reason or "history loop step failed"
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
        return "blocked", action_reason or "final next action requests repair"
    if stop_reason == "history_next_action_not_runnable":
        return (
            "not_runnable",
            action_reason or "final next action target is not runnable",
        )
    if stop_reason == "max_steps_reached":
        return "max_steps_reached", "max steps reached"
    if stop_reason is not None or action_reason is not None:
        return "stopped", action_reason or str(stop_reason)
    return None, None


def _run_loop_status(
    *,
    summary_path: Path | None,
) -> dict[str, Any]:
    exists = _exists(summary_path)
    status: dict[str, Any] = {
        "path": str(summary_path) if summary_path is not None else None,
        "exists": exists,
        "valid_json": None,
        "schema": None,
        "schema_ok": None,
        "command_dir": None,
        "handoff_status": None,
        "handoff_reason": None,
        "max_steps": None,
        "step_count": None,
        "executed_count": None,
        "success_count": None,
        "failure_count": None,
        "stop_reason": None,
        "fail_on_final_actions": None,
        "final_action_failed": None,
        "fail_on_max_steps_continuation": None,
        "max_steps_continuation_failed": None,
        "returncode": None,
        "final_next_action": None,
        "final_next_action_reason": None,
        "final_next_action_target": None,
        "final_next_action_command_source": None,
        "final_next_action_script_path": None,
        "final_next_action_default_new_seeds": None,
        "final_next_action_should_continue": None,
        "final_next_action_runnable": None,
        "continuation_command": None,
        "error": None,
    }
    if summary_path is None or not exists:
        return status
    try:
        payload = _read_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        status["valid_json"] = False
        status["error"] = str(exc)
        return status
    final_next_action = payload.get("final_next_action")
    final_next_action = final_next_action if isinstance(final_next_action, dict) else {}
    final_next_action_runnable = payload.get("final_next_action_runnable")
    if not isinstance(final_next_action_runnable, bool):
        final_next_action_runnable = (
            final_next_action.get("should_continue") is True
            and final_next_action.get("target") in RUN_LOOP_RUNNABLE_TARGETS
        )
    handoff_status, handoff_reason = _derived_run_loop_handoff(
        payload,
        final_next_action,
    )
    schema = payload.get("schema")
    status.update(
        {
            "valid_json": True,
            "schema": schema,
            "schema_ok": schema == RUN_LOOP_SCHEMA,
            "command_dir": payload.get("command_dir"),
            "handoff_status": handoff_status,
            "handoff_reason": handoff_reason,
            "max_steps": payload.get("max_steps"),
            "step_count": payload.get("step_count"),
            "executed_count": payload.get("executed_count"),
            "success_count": payload.get("success_count"),
            "failure_count": payload.get("failure_count"),
            "stop_reason": payload.get("stop_reason"),
            "fail_on_final_actions": payload.get("fail_on_final_actions"),
            "final_action_failed": payload.get("final_action_failed"),
            "fail_on_max_steps_continuation": payload.get(
                "fail_on_max_steps_continuation"
            ),
            "max_steps_continuation_failed": payload.get(
                "max_steps_continuation_failed"
            ),
            "returncode": payload.get("returncode"),
            "error": payload.get("error"),
            "final_next_action": final_next_action.get("action"),
            "final_next_action_reason": final_next_action.get("reason"),
            "final_next_action_target": final_next_action.get("target"),
            "final_next_action_command_source": final_next_action.get(
                "command_source"
            ),
            "final_next_action_script_path": final_next_action.get("script_path"),
            "final_next_action_default_new_seeds": final_next_action.get(
                "default_new_seeds"
            ),
            "final_next_action_should_continue": final_next_action.get(
                "should_continue"
            ),
            "final_next_action_runnable": final_next_action_runnable,
            "continuation_command": payload.get("continuation_command"),
        }
    )
    return status


def inspect_bundle(command_dir: Path) -> dict[str, Any]:
    command_dir = command_dir.resolve()
    manifest_path = command_dir / "recommendation.json"
    manifest = _read_json(manifest_path)

    command_scripts = manifest.get("command_scripts")
    command_scripts = command_scripts if isinstance(command_scripts, dict) else {}
    comparison = manifest.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    recommendation = manifest.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}

    comparison_json_path = _path_from(command_scripts.get("comparison_json_path"))
    comparison_markdown_path = _path_from(
        command_scripts.get("comparison_markdown_path")
    )
    readme_path = (
        _path_from(command_scripts.get("readme_path")) or command_dir / "README.md"
    )
    next_path = _path_from(command_scripts.get("next_path"))
    follow_up_path = _path_from(command_scripts.get("follow_up_path"))
    review_path = _path_from(command_scripts.get("review_path"))
    runner_path = _path_from(command_scripts.get("runner_path"))
    runner_command = _command_from(command_scripts.get("runner_command"))
    history_next_action_runner_path = _path_from(
        command_scripts.get("history_next_action_runner_path")
    )
    history_next_action_command = _command_from(
        command_scripts.get("history_next_action_command")
    )
    history_loop_runner_path = _path_from(
        command_scripts.get("history_loop_runner_path")
    )
    history_loop_command = _command_from(command_scripts.get("history_loop_command"))
    history_report_command = _command_from(
        command_scripts.get("history_report_command")
    )
    run_json_path = _path_from(command_scripts.get("run_json_path"))
    run_markdown_path = _path_from(command_scripts.get("run_markdown_path"))
    run_history_jsonl_path = _path_from(
        command_scripts.get("run_history_jsonl_path")
    )
    run_history_markdown_path = _path_from(
        command_scripts.get("run_history_markdown_path")
    )
    run_history_summary_path = _path_from(
        command_scripts.get("run_history_summary_path")
    )
    run_loop_json_path = _path_from(command_scripts.get("run_loop_json_path"))
    run_loop_markdown_path = _path_from(command_scripts.get("run_loop_markdown_path"))
    runner_wrapper_status = _runner_wrapper_status(
        runner_path,
        runner_command=runner_command,
    )
    history_next_action_runner_status = _runner_wrapper_status(
        history_next_action_runner_path,
        runner_command=history_next_action_command,
    )
    history_loop_runner_status = _runner_wrapper_status(
        history_loop_runner_path,
        runner_command=history_loop_command,
    )

    chain_sources = comparison.get("chain_sources")
    if not isinstance(chain_sources, list):
        chain_sources = []
    chain_source_paths = [
        Path(source) for source in chain_sources if isinstance(source, str)
    ]
    missing_chain_sources = [
        str(path) for path in chain_source_paths if not path.exists()
    ]

    checks = [
        _check(
            "manifest",
            path=manifest_path,
            ok=manifest_path.is_file(),
            required=True,
        ),
        _check(
            "comparison_json",
            path=comparison_json_path,
            ok=_exists(comparison_json_path),
            required=True,
        ),
        _check(
            "comparison_markdown",
            path=comparison_markdown_path,
            ok=_exists(comparison_markdown_path),
            required=True,
        ),
        _check("readme", path=readme_path, ok=_exists(readme_path), required=True),
        _check(
            "next_script",
            path=next_path,
            ok=next_path is None or _is_executable(next_path),
            required=False,
        ),
        _check(
            "follow_up_script",
            path=follow_up_path,
            ok=follow_up_path is None or _is_executable(follow_up_path),
            required=False,
        ),
        _check(
            "review_script",
            path=review_path,
            ok=review_path is None or _is_executable(review_path),
            required=False,
        ),
        _check(
            "runner_script",
            path=runner_path,
            ok=bool(runner_wrapper_status["ok"]),
            required=False,
        ),
        _check(
            "history_next_action_runner_script",
            path=history_next_action_runner_path,
            ok=bool(history_next_action_runner_status["ok"]),
            required=False,
        ),
        _check(
            "history_loop_runner_script",
            path=history_loop_runner_path,
            ok=bool(history_loop_runner_status["ok"]),
            required=False,
        ),
        _check(
            "chain_sources",
            path=None,
            ok=bool(chain_source_paths) and not missing_chain_sources,
            required=True,
        ),
    ]
    declared_outputs = [
        _declared_output("run_json", run_json_path),
        _declared_output("run_markdown", run_markdown_path),
        _declared_output("run_history_jsonl", run_history_jsonl_path),
        _declared_output("run_history_markdown", run_history_markdown_path),
        _declared_output("run_history_summary", run_history_summary_path),
        _declared_output("run_loop_json", run_loop_json_path),
        _declared_output("run_loop_markdown", run_loop_markdown_path),
    ]
    declared_commands = [
        _declared_command(
            "runner_command",
            runner_command,
            required_flags=(
                "run_char_vae_command_bundle.py",
                "--write-inspection-report",
                "--write-run-report",
                "--append-run-history",
                "--write-run-history-report",
            ),
            command_dir=command_dir,
        ),
        _declared_command(
            "history_report_command",
            history_report_command,
            required_flags=("run_char_vae_command_bundle.py", "--history-report-only"),
            forbidden_flags=("--append-run-history",),
            command_dir=command_dir,
        ),
        _declared_command(
            "history_next_action_command",
            history_next_action_command,
            required_flags=(
                "run_char_vae_command_bundle.py",
                "--use-history-next-action",
                "--write-inspection-report",
                "--write-run-report",
                "--append-run-history",
                "--write-run-history-report",
            ),
            command_dir=command_dir,
        ),
        _declared_command(
            "history_loop_command",
            history_loop_command,
            required_flags=(
                "run_char_vae_history_loop.py",
                "--max-steps",
                "--fail-on-max-steps-continuation",
                "--write-loop-report",
            ),
            command_dir=command_dir,
        ),
    ]
    declared_command_issues = [
        command["label"]
        for command in declared_commands
        if command["present"] and not command["ok"]
    ]
    missing_required = [
        check["label"] for check in checks if check["required"] and not check["ok"]
    ]
    missing_optional = [
        check["label"] for check in checks if not check["required"] and not check["ok"]
    ]
    missing_optional.extend(declared_command_issues)
    run_history_summary_status = _run_history_summary_status(
        summary_path=run_history_summary_path,
        history_path=run_history_jsonl_path,
    )
    run_loop_status = _run_loop_status(summary_path=run_loop_json_path)
    return {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "manifest_path": str(manifest_path),
        "bundle_ready": not missing_required,
        "strict_ready": not missing_required and not missing_optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "checks": checks,
        "action": recommendation.get("action"),
        "next_kind": command_scripts.get("next_kind"),
        "written_count": command_scripts.get("written_count"),
        "comparison_json_path": str(comparison_json_path)
        if comparison_json_path is not None
        else None,
        "comparison_markdown_path": str(comparison_markdown_path)
        if comparison_markdown_path is not None
        else None,
        "runner_command": runner_command,
        "runner_path": str(runner_path) if runner_path is not None else None,
        "runner_wrapper_status": runner_wrapper_status,
        "history_next_action_command": history_next_action_command,
        "history_next_action_runner_path": (
            str(history_next_action_runner_path)
            if history_next_action_runner_path is not None
            else None
        ),
        "history_next_action_runner_status": history_next_action_runner_status,
        "history_loop_command": history_loop_command,
        "history_loop_runner_path": (
            str(history_loop_runner_path)
            if history_loop_runner_path is not None
            else None
        ),
        "history_loop_runner_status": history_loop_runner_status,
        "history_report_command": history_report_command,
        "declared_commands": declared_commands,
        "declared_command_issues": declared_command_issues,
        "declared_outputs": declared_outputs,
        "run_json_path": str(run_json_path) if run_json_path is not None else None,
        "run_markdown_path": (
            str(run_markdown_path) if run_markdown_path is not None else None
        ),
        "run_history_jsonl_path": (
            str(run_history_jsonl_path)
            if run_history_jsonl_path is not None
            else None
        ),
        "run_history_markdown_path": (
            str(run_history_markdown_path)
            if run_history_markdown_path is not None
            else None
        ),
        "run_history_summary_path": (
            str(run_history_summary_path)
            if run_history_summary_path is not None
            else None
        ),
        "run_loop_json_path": (
            str(run_loop_json_path) if run_loop_json_path is not None else None
        ),
        "run_loop_markdown_path": (
            str(run_loop_markdown_path)
            if run_loop_markdown_path is not None
            else None
        ),
        "run_history_summary_status": run_history_summary_status,
        "run_loop_status": run_loop_status,
        "chain_source_count": len(chain_source_paths),
        "missing_chain_sources": missing_chain_sources,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Char VAE Command Bundle Inspection",
        "",
        f"- schema: {summary.get('schema')}",
        f"- command_dir: {_fmt(summary.get('command_dir'))}",
        f"- manifest_path: {_fmt(summary.get('manifest_path'))}",
        f"- bundle_ready: {_fmt(summary.get('bundle_ready'))}",
        f"- strict_ready: {_fmt(summary.get('strict_ready'))}",
        f"- action: {_fmt(summary.get('action'))}",
        f"- next_kind: {_fmt(summary.get('next_kind'))}",
        f"- written_count: {_fmt(summary.get('written_count'))}",
        f"- comparison_json_path: {_fmt(summary.get('comparison_json_path'))}",
        f"- comparison_markdown_path: {_fmt(summary.get('comparison_markdown_path'))}",
        f"- runner_command: {_fmt(summary.get('runner_command'))}",
        f"- runner_path: {_fmt(summary.get('runner_path'))}",
        (
            "- runner_wrapper_ok: "
            f"{_fmt(_value(summary, 'runner_wrapper_status', 'ok'))}"
        ),
        (
            "- runner_wrapper_contains_runner_command: "
            f"{_fmt(_value(summary, 'runner_wrapper_status', 'contains_runner_command'))}"
        ),
        (
            "- runner_wrapper_executes_runner_command: "
            f"{_fmt(_value(summary, 'runner_wrapper_status', 'executes_runner_command'))}"
        ),
        (
            "- runner_wrapper_forwards_arguments: "
            f"{_fmt(_value(summary, 'runner_wrapper_status', 'forwards_arguments'))}"
        ),
        (
            "- runner_wrapper_error: "
            f"{_fmt(_value(summary, 'runner_wrapper_status', 'error'))}"
        ),
        f"- history_next_action_command: {_fmt(summary.get('history_next_action_command'))}",
        f"- history_next_action_runner_path: {_fmt(summary.get('history_next_action_runner_path'))}",
        (
            "- history_next_action_runner_ok: "
            f"{_fmt(_value(summary, 'history_next_action_runner_status', 'ok'))}"
        ),
        (
            "- history_next_action_runner_executes_command: "
            f"{_fmt(_value(summary, 'history_next_action_runner_status', 'executes_runner_command'))}"
        ),
        (
            "- history_next_action_runner_error: "
            f"{_fmt(_value(summary, 'history_next_action_runner_status', 'error'))}"
        ),
        f"- history_loop_command: {_fmt(summary.get('history_loop_command'))}",
        f"- history_loop_runner_path: {_fmt(summary.get('history_loop_runner_path'))}",
        (
            "- history_loop_runner_ok: "
            f"{_fmt(_value(summary, 'history_loop_runner_status', 'ok'))}"
        ),
        (
            "- history_loop_runner_executes_command: "
            f"{_fmt(_value(summary, 'history_loop_runner_status', 'executes_runner_command'))}"
        ),
        (
            "- history_loop_runner_error: "
            f"{_fmt(_value(summary, 'history_loop_runner_status', 'error'))}"
        ),
        f"- history_report_command: {_fmt(summary.get('history_report_command'))}",
        f"- declared_command_issues: {_fmt_list(summary.get('declared_command_issues'))}",
        f"- run_json_path: {_fmt(summary.get('run_json_path'))}",
        f"- run_markdown_path: {_fmt(summary.get('run_markdown_path'))}",
        f"- run_history_jsonl_path: {_fmt(summary.get('run_history_jsonl_path'))}",
        f"- run_history_markdown_path: {_fmt(summary.get('run_history_markdown_path'))}",
        f"- run_history_summary_path: {_fmt(summary.get('run_history_summary_path'))}",
        f"- run_loop_json_path: {_fmt(summary.get('run_loop_json_path'))}",
        f"- run_loop_markdown_path: {_fmt(summary.get('run_loop_markdown_path'))}",
        f"- run_loop_valid_json: {_fmt(_value(summary, 'run_loop_status', 'valid_json'))}",
        f"- run_loop_schema_ok: {_fmt(_value(summary, 'run_loop_status', 'schema_ok'))}",
        f"- run_loop_handoff_status: {_fmt(_value(summary, 'run_loop_status', 'handoff_status'))}",
        f"- run_loop_handoff_reason: {_fmt(_value(summary, 'run_loop_status', 'handoff_reason'))}",
        f"- run_loop_step_count: {_fmt(_value(summary, 'run_loop_status', 'step_count'))}",
        f"- run_loop_executed_count: {_fmt(_value(summary, 'run_loop_status', 'executed_count'))}",
        f"- run_loop_stop_reason: {_fmt(_value(summary, 'run_loop_status', 'stop_reason'))}",
        f"- run_loop_fail_on_final_actions: {_fmt(_value(summary, 'run_loop_status', 'fail_on_final_actions'))}",
        f"- run_loop_final_action_failed: {_fmt(_value(summary, 'run_loop_status', 'final_action_failed'))}",
        f"- run_loop_fail_on_max_steps_continuation: {_fmt(_value(summary, 'run_loop_status', 'fail_on_max_steps_continuation'))}",
        f"- run_loop_max_steps_continuation_failed: {_fmt(_value(summary, 'run_loop_status', 'max_steps_continuation_failed'))}",
        f"- run_loop_returncode: {_fmt(_value(summary, 'run_loop_status', 'returncode'))}",
        f"- run_loop_final_next_action: {_fmt(_value(summary, 'run_loop_status', 'final_next_action'))}",
        f"- run_loop_final_next_action_reason: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_reason'))}",
        f"- run_loop_final_next_action_target: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_target'))}",
        f"- run_loop_final_next_action_command_source: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_command_source'))}",
        f"- run_loop_final_next_action_script_path: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_script_path'))}",
        f"- run_loop_final_next_action_default_new_seeds: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_default_new_seeds'))}",
        f"- run_loop_final_next_action_should_continue: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_should_continue'))}",
        f"- run_loop_final_next_action_runnable: {_fmt(_value(summary, 'run_loop_status', 'final_next_action_runnable'))}",
        f"- run_loop_continuation_command: {_fmt(_value(summary, 'run_loop_status', 'continuation_command'))}",
        f"- run_loop_error: {_fmt(_value(summary, 'run_loop_status', 'error'))}",
        f"- run_history_summary_valid_json: {_fmt(_value(summary, 'run_history_summary_status', 'valid_json'))}",
        f"- run_history_summary_schema_ok: {_fmt(_value(summary, 'run_history_summary_status', 'schema_ok'))}",
        f"- run_history_summary_total_runs: {_fmt(_value(summary, 'run_history_summary_status', 'total_runs'))}",
        f"- run_history_next_action: {_fmt(_value(summary, 'run_history_summary_status', 'next_action'))}",
        f"- run_history_next_action_reason: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_reason'))}",
        f"- run_history_next_action_target: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_target'))}",
        f"- run_history_next_action_command_source: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_command_source'))}",
        f"- run_history_next_action_script_path: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_script_path'))}",
        f"- run_history_next_action_default_new_seeds: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_default_new_seeds'))}",
        f"- run_history_next_action_should_continue: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_should_continue'))}",
        f"- run_history_next_action_schema_ok: {_fmt(_value(summary, 'run_history_summary_status', 'next_action_schema_ok'))}",
        f"- run_history_event_count: {_fmt(_value(summary, 'run_history_summary_status', 'history_event_count'))}",
        f"- run_history_summary_matches_jsonl: {_fmt(_value(summary, 'run_history_summary_status', 'matches_history_event_count'))}",
        f"- run_history_summary_error: {_fmt(_value(summary, 'run_history_summary_status', 'error'))}",
        f"- inspection_json_path: {_fmt(summary.get('inspection_json_path'))}",
        f"- inspection_markdown_path: {_fmt(summary.get('inspection_markdown_path'))}",
        f"- chain_source_count: {_fmt(summary.get('chain_source_count'))}",
        f"- missing_required: {_fmt(', '.join(summary.get('missing_required') or []))}",
        f"- missing_optional: {_fmt(', '.join(summary.get('missing_optional') or []))}",
        f"- missing_chain_sources: {_fmt(', '.join(summary.get('missing_chain_sources') or []))}",
        "",
        "## Checks",
        "",
        "| label | required | ok | path |",
        "| --- | --- | --- | --- |",
    ]
    checks = summary.get("checks")
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict):
            continue
        lines.append(
            "| {label} | {required} | {ok} | {path} |".format(
                label=_fmt(check.get("label")),
                required=_fmt(check.get("required")),
                ok=_fmt(check.get("ok")),
                path=_fmt(check.get("path")),
            )
        )
    lines.extend(
        [
            "",
            "## Declared Commands",
            "",
            "| label | present | ok | target_dir_ok | parse_error | "
            "missing_required_flags | forbidden_flags_present | command |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    commands = summary.get("declared_commands")
    for command in commands if isinstance(commands, list) else []:
        if not isinstance(command, dict):
            continue
        lines.append(
            "| {label} | {present} | {ok} | {target_dir_ok} | {parse_error} | "
            "{missing} | {forbidden} | {command} |".format(
                label=_fmt(command.get("label")),
                present=_fmt(command.get("present")),
                ok=_fmt(command.get("ok")),
                target_dir_ok=_fmt(command.get("target_command_dir_ok")),
                parse_error=_fmt(command.get("parse_error")),
                missing=_fmt_list(command.get("missing_required_flags")),
                forbidden=_fmt_list(command.get("forbidden_flags_present")),
                command=_fmt(command.get("command")),
            )
        )
    lines.extend(
        [
            "",
            "## Declared Run Outputs",
            "",
            "| label | exists | path |",
            "| --- | --- | --- |",
        ]
    )
    outputs = summary.get("declared_outputs")
    for output in outputs if isinstance(outputs, list) else []:
        if not isinstance(output, dict):
            continue
        lines.append(
            "| {label} | {exists} | {path} |".format(
                label=_fmt(output.get("label")),
                exists=_fmt(output.get("exists")),
                path=_fmt(output.get("path")),
            )
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_dir", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON instead of Markdown",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when required or optional declared artifacts are missing",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="write inspection.json and inspection.md into the command bundle",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = inspect_bundle(args.command_dir)
    command_dir = Path(summary["command_dir"])
    json_out = args.json_out
    markdown_out = args.markdown_out
    if args.write_report:
        if json_out is None:
            json_out = command_dir / "inspection.json"
        if markdown_out is None:
            markdown_out = command_dir / "inspection.md"
    summary["inspection_json_path"] = str(json_out) if json_out is not None else None
    summary["inspection_markdown_path"] = (
        str(markdown_out) if markdown_out is not None else None
    )
    markdown = render_markdown(summary)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(markdown, end="")
    if args.strict:
        return 0 if summary["strict_ready"] else 1
    return 0 if summary["bundle_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

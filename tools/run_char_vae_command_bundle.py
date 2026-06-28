#!/usr/bin/env python3
"""Run a generated char VAE command bundle after inspection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA = "st.llm_char_vae_context.command_bundle_run.v1"
TARGET_KEYS = {
    "next": "next_path",
    "follow-up": "follow_up_path",
    "review": "review_path",
}
TARGET_KIND_KEYS = {
    "follow_up": "follow_up_path",
    "review": "review_path",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path_from(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _load_inspector() -> Any:
    module_path = Path(__file__).resolve().with_name(
        "inspect_char_vae_command_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_spiraltorch_char_vae_command_bundle_inspector",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load command bundle inspector from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command_scripts(manifest: dict[str, Any]) -> dict[str, Any]:
    command_scripts = manifest.get("command_scripts")
    return command_scripts if isinstance(command_scripts, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _selected_script(
    command_scripts: dict[str, Any],
    *,
    target: str,
) -> tuple[str, Path | None]:
    key = TARGET_KEYS[target]
    return key, _path_from(command_scripts.get(key))


def _target_details(
    command_scripts: dict[str, Any],
    *,
    target: str,
    script_key: str,
    script_path: Path | None,
) -> dict[str, Any]:
    if target != "next":
        target_kind = "follow_up" if target == "follow-up" else target
        return {
            "target_kind": target_kind,
            "target_script_key": script_key,
            "target_script_path": script_path,
        }
    target_kind_value = command_scripts.get("next_kind")
    target_kind = (
        str(target_kind_value)
        if isinstance(target_kind_value, str) and target_kind_value
        else None
    )
    target_script_key = TARGET_KIND_KEYS.get(target_kind or "")
    target_script_path = (
        _path_from(command_scripts.get(target_script_key))
        if target_script_key is not None
        else None
    )
    return {
        "target_kind": target_kind,
        "target_script_key": target_script_key,
        "target_script_path": target_script_path,
    }


def _candidate_context(value: Any) -> dict[str, Any] | None:
    candidate = _mapping(value)
    if not candidate:
        return None
    fields = (
        "config",
        "mean_best_nll",
        "step",
        "summary_path",
    )
    compact = {key: candidate.get(key) for key in fields if key in candidate}
    return compact or None


def _recommendation_context(
    manifest: dict[str, Any],
    command_scripts: dict[str, Any],
    *,
    target_kind: str | None,
) -> dict[str, Any]:
    comparison = _mapping(manifest.get("comparison"))
    aggregate = _mapping(manifest.get("aggregate"))
    selection = _mapping(manifest.get("selection"))
    recommendation = _mapping(manifest.get("recommendation"))
    follow_up_command = _mapping(recommendation.get("follow_up_command"))
    review_command = _mapping(recommendation.get("review_command"))
    return {
        "schema": (
            "st.llm_char_vae_context.command_bundle_run_recommendation_context.v1"
        ),
        "action": recommendation.get("action"),
        "reason": recommendation.get("reason"),
        "target_kind": target_kind,
        "next_kind": command_scripts.get("next_kind"),
        "accepted_matches_best": selection.get("accepted_matches_best"),
        "best_requires_review": selection.get("best_requires_review"),
        "accepted_vs_best_nll_gap": selection.get("accepted_vs_best_nll_gap"),
        "follow_up_from_summary_path": recommendation.get(
            "follow_up_from_summary_path"
        ),
        "review_summary_path": recommendation.get("review_summary_path"),
        "champion_source": recommendation.get("champion_source"),
        "champion": _candidate_context(recommendation.get("champion")),
        "fallback_source": recommendation.get("fallback_source"),
        "fallback": _candidate_context(recommendation.get("fallback")),
        "follow_up_command_source": follow_up_command.get("command_source"),
        "follow_up_command_summary_path": follow_up_command.get("source_summary_path"),
        "review_command_source": review_command.get("command_source"),
        "review_command_summary_path": review_command.get("source_summary_path"),
        "chain_sources": _list_of_strings(comparison.get("chain_sources")),
        "chain_count": aggregate.get("chain_count"),
        "attempted_follow_ups": aggregate.get("attempted_follow_ups"),
    }


def _write_inspection_report(
    inspector: Any,
    command_dir: Path,
    inspection: dict[str, Any],
    command_scripts: dict[str, Any],
) -> dict[str, Any]:
    json_out = _path_from(command_scripts.get("inspection_json_path"))
    markdown_out = _path_from(command_scripts.get("inspection_markdown_path"))
    json_out = json_out or command_dir / "inspection.json"
    markdown_out = markdown_out or command_dir / "inspection.md"
    inspection = dict(inspection)
    inspection["inspection_json_path"] = str(json_out)
    inspection["inspection_markdown_path"] = str(markdown_out)
    markdown = inspector.render_markdown(inspection)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")
    return inspection


def _run_report_paths(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    write_report: bool,
) -> tuple[Path | None, Path | None]:
    if write_report or json_out is not None or markdown_out is not None:
        json_out = json_out or _path_from(command_scripts.get("run_json_path"))
        markdown_out = markdown_out or _path_from(
            command_scripts.get("run_markdown_path")
        )
        json_out = json_out or command_dir / "run.json"
        markdown_out = markdown_out or command_dir / "run.md"
    return json_out, markdown_out


def _run_history_path(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    append_history: bool,
    write_history_report: bool,
) -> Path | None:
    if not append_history and not write_history_report:
        return None
    return (
        _path_from(command_scripts.get("run_history_jsonl_path"))
        or command_dir / "run_history.jsonl"
    )


def _run_history_markdown_path(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    write_history_report: bool,
) -> Path | None:
    if not write_history_report:
        return None
    return (
        _path_from(command_scripts.get("run_history_markdown_path"))
        or command_dir / "run_history.md"
    )


def _runner_summary(
    *,
    command_dir: Path,
    manifest_path: Path,
    target: str,
    target_kind: str | None,
    script_key: str,
    script_path: Path | None,
    target_script_key: str | None,
    target_script_path: Path | None,
    recommendation_context: dict[str, Any],
    strict: bool,
    dry_run: bool,
    inspection: dict[str, Any],
    returncode: int | None,
    error: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
    executed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "manifest_path": str(manifest_path),
        "target": target,
        "target_kind": target_kind,
        "script_key": script_key,
        "script_path": str(script_path) if script_path is not None else None,
        "target_script_key": target_script_key,
        "target_script_path": (
            str(target_script_path) if target_script_path is not None else None
        ),
        "recommendation_context": recommendation_context,
        "command_argv": ["bash", str(script_path)]
        if script_path is not None
        else None,
        "execution_cwd": str(command_dir),
        "strict": strict,
        "dry_run": dry_run,
        "executed": bool(executed),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "bundle_ready": bool(inspection.get("bundle_ready")),
        "strict_ready": bool(inspection.get("strict_ready")),
        "missing_required": inspection.get("missing_required") or [],
        "missing_optional": inspection.get("missing_optional") or [],
        "inspection": inspection,
        "returncode": returncode,
        "error": error,
        "stdout": stdout,
        "stderr": stderr,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    context = _mapping(summary.get("recommendation_context"))
    champion = _mapping(context.get("champion"))
    fallback = _mapping(context.get("fallback"))
    lines = [
        "# Char VAE Command Bundle Runner",
        "",
        f"- command_dir: {_fmt(summary.get('command_dir'))}",
        f"- manifest_path: {_fmt(summary.get('manifest_path'))}",
        f"- target: {_fmt(summary.get('target'))}",
        f"- target_kind: {_fmt(summary.get('target_kind'))}",
        f"- script_key: {_fmt(summary.get('script_key'))}",
        f"- script_path: {_fmt(summary.get('script_path'))}",
        f"- target_script_key: {_fmt(summary.get('target_script_key'))}",
        f"- target_script_path: {_fmt(summary.get('target_script_path'))}",
        f"- command_argv: {_fmt(summary.get('command_argv'))}",
        f"- execution_cwd: {_fmt(summary.get('execution_cwd'))}",
        f"- strict: {_fmt(summary.get('strict'))}",
        f"- dry_run: {_fmt(summary.get('dry_run'))}",
        f"- executed: {_fmt(summary.get('executed'))}",
        f"- started_at: {_fmt(summary.get('started_at'))}",
        f"- finished_at: {_fmt(summary.get('finished_at'))}",
        f"- duration_seconds: {_fmt(summary.get('duration_seconds'))}",
        f"- bundle_ready: {_fmt(summary.get('bundle_ready'))}",
        f"- strict_ready: {_fmt(summary.get('strict_ready'))}",
        f"- missing_required: {_fmt(summary.get('missing_required'))}",
        f"- missing_optional: {_fmt(summary.get('missing_optional'))}",
        f"- run_json_path: {_fmt(summary.get('run_json_path'))}",
        f"- run_markdown_path: {_fmt(summary.get('run_markdown_path'))}",
        f"- run_history_jsonl_path: {_fmt(summary.get('run_history_jsonl_path'))}",
        f"- run_history_markdown_path: {_fmt(summary.get('run_history_markdown_path'))}",
        f"- returncode: {_fmt(summary.get('returncode'))}",
        f"- error: {_fmt(summary.get('error'))}",
        "",
        "## Recommendation Context",
        "",
        f"- recommendation_action: {_fmt(context.get('action'))}",
        f"- recommendation_reason: {_fmt(context.get('reason'))}",
        f"- next_kind: {_fmt(context.get('next_kind'))}",
        f"- accepted_matches_best: {_fmt(context.get('accepted_matches_best'))}",
        f"- best_requires_review: {_fmt(context.get('best_requires_review'))}",
        f"- accepted_vs_best_nll_gap: {_fmt(context.get('accepted_vs_best_nll_gap'))}",
        f"- follow_up_from_summary_path: {_fmt(context.get('follow_up_from_summary_path'))}",
        f"- review_summary_path: {_fmt(context.get('review_summary_path'))}",
        f"- champion_source: {_fmt(context.get('champion_source'))}",
        f"- champion_config: {_fmt(champion.get('config'))}",
        f"- champion_summary_path: {_fmt(champion.get('summary_path'))}",
        f"- fallback_source: {_fmt(context.get('fallback_source'))}",
        f"- fallback_config: {_fmt(fallback.get('config'))}",
        f"- fallback_summary_path: {_fmt(fallback.get('summary_path'))}",
        f"- follow_up_command_source: {_fmt(context.get('follow_up_command_source'))}",
        f"- review_command_source: {_fmt(context.get('review_command_source'))}",
        f"- chain_sources: {_fmt(context.get('chain_sources'))}",
        "",
    ]
    return "\n".join(lines)


def _history_event(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "command_dir",
        "manifest_path",
        "target",
        "target_kind",
        "script_key",
        "script_path",
        "target_script_key",
        "target_script_path",
        "command_argv",
        "execution_cwd",
        "strict",
        "dry_run",
        "executed",
        "started_at",
        "finished_at",
        "duration_seconds",
        "bundle_ready",
        "strict_ready",
        "missing_required",
        "missing_optional",
        "returncode",
        "error",
        "run_json_path",
        "run_markdown_path",
        "run_history_jsonl_path",
        "run_history_markdown_path",
        "recommendation_context",
    )
    event = {
        "schema": "st.llm_char_vae_context.command_bundle_run_history_event.v1",
    }
    event.update({key: summary.get(key) for key in fields})
    return event


def _append_run_history(summary: dict[str, Any], history_out: Path) -> None:
    history_out.parent.mkdir(parents=True, exist_ok=True)
    with history_out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_history_event(summary), sort_keys=True) + "\n")


def _read_history_events(history_out: Path) -> list[dict[str, Any]]:
    if not history_out.exists():
        return []
    events: list[dict[str, Any]] = []
    lines = history_out.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"{history_out}:{lineno} did not contain a JSON object")
        events.append(event)
    return events


def _event_status(event: dict[str, Any]) -> str:
    if event.get("dry_run") and event.get("returncode") == 0:
        return "dry-run"
    if event.get("returncode") == 0:
        return "ok"
    if event.get("error"):
        return "blocked"
    return "failed"


def _count_values(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _latest_event(
    events: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    for event in reversed(events):
        if predicate(event):
            return event
    return None


def _status_streak(events: list[dict[str, Any]]) -> tuple[str | None, int]:
    if not events:
        return None, 0
    status = _event_status(events[-1])
    count = 0
    for event in reversed(events):
        if _event_status(event) != status:
            break
        count += 1
    return status, count


def _fmt_streak(status: str | None, count: int) -> str:
    if status is None:
        return "-"
    return f"{status}:{count}"


def _md_cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_history_markdown(
    events: list[dict[str, Any]],
    *,
    history_jsonl_path: Path,
) -> str:
    success_count = sum(1 for event in events if event.get("returncode") == 0)
    dry_run_count = sum(1 for event in events if event.get("dry_run"))
    executed_count = sum(1 for event in events if event.get("executed"))
    failure_count = len(events) - success_count
    latest = events[-1] if events else {}
    latest_context = _mapping(latest.get("recommendation_context"))
    latest_champion = _mapping(latest_context.get("champion"))
    status_counts = _count_values([_event_status(event) for event in events])
    target_kind_counts = _count_values([event.get("target_kind") for event in events])
    action_counts = _count_values(
        [
            _mapping(event.get("recommendation_context")).get("action")
            for event in events
        ]
    )
    current_status, current_status_count = _status_streak(events)
    latest_executed = _latest_event(events, lambda event: bool(event.get("executed")))
    latest_executed_context = _mapping(
        latest_executed.get("recommendation_context") if latest_executed else None
    )
    latest_executed_champion = _mapping(latest_executed_context.get("champion"))
    last_problem = _latest_event(
        events,
        lambda event: _event_status(event) in {"blocked", "failed"},
    )
    lines = [
        "# Char VAE Command Bundle Run History",
        "",
        f"- history_jsonl_path: {_fmt(str(history_jsonl_path))}",
        f"- total_runs: {len(events)}",
        f"- success_count: {success_count}",
        f"- failure_count: {failure_count}",
        f"- dry_run_count: {dry_run_count}",
        f"- executed_count: {executed_count}",
        f"- latest_started_at: {_fmt(latest.get('started_at'))}",
        f"- latest_finished_at: {_fmt(latest.get('finished_at'))}",
        f"- latest_status: {_fmt(_event_status(latest) if latest else None)}",
        f"- latest_target_kind: {_fmt(latest.get('target_kind'))}",
        f"- latest_recommendation_action: {_fmt(latest_context.get('action'))}",
        f"- latest_champion_config: {_fmt(latest_champion.get('config'))}",
        "",
        "## Decision Signals",
        "",
        f"- status_counts: {_fmt_counts(status_counts)}",
        f"- target_kind_counts: {_fmt_counts(target_kind_counts)}",
        f"- recommendation_action_counts: {_fmt_counts(action_counts)}",
        f"- current_status_streak: {_fmt_streak(current_status, current_status_count)}",
        f"- latest_executed_status: {_fmt(_event_status(latest_executed) if latest_executed else None)}",
        f"- latest_executed_finished_at: {_fmt(latest_executed.get('finished_at') if latest_executed else None)}",
        f"- latest_executed_action: {_fmt(latest_executed_context.get('action'))}",
        f"- latest_executed_champion_config: {_fmt(latest_executed_champion.get('config'))}",
        f"- last_problem_status: {_fmt(_event_status(last_problem) if last_problem else None)}",
        f"- last_problem_error: {_fmt(last_problem.get('error') if last_problem else None)}",
        f"- last_problem_missing_required: {_fmt(last_problem.get('missing_required') if last_problem else None)}",
        "",
        "## Recent Events",
        "",
        "| # | status | target | kind | dry_run | executed | returncode | "
        "started_at | duration_seconds | action | champion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, event in enumerate(events[-10:], max(1, len(events) - 9)):
        context = _mapping(event.get("recommendation_context"))
        champion = _mapping(context.get("champion"))
        row = [
            index,
            _event_status(event),
            event.get("target"),
            event.get("target_kind"),
            event.get("dry_run"),
            event.get("executed"),
            event.get("returncode"),
            event.get("started_at"),
            event.get("duration_seconds"),
            context.get("action"),
            champion.get("config"),
        ]
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_run_history_report(history_out: Path, markdown_out: Path) -> None:
    events = _read_history_events(history_out)
    markdown = render_history_markdown(events, history_jsonl_path=history_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")


def write_run_artifacts(
    summary: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    history_out: Path | None = None,
    history_markdown_out: Path | None = None,
    append_history: bool = False,
) -> dict[str, Any]:
    summary = dict(summary)
    summary["run_json_path"] = str(json_out) if json_out is not None else None
    summary["run_markdown_path"] = (
        str(markdown_out) if markdown_out is not None else None
    )
    summary["run_history_jsonl_path"] = (
        str(history_out) if history_out is not None else None
    )
    summary["run_history_markdown_path"] = (
        str(history_markdown_out) if history_markdown_out is not None else None
    )
    markdown = render_markdown(summary)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    if history_out is not None and append_history:
        _append_run_history(summary, history_out)
    if history_out is not None and history_markdown_out is not None:
        _write_run_history_report(history_out, history_markdown_out)
    return summary


def run_bundle(
    command_dir: Path,
    *,
    target: str,
    strict: bool,
    dry_run: bool,
    json_mode: bool,
    write_inspection_report: bool,
    write_run_report: bool,
    append_run_history: bool,
    write_run_history_report: bool,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    started_at = _utc_now()
    started_perf = time.perf_counter()

    def timing_fields() -> dict[str, Any]:
        return {
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.perf_counter() - started_perf, 6),
        }

    command_dir = command_dir.resolve()
    manifest_path = command_dir / "recommendation.json"
    manifest = _read_json(manifest_path)
    command_scripts = _command_scripts(manifest)
    script_key, script_path = _selected_script(command_scripts, target=target)
    target_details = _target_details(
        command_scripts,
        target=target,
        script_key=script_key,
        script_path=script_path,
    )
    recommendation_context = _recommendation_context(
        manifest,
        command_scripts,
        target_kind=target_details.get("target_kind"),
    )
    json_out, markdown_out = _run_report_paths(
        command_dir,
        command_scripts,
        json_out=json_out,
        markdown_out=markdown_out,
        write_report=write_run_report,
    )
    history_out = _run_history_path(
        command_dir,
        command_scripts,
        append_history=append_run_history,
        write_history_report=write_run_history_report,
    )
    history_markdown_out = _run_history_markdown_path(
        command_dir,
        command_scripts,
        write_history_report=write_run_history_report,
    )
    inspector = _load_inspector()
    inspection = inspector.inspect_bundle(command_dir)
    if write_inspection_report:
        inspection = _write_inspection_report(
            inspector,
            command_dir,
            inspection,
            command_scripts,
        )
    required_ready = bool(inspection.get("strict_ready" if strict else "bundle_ready"))
    if not required_ready:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error="command bundle did not pass the requested inspection gate",
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            append_history=append_run_history,
        )
        return 1, summary
    if script_path is None:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error=f"manifest does not declare {script_key}",
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            append_history=append_run_history,
        )
        return 1, summary
    if dry_run:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=0,
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            append_history=append_run_history,
        )
        return 0, summary
    if json_mode:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=command_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            executed=True,
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            append_history=append_run_history,
        )
        return result.returncode, summary
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=command_dir,
        check=False,
    )
    summary = _runner_summary(
        command_dir=command_dir,
        manifest_path=manifest_path,
        target=target,
        script_key=script_key,
        script_path=script_path,
        **target_details,
        recommendation_context=recommendation_context,
        strict=strict,
        dry_run=dry_run,
        inspection=inspection,
        returncode=result.returncode,
        executed=True,
        **timing_fields(),
    )
    summary = write_run_artifacts(
        summary,
        json_out=json_out,
        markdown_out=markdown_out,
        history_out=history_out,
        history_markdown_out=history_markdown_out,
        append_history=append_run_history,
    )
    return result.returncode, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_dir", type=Path)
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_KEYS),
        default="next",
        help="which declared command script to run",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="require only bundle_ready instead of strict_ready before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect and report the selected command without executing it",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON run summary; command stdout/stderr are captured",
    )
    parser.add_argument(
        "--write-inspection-report",
        action="store_true",
        help="rewrite inspection.json and inspection.md before running",
    )
    parser.add_argument(
        "--write-run-report",
        action="store_true",
        help="write run.json and run.md into the command bundle",
    )
    parser.add_argument(
        "--append-run-history",
        action="store_true",
        help="append a compact run event to run_history.jsonl",
    )
    parser.add_argument(
        "--write-run-history-report",
        action="store_true",
        help="write a Markdown summary of run_history.jsonl",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        returncode, summary = run_bundle(
            args.command_dir,
            target=args.target,
            strict=not args.no_strict,
            dry_run=bool(args.dry_run),
            json_mode=bool(args.json),
            write_inspection_report=bool(args.write_inspection_report),
            write_run_report=bool(args.write_run_report),
            append_run_history=bool(args.append_run_history),
            write_run_history_report=bool(args.write_run_history_report),
            json_out=args.json_out,
            markdown_out=args.markdown_out,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        summary = {
            "schema": SCHEMA,
            "command_dir": str(args.command_dir),
            "target": args.target,
            "strict": not args.no_strict,
            "dry_run": bool(args.dry_run),
            "returncode": 1,
            "error": str(exc),
        }
        returncode = 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.dry_run or returncode != 0:
        stream = sys.stderr if returncode != 0 else sys.stdout
        print(render_markdown(summary), file=stream)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

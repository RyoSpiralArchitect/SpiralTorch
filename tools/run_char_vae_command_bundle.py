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
EXECUTION_NEXT_TARGET = "execution-next"
TARGET_KEYS = {
    "next": "next_path",
    "follow-up": "follow_up_path",
    "review": "review_path",
    EXECUTION_NEXT_TARGET: "execution_summary.next_command.script_path",
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
    execution_next_command: dict[str, Any] | None = None,
    previous_run_report: dict[str, Any] | None = None,
) -> tuple[str, Path | None]:
    key = TARGET_KEYS[target]
    if target == EXECUTION_NEXT_TARGET:
        next_command = _mapping(execution_next_command)
        if not next_command:
            next_command = _execution_next_command_from_report(previous_run_report)
        return key, _path_from(next_command.get("script_path"))
    return key, _path_from(command_scripts.get(key))


def _target_details(
    command_scripts: dict[str, Any],
    *,
    target: str,
    script_key: str,
    script_path: Path | None,
) -> dict[str, Any]:
    if target == EXECUTION_NEXT_TARGET:
        return {
            "target_kind": "execution_next",
            "target_script_key": script_key,
            "target_script_path": script_path,
        }
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


def _run_json_path(command_dir: Path, command_scripts: dict[str, Any]) -> Path:
    return _path_from(command_scripts.get("run_json_path")) or command_dir / "run.json"


def _previous_run_report(
    command_dir: Path,
    command_scripts: dict[str, Any],
) -> dict[str, Any]:
    path = _run_json_path(command_dir, command_scripts)
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _execution_next_command_from_report(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    report = _mapping(report)
    execution_summary = _mapping(report.get("execution_summary"))
    next_command = _mapping(execution_summary.get("next_command"))
    if next_command:
        return next_command
    return _mapping(report.get("selected_execution_next_command"))


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
    return _configured_run_history_path(command_dir, command_scripts)


def _configured_run_history_path(
    command_dir: Path,
    command_scripts: dict[str, Any],
) -> Path:
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


def _run_history_summary_path(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    write_history_report: bool,
) -> Path | None:
    if not write_history_report:
        return None
    return (
        _path_from(command_scripts.get("run_history_summary_path"))
        or command_dir / "run_history_summary.json"
    )


def _config_label(config: dict[str, Any]) -> str | None:
    feature = config.get("best_feature") or config.get("feature")
    if not isinstance(feature, str) or not feature:
        return None
    parts = [feature]
    normalize = config.get("feature_normalize")
    if normalize is not None:
        parts.append(f"normalize={normalize}")
    scale = config.get("hybrid_latent_scale")
    if scale is not None:
        parts.append(f"scale={scale}")
    return parts[0] if len(parts) == 1 else f"{parts[0]}@" + ",".join(parts[1:])


def _recommended_command_for_kind(
    manifest: dict[str, Any],
    *,
    target_kind: str | None,
) -> dict[str, Any]:
    recommendation = _mapping(manifest.get("recommendation"))
    if target_kind == "follow_up":
        return _mapping(recommendation.get("follow_up_command"))
    if target_kind == "review":
        return _mapping(recommendation.get("review_command"))
    return {}


def _expected_execution_summary_path(
    manifest: dict[str, Any],
    *,
    target_kind: str | None,
    execution_next_command: dict[str, Any] | None = None,
) -> Path | None:
    if target_kind == "execution_next":
        run_dir = _path_from(_mapping(execution_next_command).get("default_run_dir"))
        return run_dir / "summary.json" if run_dir is not None else None
    command = _recommended_command_for_kind(manifest, target_kind=target_kind)
    run_dir = _path_from(command.get("default_run_dir"))
    return run_dir / "summary.json" if run_dir is not None else None


def _execution_cwd_for_target(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    target: str,
) -> Path:
    if target == EXECUTION_NEXT_TARGET:
        return _path_from(command_scripts.get("execution_cwd")) or command_dir
    return command_dir


def _compact_execution_next_command(payload: dict[str, Any]) -> dict[str, Any]:
    guided_command = _mapping(payload.get("guided_next_follow_up_command"))
    next_command = _mapping(payload.get("next_follow_up_command"))
    if guided_command.get("enabled"):
        source = "guided_next_follow_up_command"
        command = guided_command
    elif next_command:
        source = "next_follow_up_command"
        command = next_command
    else:
        return {
            "schema": "st.llm_char_vae_context.command_bundle_execution_next_command.v1",
            "available": False,
            "source": None,
        }
    seed_policy = _mapping(command.get("seed_confirmation_policy"))
    return {
        "schema": "st.llm_char_vae_context.command_bundle_execution_next_command.v1",
        "available": True,
        "source": source,
        "enabled": command.get("enabled"),
        "action": command.get("action") or command.get("guidance_action"),
        "guidance_action": command.get("guidance_action"),
        "trajectory_action": command.get("trajectory_action"),
        "verdict": command.get("verdict"),
        "gate_failed": command.get("gate_failed"),
        "script_path": command.get("script_path"),
        "script_usage": command.get("script_usage"),
        "shell_command": command.get("shell_command"),
        "default_follow_up_from": command.get("default_follow_up_from"),
        "default_follow_up_fail_on_verdict": command.get(
            "default_follow_up_fail_on_verdict"
        ),
        "default_new_seeds": command.get("default_new_seeds"),
        "default_new_seed_count": command.get("default_new_seed_count"),
        "default_run_dir": command.get("default_run_dir"),
        "used_seed_history": command.get("used_seed_history"),
        "seed_policy_reason": seed_policy.get("reason"),
        "tie_seed_boost": seed_policy.get("uncertainty_tie_seed_boost"),
    }


def _compact_execution_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    summary: dict[str, Any] = {
        "schema": "st.llm_char_vae_context.command_bundle_execution_summary.v1",
        "summary_path": str(path),
        "exists": path.exists(),
        "valid_json": None,
        "error": None,
    }
    if not path.exists():
        summary["valid_json"] = False
        return summary
    try:
        payload = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive artifact handling.
        summary["valid_json"] = False
        summary["error"] = str(exc)
        return summary

    best_config = _mapping(payload.get("best_config"))
    follow_up_result = _mapping(payload.get("follow_up_result"))
    follow_up_gate = _mapping(payload.get("follow_up_gate"))
    follow_up_guidance = _mapping(payload.get("follow_up_guidance"))
    follow_up_trajectory = _mapping(payload.get("follow_up_trajectory"))
    next_command = _mapping(payload.get("next_follow_up_command"))
    guided_command = _mapping(payload.get("guided_next_follow_up_command"))
    execution_next_command = _compact_execution_next_command(payload)
    summary.update(
        {
            "exists": True,
            "valid_json": True,
            "status": payload.get("status"),
            "best_feature": payload.get("best_feature") or best_config.get("best_feature"),
            "best_config_label": _config_label(best_config),
            "mean_best_nll": best_config.get("mean_best_nll"),
            "mean_best_nll_delta_vs_raw": best_config.get("mean_best_nll_delta_vs_raw"),
            "runner_up_feature": best_config.get("runner_up_feature"),
            "margin_to_runner_up": best_config.get("margin_to_runner_up"),
            "combined_runner_up_margin_stderr": best_config.get(
                "combined_runner_up_margin_stderr"
            ),
            "runner_up_within_uncertainty": best_config.get(
                "runner_up_within_uncertainty"
            ),
            "follow_up_verdict": follow_up_result.get("verdict")
            or follow_up_gate.get("effective_verdict"),
            "follow_up_gate_failed": follow_up_gate.get("failed"),
            "source_best_feature_retained": follow_up_result.get(
                "source_best_feature_retained"
            ),
            "mean_best_nll_delta_vs_source": follow_up_result.get(
                "mean_best_nll_delta_vs_source"
            ),
            "guidance_action": follow_up_guidance.get("action"),
            "trajectory_action": follow_up_trajectory.get("trajectory_action"),
            "trajectory_verdict": follow_up_trajectory.get("trajectory_verdict"),
            "unsafe_promotion": follow_up_guidance.get("unsafe_promotion"),
            "next_default_new_seeds": next_command.get("default_new_seeds"),
            "next_default_new_seed_count": next_command.get("default_new_seed_count"),
            "guided_next_enabled": guided_command.get("enabled"),
            "guided_default_new_seeds": guided_command.get("default_new_seeds"),
            "next_command": execution_next_command,
            "next_command_source": execution_next_command.get("source"),
            "next_command_available": execution_next_command.get("available"),
            "next_command_default_new_seeds": execution_next_command.get(
                "default_new_seeds"
            ),
            "next_command_default_run_dir": execution_next_command.get(
                "default_run_dir"
            ),
            "next_command_script_path": execution_next_command.get("script_path"),
            "used_seed_history": next_command.get("used_seed_history")
            or guided_command.get("used_seed_history"),
        }
    )
    return summary


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
    execution_cwd: Path | None = None,
    selected_execution_next_command: dict[str, Any] | None = None,
    error: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
    executed: bool = False,
    history_report_only: bool = False,
) -> dict[str, Any]:
    runner_wrapper_status = _mapping(inspection.get("runner_wrapper_status"))
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
        "execution_cwd": str(execution_cwd or command_dir),
        "selected_execution_next_command": (
            selected_execution_next_command
            if selected_execution_next_command is not None
            else None
        ),
        "strict": strict,
        "dry_run": dry_run,
        "executed": bool(executed),
        "history_report_only": bool(history_report_only),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "expected_execution_summary_path": None,
        "execution_summary": None,
        "bundle_ready": bool(inspection.get("bundle_ready")),
        "strict_ready": bool(inspection.get("strict_ready")),
        "missing_required": inspection.get("missing_required") or [],
        "missing_optional": inspection.get("missing_optional") or [],
        "runner_wrapper_status": runner_wrapper_status,
        "runner_wrapper_ok": runner_wrapper_status.get("ok"),
        "runner_wrapper_executes_runner_command": runner_wrapper_status.get(
            "executes_runner_command"
        ),
        "runner_wrapper_forwards_arguments": runner_wrapper_status.get(
            "forwards_arguments"
        ),
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
    selected_execution_next = _mapping(summary.get("selected_execution_next_command"))
    execution_summary = _mapping(summary.get("execution_summary"))
    execution_next = _mapping(execution_summary.get("next_command"))
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
        (
            "- selected_execution_next_source: "
            f"{_fmt(selected_execution_next.get('source'))}"
        ),
        (
            "- selected_execution_next_seeds: "
            f"{_fmt(selected_execution_next.get('default_new_seeds'))}"
        ),
        (
            "- selected_execution_next_script: "
            f"{_fmt(selected_execution_next.get('script_path'))}"
        ),
        (
            "- selected_execution_next_run_dir: "
            f"{_fmt(selected_execution_next.get('default_run_dir'))}"
        ),
        f"- command_argv: {_fmt(summary.get('command_argv'))}",
        f"- execution_cwd: {_fmt(summary.get('execution_cwd'))}",
        f"- strict: {_fmt(summary.get('strict'))}",
        f"- dry_run: {_fmt(summary.get('dry_run'))}",
        f"- executed: {_fmt(summary.get('executed'))}",
        f"- history_report_only: {_fmt(summary.get('history_report_only'))}",
        f"- started_at: {_fmt(summary.get('started_at'))}",
        f"- finished_at: {_fmt(summary.get('finished_at'))}",
        f"- duration_seconds: {_fmt(summary.get('duration_seconds'))}",
        f"- bundle_ready: {_fmt(summary.get('bundle_ready'))}",
        f"- strict_ready: {_fmt(summary.get('strict_ready'))}",
        f"- missing_required: {_fmt(summary.get('missing_required'))}",
        f"- missing_optional: {_fmt(summary.get('missing_optional'))}",
        f"- runner_wrapper_ok: {_fmt(summary.get('runner_wrapper_ok'))}",
        (
            "- runner_wrapper_executes_runner_command: "
            f"{_fmt(summary.get('runner_wrapper_executes_runner_command'))}"
        ),
        (
            "- runner_wrapper_forwards_arguments: "
            f"{_fmt(summary.get('runner_wrapper_forwards_arguments'))}"
        ),
        f"- run_json_path: {_fmt(summary.get('run_json_path'))}",
        f"- run_markdown_path: {_fmt(summary.get('run_markdown_path'))}",
        f"- run_history_jsonl_path: {_fmt(summary.get('run_history_jsonl_path'))}",
        f"- run_history_markdown_path: {_fmt(summary.get('run_history_markdown_path'))}",
        f"- run_history_summary_path: {_fmt(summary.get('run_history_summary_path'))}",
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
        "## Execution Summary",
        "",
        (
            "- expected_summary_path: "
            f"{_fmt(summary.get('expected_execution_summary_path'))}"
        ),
        f"- summary_exists: {_fmt(execution_summary.get('exists'))}",
        f"- summary_valid_json: {_fmt(execution_summary.get('valid_json'))}",
        f"- summary_error: {_fmt(execution_summary.get('error'))}",
        f"- execution_status: {_fmt(execution_summary.get('status'))}",
        f"- execution_best_config: {_fmt(execution_summary.get('best_config_label'))}",
        f"- execution_best_feature: {_fmt(execution_summary.get('best_feature'))}",
        f"- execution_mean_best_nll: {_fmt(execution_summary.get('mean_best_nll'))}",
        (
            "- execution_delta_vs_raw: "
            f"{_fmt(execution_summary.get('mean_best_nll_delta_vs_raw'))}"
        ),
        f"- execution_runner_up: {_fmt(execution_summary.get('runner_up_feature'))}",
        f"- execution_margin: {_fmt(execution_summary.get('margin_to_runner_up'))}",
        (
            "- execution_margin_stderr: "
            f"{_fmt(execution_summary.get('combined_runner_up_margin_stderr'))}"
        ),
        (
            "- execution_follow_up_verdict: "
            f"{_fmt(execution_summary.get('follow_up_verdict'))}"
        ),
        (
            "- execution_gate_failed: "
            f"{_fmt(execution_summary.get('follow_up_gate_failed'))}"
        ),
        f"- execution_guidance_action: {_fmt(execution_summary.get('guidance_action'))}",
        f"- execution_next_seeds: {_fmt(execution_summary.get('next_default_new_seeds'))}",
        f"- execution_next_command_source: {_fmt(execution_next.get('source'))}",
        f"- execution_next_command_available: {_fmt(execution_next.get('available'))}",
        f"- execution_next_command_seeds: {_fmt(execution_next.get('default_new_seeds'))}",
        f"- execution_next_command_run_dir: {_fmt(execution_next.get('default_run_dir'))}",
        f"- execution_next_command_script: {_fmt(execution_next.get('script_path'))}",
        f"- execution_next_command_usage: {_fmt(execution_next.get('script_usage'))}",
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
        "runner_wrapper_status",
        "runner_wrapper_ok",
        "runner_wrapper_executes_runner_command",
        "runner_wrapper_forwards_arguments",
        "returncode",
        "error",
        "run_json_path",
        "run_markdown_path",
        "run_history_jsonl_path",
        "run_history_markdown_path",
        "run_history_summary_path",
        "recommendation_context",
        "expected_execution_summary_path",
        "selected_execution_next_command",
        "execution_summary",
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


def _execution_next_command_from_history(
    command_dir: Path,
    command_scripts: dict[str, Any],
) -> dict[str, Any]:
    history_path = _configured_run_history_path(command_dir, command_scripts)
    try:
        events = _read_history_events(history_path)
    except Exception:
        return {}
    for event in reversed(events):
        next_command = _execution_next_command_from_report(event)
        if next_command.get("script_path"):
            return next_command
    return {}


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


def _bool_count_values(values: list[Any]) -> dict[str, int]:
    return _count_values([_fmt(value) for value in values if isinstance(value, bool)])


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


def summarize_history_events(
    events: list[dict[str, Any]],
    *,
    history_jsonl_path: Path,
) -> dict[str, Any]:
    success_count = sum(1 for event in events if event.get("returncode") == 0)
    dry_run_count = sum(1 for event in events if event.get("dry_run"))
    executed_count = sum(1 for event in events if event.get("executed"))
    latest = events[-1] if events else {}
    latest_context = _mapping(latest.get("recommendation_context"))
    latest_champion = _mapping(latest_context.get("champion"))
    latest_execution = _mapping(latest.get("execution_summary"))
    latest_execution_next = _mapping(latest_execution.get("next_command"))
    status_counts = _count_values([_event_status(event) for event in events])
    target_kind_counts = _count_values([event.get("target_kind") for event in events])
    runner_wrapper_ok_counts = _bool_count_values(
        [event.get("runner_wrapper_ok") for event in events]
    )
    action_counts = _count_values(
        [
            _mapping(event.get("recommendation_context")).get("action")
            for event in events
        ]
    )
    execution_verdict_counts = _count_values(
        [
            _mapping(event.get("execution_summary")).get("follow_up_verdict")
            for event in events
        ]
    )
    execution_guidance_action_counts = _count_values(
        [
            _mapping(event.get("execution_summary")).get("guidance_action")
            for event in events
        ]
    )
    execution_next_source_counts = _count_values(
        [
            _mapping(_mapping(event.get("execution_summary")).get("next_command")).get(
                "source"
            )
            for event in events
        ]
    )
    current_status, current_status_count = _status_streak(events)
    latest_executed = _latest_event(events, lambda event: bool(event.get("executed")))
    latest_executed_context = _mapping(
        latest_executed.get("recommendation_context") if latest_executed else None
    )
    latest_executed_champion = _mapping(latest_executed_context.get("champion"))
    latest_executed_execution = _mapping(
        latest_executed.get("execution_summary") if latest_executed else None
    )
    latest_executed_execution_next = _mapping(
        latest_executed_execution.get("next_command")
    )
    last_problem = _latest_event(
        events,
        lambda event: _event_status(event) in {"blocked", "failed"},
    )
    return {
        "schema": "st.llm_char_vae_context.command_bundle_run_history_summary.v1",
        "history_jsonl_path": str(history_jsonl_path),
        "total_runs": len(events),
        "success_count": success_count,
        "failure_count": len(events) - success_count,
        "dry_run_count": dry_run_count,
        "executed_count": executed_count,
        "latest": {
            "started_at": latest.get("started_at"),
            "finished_at": latest.get("finished_at"),
            "status": _event_status(latest) if latest else None,
            "target_kind": latest.get("target_kind"),
            "recommendation_action": latest_context.get("action"),
            "champion_config": latest_champion.get("config"),
            "runner_wrapper_ok": latest.get("runner_wrapper_ok"),
            "runner_wrapper_executes_runner_command": latest.get(
                "runner_wrapper_executes_runner_command"
            ),
            "execution_summary_path": latest_execution.get("summary_path"),
            "execution_verdict": latest_execution.get("follow_up_verdict"),
            "execution_best_config": latest_execution.get("best_config_label"),
            "execution_guidance_action": latest_execution.get("guidance_action"),
            "execution_next_source": latest_execution_next.get("source"),
            "execution_next_seeds": latest_execution_next.get("default_new_seeds"),
            "execution_next_script_path": latest_execution_next.get("script_path"),
        },
        "signals": {
            "status_counts": status_counts,
            "target_kind_counts": target_kind_counts,
            "runner_wrapper_ok_counts": runner_wrapper_ok_counts,
            "recommendation_action_counts": action_counts,
            "execution_verdict_counts": execution_verdict_counts,
            "execution_guidance_action_counts": execution_guidance_action_counts,
            "execution_next_source_counts": execution_next_source_counts,
            "current_status_streak": {
                "status": current_status,
                "count": current_status_count,
            },
            "latest_executed": {
                "status": (
                    _event_status(latest_executed) if latest_executed else None
                ),
                "finished_at": (
                    latest_executed.get("finished_at") if latest_executed else None
                ),
                "recommendation_action": latest_executed_context.get("action"),
                "champion_config": latest_executed_champion.get("config"),
                "execution_summary_path": latest_executed_execution.get("summary_path"),
                "execution_verdict": latest_executed_execution.get(
                    "follow_up_verdict"
                ),
                "execution_best_config": latest_executed_execution.get(
                    "best_config_label"
                ),
                "execution_guidance_action": latest_executed_execution.get(
                    "guidance_action"
                ),
                "execution_next_source": latest_executed_execution_next.get("source"),
                "execution_next_seeds": latest_executed_execution_next.get(
                    "default_new_seeds"
                ),
                "execution_next_script_path": latest_executed_execution_next.get(
                    "script_path"
                ),
            },
            "last_problem": {
                "status": _event_status(last_problem) if last_problem else None,
                "error": last_problem.get("error") if last_problem else None,
                "missing_required": (
                    last_problem.get("missing_required") if last_problem else None
                ),
            },
        },
    }


def _md_cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_history_markdown(
    events: list[dict[str, Any]],
    *,
    history_jsonl_path: Path,
) -> str:
    summary = summarize_history_events(events, history_jsonl_path=history_jsonl_path)
    latest = _mapping(summary.get("latest"))
    signals = _mapping(summary.get("signals"))
    status_streak = _mapping(signals.get("current_status_streak"))
    latest_executed = _mapping(signals.get("latest_executed"))
    last_problem = _mapping(signals.get("last_problem"))
    lines = [
        "# Char VAE Command Bundle Run History",
        "",
        f"- history_jsonl_path: {_fmt(summary.get('history_jsonl_path'))}",
        f"- total_runs: {summary.get('total_runs')}",
        f"- success_count: {summary.get('success_count')}",
        f"- failure_count: {summary.get('failure_count')}",
        f"- dry_run_count: {summary.get('dry_run_count')}",
        f"- executed_count: {summary.get('executed_count')}",
        f"- latest_started_at: {_fmt(latest.get('started_at'))}",
        f"- latest_finished_at: {_fmt(latest.get('finished_at'))}",
        f"- latest_status: {_fmt(latest.get('status'))}",
        f"- latest_target_kind: {_fmt(latest.get('target_kind'))}",
        f"- latest_recommendation_action: {_fmt(latest.get('recommendation_action'))}",
        f"- latest_champion_config: {_fmt(latest.get('champion_config'))}",
        f"- latest_runner_wrapper_ok: {_fmt(latest.get('runner_wrapper_ok'))}",
        (
            "- latest_runner_wrapper_executes_runner_command: "
            f"{_fmt(latest.get('runner_wrapper_executes_runner_command'))}"
        ),
        f"- latest_execution_verdict: {_fmt(latest.get('execution_verdict'))}",
        f"- latest_execution_best_config: {_fmt(latest.get('execution_best_config'))}",
        (
            "- latest_execution_guidance_action: "
            f"{_fmt(latest.get('execution_guidance_action'))}"
        ),
        f"- latest_execution_next_source: {_fmt(latest.get('execution_next_source'))}",
        f"- latest_execution_next_seeds: {_fmt(latest.get('execution_next_seeds'))}",
        (
            "- latest_execution_next_script_path: "
            f"{_fmt(latest.get('execution_next_script_path'))}"
        ),
        "",
        "## Decision Signals",
        "",
        f"- status_counts: {_fmt_counts(_mapping(signals.get('status_counts')))}",
        f"- target_kind_counts: {_fmt_counts(_mapping(signals.get('target_kind_counts')))}",
        (
            "- runner_wrapper_ok_counts: "
            f"{_fmt_counts(_mapping(signals.get('runner_wrapper_ok_counts')))}"
        ),
        (
            "- recommendation_action_counts: "
            f"{_fmt_counts(_mapping(signals.get('recommendation_action_counts')))}"
        ),
        (
            "- execution_verdict_counts: "
            f"{_fmt_counts(_mapping(signals.get('execution_verdict_counts')))}"
        ),
        (
            "- execution_guidance_action_counts: "
            f"{_fmt_counts(_mapping(signals.get('execution_guidance_action_counts')))}"
        ),
        (
            "- execution_next_source_counts: "
            f"{_fmt_counts(_mapping(signals.get('execution_next_source_counts')))}"
        ),
        f"- current_status_streak: {_fmt_streak(status_streak.get('status'), int(status_streak.get('count') or 0))}",
        f"- latest_executed_status: {_fmt(latest_executed.get('status'))}",
        f"- latest_executed_finished_at: {_fmt(latest_executed.get('finished_at'))}",
        f"- latest_executed_action: {_fmt(latest_executed.get('recommendation_action'))}",
        f"- latest_executed_champion_config: {_fmt(latest_executed.get('champion_config'))}",
        (
            "- latest_executed_execution_verdict: "
            f"{_fmt(latest_executed.get('execution_verdict'))}"
        ),
        (
            "- latest_executed_execution_best_config: "
            f"{_fmt(latest_executed.get('execution_best_config'))}"
        ),
        (
            "- latest_executed_execution_guidance_action: "
            f"{_fmt(latest_executed.get('execution_guidance_action'))}"
        ),
        (
            "- latest_executed_execution_next_source: "
            f"{_fmt(latest_executed.get('execution_next_source'))}"
        ),
        (
            "- latest_executed_execution_next_seeds: "
            f"{_fmt(latest_executed.get('execution_next_seeds'))}"
        ),
        (
            "- latest_executed_execution_next_script_path: "
            f"{_fmt(latest_executed.get('execution_next_script_path'))}"
        ),
        f"- last_problem_status: {_fmt(last_problem.get('status'))}",
        f"- last_problem_error: {_fmt(last_problem.get('error'))}",
        f"- last_problem_missing_required: {_fmt(last_problem.get('missing_required'))}",
        "",
        "## Recent Events",
        "",
        "| # | status | target | kind | dry_run | executed | returncode | "
        "started_at | duration_seconds | action | champion | exec_verdict | exec_best | next_seeds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, event in enumerate(events[-10:], max(1, len(events) - 9)):
        context = _mapping(event.get("recommendation_context"))
        champion = _mapping(context.get("champion"))
        execution_summary = _mapping(event.get("execution_summary"))
        execution_next = _mapping(execution_summary.get("next_command"))
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
            execution_summary.get("follow_up_verdict"),
            execution_summary.get("best_config_label"),
            execution_next.get("default_new_seeds"),
        ]
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    lines.append("")
    return "\n".join(lines)


def write_run_artifacts(
    summary: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    history_out: Path | None = None,
    history_markdown_out: Path | None = None,
    history_summary_out: Path | None = None,
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
    summary["run_history_summary_path"] = (
        str(history_summary_out) if history_summary_out is not None else None
    )
    if history_out is not None and append_history:
        _append_run_history(summary, history_out)
    if history_out is not None:
        events = _read_history_events(history_out)
        summary["run_history_summary"] = summarize_history_events(
            events,
            history_jsonl_path=history_out,
        )
        if history_markdown_out is not None:
            markdown_history = render_history_markdown(
                events,
                history_jsonl_path=history_out,
            )
            history_markdown_out.parent.mkdir(parents=True, exist_ok=True)
            history_markdown_out.write_text(markdown_history, encoding="utf-8")
        if history_summary_out is not None:
            history_summary_out.parent.mkdir(parents=True, exist_ok=True)
            history_summary_out.write_text(
                json.dumps(summary["run_history_summary"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    markdown = render_markdown(summary)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
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
    history_report_only: bool = False,
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
    previous_run_report = _previous_run_report(command_dir, command_scripts)
    execution_next_command = _execution_next_command_from_report(previous_run_report)
    if target == EXECUTION_NEXT_TARGET and not execution_next_command:
        execution_next_command = _execution_next_command_from_history(
            command_dir,
            command_scripts,
        )
    selected_execution_next_command = (
        execution_next_command if target == EXECUTION_NEXT_TARGET else None
    )
    script_key, script_path = _selected_script(
        command_scripts,
        target=target,
        execution_next_command=execution_next_command,
        previous_run_report=previous_run_report,
    )
    target_details = _target_details(
        command_scripts,
        target=target,
        script_key=script_key,
        script_path=script_path,
    )
    execution_cwd = _execution_cwd_for_target(
        command_dir,
        command_scripts,
        target=target,
    )
    recommendation_context = _recommendation_context(
        manifest,
        command_scripts,
        target_kind=target_details.get("target_kind"),
    )
    expected_summary_path = _expected_execution_summary_path(
        manifest,
        target_kind=target_details.get("target_kind"),
        execution_next_command=execution_next_command,
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
    history_summary_out = _run_history_summary_path(
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

    def finish(
        summary: dict[str, Any],
        *,
        append_history: bool = append_run_history,
    ) -> dict[str, Any]:
        summary = dict(summary)
        summary["expected_execution_summary_path"] = (
            str(expected_summary_path) if expected_summary_path is not None else None
        )
        summary["execution_summary"] = (
            _compact_execution_summary(expected_summary_path)
            if summary.get("executed")
            else None
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            history_summary_out=history_summary_out,
            append_history=append_history,
        )
        if not write_inspection_report:
            return summary
        refreshed_inspection = inspector.inspect_bundle(command_dir)
        refreshed_inspection = _write_inspection_report(
            inspector,
            command_dir,
            refreshed_inspection,
            command_scripts,
        )
        summary = dict(summary)
        summary["bundle_ready"] = bool(refreshed_inspection.get("bundle_ready"))
        summary["strict_ready"] = bool(refreshed_inspection.get("strict_ready"))
        summary["missing_required"] = (
            refreshed_inspection.get("missing_required") or []
        )
        summary["missing_optional"] = (
            refreshed_inspection.get("missing_optional") or []
        )
        runner_wrapper_status = _mapping(
            refreshed_inspection.get("runner_wrapper_status")
        )
        summary["runner_wrapper_status"] = runner_wrapper_status
        summary["runner_wrapper_ok"] = runner_wrapper_status.get("ok")
        summary["runner_wrapper_executes_runner_command"] = (
            runner_wrapper_status.get("executes_runner_command")
        )
        summary["runner_wrapper_forwards_arguments"] = runner_wrapper_status.get(
            "forwards_arguments"
        )
        summary["inspection"] = refreshed_inspection
        return write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
            history_out=history_out,
            history_markdown_out=history_markdown_out,
            history_summary_out=history_summary_out,
            append_history=False,
        )

    if history_report_only:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            execution_cwd=execution_cwd,
            selected_execution_next_command=selected_execution_next_command,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=0,
            history_report_only=True,
            **timing_fields(),
        )
        summary = finish(summary, append_history=False)
        return 0, summary

    if not required_ready:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            **target_details,
            recommendation_context=recommendation_context,
            execution_cwd=execution_cwd,
            selected_execution_next_command=selected_execution_next_command,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error="command bundle did not pass the requested inspection gate",
            **timing_fields(),
        )
        summary = finish(summary)
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
            execution_cwd=execution_cwd,
            selected_execution_next_command=selected_execution_next_command,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error=f"manifest does not declare {script_key}",
            **timing_fields(),
        )
        summary = finish(summary)
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
            execution_cwd=execution_cwd,
            selected_execution_next_command=selected_execution_next_command,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=0,
            **timing_fields(),
        )
        summary = finish(summary)
        return 0, summary
    if json_mode:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=execution_cwd,
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
            execution_cwd=execution_cwd,
            selected_execution_next_command=selected_execution_next_command,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            executed=True,
            **timing_fields(),
        )
        summary = finish(summary)
        return result.returncode, summary
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=execution_cwd,
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
        execution_cwd=execution_cwd,
        selected_execution_next_command=selected_execution_next_command,
        strict=strict,
        dry_run=dry_run,
        inspection=inspection,
        returncode=result.returncode,
        executed=True,
        **timing_fields(),
    )
    summary = finish(summary)
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
        help="write Markdown and JSON summaries of run_history.jsonl",
    )
    parser.add_argument(
        "--history-report-only",
        action="store_true",
        help=(
            "rewrite run_history.md and run_history_summary.json without "
            "executing a command or appending a new history event"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.history_report_only and args.append_run_history:
        parser.error("--history-report-only cannot be combined with --append-run-history")
    write_run_history_report = bool(
        args.write_run_history_report or args.history_report_only
    )
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
            write_run_history_report=write_run_history_report,
            history_report_only=bool(args.history_report_only),
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
    elif args.dry_run or args.history_report_only or returncode != 0:
        stream = sys.stderr if returncode != 0 else sys.stdout
        print(render_markdown(summary), file=stream)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

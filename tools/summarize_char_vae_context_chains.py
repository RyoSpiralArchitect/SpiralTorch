#!/usr/bin/env python3
"""Summarize multiple char VAE context chain.json artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.chain_comparison.v1"


def _chain_path(path: Path) -> Path:
    return path / "chain.json" if path.is_dir() else path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _discover_chain_paths(paths: list[Path], *, recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if recursive and path.is_dir():
            matches = sorted(candidate for candidate in path.rglob("chain.json") if candidate.is_file())
            if matches:
                discovered.extend(matches)
                continue
        discovered.append(_chain_path(path))
    return _dedupe_paths(discovered)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _maybe_read_json(path: Any) -> dict[str, Any] | None:
    if not isinstance(path, str) or not path:
        return None
    json_path = Path(path)
    if not json_path.exists() or not json_path.is_file():
        return None
    try:
        return _read_json(json_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _value(payload: dict[str, Any] | None, *keys: str) -> Any:
    item: Any = payload
    for key in keys:
        if not isinstance(item, dict):
            return None
        item = item.get(key)
    return item


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _fmt_groups(groups: Any) -> str:
    if not isinstance(groups, list) or not groups:
        return "-"
    return ",".join(str(group) for group in groups)


def _fmt_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "-"
    return ", ".join(str(value) for value in values)


def _merge_counts(target: dict[str, int], counts: Any) -> None:
    if not isinstance(counts, dict):
        return
    for key, value in counts.items():
        try:
            amount = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = target.get(str(key), 0) + amount


def _chain_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    seed_summary = payload.get("follow_up_seed_resolution_summary")
    seed_summary = seed_summary if isinstance(seed_summary, dict) else {}
    accepted = payload.get("accepted_step")
    accepted = accepted if isinstance(accepted, dict) else {}
    best = payload.get("best_step")
    best = best if isinstance(best, dict) else {}
    return {
        "source": str(path),
        "preset": payload.get("preset"),
        "run_root": payload.get("run_root"),
        "dry_run": bool(payload.get("dry_run")),
        "stopped_reason": payload.get("stopped_reason"),
        "allowed_gate_stop": bool(payload.get("allowed_gate_stop")),
        "planned_follow_ups": payload.get("planned_follow_ups"),
        "attempted_follow_ups": payload.get("attempted_follow_ups"),
        "accepted_step": accepted.get("index"),
        "accepted_role": accepted.get("role"),
        "accepted_run_dir": accepted.get("run_dir"),
        "accepted_summary_path": payload.get("accepted_summary_path")
        or accepted.get("summary_path"),
        "accepted_config": accepted.get("best_config_label")
        or accepted.get("best_feature"),
        "accepted_mean_best_nll": accepted.get("mean_best_nll"),
        "accepted_delta_vs_raw": accepted.get("mean_best_nll_delta_vs_raw"),
        "accepted_runner_up_feature": accepted.get("runner_up_feature"),
        "accepted_margin_to_runner_up": accepted.get("margin_to_runner_up"),
        "accepted_runner_up_within_uncertainty": accepted.get(
            "runner_up_within_uncertainty"
        ),
        "best_step": best.get("index"),
        "best_role": best.get("role"),
        "best_run_dir": best.get("run_dir"),
        "best_summary_path": payload.get("best_summary_path")
        or best.get("summary_path"),
        "best_config": best.get("best_config_label") or best.get("best_feature"),
        "best_mean_best_nll": best.get("mean_best_nll"),
        "best_delta_vs_raw": best.get("mean_best_nll_delta_vs_raw"),
        "best_runner_up_feature": best.get("runner_up_feature"),
        "best_margin_to_runner_up": best.get("margin_to_runner_up"),
        "best_runner_up_within_uncertainty": best.get(
            "runner_up_within_uncertainty"
        ),
        "runner_up_feature": best.get("runner_up_feature"),
        "margin_to_runner_up": best.get("margin_to_runner_up"),
        "runner_up_within_uncertainty": best.get("runner_up_within_uncertainty"),
        "seed_source_counts": seed_summary.get("seed_source_counts", {}),
        "command_source_counts": seed_summary.get("command_source_counts", {}),
        "configured_seed_group_status_counts": seed_summary.get(
            "configured_seed_group_status_counts",
            {},
        ),
        "gate_failed_count": seed_summary.get("gate_failed_count", 0),
        "nonzero_exit_count": seed_summary.get("nonzero_exit_count", 0),
        "extra_explicit_seed_groups": payload.get("extra_explicit_seed_groups", []),
        "unused_explicit_seed_groups": payload.get("unused_explicit_seed_groups", []),
    }


def _leader_record(
    row: dict[str, Any],
    *,
    prefix: str,
    mean_best_nll: float,
) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "preset": row.get("preset"),
        "run_root": row.get("run_root"),
        "step": row.get(f"{prefix}_step"),
        "role": row.get(f"{prefix}_role"),
        "run_dir": row.get(f"{prefix}_run_dir"),
        "summary_path": row.get(f"{prefix}_summary_path"),
        "config": row.get(f"{prefix}_config"),
        "mean_best_nll": mean_best_nll,
        "delta_vs_raw": row.get(f"{prefix}_delta_vs_raw"),
        "stopped_reason": row.get("stopped_reason"),
        "allowed_gate_stop": row.get("allowed_gate_stop"),
        "runner_up_feature": row.get(f"{prefix}_runner_up_feature"),
        "margin_to_runner_up": row.get(f"{prefix}_margin_to_runner_up"),
        "runner_up_within_uncertainty": row.get(
            f"{prefix}_runner_up_within_uncertainty"
        ),
        "seed_source_counts": row.get("seed_source_counts"),
        "command_source_counts": row.get("command_source_counts"),
    }


def _leader(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    key = f"{prefix}_mean_best_nll"
    for row in rows:
        mean_best_nll = _number(row.get(key))
        if mean_best_nll is not None:
            candidates.append((mean_best_nll, row))
    if not candidates:
        return None
    mean_best_nll, row = min(
        candidates,
        key=lambda candidate: (candidate[0], str(candidate[1].get("source"))),
    )
    return _leader_record(row, prefix=prefix, mean_best_nll=mean_best_nll)


def _same_leader(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> bool:
    if left is None or right is None:
        return False
    return (
        left.get("source") == right.get("source")
        and left.get("step") == right.get("step")
        and left.get("config") == right.get("config")
    )


def _selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = _leader(rows, prefix="accepted")
    best = _leader(rows, prefix="best")
    matches = _same_leader(accepted, best)
    accepted_nll = _number(accepted.get("mean_best_nll")) if accepted else None
    best_nll = _number(best.get("mean_best_nll")) if best else None
    gap = (
        accepted_nll - best_nll
        if accepted_nll is not None and best_nll is not None
        else None
    )
    return {
        "accepted_candidate_count": sum(
            1 for row in rows if _number(row.get("accepted_mean_best_nll")) is not None
        ),
        "best_candidate_count": sum(
            1 for row in rows if _number(row.get("best_mean_best_nll")) is not None
        ),
        "accepted_champion": accepted,
        "best_champion": best,
        "accepted_matches_best": matches,
        "best_requires_review": best is not None and not matches,
        "accepted_vs_best_nll_gap": gap,
    }


def _summary_follow_up_command(summary_path: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": "st.llm_char_vae_context.chain_recommended_command.v1",
        "source_summary_path": summary_path if isinstance(summary_path, str) else None,
        "available": False,
        "command_source": None,
        "script_usage": None,
        "script_path": None,
        "shell_command": None,
        "default_new_seeds": None,
        "default_run_dir": None,
        "default_follow_up_from": None,
        "missing_reason": "summary path unavailable",
    }
    summary = _maybe_read_json(summary_path)
    if summary is None:
        if isinstance(summary_path, str) and summary_path:
            record["missing_reason"] = "summary json unavailable"
        return record

    command_names = [
        "guided_next_follow_up_command",
        "feature_swap_review_command",
        "best_generation_follow_up_command",
        "broadened_follow_up_command",
        "next_follow_up_command",
    ]
    for name in command_names:
        command = summary.get(name)
        if not isinstance(command, dict):
            continue
        if name == "guided_next_follow_up_command" and not command.get("enabled"):
            continue
        script_usage = command.get("script_usage")
        script_path = command.get("script_path")
        shell_command = command.get("shell_command")
        if not any(
            isinstance(item, str) and item
            for item in (script_usage, script_path, shell_command)
        ):
            continue
        record.update(
            {
                "available": True,
                "command_source": name,
                "script_usage": script_usage,
                "script_path": script_path,
                "shell_command": shell_command,
                "default_new_seeds": command.get("default_new_seeds"),
                "default_run_dir": command.get("default_run_dir"),
                "default_follow_up_from": command.get("default_follow_up_from"),
                "default_follow_up_fail_on_verdict": command.get(
                    "default_follow_up_fail_on_verdict"
                ),
                "missing_reason": None,
            }
        )
        return record

    record["missing_reason"] = "no runnable follow-up command in summary"
    return record


def _with_recommended_commands(recommendation: dict[str, Any]) -> dict[str, Any]:
    recommendation = dict(recommendation)
    recommendation["follow_up_command"] = _summary_follow_up_command(
        recommendation.get("follow_up_from_summary_path")
    )
    recommendation["review_command"] = _summary_follow_up_command(
        recommendation.get("review_summary_path")
    )
    return recommendation


def _recommendation(selection: dict[str, Any]) -> dict[str, Any]:
    accepted = selection.get("accepted_champion")
    accepted = accepted if isinstance(accepted, dict) else None
    best = selection.get("best_champion")
    best = best if isinstance(best, dict) else None
    if best is not None and selection.get("best_requires_review"):
        return _with_recommended_commands(
            {
                "schema": "st.llm_char_vae_context.chain_recommendation.v1",
                "action": "review_absolute_best",
                "reason": (
                    "absolute best has lower NLL than the accepted champion, "
                    "but it differs from the safe accepted promotion"
                ),
                "follow_up_from_summary_path": accepted.get("summary_path")
                if accepted is not None
                else None,
                "review_summary_path": best.get("summary_path"),
                "champion_source": "best_champion",
                "champion": best,
                "fallback_source": (
                    "accepted_champion" if accepted is not None else None
                ),
                "fallback": accepted,
            }
        )
    if accepted is not None:
        return _with_recommended_commands(
            {
                "schema": "st.llm_char_vae_context.chain_recommendation.v1",
                "action": "continue_from_accepted",
                "reason": (
                    "accepted champion matches the absolute best"
                    if selection.get("accepted_matches_best")
                    else "accepted champion is the best safe promotion candidate"
                ),
                "follow_up_from_summary_path": accepted.get("summary_path"),
                "review_summary_path": None,
                "champion_source": "accepted_champion",
                "champion": accepted,
                "fallback_source": None,
                "fallback": None,
            }
        )
    if best is not None:
        return _with_recommended_commands(
            {
                "schema": "st.llm_char_vae_context.chain_recommendation.v1",
                "action": "review_absolute_best",
                "reason": "only an absolute best candidate is available",
                "follow_up_from_summary_path": None,
                "review_summary_path": best.get("summary_path"),
                "champion_source": "best_champion",
                "champion": best,
                "fallback_source": None,
                "fallback": None,
            }
        )
    return _with_recommended_commands(
        {
            "schema": "st.llm_char_vae_context.chain_recommendation.v1",
            "action": "collect_more_chains",
            "reason": "no accepted or best chain candidates were found",
            "follow_up_from_summary_path": None,
            "review_summary_path": None,
            "champion_source": None,
            "champion": None,
            "fallback_source": None,
            "fallback": None,
        }
    )


def _sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "input":
        return rows
    if sort_by == "accepted":
        return sorted(
            rows,
            key=lambda row: (
                _number(row.get("accepted_mean_best_nll")) is None,
                _number(row.get("accepted_mean_best_nll")) or 0.0,
                str(row.get("source")),
            ),
        )
    if sort_by == "best":
        return sorted(
            rows,
            key=lambda row: (
                _number(row.get("best_mean_best_nll")) is None,
                _number(row.get("best_mean_best_nll")) or 0.0,
                str(row.get("source")),
            ),
        )
    if sort_by == "attempted":
        return sorted(
            rows,
            key=lambda row: (
                -int(row.get("attempted_follow_ups") or 0),
                str(row.get("source")),
            ),
        )
    raise ValueError(f"unknown sort mode: {sort_by}")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_sources: dict[str, int] = {}
    command_sources: dict[str, int] = {}
    group_statuses: dict[str, int] = {}
    stopped_reasons: dict[str, int] = {}
    for row in rows:
        _merge_counts(seed_sources, row.get("seed_source_counts"))
        _merge_counts(command_sources, row.get("command_source_counts"))
        _merge_counts(
            group_statuses,
            row.get("configured_seed_group_status_counts"),
        )
        stopped_reason = row.get("stopped_reason")
        if stopped_reason:
            label = str(stopped_reason)
            stopped_reasons[label] = stopped_reasons.get(label, 0) + 1
    return {
        "chain_count": len(rows),
        "attempted_follow_ups": sum(int(row.get("attempted_follow_ups") or 0) for row in rows),
        "gate_failed_count": sum(int(row.get("gate_failed_count") or 0) for row in rows),
        "nonzero_exit_count": sum(int(row.get("nonzero_exit_count") or 0) for row in rows),
        "allowed_gate_stop_count": sum(1 for row in rows if row.get("allowed_gate_stop")),
        "dry_run_count": sum(1 for row in rows if row.get("dry_run")),
        "seed_source_counts": seed_sources,
        "command_source_counts": command_sources,
        "configured_seed_group_status_counts": group_statuses,
        "stopped_reason_counts": stopped_reasons,
    }


def summarize_chains(
    paths: list[Path],
    *,
    sort_by: str = "input",
    recursive: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    discovered_paths = _discover_chain_paths(paths, recursive=recursive)
    for path in discovered_paths:
        rows.append(_chain_row(path, _read_json(path)))
    rows = _sort_rows(rows, sort_by)
    selection = _selection(rows)
    return {
        "schema": SCHEMA,
        "sort_by": sort_by,
        "recursive": recursive,
        "input_count": len(paths),
        "discovered_chain_count": len(discovered_paths),
        "aggregate": _aggregate(rows),
        "selection": selection,
        "recommendation": _recommendation(selection),
        "chains": rows,
    }


def _fmt_leader(record: Any) -> str:
    if not isinstance(record, dict):
        return "-"
    return (
        "{config} (source={source}, step={step}, nll={nll}, "
        "delta_vs_raw={delta}, summary={summary})"
    ).format(
        config=_fmt(record.get("config")),
        source=_fmt(record.get("source")),
        step=_fmt(record.get("step")),
        nll=_fmt(record.get("mean_best_nll")),
        delta=_fmt(record.get("delta_vs_raw")),
        summary=_fmt(record.get("summary_path")),
    )


def _fmt_recommendation(record: Any) -> str:
    if not isinstance(record, dict):
        return "-"
    return (
        "{action} (follow_up_from={follow_up}, review={review}, reason={reason})"
    ).format(
        action=_fmt(record.get("action")),
        follow_up=_fmt(record.get("follow_up_from_summary_path")),
        review=_fmt(record.get("review_summary_path")),
        reason=_fmt(record.get("reason")),
    )


def _fmt_command(record: Any) -> str:
    if not isinstance(record, dict):
        return "-"
    if not record.get("available"):
        return f"unavailable ({_fmt(record.get('missing_reason'))})"
    return (
        "{source}: {usage}"
    ).format(
        source=_fmt(record.get("command_source")),
        usage=_fmt(record.get("script_usage") or record.get("shell_command")),
    )


def _command_line(record: Any) -> str | None:
    if not isinstance(record, dict) or not record.get("available"):
        return None
    command_line = _safe_script_command_line(record)
    if command_line is not None:
        return command_line
    for key in ("script_usage", "shell_command"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _env_assignment(name: str, value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return f"{name}={shlex.quote(value)}"


def _safe_script_command_line(record: dict[str, Any]) -> str | None:
    script_path = record.get("script_path")
    if not isinstance(script_path, str) or not script_path:
        return None
    assignments = [
        _env_assignment("FOLLOW_UP_FROM", record.get("default_follow_up_from")),
        _env_assignment("NEW_SEEDS", record.get("default_new_seeds")),
        _env_assignment("NEXT_RUN_DIR", record.get("default_run_dir")),
        _env_assignment(
            "FOLLOW_UP_FAIL_ON_VERDICT",
            record.get("default_follow_up_fail_on_verdict"),
        ),
    ]
    parts = [assignment for assignment in assignments if assignment]
    parts.extend(["bash", shlex.quote(script_path)])
    return " ".join(parts)


def _write_command_script(
    path: Path,
    record: Any,
    *,
    label: str,
    execution_cwd: Path,
) -> str | None:
    command_line = _command_line(record)
    if command_line is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = shlex.quote(str(execution_cwd))
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Generated by {Path(__file__).name}: {label}",
            f"# source_summary_path: {_fmt(_value(record, 'source_summary_path'))}",
            f"# command_source: {_fmt(_value(record, 'command_source'))}",
            f"# execution_cwd: {execution_cwd}",
            f"cd {cwd}",
            command_line,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    return str(path)


def _recommended_next_target(
    recommendation: dict[str, Any],
    *,
    follow_up_path: str | None,
    review_path: str | None,
) -> tuple[str | None, str | None]:
    action = recommendation.get("action")
    if action == "review_absolute_best" and review_path:
        return review_path, "review"
    if follow_up_path:
        return follow_up_path, "follow_up"
    if review_path:
        return review_path, "review"
    return None, None


def _write_next_command_script(
    path: Path,
    target_path: str | None,
    *,
    target_kind: str | None,
) -> str | None:
    if target_path is None or target_kind is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    script_name = Path(target_path).name
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Generated by {Path(__file__).name}: recommended_next",
            f"# target_kind: {target_kind}",
            f"# target_script: {script_name}",
            'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            f'exec bash "$SCRIPT_DIR/{script_name}" "$@"',
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    return str(path)


def _write_runner_command_script(
    path: Path,
    runner_command: str | None,
    *,
    target_kind: str | None,
    label: str = "run_recommended_next",
    description: str = "# Runs the recommended next action through bundle inspection first.",
) -> str | None:
    if runner_command is None or target_kind is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Generated by {Path(__file__).name}: {label}",
            f"# target_kind: {target_kind}",
            description,
            f"exec {runner_command} \"$@\"",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    return str(path)


def _fmt_readme_value(value: Any) -> str:
    text = _fmt(value)
    if text == "-":
        return text
    return f"`{text}`"


def _run_line(path: Any, *extra_args: str) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    parts = ["bash", shlex.quote(path)]
    parts.extend(shlex.quote(arg) for arg in extra_args if arg)
    return " ".join(parts)


def _inspection_command_line(command_dir: Any) -> str | None:
    if not isinstance(command_dir, str) or not command_dir:
        return None
    script_path = Path(__file__).resolve().with_name(
        "inspect_char_vae_command_bundle.py"
    )
    return (
        "PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(script_path))} "
        f"{shlex.quote(command_dir)} --strict --write-report"
    )


def _runner_command_line(
    command_dir: Any,
    *,
    target: str | None = None,
    use_history_next_action: bool = False,
) -> str | None:
    if not isinstance(command_dir, str) or not command_dir:
        return None
    script_path = Path(__file__).resolve().with_name("run_char_vae_command_bundle.py")
    target_arg = f" --target {shlex.quote(target)}" if target else ""
    history_arg = " --use-history-next-action" if use_history_next_action else ""
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(script_path))} "
        f"{shlex.quote(command_dir)}{target_arg}{history_arg} --write-inspection-report "
        "--write-run-report --append-run-history --write-run-history-report"
    )


def _history_report_command_line(command_dir: Any) -> str | None:
    if not isinstance(command_dir, str) or not command_dir:
        return None
    script_path = Path(__file__).resolve().with_name("run_char_vae_command_bundle.py")
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(script_path))} "
        f"{shlex.quote(command_dir)} --history-report-only"
    )


def _history_loop_command_line(
    command_dir: Any,
    *,
    max_steps: int = 3,
    fail_on_final_actions: tuple[str, ...] = (
        "review_before_continuing",
        "inspect_history",
    ),
) -> str | None:
    if not isinstance(command_dir, str) or not command_dir:
        return None
    script_path = Path(__file__).resolve().with_name("run_char_vae_history_loop.py")
    fail_arg = (
        " --fail-on-final-action "
        f"{shlex.quote(','.join(fail_on_final_actions))}"
        if fail_on_final_actions
        else ""
    )
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(script_path))} "
        f"{shlex.quote(command_dir)} --max-steps {max_steps}{fail_arg} "
        "--write-loop-report"
    )


def _path_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _resolved_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def _chain_sources(summary: dict[str, Any]) -> list[str]:
    chains = summary.get("chains")
    if not isinstance(chains, list):
        return []
    sources: list[str] = []
    for row in chains:
        if not isinstance(row, dict):
            continue
        source = row.get("source")
        if isinstance(source, str) and source:
            sources.append(source)
    return sources


def _render_command_readme(
    summary: dict[str, Any],
    command_scripts: dict[str, Any],
) -> str:
    recommendation = summary.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    champion = recommendation.get("champion")
    champion = champion if isinstance(champion, dict) else {}
    fallback = recommendation.get("fallback")
    fallback = fallback if isinstance(fallback, dict) else {}
    follow_up = recommendation.get("follow_up_command")
    follow_up = follow_up if isinstance(follow_up, dict) else {}
    review = recommendation.get("review_command")
    review = review if isinstance(review, dict) else {}
    selection = summary.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    lines = [
        "# Char VAE Chain Recommended Commands",
        "",
        "Generated by `summarize_char_vae_context_chains.py --command-out-dir`.",
        "",
        "## Recommendation",
        "",
        f"- action: {_fmt_readme_value(recommendation.get('action'))}",
        f"- reason: {_fmt(recommendation.get('reason'))}",
        f"- accepted_matches_best: {_fmt_readme_value(selection.get('accepted_matches_best'))}",
        f"- best_requires_review: {_fmt_readme_value(selection.get('best_requires_review'))}",
        f"- follow_up_from_summary_path: {_fmt_readme_value(recommendation.get('follow_up_from_summary_path'))}",
        f"- review_summary_path: {_fmt_readme_value(recommendation.get('review_summary_path'))}",
        f"- execution_cwd: {_fmt_readme_value(command_scripts.get('execution_cwd'))}",
        "",
        "## Champion",
        "",
        f"- source: {_fmt_readme_value(recommendation.get('champion_source'))}",
        f"- config: {_fmt_readme_value(champion.get('config'))}",
        f"- mean_best_nll: {_fmt_readme_value(champion.get('mean_best_nll'))}",
        f"- step: {_fmt_readme_value(champion.get('step'))}",
        f"- summary_path: {_fmt_readme_value(champion.get('summary_path'))}",
        "",
        "## Fallback",
        "",
        f"- source: {_fmt_readme_value(recommendation.get('fallback_source'))}",
        f"- config: {_fmt_readme_value(fallback.get('config'))}",
        f"- mean_best_nll: {_fmt_readme_value(fallback.get('mean_best_nll'))}",
        f"- step: {_fmt_readme_value(fallback.get('step'))}",
        f"- summary_path: {_fmt_readme_value(fallback.get('summary_path'))}",
        "",
        "## Recommended Next",
        "",
        "Run this dispatcher when you want the comparison's preferred next action.",
        "",
        f"- script: {_fmt_readme_value(command_scripts.get('next_path'))}",
        f"- target_kind: {_fmt_readme_value(command_scripts.get('next_kind'))}",
        f"- run: {_fmt_readme_value(_run_line(command_scripts.get('next_path')))}",
        f"- inspected_script: {_fmt_readme_value(command_scripts.get('runner_path'))}",
        f"- inspected_script_run: {_fmt_readme_value(_run_line(command_scripts.get('runner_path')))}",
        f"- inspected_run: {_fmt_readme_value(command_scripts.get('runner_command'))}",
        f"- run_json: {_fmt_readme_value(command_scripts.get('run_json_path'))}",
        f"- run_markdown: {_fmt_readme_value(command_scripts.get('run_markdown_path'))}",
        f"- run_history_jsonl: {_fmt_readme_value(command_scripts.get('run_history_jsonl_path'))}",
        f"- run_history_markdown: {_fmt_readme_value(command_scripts.get('run_history_markdown_path'))}",
        f"- run_history_summary: {_fmt_readme_value(command_scripts.get('run_history_summary_path'))}",
        f"- history_report_only: {_fmt_readme_value(command_scripts.get('history_report_command'))}",
        "",
        "## History-Guided Continuation",
        "",
        "After run history exists, use this command to let `run_history` choose the next safe runnable target.",
        "",
        f"- inspected_script: {_fmt_readme_value(command_scripts.get('history_next_action_runner_path'))}",
        f"- inspected_script_run: {_fmt_readme_value(_run_line(command_scripts.get('history_next_action_runner_path')))}",
        f"- inspected_run: {_fmt_readme_value(command_scripts.get('history_next_action_command'))}",
        "",
        "## History-Guided Loop",
        "",
        "Run this bounded loop when you want the bundle to keep following safe run-history decisions until review, a blocker, or `--max-steps` stops it.",
        "",
        f"- inspected_script: {_fmt_readme_value(command_scripts.get('history_loop_runner_path'))}",
        f"- inspected_script_run: {_fmt_readme_value(_run_line(command_scripts.get('history_loop_runner_path')))}",
        f"- inspected_run: {_fmt_readme_value(command_scripts.get('history_loop_command'))}",
        f"- run_loop_json: {_fmt_readme_value(command_scripts.get('run_loop_json_path'))}",
        f"- run_loop_markdown: {_fmt_readme_value(command_scripts.get('run_loop_markdown_path'))}",
        "",
        "## Execution-Next Continuation",
        "",
        "After an inspected run writes `run.json`, use this target to run the next command selected from the latest execution summary or run history.",
        "",
        f"- target: {_fmt_readme_value('execution-next')}",
        f"- inspected_script_run: {_fmt_readme_value(_run_line(command_scripts.get('runner_path'), '--target', 'execution-next'))}",
        f"- inspected_run: {_fmt_readme_value(command_scripts.get('execution_next_command'))}",
        "",
        "## Safe Follow-Up",
        "",
        "Use this when the comparison recommends continuing from the accepted champion.",
        "",
        f"- script: {_fmt_readme_value(command_scripts.get('follow_up_path'))}",
        f"- run: {_fmt_readme_value(_run_line(command_scripts.get('follow_up_path')))}",
        f"- command_source: {_fmt_readme_value(follow_up.get('command_source'))}",
        f"- summary_path: {_fmt_readme_value(follow_up.get('source_summary_path'))}",
        "",
        "## Absolute-Best Review",
        "",
        "Use this when a lower-NLL candidate exists but still needs gate-stop review before promotion.",
        "",
        f"- script: {_fmt_readme_value(command_scripts.get('review_path'))}",
        f"- run: {_fmt_readme_value(_run_line(command_scripts.get('review_path')))}",
        f"- command_source: {_fmt_readme_value(review.get('command_source'))}",
        f"- summary_path: {_fmt_readme_value(review.get('source_summary_path'))}",
        "",
        "## Comparison Artifacts",
        "",
        "Use these to reopen the full comparison that generated this command directory.",
        "",
        f"- json: {_fmt_readme_value(command_scripts.get('comparison_json_path'))}",
        f"- markdown: {_fmt_readme_value(command_scripts.get('comparison_markdown_path'))}",
        f"- chain_sources: {_fmt_readme_value(', '.join(_chain_sources(summary)))}",
        "",
        "## Bundle Inspection",
        "",
        "Run this before handing the command bundle to automation.",
        "",
        f"- run: {_fmt_readme_value(command_scripts.get('inspection_command'))}",
        f"- report_json: {_fmt_readme_value(command_scripts.get('inspection_json_path'))}",
        f"- report_markdown: {_fmt_readme_value(command_scripts.get('inspection_markdown_path'))}",
        f"- generated_now: {_fmt_readme_value(command_scripts.get('inspection_generated'))}",
        f"- strict_ready: {_fmt_readme_value(command_scripts.get('inspection_strict_ready'))}",
        f"- runner_wrapper_ok: {_fmt_readme_value(command_scripts.get('inspection_runner_wrapper_ok'))}",
        f"- runner_wrapper_executes_runner_command: {_fmt_readme_value(command_scripts.get('inspection_runner_wrapper_executes_runner_command'))}",
        f"- runner_wrapper_forwards_arguments: {_fmt_readme_value(command_scripts.get('inspection_runner_wrapper_forwards_arguments'))}",
        f"- history_next_action_runner_ok: {_fmt_readme_value(command_scripts.get('inspection_history_next_action_runner_ok'))}",
        f"- history_next_action_runner_executes_command: {_fmt_readme_value(command_scripts.get('inspection_history_next_action_runner_executes_command'))}",
        f"- history_next_action_runner_forwards_arguments: {_fmt_readme_value(command_scripts.get('inspection_history_next_action_runner_forwards_arguments'))}",
        f"- history_loop_runner_ok: {_fmt_readme_value(command_scripts.get('inspection_history_loop_runner_ok'))}",
        f"- history_loop_runner_executes_command: {_fmt_readme_value(command_scripts.get('inspection_history_loop_runner_executes_command'))}",
        f"- history_loop_runner_forwards_arguments: {_fmt_readme_value(command_scripts.get('inspection_history_loop_runner_forwards_arguments'))}",
        "",
        "## Machine-Readable Manifest",
        "",
        "Includes comparison metadata, aggregate counts, selection, recommendation, and scripts.",
        "",
        f"- manifest: {_fmt_readme_value(command_scripts.get('manifest_path'))}",
    ]
    return "\n".join(lines) + "\n"


def _command_manifest(
    summary: dict[str, Any],
    command_scripts: dict[str, Any],
) -> dict[str, Any]:
    recommendation = summary.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    return {
        "schema": "st.llm_char_vae_context.chain_command_manifest.v1",
        "comparison": {
            "schema": summary.get("schema"),
            "sort_by": summary.get("sort_by"),
            "recursive": summary.get("recursive"),
            "input_count": summary.get("input_count"),
            "discovered_chain_count": summary.get("discovered_chain_count"),
            "chain_sources": _chain_sources(summary),
        },
        "aggregate": summary.get("aggregate"),
        "selection": summary.get("selection"),
        "recommendation": recommendation,
        "command_scripts": command_scripts,
    }


def _write_command_bundle_metadata(
    summary: dict[str, Any],
    command_scripts: dict[str, Any],
    *,
    manifest_path: Path,
    readme_path: Path,
) -> None:
    manifest = _command_manifest(summary, command_scripts)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme_path.write_text(
        _render_command_readme(summary, command_scripts),
        encoding="utf-8",
    )


def _write_recommended_command_scripts(
    summary: dict[str, Any],
    out_dir: Path,
    *,
    comparison_json_path: Path | None = None,
    comparison_markdown_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    execution_cwd = Path.cwd().resolve()
    recommendation = summary.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    follow_up_path = _write_command_script(
        out_dir / "recommended_follow_up.sh",
        recommendation.get("follow_up_command"),
        label="follow_up_command",
        execution_cwd=execution_cwd,
    )
    review_path = _write_command_script(
        out_dir / "recommended_review.sh",
        recommendation.get("review_command"),
        label="review_command",
        execution_cwd=execution_cwd,
    )
    next_target, next_kind = _recommended_next_target(
        recommendation,
        follow_up_path=follow_up_path,
        review_path=review_path,
    )
    next_path = _write_next_command_script(
        out_dir / "recommended_next.sh",
        next_target,
        target_kind=next_kind,
    )
    runner_command = _runner_command_line(str(out_dir))
    execution_next_command = _runner_command_line(
        str(out_dir),
        target="execution-next",
    )
    history_next_action_command = _runner_command_line(
        str(out_dir),
        use_history_next_action=True,
    )
    history_next_action_runner_path = _write_runner_command_script(
        out_dir / "run_history_next_action.sh",
        history_next_action_command,
        target_kind="history_next_action",
        label="run_history_next_action",
        description="# Runs the next safe target selected from run history.",
    )
    history_loop_command = _history_loop_command_line(str(out_dir))
    history_loop_runner_path = _write_runner_command_script(
        out_dir / "run_history_loop.sh",
        history_loop_command,
        target_kind="history_loop",
        label="run_history_loop",
        description="# Runs bounded history-guided continuation.",
    )
    runner_path = _write_runner_command_script(
        out_dir / "run_recommended_next.sh",
        runner_command if next_path else None,
        target_kind=next_kind,
    )
    manifest_path = out_dir / "recommendation.json"
    readme_path = out_dir / "README.md"
    command_scripts = {
        "schema": "st.llm_char_vae_context.chain_command_scripts.v1",
        "directory": str(out_dir),
        "next_path": next_path,
        "next_kind": next_kind,
        "follow_up_path": follow_up_path,
        "review_path": review_path,
        "written_count": sum(
            1
            for path in (next_path, follow_up_path, review_path, runner_path)
            + (history_next_action_runner_path, history_loop_runner_path)
            if path
        ),
        "execution_cwd": str(execution_cwd),
        "comparison_json_path": _resolved_path(comparison_json_path),
        "comparison_markdown_path": _resolved_path(comparison_markdown_path),
        "inspection_command": _inspection_command_line(str(out_dir)),
        "inspection_json_path": str(out_dir / "inspection.json"),
        "inspection_markdown_path": str(out_dir / "inspection.md"),
        "inspection_generated": False,
        "inspection_bundle_ready": None,
        "inspection_strict_ready": None,
        "inspection_missing_required": [],
        "inspection_missing_optional": [],
        "inspection_runner_wrapper_status": None,
        "inspection_runner_wrapper_ok": None,
        "inspection_runner_wrapper_executes_runner_command": None,
        "inspection_runner_wrapper_forwards_arguments": None,
        "inspection_history_next_action_runner_status": None,
        "inspection_history_next_action_runner_ok": None,
        "inspection_history_next_action_runner_executes_command": None,
        "inspection_history_next_action_runner_forwards_arguments": None,
        "inspection_history_loop_runner_status": None,
        "inspection_history_loop_runner_ok": None,
        "inspection_history_loop_runner_executes_command": None,
        "inspection_history_loop_runner_forwards_arguments": None,
        "runner_command": runner_command,
        "execution_next_command": execution_next_command,
        "history_next_action_command": history_next_action_command,
        "history_next_action_runner_path": history_next_action_runner_path,
        "history_loop_command": history_loop_command,
        "history_loop_runner_path": history_loop_runner_path,
        "history_report_command": _history_report_command_line(str(out_dir)),
        "runner_path": runner_path,
        "run_json_path": str(out_dir / "run.json"),
        "run_markdown_path": str(out_dir / "run.md"),
        "run_history_jsonl_path": str(out_dir / "run_history.jsonl"),
        "run_history_markdown_path": str(out_dir / "run_history.md"),
        "run_history_summary_path": str(out_dir / "run_history_summary.json"),
        "run_loop_json_path": str(out_dir / "run_loop.json"),
        "run_loop_markdown_path": str(out_dir / "run_loop.md"),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
    }
    _write_command_bundle_metadata(
        summary,
        command_scripts,
        manifest_path=manifest_path,
        readme_path=readme_path,
    )
    return command_scripts


def _load_command_bundle_inspector() -> Any:
    module_path = Path(__file__).with_name("inspect_char_vae_command_bundle.py")
    spec = importlib.util.spec_from_file_location(
        "_spiraltorch_char_vae_command_bundle_inspector",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load command bundle inspector from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_command_bundle_inspection(command_dir: Path) -> dict[str, Any]:
    inspector = _load_command_bundle_inspector()
    manifest = _read_json(command_dir / "recommendation.json")
    command_scripts = manifest.get("command_scripts")
    command_scripts = command_scripts if isinstance(command_scripts, dict) else {}
    json_out = (
        _path_value(command_scripts.get("inspection_json_path"))
        or command_dir / "inspection.json"
    )
    markdown_out = (
        _path_value(command_scripts.get("inspection_markdown_path"))
        or command_dir / "inspection.md"
    )
    summary = inspector.inspect_bundle(command_dir)
    summary["inspection_json_path"] = str(json_out)
    summary["inspection_markdown_path"] = str(markdown_out)
    markdown = inspector.render_markdown(summary)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    selection = summary.get("selection", {})
    recommendation = summary.get("recommendation", {})
    command_scripts = summary.get("command_scripts", {})
    chains = summary.get("chains", [])
    lines = [
        "# Char VAE Context Chain Comparison",
        "",
        f"- schema: {summary.get('schema')}",
        f"- sort_by: {summary.get('sort_by')}",
        f"- recursive: {_fmt(summary.get('recursive'))}",
        f"- input_count: {_fmt(summary.get('input_count'))}",
        f"- discovered_chain_count: {_fmt(summary.get('discovered_chain_count'))}",
        f"- chain_count: {_fmt(_value(aggregate, 'chain_count'))}",
        f"- attempted_follow_ups: {_fmt(_value(aggregate, 'attempted_follow_ups'))}",
        f"- gate_failed_count: {_fmt(_value(aggregate, 'gate_failed_count'))}",
        f"- nonzero_exit_count: {_fmt(_value(aggregate, 'nonzero_exit_count'))}",
        f"- allowed_gate_stop_count: {_fmt(_value(aggregate, 'allowed_gate_stop_count'))}",
        f"- dry_run_count: {_fmt(_value(aggregate, 'dry_run_count'))}",
        f"- seed_source_counts: {_fmt_counts(_value(aggregate, 'seed_source_counts'))}",
        f"- command_source_counts: {_fmt_counts(_value(aggregate, 'command_source_counts'))}",
        "- configured_seed_group_status_counts: "
        f"{_fmt_counts(_value(aggregate, 'configured_seed_group_status_counts'))}",
        f"- stopped_reason_counts: {_fmt_counts(_value(aggregate, 'stopped_reason_counts'))}",
        "",
        "## Selection",
        "",
        f"- accepted_candidate_count: {_fmt(_value(selection, 'accepted_candidate_count'))}",
        f"- best_candidate_count: {_fmt(_value(selection, 'best_candidate_count'))}",
        f"- accepted_champion: {_fmt_leader(_value(selection, 'accepted_champion'))}",
        f"- best_champion: {_fmt_leader(_value(selection, 'best_champion'))}",
        f"- accepted_matches_best: {_fmt(_value(selection, 'accepted_matches_best'))}",
        f"- best_requires_review: {_fmt(_value(selection, 'best_requires_review'))}",
        f"- accepted_vs_best_nll_gap: {_fmt(_value(selection, 'accepted_vs_best_nll_gap'))}",
        f"- recommendation: {_fmt_recommendation(recommendation)}",
        f"- follow_up_from_summary_path: {_fmt(_value(recommendation, 'follow_up_from_summary_path'))}",
        f"- review_summary_path: {_fmt(_value(recommendation, 'review_summary_path'))}",
        f"- follow_up_command: {_fmt_command(_value(recommendation, 'follow_up_command'))}",
        f"- review_command: {_fmt_command(_value(recommendation, 'review_command'))}",
        f"- command_scripts_dir: {_fmt(_value(command_scripts, 'directory'))}",
        f"- next_command_script: {_fmt(_value(command_scripts, 'next_path'))}",
        f"- next_command_kind: {_fmt(_value(command_scripts, 'next_kind'))}",
        f"- follow_up_command_script: {_fmt(_value(command_scripts, 'follow_up_path'))}",
        f"- review_command_script: {_fmt(_value(command_scripts, 'review_path'))}",
        f"- command_execution_cwd: {_fmt(_value(command_scripts, 'execution_cwd'))}",
        f"- command_manifest_path: {_fmt(_value(command_scripts, 'manifest_path'))}",
        f"- command_readme_path: {_fmt(_value(command_scripts, 'readme_path'))}",
        f"- command_inspection: {_fmt(_value(command_scripts, 'inspection_command'))}",
        f"- command_inspection_json_path: {_fmt(_value(command_scripts, 'inspection_json_path'))}",
        f"- command_inspection_markdown_path: {_fmt(_value(command_scripts, 'inspection_markdown_path'))}",
        f"- command_inspection_generated: {_fmt(_value(command_scripts, 'inspection_generated'))}",
        f"- command_inspection_bundle_ready: {_fmt(_value(command_scripts, 'inspection_bundle_ready'))}",
        f"- command_inspection_strict_ready: {_fmt(_value(command_scripts, 'inspection_strict_ready'))}",
        f"- command_inspection_missing_required: {_fmt_list(_value(command_scripts, 'inspection_missing_required'))}",
        f"- command_inspection_missing_optional: {_fmt_list(_value(command_scripts, 'inspection_missing_optional'))}",
        f"- command_inspection_runner_wrapper_ok: {_fmt(_value(command_scripts, 'inspection_runner_wrapper_ok'))}",
        f"- command_inspection_runner_wrapper_executes_runner_command: {_fmt(_value(command_scripts, 'inspection_runner_wrapper_executes_runner_command'))}",
        f"- command_inspection_runner_wrapper_forwards_arguments: {_fmt(_value(command_scripts, 'inspection_runner_wrapper_forwards_arguments'))}",
        f"- command_inspection_history_next_action_runner_ok: {_fmt(_value(command_scripts, 'inspection_history_next_action_runner_ok'))}",
        f"- command_inspection_history_next_action_runner_executes_command: {_fmt(_value(command_scripts, 'inspection_history_next_action_runner_executes_command'))}",
        f"- command_inspection_history_next_action_runner_forwards_arguments: {_fmt(_value(command_scripts, 'inspection_history_next_action_runner_forwards_arguments'))}",
        f"- command_inspection_history_loop_runner_ok: {_fmt(_value(command_scripts, 'inspection_history_loop_runner_ok'))}",
        f"- command_inspection_history_loop_runner_executes_command: {_fmt(_value(command_scripts, 'inspection_history_loop_runner_executes_command'))}",
        f"- command_inspection_history_loop_runner_forwards_arguments: {_fmt(_value(command_scripts, 'inspection_history_loop_runner_forwards_arguments'))}",
        f"- command_runner: {_fmt(_value(command_scripts, 'runner_command'))}",
        f"- command_execution_next: {_fmt(_value(command_scripts, 'execution_next_command'))}",
        f"- command_history_next_action: {_fmt(_value(command_scripts, 'history_next_action_command'))}",
        f"- command_history_next_action_runner: {_fmt(_value(command_scripts, 'history_next_action_runner_path'))}",
        f"- command_history_loop: {_fmt(_value(command_scripts, 'history_loop_command'))}",
        f"- command_history_loop_runner: {_fmt(_value(command_scripts, 'history_loop_runner_path'))}",
        f"- command_runner_script: {_fmt(_value(command_scripts, 'runner_path'))}",
        f"- command_run_json_path: {_fmt(_value(command_scripts, 'run_json_path'))}",
        f"- command_run_markdown_path: {_fmt(_value(command_scripts, 'run_markdown_path'))}",
        f"- command_run_history_jsonl_path: {_fmt(_value(command_scripts, 'run_history_jsonl_path'))}",
        f"- command_run_history_markdown_path: {_fmt(_value(command_scripts, 'run_history_markdown_path'))}",
        f"- command_run_history_summary_path: {_fmt(_value(command_scripts, 'run_history_summary_path'))}",
        f"- command_run_loop_json_path: {_fmt(_value(command_scripts, 'run_loop_json_path'))}",
        f"- command_run_loop_markdown_path: {_fmt(_value(command_scripts, 'run_loop_markdown_path'))}",
        f"- command_history_report_only: {_fmt(_value(command_scripts, 'history_report_command'))}",
        "",
        "## Chains",
        "",
        "| source | preset | stopped | planned | attempted | accepted | accepted_nll | best | best_nll | delta_vs_raw | runner_up | margin | tie | seed_sources | command_sources | group_statuses | gates | exits | extra | unused |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in chains if isinstance(chains, list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {source} | {preset} | {stopped} | {planned} | {attempted} | "
            "{accepted} | {accepted_nll} | {best} | {best_nll} | "
            "{delta} | {runner_up} | {margin} | {tie} | {seed_sources} | "
            "{command_sources} | {group_statuses} | {gates} | {exits} | "
            "{extra} | {unused} |".format(
                source=_fmt(row.get("source")),
                preset=_fmt(row.get("preset")),
                stopped=_fmt(row.get("stopped_reason")),
                planned=_fmt(row.get("planned_follow_ups")),
                attempted=_fmt(row.get("attempted_follow_ups")),
                accepted=_fmt(row.get("accepted_config")),
                accepted_nll=_fmt(row.get("accepted_mean_best_nll")),
                best=_fmt(row.get("best_config")),
                best_nll=_fmt(row.get("best_mean_best_nll")),
                delta=_fmt(row.get("best_delta_vs_raw")),
                runner_up=_fmt(row.get("runner_up_feature")),
                margin=_fmt(row.get("margin_to_runner_up")),
                tie=_fmt(row.get("runner_up_within_uncertainty")),
                seed_sources=_fmt_counts(row.get("seed_source_counts")),
                command_sources=_fmt_counts(row.get("command_source_counts")),
                group_statuses=_fmt_counts(
                    row.get("configured_seed_group_status_counts")
                ),
                gates=_fmt(row.get("gate_failed_count")),
                exits=_fmt(row.get("nonzero_exit_count")),
                extra=_fmt_groups(row.get("extra_explicit_seed_groups")),
                unused=_fmt_groups(row.get("unused_explicit_seed_groups")),
            )
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chains",
        nargs="+",
        type=Path,
        help="chain.json files, chain run directories, or roots with --recursive",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="discover all chain.json files below directory arguments",
    )
    parser.add_argument(
        "--sort-by",
        choices=["input", "accepted", "best", "attempted"],
        default="input",
        help="row ordering for the rendered comparison",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument(
        "--command-out-dir",
        type=Path,
        default=None,
        help=(
            "write recommended_next plus follow-up/review scripts, README, and manifest there"
        ),
    )
    parser.add_argument(
        "--write-command-inspection",
        action="store_true",
        help=(
            "with --command-out-dir, write inspection.json/inspection.md and fail "
            "when the generated command bundle is not strict-ready"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.write_command_inspection and args.command_out_dir is None:
        parser.error("--write-command-inspection requires --command-out-dir")
    summary = summarize_chains(
        args.chains,
        sort_by=args.sort_by,
        recursive=bool(args.recursive),
    )
    comparison_json_path = args.json_out
    comparison_markdown_path = args.markdown_out
    if args.command_out_dir is not None:
        if comparison_json_path is None:
            comparison_json_path = args.command_out_dir / "comparison.json"
        if comparison_markdown_path is None:
            comparison_markdown_path = args.command_out_dir / "comparison.md"
        summary["command_scripts"] = _write_recommended_command_scripts(
            summary,
            args.command_out_dir,
            comparison_json_path=comparison_json_path,
            comparison_markdown_path=comparison_markdown_path,
        )
    markdown = _render_markdown(summary)
    if comparison_json_path is not None:
        comparison_json_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    if comparison_markdown_path is not None:
        comparison_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_markdown_path.write_text(markdown)
    if args.write_command_inspection:
        try:
            inspection = _write_command_bundle_inspection(args.command_out_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: failed to inspect command bundle: {exc}", file=sys.stderr)
            return 1
        command_scripts = summary.get("command_scripts")
        command_scripts = command_scripts if isinstance(command_scripts, dict) else {}
        runner_wrapper_status = inspection.get("runner_wrapper_status")
        runner_wrapper_status = (
            runner_wrapper_status if isinstance(runner_wrapper_status, dict) else {}
        )
        history_next_action_runner_status = inspection.get(
            "history_next_action_runner_status"
        )
        history_next_action_runner_status = (
            history_next_action_runner_status
            if isinstance(history_next_action_runner_status, dict)
            else {}
        )
        history_loop_runner_status = inspection.get("history_loop_runner_status")
        history_loop_runner_status = (
            history_loop_runner_status
            if isinstance(history_loop_runner_status, dict)
            else {}
        )
        command_scripts.update(
            {
                "inspection_generated": True,
                "inspection_bundle_ready": bool(inspection.get("bundle_ready")),
                "inspection_strict_ready": bool(inspection.get("strict_ready")),
                "inspection_missing_required": inspection.get("missing_required") or [],
                "inspection_missing_optional": inspection.get("missing_optional") or [],
                "inspection_runner_wrapper_status": runner_wrapper_status,
                "inspection_runner_wrapper_ok": runner_wrapper_status.get("ok"),
                "inspection_runner_wrapper_executes_runner_command": (
                    runner_wrapper_status.get("executes_runner_command")
                ),
                "inspection_runner_wrapper_forwards_arguments": (
                    runner_wrapper_status.get("forwards_arguments")
                ),
                "inspection_history_next_action_runner_status": (
                    history_next_action_runner_status
                ),
                "inspection_history_next_action_runner_ok": (
                    history_next_action_runner_status.get("ok")
                ),
                "inspection_history_next_action_runner_executes_command": (
                    history_next_action_runner_status.get("executes_runner_command")
                ),
                "inspection_history_next_action_runner_forwards_arguments": (
                    history_next_action_runner_status.get("forwards_arguments")
                ),
                "inspection_history_loop_runner_status": (
                    history_loop_runner_status
                ),
                "inspection_history_loop_runner_ok": (
                    history_loop_runner_status.get("ok")
                ),
                "inspection_history_loop_runner_executes_command": (
                    history_loop_runner_status.get("executes_runner_command")
                ),
                "inspection_history_loop_runner_forwards_arguments": (
                    history_loop_runner_status.get("forwards_arguments")
                ),
            }
        )
        summary["command_scripts"] = command_scripts
        summary["command_inspection"] = inspection
        manifest_path = _path_value(command_scripts.get("manifest_path"))
        readme_path = _path_value(command_scripts.get("readme_path"))
        if manifest_path is not None and readme_path is not None:
            _write_command_bundle_metadata(
                summary,
                command_scripts,
                manifest_path=manifest_path,
                readme_path=readme_path,
            )
        markdown = _render_markdown(summary)
        if comparison_json_path is not None:
            comparison_json_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
        if comparison_markdown_path is not None:
            comparison_markdown_path.write_text(markdown)
        if not inspection.get("strict_ready"):
            return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize multiple char VAE context chain.json artifacts."""

from __future__ import annotations

import argparse
import json
import math
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


def _render_markdown(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    selection = summary.get("selection", {})
    recommendation = summary.get("recommendation", {})
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = summarize_chains(
        args.chains,
        sort_by=args.sort_by,
        recursive=bool(args.recursive),
    )
    markdown = _render_markdown(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

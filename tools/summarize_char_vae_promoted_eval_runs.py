#!/usr/bin/env python3
"""Summarize promoted char VAE recipe eval-reload reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.promoted_eval_summaries.v1"
REPORT_SCHEMA = "st.llm_char_vae_context.promoted_eval_summary.v1"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _report_path(path: Path) -> Path:
    return path / "promoted_recipe_eval_run.json" if path.is_dir() else path


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _finite_floats(values: list[Any]) -> list[float]:
    floats: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            floats.append(number)
    return floats


def _mean(values: list[Any]) -> float | None:
    floats = _finite_floats(values)
    return sum(floats) / len(floats) if floats else None


def _winner_counts(seed_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seed_results:
        feature = seed.get("best_feature")
        if feature is None:
            continue
        key = str(feature)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_winner(counts: dict[str, int]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _winner_rates(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def _ranking_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    deltas = summary.get("deltas")
    deltas = deltas if isinstance(deltas, dict) else {}
    rows: list[dict[str, Any]] = []
    for item in summary.get("ranking") or []:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature")
        rows.append(
            {
                "feature": feature,
                "best_mean_nll": item.get("best_mean_nll"),
                "best_accuracy": item.get("best_accuracy"),
                "best_nll_vs_raw": deltas.get(f"{feature}_best_nll_vs_raw"),
            }
        )
    return rows


def _seed_result(item: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    run_dir = item.get("run_dir")
    summary_path = Path(str(run_dir)) / "summary.json" if run_dir else None
    if summary_path is not None and not summary_path.is_absolute():
        summary_path = cwd / summary_path
    payload = _read_json(summary_path) if summary_path is not None else None
    ranking = _ranking_rows(payload or {})
    winner = ranking[0] if ranking else {}
    runner_up = ranking[1] if len(ranking) > 1 else {}
    margin = None
    try:
        if winner and runner_up:
            margin = float(runner_up["best_mean_nll"]) - float(winner["best_mean_nll"])
    except (TypeError, ValueError):
        margin = None
    return {
        "seed": item.get("seed"),
        "run_dir": run_dir,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "summary_exists": payload is not None,
        "returncode": item.get("returncode"),
        "executed": item.get("executed"),
        "best_feature": winner.get("feature"),
        "best_mean_nll": winner.get("best_mean_nll"),
        "best_accuracy": winner.get("best_accuracy"),
        "best_nll_vs_raw": winner.get("best_nll_vs_raw"),
        "runner_up_feature": runner_up.get("feature"),
        "margin_to_runner_up": margin,
        "ranking": ranking,
    }


def _target_delta(seed: dict[str, Any], target_feature: str | None) -> Any:
    if target_feature is None:
        return None
    for row in seed.get("ranking") or []:
        if isinstance(row, dict) and row.get("feature") == target_feature:
            return row.get("best_nll_vs_raw")
    return None


def _recommendation(
    *,
    target_feature: str | None,
    successful_count: int,
    missing_summary_count: int,
    failed_count: int,
    target_win_count: int,
    mean_target_delta_vs_raw: float | None,
) -> str:
    if failed_count:
        return "review_eval_failures"
    if missing_summary_count:
        return "complete_eval_summaries"
    if successful_count == 0:
        return "run_eval_reload"
    if target_feature and target_win_count == successful_count:
        if mean_target_delta_vs_raw is not None and mean_target_delta_vs_raw < 0:
            return "promote_reload_evidence"
        return "review_target_delta"
    return "review_reload_winners"


def summarize_report(path: Path) -> dict[str, Any]:
    report_path = _report_path(path)
    payload = _read_json(report_path)
    if payload is None:
        raise ValueError(f"{report_path} did not contain a JSON object")
    cwd_raw = payload.get("cwd")
    cwd = Path(str(cwd_raw)) if cwd_raw else report_path.parent
    if not cwd.is_absolute():
        cwd = report_path.parent / cwd
    results = [item for item in payload.get("results") or [] if isinstance(item, dict)]
    seed_results = [_seed_result(item, cwd=cwd) for item in results]
    successful = [
        seed
        for seed in seed_results
        if seed.get("returncode") == 0 and seed.get("summary_exists")
    ]
    failed_count = sum(
        1
        for seed in seed_results
        if seed.get("returncode") not in (None, 0)
    )
    missing_summary_count = sum(1 for seed in seed_results if not seed.get("summary_exists"))
    target_feature = payload.get("feature")
    target_feature = str(target_feature) if target_feature else None
    counts = _winner_counts(successful)
    top_feature, top_count = _top_winner(counts)
    target_win_count = counts.get(target_feature, 0) if target_feature else 0
    mean_target_delta_vs_raw = _mean(
        [_target_delta(seed, target_feature) for seed in successful]
    )
    successful_count = len(successful)
    return {
        "schema": REPORT_SCHEMA,
        "report_path": str(report_path),
        "source_summary_path": payload.get("summary_path"),
        "feature": target_feature,
        "feature_family": payload.get("feature_family"),
        "execute": payload.get("execute"),
        "ready_only": payload.get("ready_only"),
        "complete_only": payload.get("complete_only"),
        "selected_count": payload.get("selected_count"),
        "available_count": payload.get("available_count"),
        "returncode": payload.get("returncode"),
        "seed_results": seed_results,
        "successful_eval_count": successful_count,
        "failed_eval_count": failed_count,
        "missing_summary_count": missing_summary_count,
        "winner_counts": counts,
        "winner_rates": _winner_rates(counts, successful_count),
        "top_winner_feature": top_feature,
        "top_winner_count": top_count,
        "top_winner_rate": top_count / successful_count if successful_count else None,
        "target_feature_win_count": target_win_count,
        "target_feature_win_rate": (
            target_win_count / successful_count if successful_count else None
        ),
        "mean_best_nll": _mean([seed.get("best_mean_nll") for seed in successful]),
        "mean_best_delta_vs_raw": _mean(
            [seed.get("best_nll_vs_raw") for seed in successful]
        ),
        "mean_target_delta_vs_raw": mean_target_delta_vs_raw,
        "mean_margin_to_runner_up": _mean(
            [seed.get("margin_to_runner_up") for seed in successful]
        ),
        "recommendation": _recommendation(
            target_feature=target_feature,
            successful_count=successful_count,
            missing_summary_count=missing_summary_count,
            failed_count=failed_count,
            target_win_count=target_win_count,
            mean_target_delta_vs_raw=mean_target_delta_vs_raw,
        ),
    }


def summarize_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [summarize_report(path) for path in paths]
    return {
        "schema": SCHEMA,
        "report_count": len(reports),
        "reports": reports,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Promoted Char VAE Eval Reload Summary",
        "",
        f"- schema: {payload.get('schema')}",
        f"- report_count: {payload.get('report_count')}",
        "",
    ]
    for report in payload.get("reports", []):
        if not isinstance(report, dict):
            continue
        lines.extend(
            [
                f"## {report.get('report_path')}",
                "",
                f"- feature: {report.get('feature') or '-'}",
                f"- feature_family: {report.get('feature_family') or '-'}",
                f"- execute: {_fmt(report.get('execute'))}",
                f"- ready_only: {_fmt(report.get('ready_only'))}",
                f"- complete_only: {_fmt(report.get('complete_only'))}",
                f"- selected/available: {report.get('selected_count')}/{report.get('available_count')}",
                f"- successful_eval_count: {report.get('successful_eval_count')}",
                f"- failed_eval_count: {report.get('failed_eval_count')}",
                f"- missing_summary_count: {report.get('missing_summary_count')}",
                f"- winner_counts: {_fmt_counts(report.get('winner_counts'))}",
                f"- target_feature_win_rate: {_fmt(report.get('target_feature_win_rate'))}",
                f"- mean_target_delta_vs_raw: {_fmt(report.get('mean_target_delta_vs_raw'))}",
                f"- mean_margin_to_runner_up: {_fmt(report.get('mean_margin_to_runner_up'))}",
                f"- recommendation: {report.get('recommendation') or '-'}",
                "",
                "| seed | returncode | summary | best_feature | best_nll | delta_vs_raw | runner_up | margin |",
                "| ---: | ---: | --- | --- | ---: | ---: | --- | ---: |",
            ]
        )
        for seed in report.get("seed_results", []):
            if not isinstance(seed, dict):
                continue
            lines.append(
                "| {seed} | {returncode} | {summary} | {feature} | {nll} | {delta} | {runner} | {margin} |".format(
                    seed=_fmt(seed.get("seed")),
                    returncode=_fmt(seed.get("returncode")),
                    summary="yes" if seed.get("summary_exists") else "no",
                    feature=seed.get("best_feature") or "-",
                    nll=_fmt(seed.get("best_mean_nll")),
                    delta=_fmt(seed.get("best_nll_vs_raw")),
                    runner=seed.get("runner_up_feature") or "-",
                    margin=_fmt(seed.get("margin_to_runner_up")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = summarize_reports(args.reports)
    markdown = markdown_report(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

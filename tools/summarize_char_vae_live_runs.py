#!/usr/bin/env python3
"""Summarize in-progress char VAE context run directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.live_run_summaries.v1"
RUN_SCHEMA = "st.llm_char_vae_context.live_run_summary.v1"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _ranking_rows(seed_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deltas = seed_summary.get("deltas")
    deltas = deltas if isinstance(deltas, dict) else {}
    for item in seed_summary.get("ranking") or []:
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


def _seed_result(path: Path) -> dict[str, Any] | None:
    payload = _read_json(path)
    if payload is None:
        return None
    ranking = _ranking_rows(payload)
    winner = ranking[0] if ranking else {}
    runner_up = ranking[1] if len(ranking) > 1 else {}
    margin = None
    try:
        if winner and runner_up:
            margin = float(runner_up["best_mean_nll"]) - float(winner["best_mean_nll"])
    except (TypeError, ValueError):
        margin = None
    return {
        "summary_path": str(path),
        "run_dir": str(path.parent),
        "seed": payload.get("seed") or _seed_from_path(path),
        "best_feature": winner.get("feature"),
        "best_mean_nll": winner.get("best_mean_nll"),
        "best_accuracy": winner.get("best_accuracy"),
        "best_nll_vs_raw": winner.get("best_nll_vs_raw"),
        "runner_up_feature": runner_up.get("feature"),
        "margin_to_runner_up": margin,
        "ranking": ranking,
    }


def _log_status(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "log_exists": False,
            "log_lines": 0,
            "current_seed": None,
            "completed_best_features": 0,
            "latest_progress": None,
            "best_feature_lines": [],
            "status_lines": [],
        }
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_seed = None
    for line in reversed(lines):
        if not line.startswith("sweep_normalize="):
            continue
        match = re.search(r"sweep_seed=(\d+)", line)
        if match:
            current_seed = int(match.group(1))
            break
    latest_progress = None
    for line in reversed(lines):
        if re.match(r"^(raw|latent|raw_latent|reconstruction_latent)\[\d+\]", line):
            latest_progress = line
            break
    best_feature_lines = [line for line in lines if line.startswith("best_feature=")]
    status_lines = [line for line in lines if line.startswith("sweep_status=")]
    return {
        "log_exists": True,
        "log_lines": len(lines),
        "current_seed": current_seed,
        "completed_best_features": len(best_feature_lines),
        "latest_progress": latest_progress,
        "best_feature_lines": best_feature_lines,
        "status_lines": status_lines,
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    seed_results = [
        result
        for result in (
            _seed_result(path)
            for path in sorted(run_dir.glob("seed_*/summary.json"))
        )
        if result is not None
    ]
    log_status = _log_status(run_dir / "run.log")
    final_best = summary.get("best_config") if isinstance(summary, dict) else None
    final_best = final_best if isinstance(final_best, dict) else {}
    follow_up = summary.get("follow_up_result") if isinstance(summary, dict) else None
    follow_up = follow_up if isinstance(follow_up, dict) else {}
    guidance = summary.get("follow_up_guidance") if isinstance(summary, dict) else None
    guidance = guidance if isinstance(guidance, dict) else {}
    return {
        "schema": RUN_SCHEMA,
        "run_dir": str(run_dir),
        "summary_exists": summary is not None,
        "summary_path": str(summary_path),
        "status": summary.get("status") if isinstance(summary, dict) else None,
        "best_feature": summary.get("best_feature") if isinstance(summary, dict) else None,
        "mean_best_nll": final_best.get("mean_best_nll"),
        "mean_best_nll_delta_vs_raw": final_best.get("mean_best_nll_delta_vs_raw"),
        "follow_up_verdict": follow_up.get("verdict"),
        "guidance_action": guidance.get("action"),
        "log": log_status,
        "seed_results": seed_results,
    }


def summarize_runs(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [summarize_run(path) for path in run_dirs]
    return {
        "schema": SCHEMA,
        "run_count": len(runs),
        "runs": runs,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = ["# Char VAE Live Runs", ""]
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        log = run.get("log") if isinstance(run.get("log"), dict) else {}
        lines.extend(
            [
                f"## {run.get('run_dir')}",
                "",
                f"- summary_exists: {run.get('summary_exists')}",
                f"- status: {run.get('status') or '-'}",
                f"- best_feature: {run.get('best_feature') or '-'}",
                f"- mean_best_nll: {_fmt(run.get('mean_best_nll'))}",
                f"- mean_delta_vs_raw: {_fmt(run.get('mean_best_nll_delta_vs_raw'))}",
                f"- follow_up_verdict: {run.get('follow_up_verdict') or '-'}",
                f"- guidance_action: {run.get('guidance_action') or '-'}",
                f"- current_seed: {log.get('current_seed') or '-'}",
                f"- completed_best_features: {log.get('completed_best_features') or 0}",
                f"- latest_progress: `{log.get('latest_progress') or '-'}`",
                "",
                "| seed | best_feature | best_nll | best_acc | delta_vs_raw | runner_up | margin |",
                "| --- | --- | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for seed in run.get("seed_results", []):
            if not isinstance(seed, dict):
                continue
            lines.append(
                "| {seed} | {feature} | {nll} | {acc} | {delta} | {runner} | {margin} |".format(
                    seed=seed.get("seed") or "-",
                    feature=seed.get("best_feature") or "-",
                    nll=_fmt(seed.get("best_mean_nll")),
                    acc=_fmt(seed.get("best_accuracy")),
                    delta=_fmt(seed.get("best_nll_vs_raw")),
                    runner=seed.get("runner_up_feature") or "-",
                    margin=_fmt(seed.get("margin_to_runner_up")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = summarize_runs(args.run_dirs)
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

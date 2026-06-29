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
PROGRESS_RE = re.compile(
    r"^(?P<feature>raw|latent|raw_latent|reconstruction_latent)"
    r"\[(?P<step>init|\d+)\]"
    r"(?: train_loss=(?P<train_loss>-?\d+(?:\.\d+)?))?"
    r" val_nll=(?P<val_nll>-?\d+(?:\.\d+)?)"
    r" acc=(?P<acc>-?\d+(?:\.\d+)?)%"
)


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


def _fmt_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


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


def _winner_counts(seed_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seed_results:
        feature = seed.get("best_feature")
        if feature is None:
            continue
        key = str(feature)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _progress_record(line: str, *, seed: int | None) -> dict[str, Any] | None:
    match = PROGRESS_RE.match(line)
    if match is None:
        return None
    step_raw = match.group("step")
    step = None if step_raw == "init" else int(step_raw)
    train_loss_raw = match.group("train_loss")
    return {
        "feature": match.group("feature"),
        "seed": seed,
        "step": step,
        "step_label": step_raw,
        "train_loss": float(train_loss_raw) if train_loss_raw is not None else None,
        "val_nll": float(match.group("val_nll")),
        "accuracy_percent": float(match.group("acc")),
        "line": line,
    }


def _feature_progress(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for record in records:
        feature = str(record.get("feature"))
        if feature not in by_feature:
            by_feature[feature] = []
            order.append(feature)
        by_feature[feature].append(record)

    raw_best = None
    raw_records = by_feature.get("raw", [])
    if raw_records:
        raw_best = min(raw_records, key=lambda item: float(item["val_nll"]))

    rows: list[dict[str, Any]] = []
    for feature in order:
        items = by_feature[feature]
        latest = items[-1]
        best = min(items, key=lambda item: float(item["val_nll"]))
        delta_vs_raw = None
        if raw_best is not None:
            delta_vs_raw = float(best["val_nll"]) - float(raw_best["val_nll"])
        rows.append(
            {
                "feature": feature,
                "latest_step": latest.get("step"),
                "latest_step_label": latest.get("step_label"),
                "latest_val_nll": latest.get("val_nll"),
                "latest_accuracy_percent": latest.get("accuracy_percent"),
                "best_step": best.get("step"),
                "best_step_label": best.get("step_label"),
                "best_val_nll": best.get("val_nll"),
                "best_accuracy_percent": best.get("accuracy_percent"),
                "best_delta_vs_raw": delta_vs_raw,
            }
        )
    return rows


def _best_progress_feature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "best_so_far_feature": None,
            "best_so_far_val_nll": None,
            "best_so_far_delta_vs_raw": None,
        }
    best = min(rows, key=lambda item: float(item["best_val_nll"]))
    return {
        "best_so_far_feature": best.get("feature"),
        "best_so_far_val_nll": best.get("best_val_nll"),
        "best_so_far_delta_vs_raw": best.get("best_delta_vs_raw"),
    }


def _log_status(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "log_exists": False,
            "log_lines": 0,
            "current_seed": None,
            "current_feature": None,
            "current_epoch": None,
            "completed_best_features": 0,
            "latest_progress": None,
            "feature_progress": [],
            "best_so_far_feature": None,
            "best_so_far_val_nll": None,
            "best_so_far_delta_vs_raw": None,
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
    progress_records: list[dict[str, Any]] = []
    active_seed = None
    for line in lines:
        if line.startswith("sweep_normalize="):
            match = re.search(r"sweep_seed=(\d+)", line)
            active_seed = int(match.group(1)) if match else None
        record = _progress_record(line, seed=active_seed)
        if record is not None:
            progress_records.append(record)
    latest_record = progress_records[-1] if progress_records else {}
    latest_progress = latest_record.get("line")
    current_seed_records = [
        record
        for record in progress_records
        if current_seed is not None and record.get("seed") == current_seed
    ]
    feature_progress = _feature_progress(current_seed_records or progress_records)
    best_so_far = _best_progress_feature(feature_progress)
    best_feature_lines = [line for line in lines if line.startswith("best_feature=")]
    status_lines = [line for line in lines if line.startswith("sweep_status=")]
    return {
        "log_exists": True,
        "log_lines": len(lines),
        "current_seed": current_seed,
        "current_feature": latest_record.get("feature"),
        "current_epoch": latest_record.get("step"),
        "completed_best_features": len(best_feature_lines),
        "latest_progress": latest_progress,
        "feature_progress": feature_progress,
        **best_so_far,
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
        "completed_seed_count": len(seed_results),
        "winner_counts": _winner_counts(seed_results),
        "log": log_status,
        "seed_results": seed_results,
    }


def _totals(runs: list[dict[str, Any]]) -> dict[str, Any]:
    winner_counts: dict[str, int] = {}
    completed_seed_count = 0
    completed_run_count = 0
    for run in runs:
        if run.get("summary_exists"):
            completed_run_count += 1
        completed_seed_count += int(run.get("completed_seed_count") or 0)
        counts = run.get("winner_counts")
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            try:
                amount = int(value)
            except (TypeError, ValueError):
                continue
            winner_counts[str(key)] = winner_counts.get(str(key), 0) + amount
    return {
        "run_count": len(runs),
        "completed_run_count": completed_run_count,
        "in_progress_run_count": len(runs) - completed_run_count,
        "completed_seed_count": completed_seed_count,
        "winner_counts": winner_counts,
    }


def summarize_runs(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [summarize_run(path) for path in run_dirs]
    return {
        "schema": SCHEMA,
        "run_count": len(runs),
        "totals": _totals(runs),
        "runs": runs,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = ["# Char VAE Live Runs", ""]
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    lines.extend(
        [
            "## Overview",
            "",
            f"- run_count: {totals.get('run_count', payload.get('run_count', 0))}",
            f"- completed_run_count: {totals.get('completed_run_count', 0)}",
            f"- in_progress_run_count: {totals.get('in_progress_run_count', 0)}",
            f"- completed_seed_count: {totals.get('completed_seed_count', 0)}",
            f"- winner_counts: {_fmt_counts(totals.get('winner_counts'))}",
            "",
            "| run | summary | current_seed | current_feature | completed_seeds | winners | best_so_far | delta_vs_raw | latest_progress |",
            "| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        log = run.get("log") if isinstance(run.get("log"), dict) else {}
        lines.append(
            "| {run} | {summary} | {current} | {feature} | {completed} | {winners} | {best_feature}:{best_nll} | {delta} | `{progress}` |".format(
                run=run.get("run_dir") or "-",
                summary=run.get("summary_exists"),
                current=log.get("current_seed") or "-",
                feature=log.get("current_feature") or "-",
                completed=run.get("completed_seed_count") or 0,
                winners=_fmt_counts(run.get("winner_counts")),
                best_feature=log.get("best_so_far_feature") or "-",
                best_nll=_fmt(log.get("best_so_far_val_nll")),
                delta=_fmt(log.get("best_so_far_delta_vs_raw")),
                progress=log.get("latest_progress") or "-",
            )
        )
    lines.append("")
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
                f"- current_feature: {log.get('current_feature') or '-'}",
                f"- current_epoch: {_fmt(log.get('current_epoch'))}",
                f"- best_so_far: {log.get('best_so_far_feature') or '-'}@{_fmt(log.get('best_so_far_val_nll'))}",
                f"- best_so_far_delta_vs_raw: {_fmt(log.get('best_so_far_delta_vs_raw'))}",
                f"- completed_best_features: {log.get('completed_best_features') or 0}",
                f"- latest_progress: `{log.get('latest_progress') or '-'}`",
                "",
                "| feature | latest_epoch | latest_nll | best_epoch | best_nll | best_acc_pct | delta_vs_raw |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature in log.get("feature_progress", []):
            if not isinstance(feature, dict):
                continue
            lines.append(
                "| {feature} | {latest_epoch} | {latest_nll} | {best_epoch} | {best_nll} | {best_acc} | {delta} |".format(
                    feature=feature.get("feature") or "-",
                    latest_epoch=_fmt(feature.get("latest_step")),
                    latest_nll=_fmt(feature.get("latest_val_nll")),
                    best_epoch=_fmt(feature.get("best_step")),
                    best_nll=_fmt(feature.get("best_val_nll")),
                    best_acc=_fmt(feature.get("best_accuracy_percent"), digits=2),
                    delta=_fmt(feature.get("best_delta_vs_raw")),
                )
            )
        lines.extend(
            [
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

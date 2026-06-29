#!/usr/bin/env python3
"""Summarize in-progress char VAE context run directories."""

from __future__ import annotations

import argparse
import json
import math
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


def _int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    values: list[int] = []
    for value in raw:
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return values


def _str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if str(value)]


def _positive_int(raw: Any) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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


def _finite_floats(values: list[Any]) -> list[float]:
    floats: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        floats.append(number)
    return floats


def _mean(values: list[Any]) -> float | None:
    floats = _finite_floats(values)
    return sum(floats) / len(floats) if floats else None


def _winner_rates(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def _top_winner(counts: dict[str, int]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    feature, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return feature, count


def _completed_seed_evidence(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(seed_results)
    counts = _winner_counts(seed_results)
    top_feature, top_count = _top_winner(counts)
    latest = seed_results[-1] if seed_results else {}
    return {
        "schema": "st.llm_char_vae_context.completed_seed_evidence.v1",
        "completed_seed_count": completed,
        "winner_counts": counts,
        "winner_rates": _winner_rates(counts, completed),
        "top_winner_feature": top_feature,
        "top_winner_count": top_count,
        "top_winner_rate": top_count / completed if completed else None,
        "mean_best_nll": _mean([seed.get("best_mean_nll") for seed in seed_results]),
        "mean_best_accuracy": _mean(
            [seed.get("best_accuracy") for seed in seed_results]
        ),
        "mean_delta_vs_raw": _mean(
            [seed.get("best_nll_vs_raw") for seed in seed_results]
        ),
        "mean_margin_to_runner_up": _mean(
            [seed.get("margin_to_runner_up") for seed in seed_results]
        ),
        "latest_completed_seed": latest.get("seed"),
        "latest_completed_best_feature": latest.get("best_feature"),
        "latest_completed_best_nll": latest.get("best_mean_nll"),
        "latest_completed_delta_vs_raw": latest.get("best_nll_vs_raw"),
        "latest_completed_margin_to_runner_up": latest.get("margin_to_runner_up"),
    }


def _completed_feature_evidence(
    seed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_feature: dict[str, list[dict[str, Any]]] = {}
    for seed in seed_results:
        ranking = seed.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            continue
        winner = ranking[0] if isinstance(ranking[0], dict) else {}
        winner_nll = winner.get("best_mean_nll")
        seed_id = seed.get("seed")
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict):
                continue
            feature = item.get("feature")
            if feature is None:
                continue
            gap_to_winner = None
            try:
                if winner_nll is not None:
                    gap_to_winner = float(item["best_mean_nll"]) - float(winner_nll)
            except (TypeError, ValueError):
                gap_to_winner = None
            rows_by_feature.setdefault(str(feature), []).append(
                {
                    "seed": seed_id,
                    "rank": rank,
                    "best_mean_nll": item.get("best_mean_nll"),
                    "best_accuracy": item.get("best_accuracy"),
                    "best_nll_vs_raw": item.get("best_nll_vs_raw"),
                    "gap_to_winner": gap_to_winner,
                    "is_winner": rank == 1,
                }
            )

    completed = len(seed_results)
    evidence: list[dict[str, Any]] = []
    for feature, rows in rows_by_feature.items():
        win_count = sum(1 for row in rows if row.get("is_winner"))
        evidence.append(
            {
                "feature": feature,
                "seed_count": len(rows),
                "coverage_rate": len(rows) / completed if completed else None,
                "win_count": win_count,
                "win_rate": win_count / completed if completed else None,
                "mean_rank": _mean([row.get("rank") for row in rows]),
                "mean_best_nll": _mean([row.get("best_mean_nll") for row in rows]),
                "mean_best_accuracy": _mean(
                    [row.get("best_accuracy") for row in rows]
                ),
                "mean_delta_vs_raw": _mean(
                    [row.get("best_nll_vs_raw") for row in rows]
                ),
                "mean_gap_to_winner": _mean(
                    [row.get("gap_to_winner") for row in rows]
                ),
            }
        )
    return sorted(
        evidence,
        key=lambda row: (
            -(row.get("win_count") or 0),
            row.get("mean_rank") if row.get("mean_rank") is not None else float("inf"),
            row.get("feature") or "",
        ),
    )


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
            "best_so_far_runner_up_feature": None,
            "best_so_far_margin_to_runner_up": None,
        }
    ranked = sorted(rows, key=lambda item: float(item["best_val_nll"]))
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else {}
    margin = None
    if runner_up:
        margin = float(runner_up["best_val_nll"]) - float(best["best_val_nll"])
    return {
        "best_so_far_feature": best.get("feature"),
        "best_so_far_val_nll": best.get("best_val_nll"),
        "best_so_far_delta_vs_raw": best.get("best_delta_vs_raw"),
        "best_so_far_runner_up_feature": runner_up.get("feature"),
        "best_so_far_margin_to_runner_up": margin,
    }


def _active_seed_progress(
    feature_progress: list[dict[str, Any]],
    *,
    planned_features: list[str],
    expected_epoch_count: int | None,
) -> dict[str, Any]:
    if not planned_features:
        planned_features = [
            str(row.get("feature"))
            for row in feature_progress
            if row.get("feature") is not None
        ]
    by_feature = {str(row.get("feature")): row for row in feature_progress}
    expected_final_epoch = (
        expected_epoch_count - 1 if expected_epoch_count is not None else None
    )
    feature_fractions: list[float] = []
    completed_features: list[str] = []
    for feature in planned_features:
        row = by_feature.get(feature)
        latest_step = row.get("latest_step") if isinstance(row, dict) else None
        if expected_epoch_count is None:
            fraction = 1.0 if row is not None else 0.0
        elif latest_step is None:
            fraction = 0.0
        else:
            fraction = min(1.0, max(0.0, (int(latest_step) + 1) / expected_epoch_count))
        feature_fractions.append(fraction)
        if expected_final_epoch is not None and latest_step is not None:
            if int(latest_step) >= expected_final_epoch:
                completed_features.append(feature)
        elif expected_epoch_count is None and row is not None:
            completed_features.append(feature)
    progress_fraction = (
        sum(feature_fractions) / len(planned_features) if planned_features else None
    )
    return {
        "planned_features": planned_features,
        "planned_feature_count": len(planned_features),
        "expected_epoch_count": expected_epoch_count,
        "expected_final_epoch": expected_final_epoch,
        "active_seed_completed_features": completed_features,
        "active_seed_completed_feature_count": len(completed_features),
        "active_seed_progress_fraction": progress_fraction,
    }


def _active_feature_evidence(
    feature_progress: list[dict[str, Any]],
    *,
    planned_features: list[str],
    expected_epoch_count: int | None,
    current_feature: str | None,
) -> list[dict[str, Any]]:
    if not planned_features:
        planned_features = [
            str(row.get("feature"))
            for row in feature_progress
            if row.get("feature") is not None
        ]
    by_feature = {str(row.get("feature")): row for row in feature_progress}
    started_rows = [
        row
        for row in feature_progress
        if row.get("best_val_nll") is not None and row.get("feature") is not None
    ]
    ranked = sorted(started_rows, key=lambda row: float(row["best_val_nll"]))
    rank_by_feature = {
        str(row.get("feature")): index for index, row in enumerate(ranked, start=1)
    }
    active_best = ranked[0] if ranked else {}
    active_best_nll = active_best.get("best_val_nll")
    expected_final_epoch = (
        expected_epoch_count - 1 if expected_epoch_count is not None else None
    )
    rows: list[dict[str, Any]] = []
    for feature in planned_features:
        row = by_feature.get(feature)
        latest_step = row.get("latest_step") if isinstance(row, dict) else None
        if row is None:
            status = "not_started"
            progress_fraction = 0.0 if expected_epoch_count is not None else None
        elif expected_final_epoch is not None and latest_step is not None:
            status = (
                "completed" if int(latest_step) >= expected_final_epoch else "active"
            )
            progress_fraction = min(
                1.0,
                max(0.0, (int(latest_step) + 1) / expected_epoch_count),
            )
        elif expected_epoch_count is not None:
            status = "active"
            progress_fraction = 0.0
        else:
            status = "active"
            progress_fraction = None

        gap_to_active_best = None
        try:
            if row is not None and active_best_nll is not None:
                gap_to_active_best = float(row["best_val_nll"]) - float(active_best_nll)
        except (TypeError, ValueError):
            gap_to_active_best = None

        rows.append(
            {
                "feature": feature,
                "status": status,
                "is_current": current_feature == feature,
                "rank_so_far": rank_by_feature.get(feature),
                "progress_fraction": progress_fraction,
                "latest_step": latest_step if row is not None else None,
                "best_step": row.get("best_step") if isinstance(row, dict) else None,
                "best_val_nll": (
                    row.get("best_val_nll") if isinstance(row, dict) else None
                ),
                "best_delta_vs_raw": (
                    row.get("best_delta_vs_raw") if isinstance(row, dict) else None
                ),
                "gap_to_active_best": gap_to_active_best,
            }
        )
    return rows


def _remaining_active_features(
    rows: list[dict[str, Any]],
) -> list[str]:
    remaining: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = row.get("feature")
        if feature is None or row.get("status") == "completed":
            continue
        remaining.append(str(feature))
    return remaining


def _log_status(
    log_path: Path,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_metadata = run_metadata if isinstance(run_metadata, dict) else {}
    planned_features = _str_list(run_metadata.get("features"))
    expected_epoch_count = _positive_int(run_metadata.get("epochs"))
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
            "best_so_far_runner_up_feature": None,
            "best_so_far_margin_to_runner_up": None,
            **_active_seed_progress(
                [],
                planned_features=planned_features,
                expected_epoch_count=expected_epoch_count,
            ),
            "active_seed_remaining_features": planned_features,
            "active_seed_remaining_feature_count": len(planned_features),
            "active_seed_next_feature": (
                planned_features[0] if planned_features else None
            ),
            "active_seed_completed_fraction": 0.0 if planned_features else None,
            "active_feature_evidence": _active_feature_evidence(
                [],
                planned_features=planned_features,
                expected_epoch_count=expected_epoch_count,
                current_feature=None,
            ),
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
    active_progress = _active_seed_progress(
        feature_progress,
        planned_features=planned_features,
        expected_epoch_count=expected_epoch_count,
    )
    active_feature_evidence = _active_feature_evidence(
        feature_progress,
        planned_features=active_progress["planned_features"],
        expected_epoch_count=expected_epoch_count,
        current_feature=(
            str(latest_record.get("feature"))
            if latest_record.get("feature") is not None
            else None
        ),
    )
    remaining_features = _remaining_active_features(active_feature_evidence)
    active_feature_index = None
    current_feature = latest_record.get("feature")
    if current_feature is not None and active_progress["planned_features"]:
        try:
            active_feature_index = active_progress["planned_features"].index(
                str(current_feature)
            ) + 1
        except ValueError:
            active_feature_index = None
    best_feature_lines = [line for line in lines if line.startswith("best_feature=")]
    status_lines = [line for line in lines if line.startswith("sweep_status=")]
    return {
        "log_exists": True,
        "log_lines": len(lines),
        "current_seed": current_seed,
        "current_feature": current_feature,
        "active_feature_index": active_feature_index,
        "current_epoch": latest_record.get("step"),
        "completed_best_features": len(best_feature_lines),
        "latest_progress": latest_progress,
        "feature_progress": feature_progress,
        **best_so_far,
        **active_progress,
        "active_seed_remaining_features": remaining_features,
        "active_seed_remaining_feature_count": len(remaining_features),
        "active_seed_next_feature": (
            remaining_features[0] if remaining_features else None
        ),
        "active_seed_completed_fraction": (
            1.0 - (len(remaining_features) / len(active_progress["planned_features"]))
            if active_progress["planned_features"]
            else None
        ),
        "active_feature_evidence": active_feature_evidence,
        "best_feature_lines": best_feature_lines,
        "status_lines": status_lines,
    }


def _run_progress_record(
    *,
    run_metadata: dict[str, Any],
    seed_results: list[dict[str, Any]],
    log_status: dict[str, Any],
) -> dict[str, Any]:
    planned_seeds = _int_list(run_metadata.get("seeds"))
    completed_seed_count = len(seed_results)
    planned_seed_count = len(planned_seeds)
    completed_seed_ids = {
        int(seed["seed"])
        for seed in seed_results
        if seed.get("seed") is not None
    }
    remaining_seeds = [
        seed for seed in planned_seeds if int(seed) not in completed_seed_ids
    ]
    current_seed = log_status.get("current_seed")
    active_seed_index = None
    if current_seed is not None and planned_seeds:
        try:
            active_seed_index = planned_seeds.index(int(current_seed)) + 1
        except (TypeError, ValueError):
            active_seed_index = None
    active_seed_progress_fraction = log_status.get("active_seed_progress_fraction")
    overall_progress_fraction = None
    try:
        if planned_seed_count:
            active_fraction = float(active_seed_progress_fraction or 0.0)
            active_seed_is_incomplete = (
                current_seed is not None
                and int(current_seed) not in completed_seed_ids
            )
            active_progress = (
                min(1.0, max(0.0, active_fraction))
                if active_seed_is_incomplete
                else 0.0
            )
            overall_progress_fraction = min(
                1.0,
                max(
                    0.0,
                    (completed_seed_count + active_progress) / planned_seed_count,
                ),
            )
    except (TypeError, ValueError):
        overall_progress_fraction = None
    latest_completed_seed = seed_results[-1] if seed_results else {}
    return {
        "schema": "st.llm_char_vae_context.live_progress.v1",
        "planned_seeds": planned_seeds,
        "planned_seed_count": planned_seed_count,
        "remaining_seeds": remaining_seeds,
        "remaining_seed_count": len(remaining_seeds),
        "completed_seed_count": completed_seed_count,
        "completed_seed_fraction": (
            completed_seed_count / planned_seed_count if planned_seed_count else None
        ),
        "overall_progress_fraction": overall_progress_fraction,
        "remaining_overall_fraction": (
            max(0.0, 1.0 - overall_progress_fraction)
            if overall_progress_fraction is not None
            else None
        ),
        "current_seed": current_seed,
        "active_seed_index": active_seed_index,
        "active_seed_progress_fraction": active_seed_progress_fraction,
        "active_seed_completed_feature_count": log_status.get(
            "active_seed_completed_feature_count"
        ),
        "planned_feature_count": log_status.get("planned_feature_count"),
        "latest_completed_seed": latest_completed_seed.get("seed"),
        "latest_completed_best_feature": latest_completed_seed.get("best_feature"),
        "latest_completed_margin_to_runner_up": latest_completed_seed.get(
            "margin_to_runner_up"
        ),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    run_metadata = _read_json(run_dir / "run.json") or {}
    seed_results = [
        result
        for result in (
            _seed_result(path)
            for path in sorted(run_dir.glob("seed_*/summary.json"))
        )
        if result is not None
    ]
    log_status = _log_status(run_dir / "run.log", run_metadata=run_metadata)
    final_best = summary.get("best_config") if isinstance(summary, dict) else None
    final_best = final_best if isinstance(final_best, dict) else {}
    follow_up = summary.get("follow_up_result") if isinstance(summary, dict) else None
    follow_up = follow_up if isinstance(follow_up, dict) else {}
    guidance = summary.get("follow_up_guidance") if isinstance(summary, dict) else None
    guidance = guidance if isinstance(guidance, dict) else {}
    completed_seed_evidence = _completed_seed_evidence(seed_results)
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
        "completed_seed_evidence": completed_seed_evidence,
        "completed_feature_evidence": _completed_feature_evidence(seed_results),
        "progress": _run_progress_record(
            run_metadata=run_metadata,
            seed_results=seed_results,
            log_status=log_status,
        ),
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
            "| run | summary | seed_progress | overall_progress | current_seed | current_feature | feature_progress | winners | completed_mean_delta | completed_mean_margin | best_so_far | margin | delta_vs_raw | latest_progress |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        log = run.get("log") if isinstance(run.get("log"), dict) else {}
        progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
        evidence = (
            run.get("completed_seed_evidence")
            if isinstance(run.get("completed_seed_evidence"), dict)
            else {}
        )
        seed_progress = "{completed}/{planned}".format(
            completed=progress.get("completed_seed_count") or 0,
            planned=progress.get("planned_seed_count") or "-",
        )
        feature_progress = "{completed}/{planned}".format(
            completed=log.get("active_seed_completed_feature_count") or 0,
            planned=log.get("planned_feature_count") or "-",
        )
        lines.append(
            "| {run} | {summary} | {seed_progress} | {overall} | {current} | {feature} | {feature_progress} | {winners} | {completed_delta} | {completed_margin} | {best_feature}:{best_nll} | {margin} | {delta} | `{latest}` |".format(
                run=run.get("run_dir") or "-",
                summary=run.get("summary_exists"),
                seed_progress=seed_progress,
                overall=_fmt(progress.get("overall_progress_fraction")),
                current=log.get("current_seed") or "-",
                feature=log.get("current_feature") or "-",
                feature_progress=feature_progress,
                winners=_fmt_counts(run.get("winner_counts")),
                completed_delta=_fmt(evidence.get("mean_delta_vs_raw")),
                completed_margin=_fmt(evidence.get("mean_margin_to_runner_up")),
                best_feature=log.get("best_so_far_feature") or "-",
                best_nll=_fmt(log.get("best_so_far_val_nll")),
                margin=_fmt(log.get("best_so_far_margin_to_runner_up")),
                delta=_fmt(log.get("best_so_far_delta_vs_raw")),
                latest=log.get("latest_progress") or "-",
            )
        )
    lines.append("")
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        log = run.get("log") if isinstance(run.get("log"), dict) else {}
        progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
        evidence = (
            run.get("completed_seed_evidence")
            if isinstance(run.get("completed_seed_evidence"), dict)
            else {}
        )
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
                f"- seed_progress: {progress.get('completed_seed_count') or 0}/{progress.get('planned_seed_count') or '-'}",
                f"- overall_progress_fraction: {_fmt(progress.get('overall_progress_fraction'))}",
                f"- remaining_overall_fraction: {_fmt(progress.get('remaining_overall_fraction'))}",
                "- remaining_seeds: "
                f"{', '.join(str(seed) for seed in progress.get('remaining_seeds') or []) or '-'}",
                f"- remaining_seed_count: {_fmt(progress.get('remaining_seed_count'), digits=0)}",
                f"- completed_seed_leader: {evidence.get('top_winner_feature') or '-'} ({evidence.get('top_winner_count') or 0}/{evidence.get('completed_seed_count') or 0}, rate={_fmt(evidence.get('top_winner_rate'))})",
                f"- completed_seed_mean_best_nll: {_fmt(evidence.get('mean_best_nll'))}",
                f"- completed_seed_mean_delta_vs_raw: {_fmt(evidence.get('mean_delta_vs_raw'))}",
                f"- completed_seed_mean_margin_to_runner_up: {_fmt(evidence.get('mean_margin_to_runner_up'))}",
                f"- latest_completed_seed: {evidence.get('latest_completed_seed') or '-'}",
                f"- latest_completed_best_feature: {evidence.get('latest_completed_best_feature') or '-'}",
                f"- current_seed: {log.get('current_seed') or '-'}",
                f"- active_seed_index: {_fmt(progress.get('active_seed_index'), digits=0)}/{progress.get('planned_seed_count') or '-'}",
                f"- current_feature: {log.get('current_feature') or '-'}",
                f"- active_feature_index: {_fmt(log.get('active_feature_index'), digits=0)}/{log.get('planned_feature_count') or '-'}",
                f"- current_epoch: {_fmt(log.get('current_epoch'))}",
                f"- active_seed_progress_fraction: {_fmt(log.get('active_seed_progress_fraction'))}",
                "- active_seed_remaining_features: "
                f"{', '.join(log.get('active_seed_remaining_features') or []) or '-'}",
                f"- active_seed_remaining_feature_count: {_fmt(log.get('active_seed_remaining_feature_count'), digits=0)}",
                f"- active_seed_next_feature: {log.get('active_seed_next_feature') or '-'}",
                f"- active_seed_completed_fraction: {_fmt(log.get('active_seed_completed_fraction'))}",
                f"- best_so_far: {log.get('best_so_far_feature') or '-'}@{_fmt(log.get('best_so_far_val_nll'))}",
                f"- best_so_far_runner_up: {log.get('best_so_far_runner_up_feature') or '-'}",
                f"- best_so_far_margin_to_runner_up: {_fmt(log.get('best_so_far_margin_to_runner_up'))}",
                f"- best_so_far_delta_vs_raw: {_fmt(log.get('best_so_far_delta_vs_raw'))}",
                f"- completed_best_features: {log.get('completed_best_features') or 0}",
                f"- active_seed_completed_features: {', '.join(log.get('active_seed_completed_features') or []) or '-'}",
                f"- latest_progress: `{log.get('latest_progress') or '-'}`",
                "",
                "| completed_feature | seeds | wins | win_rate | mean_rank | mean_nll | mean_delta_vs_raw | mean_gap_to_winner |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature in run.get("completed_feature_evidence", []):
            if not isinstance(feature, dict):
                continue
            lines.append(
                "| {feature} | {seeds} | {wins} | {win_rate} | {rank} | {nll} | {delta} | {gap} |".format(
                    feature=feature.get("feature") or "-",
                    seeds=feature.get("seed_count") or 0,
                    wins=feature.get("win_count") or 0,
                    win_rate=_fmt(feature.get("win_rate")),
                    rank=_fmt(feature.get("mean_rank")),
                    nll=_fmt(feature.get("mean_best_nll")),
                    delta=_fmt(feature.get("mean_delta_vs_raw")),
                    gap=_fmt(feature.get("mean_gap_to_winner")),
                )
            )
        lines.extend(
            [
                "",
                "| active_feature | status | current | rank_so_far | progress | best_nll | delta_vs_raw | gap_to_active_best |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for feature in log.get("active_feature_evidence", []):
            if not isinstance(feature, dict):
                continue
            lines.append(
                "| {feature} | {status} | {current} | {rank} | {progress} | {nll} | {delta} | {gap} |".format(
                    feature=feature.get("feature") or "-",
                    status=feature.get("status") or "-",
                    current="yes" if feature.get("is_current") else "no",
                    rank=_fmt(feature.get("rank_so_far"), digits=0),
                    progress=_fmt(feature.get("progress_fraction")),
                    nll=_fmt(feature.get("best_val_nll")),
                    delta=_fmt(feature.get("best_delta_vs_raw")),
                    gap=_fmt(feature.get("gap_to_active_best")),
                )
            )
        lines.extend(
            [
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

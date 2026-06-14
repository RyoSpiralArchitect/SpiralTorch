#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

"""Compare SpiralTorch char-LM model-zoo run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def run_paths(raw: str) -> tuple[Path, Path, Path | None]:
    path = Path(raw)
    if path.is_dir():
        summary_path = path / "summary.json"
        run_path = path / "run.json"
        return path, summary_path, run_path if run_path.exists() else None
    return path.parent, path, None


def metric_value(metric: dict[str, Any] | None, field: str) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.2f}%"


def learnability_value(metric: dict[str, Any], field: str) -> float | None:
    learnability = metric.get("learnability")
    if not isinstance(learnability, dict):
        return None
    value = learnability.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def row_for(raw: str) -> tuple[dict[str, str], Path]:
    run_dir, summary_path, run_path = run_paths(raw)
    summary = read_json(summary_path)
    run = read_json(run_path) if run_path is not None else {}
    initial = summary.get("initial_validation")
    final = summary.get("final_validation")
    unigram = summary.get("unigram_validation")
    if not isinstance(initial, dict):
        initial = None
    if not isinstance(final, dict):
        final = None
    if not isinstance(unigram, dict):
        unigram = None

    label = run_dir.name or str(run_dir)
    arch = str(run.get("arch", "-"))
    init_nll = metric_value(initial, "mean_nll")
    final_nll = metric_value(final, "mean_nll")
    unigram_nll = metric_value(unigram, "mean_nll")
    delta_nll = summary.get("validation_nll_delta")
    final_vs_unigram = summary.get("final_vs_unigram_nll_delta")
    delta_acc = summary.get("validation_accuracy_delta")
    if not isinstance(delta_nll, (int, float)):
        delta_nll = None
    if not isinstance(final_vs_unigram, (int, float)):
        final_vs_unigram = None
    if not isinstance(delta_acc, (int, float)):
        delta_acc = None

    return (
        {
            "run": label,
            "arch": arch,
            "init_nll": fmt_float(init_nll),
            "final_nll": fmt_float(final_nll),
            "delta_nll": fmt_float(float(delta_nll) if delta_nll is not None else None),
            "unigram_nll": fmt_float(unigram_nll),
            "final_vs_unigram": fmt_float(
                float(final_vs_unigram) if final_vs_unigram is not None else None
            ),
            "final_logprob_lift": fmt_float(
                metric_value(final, "mean_target_logprob_lift")
            ),
            "final_rank_lift": fmt_float(
                metric_value(final, "mean_target_rank_lift"), digits=2
            ),
            "final_kl_unigram": fmt_float(metric_value(final, "mean_kl_to_unigram")),
            "final_top5_overlap": fmt_percent(
                metric_value(final, "mean_top5_overlap_with_unigram")
            ),
            "final_ppl": fmt_float(metric_value(final, "perplexity")),
            "final_acc": fmt_percent(metric_value(final, "accuracy")),
            "final_entropy": fmt_float(metric_value(final, "mean_entropy")),
            "final_rank": fmt_float(metric_value(final, "mean_target_rank"), digits=2),
            "delta_acc": fmt_percent(float(delta_acc) if delta_acc is not None else None),
            "best_epoch": str(summary.get("best_validation_epoch", "-")),
        },
        run_dir,
    )


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "init_nll",
        "final_nll",
        "delta_nll",
        "unigram_nll",
        "final_vs_unigram",
        "final_logprob_lift",
        "final_rank_lift",
        "final_kl_unigram",
        "final_top5_overlap",
        "final_ppl",
        "final_acc",
        "final_entropy",
        "final_rank",
        "delta_acc",
        "best_epoch",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def curve_rows_for(summary_row: dict[str, str], run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for metric in read_jsonl(run_dir / "metrics.jsonl"):
        validation = metric.get("validation")
        if not isinstance(validation, dict):
            validation = None
        rows.append(
            {
                "run": summary_row["run"],
                "arch": summary_row["arch"],
                "epoch": str(metric.get("epoch", "-")),
                "train_loss": fmt_float(
                    float(metric["average_loss"])
                    if isinstance(metric.get("average_loss"), (int, float))
                    else None
                ),
                "val_nll": fmt_float(metric_value(validation, "mean_nll")),
                "val_ppl": fmt_float(metric_value(validation, "perplexity")),
                "val_acc": fmt_percent(metric_value(validation, "accuracy")),
                "val_entropy": fmt_float(metric_value(validation, "mean_entropy")),
                "val_rank": fmt_float(metric_value(validation, "mean_target_rank"), digits=2),
                "logprob_lift": fmt_float(
                    metric_value(validation, "mean_target_logprob_lift")
                ),
                "rank_lift": fmt_float(
                    metric_value(validation, "mean_target_rank_lift"), digits=2
                ),
                "kl_unigram": fmt_float(metric_value(validation, "mean_kl_to_unigram")),
                "update_l2": fmt_float(learnability_value(metric, "total_update_l2"), digits=6),
                "update_ratio": fmt_float(
                    learnability_value(metric, "mean_update_to_value_l2"), digits=6
                ),
            }
        )
    return rows


def curve_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "epoch",
        "train_loss",
        "val_nll",
        "val_ppl",
        "val_acc",
        "val_entropy",
        "val_rank",
        "logprob_lift",
        "rank_lift",
        "kl_unigram",
        "update_l2",
        "update_ratio",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def parameter_rows_for(
    summary_row: dict[str, str], run_dir: Path, limit: int
) -> list[dict[str, str]]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        return []
    metric = metrics[-1]
    learnability = metric.get("learnability")
    if not isinstance(learnability, dict):
        return []
    parameters = learnability.get("parameters")
    if not isinstance(parameters, list):
        return []

    rows: list[dict[str, str]] = []
    for param in parameters:
        if not isinstance(param, dict):
            continue
        update_l2 = param.get("update_l2")
        update_ratio = param.get("update_to_value_l2")
        rows.append(
            {
                "run": summary_row["run"],
                "arch": summary_row["arch"],
                "epoch": str(metric.get("epoch", "-")),
                "parameter": str(param.get("name", "-")),
                "shape": f"{param.get('rows', '-')}x{param.get('cols', '-')}",
                "value_l2": fmt_float(
                    float(param["value_l2"])
                    if isinstance(param.get("value_l2"), (int, float))
                    else None,
                    digits=6,
                ),
                "update_l2": fmt_float(
                    float(update_l2) if isinstance(update_l2, (int, float)) else None,
                    digits=6,
                ),
                "update_ratio": fmt_float(
                    float(update_ratio) if isinstance(update_ratio, (int, float)) else None,
                    digits=6,
                ),
            }
        )
    rows.sort(
        key=lambda row: float(row["update_l2"]) if row["update_l2"] != "-" else -1.0,
        reverse=True,
    )
    return rows[: max(limit, 0)]


def parameter_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "epoch",
        "parameter",
        "shape",
        "value_l2",
        "update_l2",
        "update_ratio",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare summary.json artifacts from char-LM model-zoo runs."
    )
    parser.add_argument(
        "--curves",
        action="store_true",
        help="also print epoch-level validation curves from metrics.jsonl",
    )
    parser.add_argument(
        "--params",
        type=int,
        default=0,
        metavar="N",
        help="also print the top N parameter updates from the last metrics epoch",
    )
    parser.add_argument("runs", nargs="+", help="run directories or summary.json files")
    args = parser.parse_args()
    pairs = [row_for(raw) for raw in args.runs]
    print(markdown_table([row for row, _ in pairs]))
    if args.curves:
        curve_rows: list[dict[str, str]] = []
        for row, run_dir in pairs:
            curve_rows.extend(curve_rows_for(row, run_dir))
        if curve_rows:
            print()
            print(curve_table(curve_rows))
    if args.params > 0:
        parameter_rows: list[dict[str, str]] = []
        for row, run_dir in pairs:
            parameter_rows.extend(parameter_rows_for(row, run_dir, args.params))
        if parameter_rows:
            print()
            print(parameter_table(parameter_rows))


if __name__ == "__main__":
    main()

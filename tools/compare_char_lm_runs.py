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


def row_for(raw: str) -> dict[str, str]:
    run_dir, summary_path, run_path = run_paths(raw)
    summary = read_json(summary_path)
    run = read_json(run_path) if run_path is not None else {}
    initial = summary.get("initial_validation")
    final = summary.get("final_validation")
    if not isinstance(initial, dict):
        initial = None
    if not isinstance(final, dict):
        final = None

    label = run_dir.name or str(run_dir)
    arch = str(run.get("arch", "-"))
    init_nll = metric_value(initial, "mean_nll")
    final_nll = metric_value(final, "mean_nll")
    delta_nll = summary.get("validation_nll_delta")
    delta_acc = summary.get("validation_accuracy_delta")
    if not isinstance(delta_nll, (int, float)):
        delta_nll = None
    if not isinstance(delta_acc, (int, float)):
        delta_acc = None

    return {
        "run": label,
        "arch": arch,
        "init_nll": fmt_float(init_nll),
        "final_nll": fmt_float(final_nll),
        "delta_nll": fmt_float(float(delta_nll) if delta_nll is not None else None),
        "final_ppl": fmt_float(metric_value(final, "perplexity")),
        "final_acc": fmt_percent(metric_value(final, "accuracy")),
        "delta_acc": fmt_percent(float(delta_acc) if delta_acc is not None else None),
        "best_epoch": str(summary.get("best_validation_epoch", "-")),
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "arch",
        "init_nll",
        "final_nll",
        "delta_nll",
        "final_ppl",
        "final_acc",
        "delta_acc",
        "best_epoch",
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
    parser.add_argument("runs", nargs="+", help="run directories or summary.json files")
    args = parser.parse_args()
    print(markdown_table([row_for(raw) for raw in args.runs]))


if __name__ == "__main__":
    main()

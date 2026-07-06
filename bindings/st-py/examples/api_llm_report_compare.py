# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Compare multiple API LLM live provider matrix reports.

Pass one or more ``report.json`` files written by
``api_llm_live_provider_matrix.py``.  With no arguments, this script runs a
small offline demo so the report-comparison surface can be smoke-tested without
provider keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import spiraltorch as st


def _demo_report(
    path: Path,
    *,
    compact_score: float,
    expanded_score: float,
) -> None:
    report: dict[str, Any] = {
        "kind": "spiraltorch.api_llm_live_provider_matrix",
        "prompt_count": 12,
        "route_count": 2,
        "skipped": {},
        "client_errors": [],
        "comparison": {
            "winners": {
                "best_score": "openai-compact",
                "highest_quality": "openai-expanded",
                "highest_text_quality": "openai-expanded",
                "highest_efficiency": "openai-compact",
                "lowest_latency": "openai-compact",
                "lowest_total_tokens": "openai-compact",
            },
            "selection_profiles": {
                "balanced": {"label": "openai-compact", "score": compact_score},
                "quality": {"label": "openai-expanded", "score": expanded_score},
                "grounded": {
                    "label": "openai-expanded",
                    "score": expanded_score - 0.01,
                },
                "efficiency": {
                    "label": "openai-compact",
                    "score": compact_score - 0.04,
                },
                "latency": {
                    "label": "openai-compact",
                    "score": compact_score - 0.03,
                },
            },
            "runs": [
                {
                    "label": "openai-compact",
                    "count": 12,
                    "route_score": compact_score,
                    "quality_score": 0.92,
                    "text_quality_score": 0.68,
                    "efficiency_score": 0.58,
                    "completion_rate": 1.0,
                    "latency_ms_mean": 2500.0,
                    "total_tokens": 2400.0,
                    "empty_text_rate": 0.0,
                    "refusal_rate": 0.0,
                },
                {
                    "label": "openai-expanded",
                    "count": 12,
                    "route_score": expanded_score,
                    "quality_score": 0.91,
                    "text_quality_score": 0.74,
                    "efficiency_score": 0.55,
                    "completion_rate": 1.0,
                    "latency_ms_mean": 3300.0,
                    "total_tokens": 2700.0,
                    "empty_text_rate": 0.0,
                    "refusal_rate": 0.0,
                },
            ],
        },
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", help="report.json files to compare")
    parser.add_argument("--label", action="append", default=[], help="label for each report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reports:
        labels = args.label or None
        comparison = st.compare_api_llm_matrix_reports(args.reports, labels=labels)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "first-report.json"
        second = root / "second-report.json"
        _demo_report(first, compact_score=0.83, expanded_score=0.82)
        _demo_report(second, compact_score=0.84, expanded_score=0.81)
        comparison = st.compare_api_llm_matrix_reports(
            {"first": first, "second": second}
        )
        print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

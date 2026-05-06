#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Compare two Z-space experiment manifests and render an HTML report."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_spiraltorch():
    try:
        import spiraltorch as st
        return st
    except ModuleNotFoundError:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        importlib.invalidate_caches()
        sys.modules.pop("spiraltorch", None)
        import spiraltorch as st
        return st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_manifest", help="Baseline artifact manifest JSON")
    parser.add_argument("candidate_manifest", help="Candidate artifact manifest JSON")
    parser.add_argument("--html", default=None, help="Output comparison HTML path")
    parser.add_argument("--title", default=None, help="Comparison page title")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warn-stability-drop", type=float, default=0.03)
    parser.add_argument("--fail-stability-drop", type=float, default=0.10)
    parser.add_argument("--min-frame-ratio", type=float, default=0.80)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when comparison_status is fail",
    )
    parser.add_argument("--json", action="store_true", help="Print the comparison packet JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    st = _load_spiraltorch()
    kwargs = {
        "top_k": args.top_k,
        "title": args.title,
        "warn_stability_drop": args.warn_stability_drop,
        "fail_stability_drop": args.fail_stability_drop,
        "min_frame_ratio": args.min_frame_ratio,
    }
    comparison = st.compare_zspace_experiment_manifests(
        args.baseline_manifest,
        args.candidate_manifest,
        **kwargs,
    )
    html_path = st.write_zspace_experiment_comparison_html(
        args.baseline_manifest,
        args.candidate_manifest,
        args.html,
        **kwargs,
    )
    print(f"comparison_html={html_path}")
    print(f"comparison_status={comparison['status']}")
    print(f"comparison_stability_delta={comparison['summary']['stability_delta']:.3f}")
    if args.json:
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
    if args.fail_on_regression and comparison["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

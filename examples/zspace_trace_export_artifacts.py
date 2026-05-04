#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Render Z-space trace artifacts from an existing JSONL trace.

This helper writes:
- trace HTML viewer
- Atlas non-collapse comparison HTML
- artifact manifest JSON
"""

from __future__ import annotations

import argparse
import importlib
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
    parser.add_argument("trace_jsonl", help="Path to ZSpaceTrace JSONL")
    parser.add_argument("--trace-html", default=None, help="Output trace HTML path")
    parser.add_argument(
        "--atlas-html",
        default=None,
        help="Output Atlas non-collapse HTML path",
    )
    parser.add_argument("--manifest", default=None, help="Output manifest JSON path")
    parser.add_argument("--title", default="SpiralTorch Z-Space Trace")
    parser.add_argument("--district", default="Concourse")
    parser.add_argument("--event-type", default="ZSpaceTrace")
    parser.add_argument("--bound", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--planner-backend", default="auto")
    parser.add_argument("--plan-rows", type=int, default=None)
    parser.add_argument("--plan-cols", type=int, default=None)
    parser.add_argument("--plan-k", type=int, default=None)
    parser.add_argument("--no-planner-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    st = _load_spiraltorch()

    trace_jsonl = Path(args.trace_jsonl)
    trace_html = (
        Path(args.trace_html)
        if args.trace_html is not None
        else trace_jsonl.with_suffix(".html")
    )
    atlas_html = (
        Path(args.atlas_html)
        if args.atlas_html is not None
        else trace_jsonl.with_suffix(".atlas_noncollapse.html")
    )
    manifest = (
        Path(args.manifest)
        if args.manifest is not None
        else trace_jsonl.with_suffix(".artifacts.json")
    )

    manifest_payload = st.write_zspace_experiment_artifacts(
        trace_jsonl,
        trace_html=trace_html,
        atlas_html=atlas_html,
        manifest=manifest,
        title=args.title,
        district=args.district,
        event_type=args.event_type,
        bound=args.bound,
        top_k=args.top_k,
        capture_planner=not args.no_planner_snapshot,
        planner_backend=args.planner_backend,
        planner_rows=args.plan_rows,
        planner_cols=args.plan_cols,
        planner_k=args.plan_k,
    )

    print(f"trace_html={manifest_payload['trace_html']}")
    print(f"atlas_noncollapse_html={manifest_payload['atlas_noncollapse_html']}")
    print(f"artifact_manifest={manifest}")
    planner_snapshot = manifest_payload.get("planner_snapshot")
    if isinstance(planner_snapshot, dict):
        print(f"planner_snapshot_available={planner_snapshot.get('available')}")
    perspective = manifest_payload.get("noncollapse_perspective")
    if isinstance(perspective, dict):
        guidance = perspective.get("guidance")
        if isinstance(guidance, str) and guidance:
            print(f"atlas_guidance={guidance}")


if __name__ == "__main__":
    main()

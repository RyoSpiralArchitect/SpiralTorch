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
import json
import os
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

    def rel(from_path: Path, target: Path) -> str:
        return Path(os.path.relpath(target, start=from_path.parent)).as_posix()

    trace_related_links = {
        "Atlas view": rel(trace_html, atlas_html),
        "Artifact manifest": rel(trace_html, manifest),
        "Trace JSONL": rel(trace_html, trace_jsonl),
    }
    atlas_related_links = {
        "Trace viewer": rel(atlas_html, trace_html),
        "Artifact manifest": rel(atlas_html, manifest),
        "Trace JSONL": rel(atlas_html, trace_jsonl),
    }

    trace_html_out = st.write_zspace_trace_html(
        trace_jsonl,
        trace_html,
        title=args.title,
        event_type=args.event_type,
        related_links=trace_related_links,
    )
    route = st.zspace_trace_to_atlas_route(
        trace_jsonl,
        district=args.district,
        bound=args.bound,
        event_type=args.event_type,
    )
    atlas_html_out = st.write_zspace_atlas_noncollapse_html(
        route,
        atlas_html,
        title=f"{args.title} Atlas Non-Collapse",
        district=args.district,
        top_k=args.top_k,
        related_links=atlas_related_links,
    )
    summary = route.summary()
    perspective = route.perspective_for(args.district, focus_prefixes=["noncollapse."])

    manifest_payload = {
        "trace_jsonl": str(trace_jsonl),
        "trace_html": str(trace_html_out),
        "atlas_noncollapse_html": str(atlas_html_out),
        "artifact_manifest": str(manifest),
        "district": args.district,
        "event_type": args.event_type,
        "summary": summary,
        "noncollapse_perspective": perspective,
    }
    manifest_payload["downstream_hook"] = st.build_zspace_downstream_hook(manifest_payload)
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"trace_html={trace_html_out}")
    print(f"atlas_noncollapse_html={atlas_html_out}")
    print(f"artifact_manifest={manifest}")
    if isinstance(perspective, dict):
        guidance = perspective.get("guidance")
        if isinstance(guidance, str) and guidance:
            print(f"atlas_guidance={guidance}")


if __name__ == "__main__":
    main()

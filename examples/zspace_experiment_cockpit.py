#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Render a cockpit HTML page from a Z-space experiment manifest."""

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
    parser.add_argument("artifact_manifest", help="Path to zspace artifact manifest JSON")
    parser.add_argument("--html", default=None, help="Output cockpit HTML path")
    parser.add_argument("--title", default=None, help="Override cockpit title")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="Print the story packet JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    st = _load_spiraltorch()
    story = st.summarize_zspace_experiment_manifest(args.artifact_manifest, top_k=args.top_k)
    html_path = st.write_zspace_experiment_cockpit_html(
        args.artifact_manifest,
        args.html,
        title=args.title,
        top_k=args.top_k,
    )
    print(f"cockpit_html={html_path}")
    print(f"story_frames={story['summary']['frames']}")
    print(f"story_planner={story['planner']['effective_backend']}")
    focus_metric = story["noncollapse"].get("focus_metric")
    if focus_metric:
        print(f"story_focus={focus_metric}")
    if args.json:
        print(json.dumps(story, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

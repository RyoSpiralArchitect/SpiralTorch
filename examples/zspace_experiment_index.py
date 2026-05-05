#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Render an index HTML page from multiple Z-space experiment manifests."""

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
    parser.add_argument(
        "artifact_manifests",
        nargs="+",
        help="Paths to zspace artifact manifest JSON files",
    )
    parser.add_argument("--html", default=None, help="Output index HTML path")
    parser.add_argument(
        "--title",
        default="SpiralTorch Z-Space Experiment Index",
        help="Index page title",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="Print the index packet JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    st = _load_spiraltorch()
    index = st.summarize_zspace_experiment_index(
        args.artifact_manifests,
        top_k=args.top_k,
        title=args.title,
    )
    html_path = st.write_zspace_experiment_index_html(
        args.artifact_manifests,
        args.html,
        title=args.title,
        top_k=args.top_k,
    )
    print(f"index_html={html_path}")
    print(f"index_runs={index['summary']['runs']}")
    mean_stability = index["summary"].get("mean_stability")
    if mean_stability is not None:
        print(f"index_mean_stability={mean_stability:.3f}")
    if args.json:
        print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

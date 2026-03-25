#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Read a Z-space artifact manifest and emit the downstream hook packet."""

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
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    st = _load_spiraltorch()
    hook = st.build_zspace_downstream_hook(args.artifact_manifest, top_k=args.top_k)
    print(json.dumps(hook, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

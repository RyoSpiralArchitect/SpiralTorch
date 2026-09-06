#!/usr/bin/env python3
"""Keep complete cataloged rank modules synchronized with their shared prelude."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "crates/st-backend-wgpu/src/shaders"
    prelude = (root / "rankk_exact_2ce_common.wgsl").read_text()
    changed = []
    for name, marker in (
        ("rankk_exact_2ce.wgsl", "fn swap_scratch("),
        ("rankk_exact_2ce_midk_tournament.wgsl", "var<workgroup> tournament_nodes:"),
    ):
        path = root / name
        source = path.read_text()
        if source.count(marker) != 1:
            raise ValueError(f"{name}: expected exactly one body marker")
        generated = prelude + "\n" + source[source.index(marker) :]
        if source != generated:
            changed.append(name)
            if not args.check:
                path.write_text(generated)
    if changed:
        print(("out of date: " if args.check else "updated: ") + ", ".join(changed))
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())

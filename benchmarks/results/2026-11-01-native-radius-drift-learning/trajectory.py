"""Secondary, file-only localization; never changes the registered endpoints or decision."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import audit  # noqa: E402
from protocol import PLAN, STOPS, require  # noqa: E402


def compare(free, bounded, counts, stops):
    require(
        len(free) == len(bounded)
        and len(counts) == len(stops)
        and stops[-1] == len(free),
        "incomplete paired update stream",
    )
    first = next(
        (i for i, (a, b) in enumerate(zip(free, bounded, strict=True), 1) if a != b),
        None,
    )
    if first is None:
        return {
            "matched_prior_reports": len(free),
            "first_different_update": None,
            "first_different_fields": [],
            "change_bracketed_by_projection": None,
            "final_projection_count": counts[-1],
        }
    a, b = (json.loads(rows[first - 1]) for rows in (free, bounded))
    require(a["update"] == b["update"] == first, "update owners differ")
    idx = next(i for i, u in enumerate(stops) if u >= first)
    return {
        "matched_prior_reports": first - 1,
        "first_different_update": first,
        "first_different_fields": sorted(
            k for k in a.keys() | b.keys() if k not in a or k not in b or a[k] != b[k]
        ),
        "change_bracketed_by_projection": (idx == 0 or counts[idx - 1] == 0)
        and counts[idx] > 0,
        "final_projection_count": counts[-1],
    }


def inspect(root):
    result = audit(root)
    by_seed = {}
    for s in PLAN["seeds"]:
        f, b = f"{s}-free", f"{s}-bounded"
        a, z = result["fits"][f], result["fits"][b]
        require(
            a["initial_geometry_id"] == z["initial_geometry_id"]
            and a["data_id"] == z["data_id"],
            "not a matched start",
        )
        by_seed[str(s)] = compare(
            (root / f / "steps.jsonl").read_bytes().splitlines(),
            (root / b / "steps.jsonl").read_bytes().splitlines(),
            z["projection_counts"],
            STOPS,
        )
    return {
        "schema": "spiraltorch.native_lm.radius_trajectory_diagnostic.v1",
        "registered_primary": False,
        "declaration_sha256": result["declaration_sha256"],
        "seeds": by_seed,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2, sort_keys=True))

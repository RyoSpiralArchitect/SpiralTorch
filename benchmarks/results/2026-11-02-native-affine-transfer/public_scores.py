"""Reconstruct all scalar receipt hashes and paired contrasts without weights or text."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer import BOOKS, C1, C2, NAMES, PLAN, read, require, weighted


def wire(obj):
    return (
        json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    )


def seal(obj):
    return hashlib.sha256(wire(obj).encode()).hexdigest()


def extract(run):
    receipt = read(run / "scored.json")
    states = {n: read(run / "scores" / (n + ".json")) for n in NAMES}
    identity = {n: v["geometry_id"] for n, v in states.items()}
    groups = {
        b: {k: r["data_id"] for k, r in states["base"]["scores"][b]["sections"].items()}
        for b in BOOKS
    }
    losses = {
        b: {
            k: [states[n]["scores"][b]["sections"][k]["nll"] for n in NAMES]
            for k in groups[b]
        }
        for b in BOOKS
    }
    return {
        "schema": "spiraltorch.native_lm.affine_transfer_section_scores.v1",
        "order": NAMES,
        "geometry_ids": identity,
        "data_ids": groups,
        "nll": losses,
        "scalar_receipt": receipt,
    }


def check(public, result):
    require(
        public["schema"] == "spiraltorch.native_lm.affine_transfer_section_scores.v1",
        "unknown score table",
    )
    receipt = public["scalar_receipt"]
    require(
        public["order"] == NAMES
        and set(receipt["files"]) == set(NAMES)
        and seal(receipt) == result["scores_sha256"]
        and receipt["declaration_sha256"] == result["declaration_sha256"]
        and receipt["package"]["wheel_sha256"] == PLAN["development_wheel_sha256"],
        "scalar roster digest/engine",
    )
    expected = read(HERE / "execution.json")["sections"]
    groups, losses, owners = public["data_ids"], public["nll"], public["geometry_ids"]
    require(
        set(groups) == set(losses) == set(BOOKS)
        and set(owners) == set(NAMES)
        and owners["base"] is None
        and all(re.fullmatch("sha256:[0-9a-f]{64}", owners[n]) for n in NAMES[1:]),
        "book/owner roster",
    )
    for book in BOOKS:
        keys = {str(i) for i in expected[book]}
        require(
            set(groups[book]) == set(losses[book]) == keys
            and all(
                re.fullmatch("sha256:[0-9a-f]{64}", key)
                for key in groups[book].values()
            ),
            "source section identities",
        )
        for values in losses[book].values():
            require(
                len(values) == 15
                and values[1] == values[2]
                and all(math.isfinite(v) for v in values),
                "section scores",
            )
    count = 0
    for i, name in enumerate(NAMES):
        scores = {}
        for book in BOOKS:
            sections = {
                k: {"nll": losses[book][k][i], "targets": 756, "data_id": data_id}
                for k, data_id in groups[book].items()
            }
            count += len(sections)
            mean = weighted([(row["nll"], row["targets"]) for row in sections.values()])
            scores[book] = {"nll": mean, "targets": 4536, "sections": sections}
            summary = result["comparison"][book]
            if name == "base" or name.startswith("initial-"):
                require(
                    mean == summary["base" if name == "base" else "matched_initial"],
                    "reference NLL",
                )
            else:
                seed, arm = name.split("-")
                row = summary["seeds"][seed]
                require(
                    owners[name] == result["cases"][name]["geometry_id"]
                    and mean == row[arm + "_nll"]
                    and mean - summary["base"] == row[arm + "_minus_base"]
                    and mean - summary["matched_initial"]
                    == row[arm + "_minus_initial"],
                    "terminal NLL",
                )
                if arm == "free":
                    require(
                        row["free_minus_affine"] == mean - row["affine_nll"]
                        and row["section_free_minus_affine"]
                        == {
                            k: v[i] - v[NAMES.index(seed + "-affine")]
                            for k, v in losses[book].items()
                        },
                        "paired contrast",
                    )
        require(
            seal({"model": name, "geometry_id": owners[name], "scores": scores})
            == receipt["files"][name],
            "scalar receipt: " + name,
        )
    for book, summary in result["comparison"].items():
        rows = summary["seeds"]
        require(
            summary["mean_free_minus_affine_by_cohort"]
            == {
                k: math.fsum(rows[str(s)]["free_minus_affine"] for s in seeds)
                / len(seeds)
                for k, seeds in PLAN["cohorts"].items()
            }
            and summary["free_wins"]
            == sum(rows[str(s)]["free_minus_affine"] < 0 for s in C1 + C2),
            "cohort aggregation",
        )
    require(
        count == result["section_evaluations"] == 270
        and result["primary_all_six_free_better"]
        == (result["comparison"]["austen"]["free_wins"] == 6),
        "primary outcome",
    )
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-private",
        type=Path,
        help="emit only the compact, verified public score table",
    )
    a = p.parse_args()
    data = (
        extract(a.from_private)
        if a.from_private
        else read(HERE / "section_scores.json")
    )
    count = check(data, read(HERE / "results.json"))
    if a.from_private:
        sys.stdout.write(wire(data))
    else:
        print(
            f"{count} section scores; all {len(NAMES)} original scalar receipt hashes and both cohort contrasts verified"
        )

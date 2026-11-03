"""Verify saved replay files and emit only the predeclared scalar projection."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import (
    HERE,
    PARENT,
    PATHS,
    PLAN,
    cases,
    check_observation,
    read,
    require,
    sha,
    weighted,
)


def summarize_sites(samples):
    denominator = math.fsum(s["elements"] * s["input_rms"] ** 2 for s in samples)
    numerator = math.fsum(s["elements"] * s["residual_rms"] ** 2 for s in samples)
    require(denominator > 0 and numerator >= 0, "invalid observed energy")
    gradients = [
        s["log_radius_gradient"]
        for s in samples
        if s["log_radius_gradient"] is not None
    ]
    require(
        len(gradients) in (0, len(samples)),
        "radius ownership changed across observations",
    )
    return {
        "pooled_relative_residual_rms": math.sqrt(numerator / denominator),
        "mean_trainable_gradient_l2": fmean(
            s["trainable_gradient_l2"] for s in samples
        ),
        "mean_abs_log_radius_gradient": fmean(map(abs, gradients))
        if gradients
        else None,
        "mean_signed_log_radius_gradient": fmean(gradients) if gradients else None,
        "negative_log_radius_gradient_fraction": fmean(g < 0 for g in gradients)
        if gradients
        else None,
    }


def first_separation(differences):
    return next(
        (
            u
            for u in PLAN["evaluation_updates"]
            if u and all(d > 0.001 for d in differences[str(u)])
        ),
        None,
    )


def audit(root, original):
    declared = read(root / "declaration.json")
    keys = [f"{s}-{a}" for s, a in cases()]
    require(
        read(root / "status.json")
        == {
            "status": "complete",
            "cases": keys,
            "declaration_sha256": sha(root / "declaration.json"),
        }
        and declared["cases"] == keys
        and declared["plan_sha256"] == sha(HERE / "plan.json")
        and all(sha(HERE / name) == pin for name, pin in declared["source"].items()),
        "incomplete study/declaration changed",
    )
    require(
        sha(PARENT / "results.json") == PLAN["prior_public_results_sha256"]
        and sha(original / "run/results.json") == PLAN["prior_private_results_sha256"],
        "original endpoints changed",
    )
    published = read(PARENT / "results.json")["case_evidence"]
    by_case, curves, evidence = {}, {}, {}
    for seed, arm in cases():
        case = f"{seed}-{arm}"
        previous = original / "run" / case
        d = root / case
        result = read(d / "result.json")
        expected = published[case]
        require(
            sha(previous / "steps.jsonl") == expected["steps_sha256"]
            and "sha256:" + sha(previous / "checkpoint.json")
            == expected["checkpoint_id"]
            == result["matched_checkpoint_id"]
            and result["matched_steps"] == PLAN["updates_per_case"]
            and result["seed"] == seed
            and result["arm"] == arm
            and result["data_id"] == expected["data_id"]
            and result["foreign_loaded"] is False
            and result["package"]["wheel_sha256"] == declared["wheel_sha256"],
            "case ownership differs",
        )
        require(
            sha(d / "observations.jsonl") == result["observations_sha256"],
            "observation bytes changed",
        )
        observed = [
            json.loads(line)
            for line in (d / "observations.jsonl").read_bytes().splitlines()
        ]
        expected_steps = [
            json.loads(line)
            for line in (previous / "steps.jsonl").read_bytes().splitlines()
        ]
        require(
            len(observed) == len(expected_steps) == PLAN["updates_per_case"],
            "wrong observation count",
        )
        for i, (row, step) in enumerate(zip(observed, expected_steps, strict=True), 1):
            require(step["update"] == i, "original step order changed")
            check_observation(row, step, expected["data_id"], arm)
        by_case[case] = {
            f"{a}:{b}": {
                p: summarize_sites([r["sites"][p] for r in observed[a - 1 : b]])
                for p in PATHS
            }
            for a, b in PLAN["bands"]
        }
        curve = curves[case] = {}
        for u in PLAN["evaluation_updates"]:
            value = read(d / f"score-{u:03d}.json")
            require(
                value == result["scores"][str(u)]
                and set(value["chapters"]) == {str(i) for i in range(50, 56)}
                and value["targets"] == 3024
                and all(v["targets"] == 504 for v in value["chapters"].values())
                and value["nll"]
                == weighted(
                    [(v["nll"], v["targets"]) for v in value["chapters"].values()]
                ),
                "development score differs",
            )
            ref = read(
                original / "run/score" / (("initial" if u == 0 else case) + ".json")
            )["development"]
            require(
                all(
                    x["data_id"] == ref["chapters"][k]["data_id"]
                    for k, x in value["chapters"].items()
                ),
                "development sample changed",
            )
            if u in (0, PLAN["updates_per_case"]):
                require(value == ref, "original development endpoint changed")
            curve[str(u)] = value["nll"]
        evidence[case] = {
            "matched_checkpoint_id": expected["checkpoint_id"],
            "result_sha256": sha(d / "result.json"),
            "observations_sha256": result["observations_sha256"],
            "original_steps_sha256": expected["steps_sha256"],
        }
    sites = {}
    for arm in PLAN["arms"]:
        table = sites[arm] = {}
        for a, b in PLAN["bands"]:
            band = f"{a}:{b}"
            table[band] = {}
            for path in PATHS:
                rows = [by_case[f"{s}-{arm}"][band][path] for s in PLAN["seeds"]]
                table[band][path] = {
                    k: (
                        None
                        if all(x[k] is None for x in rows)
                        else fmean(x[k] for x in rows)
                    )
                    for k in rows[0]
                }
    comparisons = {}
    for left, right in (
        ("learned4", "affine"),
        ("learned4", "fixed4"),
        ("fixed4", "affine"),
    ):
        comparisons[f"{left}-{right}"] = {
            str(u): [
                curves[f"{s}-{left}"][str(u)] - curves[f"{s}-{right}"][str(u)]
                for s in PLAN["seeds"]
            ]
            for u in PLAN["evaluation_updates"]
        }
    return {
        "schema": "spiraltorch.native_book.step_observation_results.v1",
        "plan_sha256": declared["plan_sha256"],
        "declaration_sha256": sha(root / "declaration.json"),
        "developer_wheel_sha256": declared["wheel_sha256"],
        "published_wheel_rebuilt": False,
        "native_source": PLAN["native_source"],
        "seeds": PLAN["seeds"],
        "matched_original_steps": len(keys) * PLAN["updates_per_case"],
        "matched_final_checkpoints": len(keys),
        "site_observations": len(keys) * PLAN["updates_per_case"] * len(PATHS),
        "development_targets_per_measurement": 3024,
        "development_nll": curves,
        "paired_development_nll": comparisons,
        "first_sampled_all_seed_gap_over_0_001": first_separation(
            comparisons["learned4-affine"]
        ),
        "seed_mean_training_sites": sites,
        "case_evidence": evidence,
        "external_book_rescored": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("original", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            audit(args.replay, args.original), sort_keys=True, indent=2, allow_nan=False
        )
    )

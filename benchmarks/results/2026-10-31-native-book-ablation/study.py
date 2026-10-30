"""First-attempt local CPU supervisor; no build, network, GPU or publishing."""

import argparse
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "tools")]
import owned_process  # noqa: E402 - support isolated Python without PYTHONPATH
from common import PLAN, SPLITS, cases, read, require, save, sha  # noqa: E402


def contrast(scores, split, left, right):
    diffs, chapter_counts = [], []
    for seed in PLAN["seeds"]:

        def get(arm):
            key = arm if arm in ("base", "initial") else f"{seed}-{arm}"
            return scores[key][split]

        a, b = get(left), get(right)
        require(
            a["targets"] == b["targets"]
            and a["chapters"].keys() == b["chapters"].keys(),
            "contrast cohorts differ",
        )
        changes = []
        for num, ca in a["chapters"].items():
            cb = b["chapters"][num]
            require(
                ca["data_id"] == cb["data_id"] and ca["targets"] == cb["targets"],
                "unpaired chapter",
            )
            changes.append(ca["nll"] - cb["nll"])
        diff = a["nll"] - b["nll"]
        require(math.isfinite(diff), "invalid paired result")
        diffs.append(diff)
        chapter_counts.append(sum(x < 0 for x in changes))
    return {
        "left_minus_right_by_seed": diffs,
        "mean": math.fsum(diffs) / len(diffs),
        "chapters_favoring_left_by_seed": chapter_counts,
        "chapters_total": len(a["chapters"]),
    }


def summarize(raw, roster):
    require(
        set(raw) == set(roster["order"]) | {"base", "initial"},
        "incomplete score roster",
    )
    splits = {}
    arms = PLAN["arms"]
    for split in SPLITS:
        deltas = {
            (a + "-" + b): contrast(raw, split, a, b)
            for a, b in [
                ("learned4", "affine"),
                ("fixed4", "affine"),
                ("learned4", "fixed4"),
            ]
            + [(a, b) for a in arms for b in ("base", "initial")]
        }
        splits[split] = {
            "base_nll": raw["base"][split]["nll"],
            "initial_nll": raw["initial"][split]["nll"],
            "targets": raw["base"][split]["targets"],
            "nll_by_seed": {
                a: [raw[f"{seed}-{a}"][split]["nll"] for seed in PLAN["seeds"]]
                for a in arms
            },
            "contrasts": deltas,
        }
    return {
        "schema": "spiraltorch.native_book_ablation.results.v1",
        "base_id": PLAN["base_id"],
        "package": PLAN["package"],
        "seeds": PLAN["seeds"],
        "case_order": roster["order"],
        "updates_per_arm_seed": PLAN["updates"],
        "training_targets_per_arm_seed": 32256,
        "case_evidence": roster["cases"],
        "splits": splits,
        "no_automatic_promotion": True,
    }


def main(args):
    os.umask(0o077)
    os.environ.update(
        RAYON_NUM_THREADS=str(PLAN["resources"]["rayon_threads"]),
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    require(sha(HERE / "plan.json") == args.plan_pin, "external protocol pin differs")
    require(
        sha(args.data / "data-manifest.json") == args.data_pin,
        "external data pin differs",
    )
    if args.mode == "run":
        require(
            args.preflight
            and args.preflight_pin
            and sha(args.preflight / "result.json") == args.preflight_pin,
            "external preflight pin required",
        )
    args.output.mkdir(mode=0o700, exist_ok=False)
    deadline = time.monotonic() + PLAN["resources"]["study_timeout_seconds"]
    order = [f"{s}-{a}" for s, a in cases()]
    completed, active = {}, None
    save(
        args.output / "claim.json",
        {
            "mode": args.mode,
            "plan_pin": args.plan_pin,
            "data_pin": args.data_pin,
            "preflight_pin": args.preflight_pin,
            "order": order if args.mode == "run" else [],
            "retry": False,
        },
    )

    def child(mode, name, extra=()):
        nonlocal active
        active = name
        command = [
            args.python,
            "-I",
            "-B",
            str(HERE / "worker.py"),
            mode,
            "--data",
            args.data,
            "--data-pin",
            args.data_pin,
            "--model",
            args.model,
            "--wheel",
            args.wheel,
            "--output",
            args.output / name,
            *extra,
        ]
        if args.mode == "run":
            command += [
                "--preflight",
                args.preflight,
                "--preflight-pin",
                args.preflight_pin,
            ]
        remaining = deadline - time.monotonic()
        require(remaining > 0, "common study deadline reached")
        with (
            (args.output / (name + ".stdout")).open("xb") as out,
            (args.output / (name + ".stderr")).open("xb") as err,
        ):
            result = owned_process.run(
                [str(x) for x in command],
                timeout=min(remaining, PLAN["resources"]["case_timeout_seconds"]),
                stdout=out,
                stderr=err,
            )
        require(result.returncode == 0, "worker failed; retain first output: " + name)
        print("completed " + name, flush=True)
        return read(args.output / name / "result.json")

    try:
        if args.mode == "preflight":
            child("preflight", "preflight")
        else:
            for seed, arm in cases():
                name = f"{seed}-{arm}"
                row = child("fit", name, ["--arm", arm, "--seed", str(seed)])
                require(
                    row["arm"] == arm
                    and row["seed"] == seed
                    and row["updates"] == PLAN["updates"]
                    and row["target_tokens"] == 32256
                    and row["evaluation_scored"] is False
                    and "sha256:" + sha(args.output / name / "checkpoint.json")
                    == row["checkpoint_id"]
                    and "sha256:" + sha(args.output / name / "geometry.json")
                    == row["geometry_id"],
                    "case does not bind complete native state",
                )
                completed[name] = row
            seed_orders = []
            for seed in PLAN["seeds"]:
                rows = [completed[f"{seed}-{arm}"] for arm in PLAN["arms"]]
                require(
                    len({r["order_sha256"] for r in rows})
                    == len({r["data_id"] for r in rows})
                    == 1,
                    "arms did not see identical training data",
                )
                seed_orders.append(rows[0]["order_sha256"])
            require(
                len(set(seed_orders)) == len(PLAN["seeds"]), "seed orders collapsed"
            )
            roster = {
                "order": order,
                "cases": completed,
                "preflight_pin": args.preflight_pin,
            }
            save(args.output / "fitted.json", roster)
            raw = child(
                "score",
                "score",
                ["--fits", args.output, "--fit-pin", sha(args.output / "fitted.json")],
            )
            report = summarize(raw, roster)
            report.update(
                plan_sha256=args.plan_pin,
                data_manifest_sha256=args.data_pin,
                preflight_sha256=args.preflight_pin,
                fitted_sha256=sha(args.output / "fitted.json"),
                all_scores_sha256=sha(args.output / "score/result.json"),
            )
            save(args.output / "results.json", report)
        save(
            args.output / "status.json",
            {"status": "complete", "completed_fit_cases": list(completed)},
        )
    except BaseException as error:
        save(
            args.output / "status.json",
            {
                "status": "failed",
                "stage": active,
                "error_type": type(error).__name__,
                "completed_fit_cases": list(completed),
                "uncompleted_fit_cases": [k for k in order if k not in completed],
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight", "run"])
    for name in ("python", "wheel", "model", "data", "output", "preflight"):
        p.add_argument("--" + name, type=Path, required=name != "preflight")
    for name in ("data-pin", "plan-pin", "preflight-pin"):
        p.add_argument("--" + name, required=name != "preflight-pin")
    main(p.parse_args())

"""Audit the nine saved checkpoints, update chains and chapter scores without inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, PLAN, SPLITS, cases, digest, ordering, read, require, sha
from study import summarize
from worker import source_pins


def checked_steps(rows, seed, arm, before, after):
    n = PLAN["updates"]
    require(len(rows) == n, "incomplete update history")
    previous = before
    for i, row in enumerate(rows):
        require(
            row["update"] == i + 1
            and row["epoch"] == 0
            and [row["window_start"], row["window_end"]] == [2 * i, 2 * i + 2]
            and row["microbatches"] == 1
            and row["target_tokens"] == 126
            and math.isfinite(row["loss"])
            and row["pre_geometry_artifact_id"] == previous,
            f"broken update chain: {seed}/{arm}/{i + 1}",
        )
        previous = row["post_geometry_artifact_id"]
    require(previous == after, "last update differs from final artifact")


def check_saved(case, row, initial, preflight):
    checkpoint = read(case / "checkpoint.json")
    require(
        row["checkpoint_id"] == "sha256:" + sha(case / "checkpoint.json")
        and row["geometry_id"] == "sha256:" + sha(case / "geometry.json")
        and sha(case / "steps.jsonl") == row["steps_sha256"],
        "saved bytes changed",
    )
    require(
        checkpoint["updates"] == PLAN["updates"]
        and checkpoint["data_id"] == row["data_id"]
        and checkpoint["geometry_id"] == row["geometry_id"]
        and checkpoint["base_model_identity"] == PLAN["base_id"]
        and checkpoint["geometry_json"].encode()
        == (case / "geometry.json").read_bytes(),
        "checkpoint state differs",
    )
    config = checkpoint["config"]
    require(
        all(config[k] == v for k, v in PLAN["recipe"].items()),
        "optimizer/window recipe differs",
    )
    require(not config.get("repetition"), "undeclared repetition objective")
    optimizer = json.loads(checkpoint["optimizer_json"])
    require(
        "sha256:" + hashlib.sha256(checkpoint["optimizer_json"].encode()).hexdigest()
        == checkpoint["optimizer_id"],
        "nested optimizer bytes changed",
    )
    require(
        optimizer["geometry_artifact_id"] == row["geometry_id"]
        and optimizer["revision"] == PLAN["updates"]
        and optimizer["max_gradient_norm"] == PLAN["max_gradient_norm"],
        "optimizer state/clip differs",
    )
    current = json.loads(checkpoint["geometry_json"])
    slots = {
        (slot["placement"], slot["parameter"]): slot for slot in optimizer["slots"]
    }
    trainable = {}
    changed_sites, changed_radii = 0, 0
    for start, end in zip(initial["placements"], current["placements"], strict=True):
        require(
            start["path"] == end["path"] and start["recipe"] == end["recipe"],
            "placement/recipe mutation",
        )
        changed = False
        for x, y in zip(start["tensors"], end["tensors"], strict=True):
            require(
                {k: v for k, v in x.items() if k != "values"}
                == {k: v for k, v in y.items() if k != "values"},
                "tensor metadata mutation",
            )
            require(
                len(x["values"]) == len(y["values"])
                and all(math.isfinite(v) for v in y["values"]),
                "invalid fitted tensor",
            )
            if y["requires_grad"]:
                trainable[(end["path"], y["name"])] = y
            else:
                require(x["values"] == y["values"], "frozen tensor changed")
            if x["values"] != y["values"]:
                changed |= y["name"] in ("gate", "bias")
                changed_radii += y["name"] == "log_radius"
        changed_sites += changed
    require(
        set(slots) == set(trainable) and len(slots) == len(optimizer["slots"]),
        "optimizer slot roster differs",
    )
    for key, tensor in trainable.items():
        slot = slots[key]
        require(
            slot["steps"] == PLAN["updates"]
            and slot["shape"] == tensor["shape"]
            and all(
                len(slot[k]) == len(tensor["values"])
                and all(math.isfinite(v) for v in slot[k])
                for k in ("first", "second")
            ),
            "incomplete moment state",
        )
    require(
        changed_sites == row["channel_sites_changed"]
        and changed_radii == row["radius_sites_changed"],
        "change summary differs",
    )
    require(
        row["order_sha256"] == digest(ordering(row["seed"], PLAN["train_windows"])),
        "data-order receipt differs",
    )
    checked_steps(
        [json.loads(line) for line in (case / "steps.jsonl").read_text().splitlines()],
        row["seed"],
        row["arm"],
        preflight["initial"][row["arm"]],
        row["geometry_id"],
    )
    return sum(len(t["values"]) for t in trainable.values())


def verify_chapters(scores):
    for name, splits in scores.items():
        require(set(splits) == set(SPLITS), "score split roster differs")
        for split, values in splits.items():
            ids = range(1, 25) if split == "external" else range(50, 56)
            chapters = values["chapters"]
            require(set(chapters) == {str(i) for i in ids}, "chapter roster differs")
            total = 0
            sums = []
            for number, chapter in chapters.items():
                require(
                    chapter["targets"] == 504
                    and math.isfinite(chapter["nll"])
                    and chapter["data_id"]
                    == scores["base"][split]["chapters"][number]["data_id"],
                    "chapter identity/count differs",
                )
                total += chapter["targets"]
                sums.append(chapter["targets"] * chapter["nll"])
            require(
                values["targets"] == total and values["nll"] == math.fsum(sums) / total,
                "reported NLL differs from targets",
            )


def audit(root):
    before, run = root / "before/preflight", root / "run"
    pins = read(HERE / "execution.json")
    meta, frozen = read(root / "data/data-manifest.json"), read(before / "result.json")
    require(
        pins["plan_sha256"] == sha(HERE / "plan.json")
        and pins["preflight_sha256"] == sha(before / "result.json")
        and pins["data_manifest_sha256"] == sha(root / "data/data-manifest.json"),
        "declaration pin differs",
    )
    for name, value in meta["files"].items():
        require(sha(root / "data" / name) == value, "encoded data changed")
    roster = read(run / "fitted.json")
    expected_order = [f"{seed}-{arm}" for seed, arm in cases()]
    require(
        roster["order"] == expected_order
        and set(roster["cases"]) == set(expected_order),
        "undeclared fit roster",
    )
    require(
        frozen["source_sha256"] == source_pins(), "executed protocol source changed"
    )
    for folder in [before, *(run / k for k in expected_order), run / "score"]:
        runtime = read(folder / "runtime.json")
        require(
            runtime["foreign_loaded"] is False
            and runtime["native"] == PLAN["native_sha256"],
            "native runtime receipt differs",
        )
    require(
        read(run / "status.json")
        == {"status": "complete", "completed_fit_cases": roster["order"]},
        "study incomplete",
    )
    counts = {}
    for seed, arm in cases():
        name = f"{seed}-{arm}"
        row = roster["cases"][name]
        require(read(run / name / "result.json") == row, "case/report mismatch")
        require(
            "sha256:" + sha(before / (arm + ".json")) == frozen["initial"][arm],
            "initial geometry differs",
        )
        value = check_saved(run / name, row, read(before / (arm + ".json")), frozen)
        require(
            value == (6148 if arm == "learned4" else 6144), "parameter budget differs"
        )
        counts[name] = value
    scores = read(run / "score/result.json")
    for name, value in scores.items():
        require(
            read(run / "score" / (name + ".json")) == value,
            "per-model score file differs",
        )
    verify_chapters(scores)
    expected = summarize(scores, roster)
    expected.update(
        plan_sha256=pins["plan_sha256"],
        data_manifest_sha256=pins["data_manifest_sha256"],
        preflight_sha256=pins["preflight_sha256"],
        fitted_sha256=sha(run / "fitted.json"),
        all_scores_sha256=sha(run / "score/result.json"),
    )
    require(
        expected == read(run / "results.json"), "summary differs from complete scores"
    )
    require(
        expected == read(HERE / "results.json"),
        "public summary differs from private results",
    )
    return {
        "schema": "spiraltorch.native_book_ablation.audit.v1",
        "checkpoints_verified": len(counts),
        "updates_verified": len(counts) * PLAN["updates"],
        "scored_variants": len(scores),
        "chapter_scores_verified": len(scores) * 30,
        "trainable_per_case": counts,
        "reserve_read_by_audit": False,
        "private_results_sha256": sha(run / "results.json"),
        "public_results_sha256": sha(HERE / "results.json"),
        "native_process_receipts_verified": 11,
        "trained_or_scored_by_audit": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("private_root", type=Path)
    print(json.dumps(audit(p.parse_args().private_root), indent=2, sort_keys=True))

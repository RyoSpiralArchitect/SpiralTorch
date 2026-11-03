"""Inspect retained files only. This is not the nine-case trajectory analysis."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import (
    HERE,
    PARENT,
    PATHS,
    PLAN,
    audit_original,
    cases,
    check_observation,
    read,
    require,
    sha,
    weighted,
    wheels,
)


def inventory(root):
    files = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), "unexpected symlink in first-attempt evidence")
        if not p.is_dir():
            require(p.is_file(), "unexpected non-file evidence")
            files[p.relative_to(root).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": sha(p),
            }
    return files


def retained_rows(path, reference, data_id, arm):
    count = 0
    with path.open("rb") as stream:
        for count, line in enumerate(stream, 1):
            require(line.endswith(b"\n"), "torn last observation; no prefix certified")
            require(
                count <= len(reference) and reference[count - 1]["update"] == count,
                "unexpected observation count/order",
            )
            check_observation(json.loads(line), reference[count - 1], data_id, arm)
    return count


def inspect_case(root, original, case, reference, expected, execution):
    folder = root / case
    seed, arm = case.split("-")
    if not folder.exists():
        require(
            not any((root / (case + ext)).exists() for ext in (".stdout", ".stderr")),
            "worker output exists for case without directory",
        )
        return {"status": "unstarted", "decoded_steps": 0}, {}
    log = folder / "observations.jsonl"
    count = retained_rows(log, reference, expected["data_id"], arm)
    complete = (folder / "result.json").exists()
    scores = {}
    for u in PLAN["evaluation_updates"]:
        path = folder / f"score-{u:03d}.json"
        if not path.exists():
            require(not complete and u >= count and u != 0, "missing retained score")
            continue
        require(u <= count, "score beyond retained prefix")
        score = scores[str(u)] = read(path)
        ref = read(original / "run/score" / (("initial" if not u else case) + ".json"))[
            "development"
        ]
        require(
            score["chapters"].keys() == ref["chapters"].keys()
            and all(
                x["targets"] == 504
                and x["nll"] >= 0
                and x["data_id"] == ref["chapters"][k]["data_id"]
                for k, x in score["chapters"].items()
            )
            and score["targets"] == 3024
            and score["nll"]
            == weighted([(x["nll"], x["targets"]) for x in score["chapters"].values()]),
            "development sample or weighted score changed",
        )
        if u in (0, PLAN["updates_per_case"]):
            require(score == ref, "original endpoint changed")
    allowed = {log.name, *(f"score-{int(u):03d}.json" for u in scores)}
    row = {
        "status": "partial",
        "decoded_steps": count,
        "score_cursors": list(map(int, scores)),
        "observations_sha256": sha(log),
    }
    if complete:
        allowed.add("result.json")
        result = read(folder / "result.json")
        package = result["package"]
        require(
            count == result["matched_steps"] == PLAN["updates_per_case"]
            and result["seed"] == int(seed)
            and result["arm"] == arm
            and result["data_id"] == expected["data_id"]
            and result["foreign_loaded"] is False
            and result["matched_checkpoint_id"] == expected["checkpoint_id"]
            and result["observations_sha256"] == row["observations_sha256"]
            and package["wheel_sha256"] == execution["wheel_sha256"]
            and package["native_sha256"] == execution["native_sha256"]
            and package["version"] == execution["package_metadata_version"]
            and package["all_original_files_match"] is True
            and package["python_bytecode_absent"] is True
            and result["scores"] == scores,
            "worker completion receipt does not match",
        )
        row.update(
            status="complete",
            matched_checkpoint_id=result["matched_checkpoint_id"],
            result_sha256=sha(folder / "result.json"),
        )
    require(
        {p.name for p in folder.iterdir()} == allowed,
        "unexpected case evidence (including possible first mismatch); stop",
    )
    return row, {u: s["nll"] for u, s in scores.items()}


def paired_training_bands(steps, expected):
    """Post-hoc summaries of the ORIGINAL training batches; never development scores."""
    out = {}
    for seed in PLAN["seeds"]:
        arms = {arm: f"{seed}-{arm}" for arm in PLAN["arms"]}
        require(
            len({expected[case]["data_id"] for case in arms.values()}) == 1,
            "unpaired training data",
        )
        table = out[str(seed)] = {}
        for a, b in PLAN["bands"]:
            means = {}
            for arm, case in arms.items():
                rows = steps[case][a - 1 : b]
                require(
                    len(rows) == b - a + 1
                    and all(
                        (
                            r["update"],
                            r["epoch"],
                            r["window_start"],
                            r["window_end"],
                            r["target_tokens"],
                        )
                        == (i, 0, 2 * i - 2, 2 * i, 126)
                        for i, r in enumerate(rows, a)
                    ),
                    "unpaired training windows/targets",
                )
                means[arm] = weighted([(r["loss"], r["target_tokens"]) for r in rows])
            table[f"{a}:{b}"] = {
                f"learned4-{arm}": means["learned4"] - means[arm]
                for arm in ("affine", "fixed4")
            }
    return out


def inspect(root, original, wheel):
    require(
        not (root / "status.json").exists(),
        "not the interrupted attempt; use analysis.py",
    )
    files = inventory(root)
    execution, declaration = (
        read(HERE / "execution.json"),
        read(root / "declaration.json"),
    )
    original_audit = audit_original(original)
    roster = [f"{s}-{a}" for s, a in cases()]
    require(
        declaration["cases"] == roster
        and declaration["retries"] == 0
        and declaration["prior_audit"] == original_audit == read(PARENT / "audit.json")
        and original_audit["private_results_sha256"]
        == PLAN["prior_private_results_sha256"]
        and original_audit["public_results_sha256"]
        == PLAN["prior_public_results_sha256"]
        and declaration["plan_sha256"]
        == execution["plan_sha256"]
        == sha(HERE / "plan.json")
        and execution["native_source"] == PLAN["native_source"]
        and declaration["source"]
        == {p: sha(HERE / p) for p in ("plan.json", "replay.py", "analysis.py")}
        and declaration["train_sha256"] == sha(original / "data/train.json")
        and declaration["development_sha256"] == sha(root / "development.json")
        and read(root / "development.json")
        == read(original / "data/evaluation.json")["development"]
        and declaration["initial_ids"]
        == read(original / "before/preflight/result.json")["initial"]
        and declaration["score_file_sha256"]
        == {
            k: sha(original / "run/score" / (k + ".json")) for k in ["initial", *roster]
        },
        "first-attempt declaration or original results changed",
    )
    package = wheels.wheel_archive_inventory(wheel)
    require(
        package["sha256"] == declaration["wheel_sha256"] == execution["wheel_sha256"]
        and package["native"]["sha256"] == execution["native_sha256"]
        and package["version"] == execution["package_metadata_version"]
        and execution["development_only"],
        "not the pinned developer wheel",
    )
    expected = read(PARENT / "results.json")["case_evidence"]
    rows, curves, steps = {}, {}, {}
    for case in roster:
        p = original / "run" / case / "steps.jsonl"
        require(sha(p) == expected[case]["steps_sha256"], "original step bytes changed")
        steps[case] = [json.loads(line) for line in p.read_bytes().splitlines()]
        rows[case], curves[case] = inspect_case(
            root, original, case, steps[case], expected[case], execution
        )
    states = [v["status"] for v in rows.values()]
    require(
        states
        == ["complete"] * states.count("complete")
        + ["partial"]
        + ["unstarted"] * states.count("unstarted"),
        "unexpected interrupted case ordering",
    )
    started = [k for k in rows if rows[k]["status"] != "unstarted"]
    require(
        {p.name for p in root.iterdir()}
        == {
            "declaration.json",
            "development.json",
            *(k + ext for k in started for ext in ("", ".stdout", ".stderr")),
        },
        "unknown run evidence",
    )
    complete_pairs = {}
    for seed in PLAN["seeds"]:
        left, right = f"{seed}-learned4", f"{seed}-affine"
        if rows[left]["status"] == rows[right]["status"] == "complete":
            complete_pairs[str(seed)] = {
                u: v - curves[right][u] for u, v in curves[left].items()
            }
    manifest = (
        json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    require(files == inventory(root), "source evidence changed during inspection")
    return {
        "schema": "spiraltorch.native_book.interrupted_inspection.v1",
        "status": "incomplete",
        "termination_cause": "not established",
        "supervisor_status_present": False,
        "plan_sha256": execution["plan_sha256"],
        "declaration_sha256": sha(root / "declaration.json"),
        "wheel_sha256": package["sha256"],
        "first_attempt_files": len(files),
        "first_attempt_inventory_sha256": hashlib.sha256(manifest).hexdigest(),
        "cases": rows,
        "decoded_observations": sum(v["decoded_steps"] for v in rows.values()),
        "observed_sites_per_step": len(PATHS),
        "worker_final_checkpoint_receipts": states.count("complete"),
        "all_seed_development_milestone": {"status": "not_evaluable", "cursor": None},
        "development_learned4_minus_affine_complete_pairs_only": complete_pairs,
        "original_study_audit": original_audit,
        "original_training_batch_nll_post_hoc": paired_training_bands(steps, expected),
    }, manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "original", "wheel"):
        p.add_argument(name, type=Path)
    p.add_argument(
        "--private-inventory", type=Path, help="new file outside both evidence roots"
    )
    args = p.parse_args()
    report, manifest = inspect(args.replay, args.original, args.wheel)
    if args.private_inventory:
        path = args.private_inventory.resolve()
        require(
            all(
                not path.is_relative_to(r.resolve())
                for r in (args.replay, args.original)
            ),
            "do not write inside first-attempt evidence",
        )
        os.umask(0o077)
        with path.open("xb") as f:
            f.write(manifest)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))

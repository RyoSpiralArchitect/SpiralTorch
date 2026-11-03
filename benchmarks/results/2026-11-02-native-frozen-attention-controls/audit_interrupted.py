"""File-only audit of a first attempt that stopped before the scoring gate."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_contract import (
    CASES,
    PARENT,
    PLAN,
    audit_fit,
    check_step,
    claim,
    first_match,
    verify_primary,
)
from joint_protocol import digest, ordering, read, require, rows, sha
from joint_worker import paths


def complete_lines(payload):
    complete, _, tail = payload.rpartition(b"\n")
    return [json.loads(v) for v in complete.splitlines()], tail


def inspect(a):
    c = claim(a)
    cp = sha(a.run / "claim.json")
    require(c == read(a.run / "claim.json"), "changed claim")
    verify_primary(a)
    require(
        not any(
            (a.run / n).exists()
            for n in (
                "results.json",
                "status.json",
                "fitted.json",
                "scored.json",
                "scalar.json",
                "native-scores",
                "score.admit.json",
                "score.started.json",
            )
        ),
        "not an unscored interruption",
    )
    init = read(a.out / "initial" / "receipt.json")
    parent = read(PARENT / "results.json")
    finished, pending, count = {}, None, 0
    for case in CASES:
        folder = a.run / case
        if not folder.exists():
            require(
                pending is not None and not any(a.run.glob(case + "-*.admit.json")),
                "unexpected case gap",
            )
            continue
        require(pending is None, "another fit started after failure")
        seed, arm = case.split("-")
        first, pin1 = read(folder / "first.json"), sha(folder / "first.json")
        for part, upstream in (("first", None), ("second", pin1)):
            stem = case + "-" + part
            require(
                read(a.run / (stem + ".admit.json"))
                == read(a.run / (stem + ".started.json"))
                == {"claim_sha256": cp, "upstream": upstream},
                "unbound worker",
            )
        if (folder / "second.json").exists():
            audited = audit_fit(
                a, case, {"first": pin1, "second": sha(folder / "second.json")}
            )
            finished[case] = audited["checkpoint_id"]
            count += 512
            continue
        geom, lora = init["owners"][arm], init["owners"]["lora"]
        full = rows(folder / "first.jsonl")
        require(
            len(full) == 256
            and first["case"] == case
            and first["claim_sha256"] == cp
            and first["package"] == init["package"]
            and first["before"] == init["parameter_hashes"][arm]
            and first["previous"] is None
            and first["order_sha256"] == digest(ordering(int(seed), 512))
            and first["steps_sha256"] == sha(folder / "first.jsonl"),
            "invalid completed first half",
        )
        first_match(full[0], rows(a.out / case / "first.jsonl")[0])
        for u, row in enumerate(full, 1):
            lora = check_step(row, u, lora, geom)
        ck = folder / "checkpoint-256.json"
        state, b = read(ck), first["boundary"]
        require(
            "sha256:" + sha(ck) == b["checkpoint_id"]
            and state["config"]
            == {
                "context_length": 64,
                "epochs": 1,
                "learning_rate": 0.01,
                "mode": "lora",
            }
            and state["updates"] == b["update"] == 256
            and state["data_id"]
            == b["recorded_data_id"]
            == parent["fits"][case]["data_id"]
            and state["geometry_id"] == b["post_geometry_id"] == geom
            and state["lora_id"] == b["post_lora_id"] == lora
            and state["geometry_json"].encode()
            == (a.out / "initial" / (arm + ".json")).read_bytes()
            and not (folder / "checkpoint-512.json").exists(),
            "invalid last durable boundary",
        )
        log = folder / "second.jsonl"
        updates, tail = complete_lines(log.read_bytes())
        require(len(updates) < 256, "not a partial training log")
        for u, row in enumerate(updates, 257):
            lora = check_step(row, u, lora, geom)
        err = a.run / (case + "-second.stderr")
        require(
            b"stream.write(" in err.read_bytes()
            and b"[Errno 28] No space left on device" in err.read_bytes(),
            "no retained ENOSPC evidence",
        )
        pending = {
            "case": case,
            "last_checkpoint_update": 256,
            "last_checkpoint_id": b["checkpoint_id"],
            "last_complete_logged_update": 256 + len(updates),
            "log_sha256": sha(log),
            "stderr_sha256": sha(err),
            "uncommitted_tail_bytes": len(tail),
            "uncommitted_tail_sha256": hashlib.sha256(tail).hexdigest(),
        }
        count += 256 + len(updates)
    require(
        pending is not None
        and len(list(a.run.glob("*.started.json"))) == 2 * (len(finished) + 1),
        "unexpected worker count",
    )
    inventory = {
        p.relative_to(a.run).as_posix(): sha(p) for p in a.run.rglob("*") if p.is_file()
    }
    return {
        "schema": "spiraltorch.native_lm.fixed_control_interrupted.v1",
        "status": "incomplete_no_scores",
        "plan_sha256": c["plan_sha256"],
        "claim_sha256": cp,
        "primary_result_sha256": PLAN["primary_result"],
        "completed_terminal_checkpoints": finished,
        "partial_case": pending,
        "never_started": CASES[len(finished) + 1 :],
        "verified_complete_step_records": count,
        "executed_but_unlogged_updates": "unknown",
        "supervisor_first_error_file_present": (a.run / "first-error.json").is_file(),
        "private_file_count": len(inventory),
        "private_inventory_sha256": digest(inventory),
        "held_out_scores": 0,
        "primary_quality_result": None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    print(json.dumps(inspect(p.parse_args()), indent=2, sort_keys=True))

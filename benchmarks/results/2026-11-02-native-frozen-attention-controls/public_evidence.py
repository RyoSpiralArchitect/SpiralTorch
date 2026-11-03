"""Export and verify numerical evidence without distributing books or checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from control_contract import (
    CASES,
    PARENT,
    PLAN,
    PLAN_PIN,
    claim,
    digest,
    ordering,
    read,
    require,
    sha,
    source_pins,
)
from control_score import summarize
from control_study import audit
from fixed_trajectories import paired
from joint_protocol import save
from joint_worker import paths


def seal(value):
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return hashlib.sha256(data.encode()).hexdigest()


def verify(meta, native):
    c, sc, fits, stages, init = (
        meta[k] for k in ("claim", "scored", "fits", "stages", "initial")
    )
    check = meta["status"]
    cp = seal(c)
    parent = read(PARENT / "results.json")
    require(
        sha(HERE / "plan.json") == c["plan_sha256"] == PLAN_PIN
        and sha(PARENT / "results.json")
        == c["primary_result_sha256"]
        == PLAN["primary_result"]
        and seal(init) == c["initial_sha256"] == PLAN["initial_receipt"]
        and c["order"] == CASES
        and c["wheel_sha256"] == parent["wheel_sha256"]
        and c["native_sha256"] == parent["native_sha256"],
        "different experiment",
    )
    files = [
        HERE / n
        for n in (
            "plan.json",
            "control_contract.py",
            "control_study.py",
            "control_score.py",
        )
    ]
    files.extend(
        PARENT / n
        for n in (
            "freeze_binding.py",
            "execution.json",
            "input_binding.json",
            "score_roster.json",
            "results.json",
        )
    )
    require(
        c["source"]
        == {
            **source_pins(),
            **{str(p.relative_to(HERE.parents[2])): sha(p) for p in files},
        },
        "different numerical or validation source",
    )
    require(
        set(stages) == set(fits) == set(CASES)
        and sc["fitted_sha256"] == seal(fits)
        and sc["claim_sha256"] == cp
        and sc["package"] == init["package"],
        "unbound fit table",
    )
    for key in CASES:
        s, arm = key.split("-")
        f = fits[key]
        geom = init["owners"][arm]
        initial = init["parameter_hashes"][arm]
        prev_params = initial
        for name, step in (("first", 256), ("second", 512)):
            receipt = stages[key][name]
            b = receipt["boundary"]
            require(
                seal(receipt) == f["stages"][name]
                and receipt["case"] == key
                and receipt["claim_sha256"] == cp
                and receipt["order_sha256"] == digest(ordering(int(s), 512))
                and receipt["package"] == init["package"]
                and receipt["previous"]
                == (None if step == 256 else f["stages"]["first"])
                and receipt["before"] == prev_params
                and b["update"] == step
                and b["post_geometry_id"] == geom
                and b["recorded_data_id"]
                == f["data_id"]
                == parent["fits"][key]["data_id"]
                and set(b["parameters"]) == set(initial)
                and all(
                    b["parameters"][n] == initial[n] for n in set(initial) - {"a", "b"}
                ),
                "different start, order, freeze or boundary",
            )
            prev_params = b["parameters"]
        require(
            (f["checkpoint_id"], f["lora_id"], f["geometry_id"])
            == (b["checkpoint_id"], b["post_lora_id"], geom)
            and all(prev_params[k] != initial[k] for k in ("a", "b"))
            and f["step1_joint_loss_and_lora_match"] is True
            and f["frozen_geometry_verified_updates"] == 512,
            "different terminal owner or update coverage",
        )
    reports, scalar = native["sections"], native["scalar"]
    traces = meta["fixed_trajectory_comparison"]
    require(set(traces) == {k.split("-")[0] for k in CASES}, "missing fixed pair")
    for seed, pair in traces.items():
        require(
            set(pair) == {"post_lora", "loss_f32_le"}, "missing trajectory projection"
        )
        for v in pair.values():
            n = v["different_updates"]
            require(
                type(n) is int
                and 0 <= n <= 512
                and (v["affine_sequence_sha256"] == v["wave_sequence_sha256"])
                == (n == 0)
                and (
                    (n == 0 and v["first_different_update"] is None)
                    or (n > 0 and 1 <= v["first_different_update"] <= 513 - n)
                ),
                "inconsistent trajectory digest/count",
            )
        if pair["post_lora"]["different_updates"] == 0:
            require(
                fits[seed + "-affine"]["lora_id"] == fits[seed + "-wave"]["lora_id"],
                "different terminal LoRA",
            )
    require(
        sc["native_scores"] == {k: seal(r) for k, r in reports.items()}
        and sc["scalar_sha256"] == seal(scalar),
        "native receipt bytes differ",
    )
    effects = summarize(fits, reports, scalar)
    require(
        effects == meta["effects"]
        and check
        == {
            "status": "complete",
            "attempts": 1,
            "claim_sha256": cp,
            "result_sha256": seal(effects),
        },
        "missing first success or changed contrasts",
    )
    return effects


def export(a):
    claim(a)
    require(
        not (a.run / "first-error.json").exists(), "failed attempt is not exportable"
    )
    effects = audit(a)
    meta = {
        "claim": read(a.run / "claim.json"),
        "scored": read(a.run / "scored.json"),
        "fits": read(a.run / "fitted.json"),
        "initial": read(a.out / "initial" / "receipt.json"),
        "status": read(a.run / "status.json"),
        "effects": effects,
        "stages": {
            key: {s: read(a.run / key / (s + ".json")) for s in ("first", "second")}
            for key in CASES
        },
    }
    native = {
        "sections": {p.stem: read(p) for p in (a.run / "native-scores").glob("*.json")},
        "scalar": read(a.run / "scalar.json"),
    }
    meta["fixed_trajectory_comparison"] = paired(a.run, meta["stages"])
    require(read(a.run / "results.json") == effects, "not the completed private result")
    verify(meta, native)
    a.dest.mkdir(mode=0o700, exist_ok=False)
    save(a.dest / "native_reports.json", native)
    save(a.dest / "results.json", meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    e = commands.add_parser("export")
    paths(e)
    e.add_argument("--dest", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("--dir", type=Path, default=HERE)
    a = p.parse_args()
    if a.action == "export":
        export(a)
    else:
        verify(read(a.dir / "results.json"), read(a.dir / "native_reports.json"))
        print(
            "12 sealed training stages, 18 native reports, 108 exact anchors, 108 fixed scores and six scalar checks verified."
        )

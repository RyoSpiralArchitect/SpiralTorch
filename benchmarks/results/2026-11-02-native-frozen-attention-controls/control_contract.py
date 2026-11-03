"""File-only lineage, progress and matched NLL checks for the frozen-group control."""

import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "2026-11-02-native-joint-attention-transfer"
sys.path.insert(0, str(PARENT))
from joint_protocol import (
    BOOKS,
    SEEDS,
    ORIGINAL,
    digest,
    ordering,
    read,
    require,
    rows,
    sha,
    source_pins,
    weighted,
)
from freeze_binding import bind_inputs, bind_run, frozen
from joint_study import audit as audit_primary
from joint_worker import checked_stage

PLAN_PIN = "0f97c980fe86fd2bfb15cf4bf149639e8c678d9674771684ce3190b06635943c"
PLAN = read(HERE / "plan.json")
CASES = PLAN["order"]
STATES = [key + ":" + mode for key in CASES for mode in ("joint", "fixed")]


def claim(a):
    require(
        sha(HERE / "plan.json") == PLAN_PIN
        and sha(PARENT / "score_roster.json")
        == "839ecffd03f6c7e1e475f18a88dfcea68b7e424f592ce1a3a1846ce804d801fe"
        and sha(PARENT / "input_binding.json")
        == "a156cdf04cc2c524d721e9d689ec7588e94400c5dd702e2fc9c51e71185245ab"
        and sha(a.out / "results.json")
        == sha(PARENT / "results.json")
        == PLAN["primary_result"]
        and read(a.out / "status.json")
        == {"status": "complete", "results_sha256": PLAN["primary_result"]}
        and sha(a.out / "initial" / "receipt.json") == PLAN["initial_receipt"],
        "unbound protocol, primary result or initial geometry",
    )
    a.run = a.out.resolve(strict=True) / ".fixed-attention-controls" / PLAN_PIN
    require(
        not (a.run.parent.is_symlink() or a.run.is_symlink()),
        "relocated claim registry",
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
    files += [
        PARENT / n
        for n in (
            "freeze_binding.py",
            "execution.json",
            "input_binding.json",
            "score_roster.json",
            "results.json",
        )
    ]
    return {
        "schema": "spiraltorch.native_lm.fixed_control_claim.v1",
        "plan_sha256": PLAN_PIN,
        "primary_result_sha256": PLAN["primary_result"],
        "initial_sha256": PLAN["initial_receipt"],
        "source": {
            **source_pins(),
            **{str(p.relative_to(HERE.parents[2])): sha(p) for p in files},
        },
        "order": CASES,
        "wheel_sha256": frozen()["development_wheel_sha256"],
        "native_sha256": frozen()["native_sha256"],
    }


def verify_primary(a):
    mapping = bind_inputs(a)
    require(
        mapping == read(PARENT / "input_binding.json"),
        "wrong held-out section or engine",
    )
    bind_run(a, mapping)
    require(
        audit_primary(a) == read(PARENT / "results.json"),
        "retained joint trajectories changed",
    )


def check_step(r, u, previous, geom):
    require(
        (
            r["update"],
            r["epoch"],
            r["window_start"],
            r["window_end"],
            r["target_tokens"],
            r["mode"],
        )
        == (u, 0, u - 1, u, 63, "lora")
        and r["pre_lora_artifact_id"] == previous
        and r["pre_geometry_artifact_id"] == r["post_geometry_artifact_id"] == geom
        and isinstance(r["post_lora_artifact_id"], str)
        and math.isfinite(r["loss"]),
        "changed geometry, order, budget or LoRA lineage",
    )
    return r["post_lora_artifact_id"]


def first_match(r, reference):
    fields = (
        "loss",
        "target_tokens",
        "pre_lora_artifact_id",
        "pre_geometry_artifact_id",
        "post_lora_artifact_id",
    )
    require(
        all(r[k] == reference[k] for k in fields),
        "first gradient step differs from matched joint arm",
    )


def audit_fit(a, case, stages):
    seed, arm = case.split("-")
    init = read(a.out / "initial" / "receipt.json")
    ref = read(PARENT / "results.json")["fits"][case]
    geom = init["owners"][arm]
    lora = init["owners"]["lora"]
    initial = init["parameter_hashes"][arm]
    params = initial
    folder = a.run / case
    first_lora_difference = None
    parent_fitted = read(a.out / "fitted.json")["cases"][case]
    for start, stop in ((0, 256), (256, 512)):
        part = "first" if not start else "second"
        p = folder / (part + ".json")
        require(sha(p) == stages[part], "changed fit stage")
        rec = read(p)
        ck = folder / f"checkpoint-{stop}.json"
        b = rec["boundary"]
        require(
            rec["case"] == case
            and rec["claim_sha256"] == sha(a.run / "claim.json")
            and rec["previous"] == (None if not start else stages["first"])
            and rec["order_sha256"] == digest(ordering(int(seed), 512))
            and rec["before"] == params
            and rec["package"] == init["package"]
            and rec["steps_sha256"] == sha(folder / (part + ".jsonl")),
            "unbound stage",
        )
        checked_stage(a.out, case, part, parent_fitted[part])
        fixed_rows, old_rows = (
            rows(folder / (part + ".jsonl")),
            rows(a.out / case / (part + ".jsonl")),
        )
        require(len(fixed_rows) == len(old_rows) == 256, "incomplete half")
        for u, (r, joint) in enumerate(
            zip(fixed_rows, old_rows, strict=True), start + 1
        ):
            lora = check_step(r, u, lora, geom)
            if u == 1:
                first_match(r, joint)
            if first_lora_difference is None and lora != joint["post_lora_artifact_id"]:
                first_lora_difference = u
        state = read(ck)
        params = b["parameters"]
        require(
            "sha256:" + sha(ck) == b["checkpoint_id"]
            and state["schema"] == "spiraltorch.native_causal_lm_training_session.v1"
            and state["base_model_identity"] == ORIGINAL["base_id"]
            and state["readout_accumulator"] == "f32"
            and state["config"]
            == {
                "context_length": 64,
                "epochs": 1,
                "learning_rate": 0.01,
                "mode": "lora",
            }
            and state["updates"] == b["update"] == stop
            and state["data_id"] == b["recorded_data_id"] == ref["data_id"]
            and state["geometry_id"] == b["post_geometry_id"] == geom
            and state["lora_id"] == b["post_lora_id"] == lora
            and state["geometry_json"].encode()
            == (a.out / "initial" / (arm + ".json")).read_bytes()
            and set(params) == set(initial)
            and all(params[k] == initial[k] for k in initial.keys() - {"a", "b"}),
            "checkpoint does not preserve the matched frozen geometry/data",
        )
    require(
        all(params[k] != initial[k] for k in ("a", "b")),
        "terminal LoRA matrices did not change",
    )
    return {
        "data_id": ref["data_id"],
        "checkpoint_id": b["checkpoint_id"],
        "lora_id": lora,
        "geometry_id": geom,
        "first_joint_lora_difference": first_lora_difference,
        "step1_joint_loss_and_lora_match": True,
        "frozen_geometry_verified_updates": 512,
        "lora_matrices_changed": ["a", "b"],
        "stages": stages,
    }


def contrasts(scores):
    books = {}
    for book, sections in scores.items():
        means = {n: weighted([(s[n], 756) for s in sections.values()]) for n in STATES}
        seeds = {}
        for seed in SEEDS:
            ja, fa, jw, fw = (
                means[f"{seed}-{arm}:{mode}"]
                for arm in ("affine", "wave")
                for mode in ("joint", "fixed")
            )
            seeds[str(seed)] = {
                "joint_minus_fixed_affine": ja - fa,
                "joint_minus_fixed_wave": jw - fw,
                "fixed_wave_minus_fixed_affine": fw - fa,
                "joint_wave_minus_joint_affine": jw - ja,
                "interaction": (jw - fw) - (ja - fa),
            }
        books[book] = {
            "state_nll": means,
            "seeds": seeds,
            "mean_deltas": {
                k: math.fsum(s[k] for s in seeds.values()) / len(SEEDS)
                for k in seeds[str(SEEDS[0])]
            },
            "wave_learning_wins": sum(
                s["joint_minus_fixed_wave"] < 0 for s in seeds.values()
            ),
            "affine_learning_wins": sum(
                s["joint_minus_fixed_affine"] < 0 for s in seeds.values()
            ),
        }
    require(set(books) == set(BOOKS), "book roster")
    return books

"""Own a single bounded study; independently audit the preserved files without inference."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import (
    ARMS,
    BOOKS,
    CASES,
    KEYS,
    ORIGINAL,
    PLAN,
    SEEDS,
    STATES,
    digest,
    divergence,
    ordering,
    read,
    require,
    rows,
    save,
    sha,
    source_pins,
    weighted,
)
from joint_worker import checked_stage, paths, wheels
import owned_process


def audit_case(a, key, init, proof, decl_pin):
    seed, arm = key.split("-")
    mode = "lora" if arm == "lora" else "joint"
    lora, geom = (
        init["owners"]["lora"],
        init["owners"].get(arm) if arm != "lora" else None,
    )
    old_params = init["parameter_hashes"][arm]
    data = None
    for stage, start, end in (("first", 0, 256), ("second", 256, 512)):
        rec = checked_stage(a.out, key, stage, proof[stage])
        b = rec["boundary"]
        require(
            rec["declaration_sha256"] == decl_pin
            and rec["init_sha256"] == sha(a.out / "initial" / "receipt.json")
            and rec["package"] == init["package"]
            and rec["before_parameters"] == old_params
            and rec["previous_sha256"] == (None if start == 0 else proof["first"])
            and rec["order_sha256"] == digest(ordering(int(seed), 512)),
            "fit start/ancestry",
        )
        events = rows(a.out / key / (stage + ".jsonl"))
        require(len(events) == end - start, "incomplete half")
        for u, row in enumerate(events, start + 1):
            require(
                (
                    row["update"],
                    row["epoch"],
                    row["window_start"],
                    row["window_end"],
                    row["target_tokens"],
                    row["mode"],
                )
                == (u, 0, u - 1, u, 63, mode)
                and (row["pre_lora_artifact_id"], row["pre_geometry_artifact_id"])
                == (lora, geom)
                and math.isfinite(row["loss"]),
                "step lineage",
            )
            lora, geom = row["post_lora_artifact_id"], row["post_geometry_artifact_id"]
        ck = a.out / key / f"checkpoint-{end}.json"
        require(
            "sha256:" + sha(ck) == b["checkpoint_id"]
            and (b["update"], b["post_lora_id"], b["post_geometry_id"])
            == (end, lora, geom),
            "checkpoint/terminal receipt",
        )
        state = read(ck)
        require(
            state["schema"] == "spiraltorch.native_causal_lm_training_session.v1"
            and state["base_model_identity"] == ORIGINAL["base_id"]
            and state["updates"] == end
            and state["readout_accumulator"] == "f32"
            and state["data_id"] == b["recorded_data_id"]
            and state["config"]
            == {
                "context_length": 64,
                "epochs": 1,
                "learning_rate": PLAN["learning_rate"],
                "mode": mode,
            }
            and (state["lora_id"], state["geometry_id"]) == (lora, geom),
            "checkpoint fields",
        )
        require(not data or data == state["data_id"], "resume changed training data")
        data = state["data_id"]
        old_params = b["parameters"]
    changed = {k: v != old_params[k] for k, v in init["parameter_hashes"][arm].items()}
    return {
        "data_id": data,
        "checkpoint_id": b["checkpoint_id"],
        "lora_id": lora,
        "geometry_id": geom,
        "parameter_groups_changed": changed,
        "halfway_receipt_sha256": proof["first"],
        "terminal_receipt_sha256": proof["second"],
    }


def audit(a):
    decl = read(a.out / "declaration.json")
    dp = sha(a.out / "declaration.json")
    require(
        decl["source"] == source_pins()
        and decl["data_sha256"] == sha(a.data / "manifest.json"),
        "changed declaration",
    )
    manifest = read(a.data / "manifest.json")
    require(
        sha(a.data / "windows.json") == manifest["windows_sha256"]
        and sha(a.data / "prompts.json") == manifest["prompts_sha256"]
        and sha(a.old / "data" / "train.json") == PLAN["train_sha256"],
        "changed study inputs",
    )
    fitted = read(a.out / "fitted.json")
    require(
        fitted["init_sha256"] == sha(a.out / "initial" / "receipt.json")
        and fitted["declaration_sha256"] == dp
        and list(fitted["order"]) == KEYS
        and set(fitted["cases"]) == set(KEYS),
        "fit roster",
    )
    init = read(a.out / "initial" / "receipt.json")
    require(
        init["declaration_sha256"] == dp
        and init["package"]["wheel_sha256"] == decl["wheel_sha256"]
        and init["package"]["native_sha256"] == decl["native_sha256"]
        and init["logits"]["base"] == init["logits"]["lora"]
        and init["logits"]["affine"] == init["logits"]["wave"]
        and init["site_hashes"]["affine"] == init["site_hashes"]["wave"]
        and init["owners"]["affine"]
        == "sha256:" + sha(a.out / "initial" / "affine.json")
        and init["owners"]["wave"] == "sha256:" + sha(a.out / "initial" / "wave.json"),
        "initial controls",
    )
    cases = {k: audit_case(a, k, init, fitted["cases"][k], dp) for k in KEYS}
    for seed in SEEDS:
        require(
            len({cases[f"{seed}-{k}"]["data_id"] for k in ARMS}) == 1,
            "unmatched training order",
        )
    require(
        len({cases[f"{s}-lora"]["data_id"] for s in SEEDS}) == len(SEEDS),
        "seed orders not distinct",
    )
    scored = read(a.out / "scored.json")
    require(
        scored["fitted_sha256"] == sha(a.out / "fitted.json")
        and scored["states"] == STATES
        and scored["package"] == init["package"]
        and scored["declaration_sha256"] == dp,
        "scored roster",
    )
    owners = {"base": (None, None)}
    owners.update(
        {
            "initial-" + k: (
                init["owners"]["lora"],
                init["owners"][k] if k != "lora" else None,
            )
            for k in ARMS
        }
    )
    owners.update({k: (c["lora_id"], c["geometry_id"]) for k, c in cases.items()})
    table, paired, section_scores = {}, {}, {}
    for book in BOOKS:
        secs = manifest["offsets"][book]
        require(len(secs) == 6, "section roster")
        chapter = {}
        for k in secs:
            stem = f"{book}-{k}"
            f = a.out / "scores" / (stem + ".json")
            require(sha(f) == scored["sections"][stem], "changed section receipt")
            score = read(f)
            reports = score["evaluations"]
            require(
                score["shared_prefix_layers"] == 8
                and len(reports) == len(STATES)
                and len({(x["data_id"], x["plan_id"]) for x in reports}) == 1,
                "variant data or result count",
            )
            for s, v in zip(STATES, reports, strict=True):
                require(
                    v["base_model_identity"] == ORIGINAL["base_id"]
                    and (v.get("lora_artifact_id"), v["geometry_artifact_id"])
                    == owners[s]
                    and v["readout_accumulator"] == "f32"
                    and (
                        v["target_tokens"],
                        v["window_count"],
                        v["skipped_windows"],
                        v["ignored_targets"],
                    )
                    == (756, 12, 0, 0)
                    and math.isfinite(v["nll"]),
                    "score owner/coverage",
                )
            chapter[k] = {s: v["nll"] for s, v in zip(STATES, reports, strict=True)}
            require(
                chapter[k]["base"] == chapter[k]["initial-lora"]
                and chapter[k]["initial-affine"] == chapter[k]["initial-wave"],
                "initial scores",
            )
        section_scores[book] = chapter
        t = table[book] = {
            s: weighted([(ch[s], 756) for ch in chapter.values()]) for s in STATES
        }
        per_seed = {}
        for seed in SEEDS:
            l, a0, w = (t[f"{seed}-{k}"] for k in ARMS)
            per_seed[str(seed)] = {
                "wave_minus_affine": w - a0,
                "affine_minus_lora": a0 - l,
                "wave_minus_lora": w - l,
                "trained_minus_own_initial": {
                    k: t[f"{seed}-{k}"] - t["initial-" + k] for k in ARMS
                },
            }
        deltas = [r["wave_minus_affine"] for r in per_seed.values()]
        paired[book] = {
            "seeds": per_seed,
            "mean_wave_minus_affine": math.fsum(deltas) / len(deltas),
            "wave_wins": sum(d < 0 for d in deltas),
        }
    require(len(scored["sections"]) == 18, "extra/missing sections")
    gen = read(a.out / "generated.json")
    require(
        gen["scored_sha256"] == sha(a.out / "scored.json")
        and gen["declaration_sha256"] == dp
        and gen["package"] == init["package"],
        "generation lineage",
    )
    texts = gen["cases"]
    require(
        set(texts) == set(STATES)
        and texts["base"] == texts["initial-lora"]
        and texts["initial-affine"] == texts["initial-wave"],
        "generation controls",
    )
    compact = {}
    for k, items in texts.items():
        require(len(items) == 3, "prompt count")
        for item in items:
            require(
                0 < len(item["ids"]) <= 24
                and all(type(i) is int and 0 <= i < 50257 for i in item["ids"]),
                "generated tokens",
            )
        compact[k] = [
            {
                "ids_sha256": digest(x["ids"]),
                "tokens": len(x["ids"]),
                "finish_reason": x["reason"],
            }
            for x in items
        ]
    pairs = {
        str(s): [
            divergence(x["ids"], y["ids"])
            for x, y in zip(texts[f"{s}-affine"], texts[f"{s}-wave"], strict=True)
        ]
        for s in SEEDS
    }
    return {
        "schema": "spiraltorch.native_lm.joint_attention_transfer_result.v1",
        "declaration_sha256": dp,
        "wheel_sha256": decl["wheel_sha256"],
        "native_sha256": decl["native_sha256"],
        "scored_sha256": sha(a.out / "scored.json"),
        "generated_sha256": sha(a.out / "generated.json"),
        "fitted_sha256": sha(a.out / "fitted.json"),
        "updates": len(cases) * 512,
        "fresh_process_resumes": len(cases),
        "targets_per_fit": 32256,
        "targets_per_state_per_book": 4536,
        "shared_prefix_layers": 8,
        "scalar_receipts_checked": len(STATES),
        "states": STATES,
        "fits": cases,
        "book_nll": table,
        "section_nll": section_scores,
        "pairs": paired,
        "primary_all_seeds": paired["mansfield"]["wave_wins"] == len(SEEDS),
        "generated": compact,
        "first_joint_continuation_difference": pairs,
    }


def run(a):
    os.umask(0o077)
    start = time.monotonic()
    manifest = read(a.data / "manifest.json")
    require(
        sha(HERE / "plan.json") == a.plan_pin == manifest["plan_sha256"]
        and sha(a.data / "manifest.json") == a.data_pin
        and manifest["windows_sha256"] == sha(a.data / "windows.json")
        and manifest["prompts_sha256"] == sha(a.data / "prompts.json")
        and PLAN["train_sha256"] == sha(a.old / "data" / "train.json"),
        "unbound inputs",
    )
    inv = wheels.wheel_archive_inventory(a.wheel)
    require(
        inv["sha256"] == a.wheel_pin and inv["native"]["sha256"] == a.native_pin,
        "unbound engine",
    )
    a.out.mkdir(mode=0o700, exist_ok=False)
    save(
        a.out / "declaration.json",
        {
            "source": source_pins(),
            "plan_sha256": a.plan_pin,
            "data_sha256": a.data_pin,
            "wheel_sha256": a.wheel_pin,
            "native_sha256": a.native_pin,
            "order": KEYS,
            "limits": PLAN["limits"],
        },
    )
    dp = sha(a.out / "declaration.json")
    os.environ.update(
        RAYON_NUM_THREADS="4",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    current = None

    def call(stage, case=None, init=None, upstream=None):
        nonlocal current
        current = f"{case}-{stage}" if case else stage
        cmd = [
            str(a.python),
            "-I",
            "-B",
            str(HERE / "joint_worker.py"),
            stage,
            "--pin",
            dp,
        ]
        for k in ("out", "old", "model", "wheel", "data"):
            cmd.extend(["--" + k, str(getattr(a, k).resolve())])
        for k, v in (("case", case), ("init-pin", init), ("upstream", upstream)):
            if v is not None:
                cmd.extend(["--" + k, v])
        remaining = min(
            PLAN["limits"]["per_process_seconds"],
            start + PLAN["limits"]["total_seconds"] - time.monotonic(),
        )
        require(remaining > 0, "study deadline")
        with (
            (a.out / (current + ".stdout")).open("xb") as out,
            (a.out / (current + ".stderr")).open("xb") as err,
        ):
            job = owned_process.run(cmd, timeout=remaining, stdout=out, stderr=err)
        require(job.returncode == 0, "failed first worker: " + current)
        print("complete: " + current, flush=True)

    try:
        call("init")
        ip = sha(a.out / "initial" / "receipt.json")
        refs = {}
        for key in KEYS:
            call("first", key, ip)
            first = sha(a.out / key / "first.json")
            call("second", key, ip, first)
            refs[key] = {"first": first, "second": sha(a.out / key / "second.json")}
            audit_case(a, key, read(a.out / "initial" / "receipt.json"), refs[key], dp)
        save(
            a.out / "fitted.json",
            {"declaration_sha256": dp, "init_sha256": ip, "order": KEYS, "cases": refs},
        )
        call("score", upstream=sha(a.out / "fitted.json"))
        call("generate", upstream=sha(a.out / "scored.json"))
        current = "audit"
        save(a.out / "results.json", audit(a))
        save(
            a.out / "status.json",
            {"status": "complete", "results_sha256": sha(a.out / "results.json")},
        )
    except BaseException as e:
        save(
            a.out / "first-error.json",
            {"stage": current, "type": type(e).__name__, "message": str(e)},
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    p.add_argument("action", choices=["run", "audit"])
    p.add_argument("--python", type=Path)
    for k in ("plan-pin", "data-pin", "wheel-pin", "native-pin"):
        p.add_argument("--" + k)
    a = p.parse_args()
    if a.action == "run":
        run(a)
    else:
        summary = audit(a)
        require(
            summary == read(a.out / "results.json")
            and sha(a.out / "results.json")
            == read(a.out / "status.json")["results_sha256"],
            "final audit mismatch",
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))

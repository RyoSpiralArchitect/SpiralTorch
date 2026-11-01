"""Recompute the declared projection from complete saved fits and scores; no model execution."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import KEYS, ORIGINAL, PLAN, ROSTER, STOPS, criterion, read, require, sha  # noqa: E402
from common import weighted  # noqa: E402
from fit import f32, radii, source_pins  # noqa: E402


def pin(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def receipt(value, decl, digest):
    require(
        value["foreign_loaded"] is False
        and value["package"]["wheel_sha256"] == decl["wheel_sha256"]
        and value["package"]["native_sha256"] == decl["native_sha256"]
        and value["declaration_sha256"] == digest,
        "unbound native execution receipt",
    )


def audit_case(root, case, arm, decl):
    out = root / case
    row = read(out / "result.json")
    receipt(row, decl, sha(root / "declaration.json"))
    require(
        row["case"] == case
        and row["initial_geometry_id"] == decl["initial_ids"][arm]
        and sha(out / "steps.jsonl") == row["steps_sha256"]
        and sha(out / "observed-index.jsonl") == row["index_sha256"],
        "fit identity differs",
    )
    steps = [json.loads(x) for x in (out / "steps.jsonl").read_bytes().splitlines()]
    require(len(steps) == PLAN["updates"], "wrong fit length")
    previous = row["initial_geometry_id"]
    for u, step in enumerate(steps, 1):
        require(
            step["update"] == u
            and step["epoch"] == 0
            and step["window_start"] == 2 * u - 2
            and step["window_end"] == 2 * u
            and step["target_tokens"] == 126
            and step["microbatches"] == 1
            and math.isfinite(step["loss"])
            and step["pre_geometry_artifact_id"] == previous,
            "step sequence differs",
        )
        previous = step["post_geometry_artifact_id"]
    require(previous == row["geometry_id"], "wrong terminal geometry")
    entries = [
        json.loads(x) for x in (out / "observed-index.jsonl").read_bytes().splitlines()
    ]
    require(
        entries == row["stops"] and [x["updates"] for x in entries] == STOPS,
        "incomplete observed boundaries",
    )
    counts, maxima = [], []
    width = struct.unpack("f", struct.pack("f", PLAN["max_log_radius_drift"]))[0]
    paths = {f"transformer.h.{layer}.mlp" for layer in ORIGINAL["layers"]}
    for entry in entries:
        u = entry["updates"]
        require(
            entry["file"] == f"checkpoint-{u:06d}.json"
            and entry["observation_file"] == f"observation-{u:06d}.json",
            "unexpected receipt path",
        )
        raw = (out / entry["file"]).read_bytes()
        require(
            pin(raw) == entry["checkpoint_id"]
            and "sha256:" + sha(out / entry["observation_file"])
            == entry["observation_id"],
            "changed paired boundary",
        )
        obs = read(out / entry["observation_file"])
        require(
            obs["checkpoint"]
            == {k: entry[k] for k in ("updates", "file", "checkpoint_id")}
            and obs["observation"]["step"] == steps[u - 1]
            and obs["observation"]["data_id"] == row["data_id"]
            and obs["observation"]["base_model_identity"] == ORIGINAL["base_id"]
            and set(obs["observation"]["sites"]) == paths,
            "observation does not own update",
        )
        state = json.loads(raw)
        geometry = json.loads(state["geometry_json"])
        optimizer = json.loads(state["optimizer_json"])
        require(
            state["base_model_identity"] == ORIGINAL["base_id"]
            and state["updates"] == u
            and state["data_id"] == row["data_id"]
            and pin(state["geometry_json"].encode())
            == state["geometry_id"]
            == steps[u - 1]["post_geometry_artifact_id"]
            and pin(state["optimizer_json"].encode()) == state["optimizer_id"]
            and optimizer["geometry_artifact_id"] == state["geometry_id"]
            and optimizer["revision"] == u
            and optimizer["max_gradient_norm"] == ORIGINAL["max_gradient_norm"]
            and all(s["steps"] == u for s in optimizer["slots"]),
            "checkpoint/optimizer ownership differs",
        )
        current = radii(geometry)
        initial = decl["initial_radii_f32"]
        require(
            set(current) == paths and set(initial) == paths, "radius roster differs"
        )
        maximum = max(abs(current[k] - initial[k]) for k in paths)
        require(maximum == entry["max_abs_log_drift"], "recorded displacement differs")
        if arm == "bounded":
            policy = optimizer["radius_drift"]
            sites = {s["placement"]: s for s in policy["sites"]}
            require(
                optimizer["schema"] == "spiraltorch.native_geometry_adamw.v2"
                and len(policy["sites"]) == len(paths)
                and set(sites) == paths
                and all(
                    f32(sites[k]["initial_log_radius"]) == initial[k] for k in paths
                )
                and struct.pack("f", policy["max_delta"]) == struct.pack("f", width)
                and maximum <= width
                and all(0 <= s["projections"] <= u for s in sites.values()),
                "invalid projection state",
            )
            count = sum(s["projections"] for s in sites.values())
            require(
                count == entry["projection_count"]
                and (not counts or counts[-1] <= count),
                "projection count differs",
            )
        else:
            require(
                optimizer["schema"] == "spiraltorch.native_geometry_adamw.v1"
                and "radius_drift" not in optimizer
                and entry["projection_count"] is None
                and (arm != "fixed" or maximum == 0.0),
                "unexpected drift policy",
            )
            count = None
        counts.append(count)
        maxima.append(maximum)
    require(
        entries[-1]["checkpoint_id"] == row["checkpoint_id"],
        "terminal checkpoint differs",
    )
    return {
        "data_id": row["data_id"],
        "initial_geometry_id": row["initial_geometry_id"],
        "geometry_id": row["geometry_id"],
        "checkpoint_id": row["checkpoint_id"],
        "steps_sha256": row["steps_sha256"],
        "index_sha256": row["index_sha256"],
        "checkpoint_ids": [e["checkpoint_id"] for e in entries],
        "projection_counts": counts,
        "max_abs_log_drift_at_sampled_stops": max(maxima),
        "final_max_abs_log_drift": maxima[-1],
    }


def audit(root, *, finalized=True):
    require(not (root / "first-error.json").exists(), "first failure retained")
    if finalized:
        require(
            read(root / "status.json")
            == {"status": "complete", "results_sha256": sha(root / "results.json")},
            "incomplete study",
        )
    decl = read(root / "declaration.json")
    d = sha(root / "declaration.json")
    require(
        decl["plan_sha256"] == sha(HERE / "plan.json")
        and decl["source"] == source_pins()
        and decl["roster"] == KEYS
        and decl["train_sha256"] == PLAN["train_sha256"]
        and decl["retries"] == 0
        and decl["initial_ids"]["free"] == decl["initial_ids"]["bounded"],
        "changed declaration",
    )
    prep = read(root / "preflight.json")
    receipt(prep, decl, d)
    fits = read(root / "fitted.json")
    require(
        prep["updates"] == 0
        and prep["initial_ids"] == decl["initial_ids"]
        and len(prep["matched_train_logits"]) == 4
        and fits
        == {
            "order": KEYS,
            "preflight_sha256": sha(root / "preflight.json"),
            "cases": {k: sha(root / k / "result.json") for k in KEYS},
        },
        "unsealed fit roster",
    )
    fitrows = {f"{s}-{a}": audit_case(root, f"{s}-{a}", a, decl) for s, a in ROSTER}
    for s in PLAN["seeds"]:
        require(
            len({fitrows[f"{s}-{a}"]["data_id"] for a in PLAN["arms"]}) == 1,
            "unpaired training cases",
        )
    scored = read(root / "scored.json")
    receipt(scored, decl, d)
    variants = ["base", "initial", *KEYS]
    require(
        scored["cases"] == variants
        and scored["files"]
        == {k: sha(root / "scores" / (k + ".json")) for k in variants},
        "incomplete final scoring",
    )
    by_model = {}
    for k in variants:
        payload = read(root / "scores" / (k + ".json"))
        gid = (
            None
            if k == "base"
            else (
                decl["initial_ids"]["fixed"]
                if k == "initial"
                else fitrows[k]["geometry_id"]
            )
        )
        require(
            payload["geometry_id"] == gid
            and set(payload["scores"]) == set(PLAN["books"]),
            "wrong scored state",
        )
        scores = by_model[k] = payload["scores"]
        for book, result in scores.items():
            chapters = result["chapters"]
            require(
                set(chapters) == set(map(str, decl["chapters"][book]))
                and len(chapters) == 14
                and result["targets"] == 7056
                and all(
                    x["targets"] == 504
                    and math.isfinite(x["nll"])
                    and x["nll"] >= 0
                    and x["data_id"] == by_model["base"][book]["chapters"][i]["data_id"]
                    for i, x in chapters.items()
                )
                and weighted([(x["nll"], 504) for x in chapters.values()])
                == result["nll"],
                "unpaired score",
            )
    contrasts = {}
    for s in PLAN["seeds"]:

        def n(a, book):
            return by_model[f"{s}-{a}"][book]["nll"]

        contrasts[str(s)] = criterion(
            n("fixed", "emma"),
            n("free", "emma"),
            n("bounded", "emma"),
            n("free", "darwin"),
            n("bounded", "darwin"),
            fitrows[f"{s}-bounded"]["projection_counts"][-1],
        )
        contrasts[str(s)]["from_frozen_base"] = {
            a: {b: n(a, b) - by_model["base"][b]["nll"] for b in PLAN["books"]}
            for a in PLAN["arms"]
        }
    report = {
        "schema": "spiraltorch.native_lm.radius_drift_learning_results.v1",
        "plan_sha256": decl["plan_sha256"],
        "declaration_sha256": d,
        "fitted_sha256": sha(root / "fitted.json"),
        "wheel_sha256": decl["wheel_sha256"],
        "manifest_sha256": decl["manifest_sha256"],
        "fits": fitrows,
        "scores": by_model,
        "paired": contrasts,
        "all_seeds_joint_pass": all(c["joint_pass"] for c in contrasts.values()),
        "verified_updates": len(KEYS) * PLAN["updates"],
        "verified_observed_checkpoints": len(KEYS) * len(STOPS),
    }
    if finalized:
        require(report == read(root / "results.json"), "saved result differs")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(
        json.dumps(
            audit(p.parse_args().root), indent=2, sort_keys=True, allow_nan=False
        )
    )

"""Reconstruct the 18 original scalar/owner rosters, without a wheel, corpus or model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import (
    ARMS,
    BOOKS,
    PLAN,
    SEEDS,
    STATES,
    read,
    require,
    sha,
    weighted,
)
from joint_interchange import contrasts, returned
from freeze_binding import FREEZE_SHA256, frozen, match_rows


def wire(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def seal(v):
    return hashlib.sha256(wire(v).encode()).hexdigest()


def extract(out):
    roster = read(out / "scored.json")
    groups = {}
    owners = None
    fixed = {
        "schema",
        "base_model_identity",
        "readout_accumulator",
        "data_id",
        "plan_id",
        "config",
        "window_count",
        "skipped_windows",
        "microbatches",
        "target_tokens",
        "ignored_targets",
    }
    for k in roster["sections"]:
        r = read(out / "scores" / (k + ".json"))
        states = r["evaluations"]
        shared = {n: states[0][n] for n in fixed}
        ids = [
            {k: e[k] for k in ("geometry_artifact_id", "lora_artifact_id") if k in e}
            for e in states
        ]
        if owners is None:
            owners = ids
        require(
            ids == owners
            and all(all(e[n] == v for n, v in shared.items()) for e in states),
            "model roster or common section fields changed",
        )
        groups[k] = {
            "common": shared,
            "perplexities": [e["perplexity"] for e in states],
        }
    return {
        "schema": "spiraltorch.native_lm.joint_attention_public_scores.v1",
        "result_sha256": sha(out / "results.json"),
        "owners": owners,
        "groups": groups,
        "scored_receipt": roster,
    }


def check(r, public):
    require(
        public["schema"] == "spiraltorch.native_lm.joint_attention_public_scores.v1"
        and public["result_sha256"] == seal(r),
        "primary result digest",
    )
    roster = public["scored_receipt"]
    f = frozen()
    binding = read(HERE / "input_binding.json")
    require(
        binding["execution_sha256"] == FREEZE_SHA256
        and binding["windows_sha256"] == f["held_out_windows_sha256"]
        and r["wheel_sha256"]
        == roster["package"]["wheel_sha256"]
        == f["development_wheel_sha256"]
        and r["native_sha256"]
        == roster["package"]["native_sha256"]
        == f["native_sha256"],
        "not the frozen study engine/data",
    )
    keys = {
        f"{b}-{k}"
        for b, v in read(HERE / "execution.json")["sections"].items()
        for k in v
    }
    require(
        STATES == r["states"] == roster["states"]
        and seal(roster) == r["scored_sha256"]
        and set(public["groups"]) == set(roster["sections"]) == keys,
        "section/state roster",
    )
    for name, o in zip(STATES, public["owners"], strict=True):
        if name in r["fits"]:
            require(
                (o.get("lora_artifact_id"), o["geometry_artifact_id"])
                == (r["fits"][name]["lora_id"], r["fits"][name]["geometry_id"]),
                "trained owner",
            )
    for book in BOOKS:
        chapters = r["section_nll"][book]
        require(
            {int(k) for k in chapters}
            == set(read(HERE / "execution.json")["sections"][book]),
            "chapter roster",
        )
        for k, values in chapters.items():
            key = f"{book}-{k}"
            common = public["groups"][key]["common"]
            match_rows(binding["sections"][key], [common], 1)
            require(
                (
                    common["target_tokens"],
                    common["window_count"],
                    common["microbatches"],
                    common["skipped_windows"],
                    common["ignored_targets"],
                )
                == (756, 12, 6, 0, 0)
                and values["base"] == values["initial-lora"]
                and values["initial-affine"] == values["initial-wave"],
                "coverage or initial mismatch",
            )
            rows = [
                {**common, **o, "nll": values[n], "perplexity": p}
                for n, o, p in zip(
                    STATES,
                    public["owners"],
                    public["groups"][key]["perplexities"],
                    strict=True,
                )
            ]
            require(
                seal(
                    {
                        "schema": "spiraltorch.native_causal_lm_evaluation_comparison.v1",
                        "shared_prefix_layers": 8,
                        "evaluations": rows,
                    }
                )
                == roster["sections"][key],
                "section receipt changed",
            )
        mean = {n: weighted([(v[n], 756) for v in chapters.values()]) for n in STATES}
        require(mean == r["book_nll"][book], "book mean")
        deltas = []
        for s in SEEDS:
            l, a, w = (mean[f"{s}-{k}"] for k in ARMS)
            diff = w - a
            require(
                r["pairs"][book]["seeds"][str(s)]
                == {
                    "wave_minus_affine": diff,
                    "affine_minus_lora": a - l,
                    "wave_minus_lora": w - l,
                    "trained_minus_own_initial": {
                        k: mean[f"{s}-{k}"] - mean["initial-" + k] for k in ARMS
                    },
                },
                "paired effects",
            )
            deltas.append(diff)
        require(
            r["pairs"][book]["wave_wins"] == sum(d < 0 for d in deltas)
            and r["pairs"][book]["mean_wave_minus_affine"]
            == math.fsum(deltas) / len(SEEDS),
            "seed summary",
        )
    require(
        r["primary_all_seeds"] == (r["pairs"]["mansfield"]["wave_wins"] == len(SEEDS)),
        "primary criterion",
    )


def check_cross(primary, x):
    binding = read(HERE / "input_binding.json")
    order = [f"{s}-{l}{g}" for s in SEEDS for l in ("A", "W") for g in ("A", "W")]
    require(
        x["returned_order"] == order
        and x["input_binding_sha256"] == sha(HERE / "input_binding.json")
        and set(x["returned_score_projection_sha256"]) == set(binding["sections"]),
        "crossed projection roster",
    )
    require(
        x["primary_result_sha256"] == seal(primary)
        and x["plan_sha256"] == sha(HERE / "interchange_plan.json"),
        "wrong interchange inputs",
    )
    for s in SEEDS:
        for l, lk in (("A", "affine"), ("W", "wave")):
            for g, gk in (("A", "affine"), ("W", "wave")):
                require(
                    x["owners_lora_then_geometry"][f"{s}-{l}{g}"]
                    == [
                        primary["fits"][f"{s}-{lk}"]["lora_id"],
                        primary["fits"][f"{s}-{gk}"]["geometry_id"],
                    ],
                    "wrong crossed component",
                )
    for b in BOOKS:
        raw = x["section_nll"][b]
        for k, values in raw.items():
            key = f"{b}-{k}"
            expected = binding["sections"][key]
            rows = [
                {
                    **expected,
                    "lora_artifact_id": x["owners_lora_then_geometry"][n][0],
                    "geometry_artifact_id": x["owners_lora_then_geometry"][n][1],
                    "nll": values[n],
                }
                for n in order
            ]
            require(
                returned(order, x["owners_lora_then_geometry"], expected, rows)
                == x["returned_score_projection_sha256"][key],
                "altered crossed row attribution",
            )
        require(
            set(raw) == set(primary["section_nll"][b]), "different crossover sample"
        )
        means = {
            n: weighted([(v[n], 756) for v in raw.values()])
            for n in x["owners_lora_then_geometry"]
        }
        require(means == x["book_nll"][b], "crossover book mean")
        for s in SEEDS:
            for k, v in raw.items():
                require(
                    v[f"{s}-AA"] == primary["section_nll"][b][k][f"{s}-affine"]
                    and v[f"{s}-WW"] == primary["section_nll"][b][k][f"{s}-wave"],
                    "crossed anchor",
                )
            require(
                contrasts({p: means[f"{s}-{p}"] for p in ("AA", "AW", "WA", "WW")})
                == x["contrasts"][b][str(s)],
                "crossover contrasts",
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-private", type=Path)
    a = p.parse_args()
    primary = read(HERE / "results.json")
    doc = (
        extract(a.from_private) if a.from_private else read(HERE / "score_roster.json")
    )
    check(primary, doc)
    if a.from_private:
        sys.stdout.write(wire(doc))
    else:
        print(
            "234 NLL values, 18 original section receipt hashes, both matched starts and all contrasts verified"
        )
        if (HERE / "interchange_results.json").exists():
            check_cross(primary, read(HERE / "interchange_results.json"))
            print("108 crossover diagonals and 108 off-diagonals verified")

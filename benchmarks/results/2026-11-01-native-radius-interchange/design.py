"""Pairing, masking and projection; SpiralTorch owns geometry, decoding and NLL."""

from copy import deepcopy
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "2026-10-31-native-book-ablation"
sys.path.insert(0, str(PRIOR))
from common import PLAN as ORIGINAL, digest, read, require, weighted  # noqa: E402

PLAN = read(HERE / "plan.json")
PATHS = [f"transformer.h.{i}.mlp" for i in ORIGINAL["layers"]]
ROSTER = ["base", *(f"{s}-{c}" for s in PLAN["seeds"] for c in PLAN["cases_per_seed"])]
PROMPTS = [p["id"] for p in PLAN["prompts"]]
PAIRS = (("FL", "FF"), ("LL", "LF"), ("LL", "FF"), ("LL", "affine"))


def masked(prefix, whole):
    require(
        0 < len(prefix) < len(whole) <= PLAN["evaluation_context"]
        and whole[: len(prefix)] == prefix,
        "tokenization did not preserve an exact prompt boundary",
    )
    return [-100] * len(prefix) + whole[len(prefix) :]


def swap_record(receiver, donor):
    """Copy only scalar VALUES; never copy donor recipe, trainability or optimizer."""

    def split(state):
        require(
            state["base_model_identity"] == ORIGINAL["base_id"]
            and state["training"] is False
            and [p["path"] for p in state["placements"]] == PATHS,
            "wrong model, mode or placement roster",
        )
        return {k: v for k, v in state.items() if k != "placements"}, state[
            "placements"
        ]

    out = deepcopy(receiver)
    shell, sites = split(out)
    other, src = split(donor)
    require(shell == other, "incompatible geometry headers")
    changed = 0
    for site, donor_site in zip(sites, src, strict=True):

        def radius(p):
            recipe = dict(p["recipe"])
            learned = recipe.pop("learnable_radius")
            require(
                type(learned) is bool
                and recipe["schema"] == "spiraltorch.wave_gate_adapter.v2"
                and p["training"] is False,
                "not an inference WaveRadius site",
            )
            tensors = {t["name"]: t for t in p["tensors"]}
            require(
                len(p["tensors"]) == 3
                and set(tensors) == {"bias", "gate", "log_radius"},
                "unexpected tensors",
            )
            r = tensors["log_radius"]
            require(
                r["shape"] == []
                and len(r["values"]) == 1
                and math.isfinite(r["values"][0])
                and r["requires_grad"] is learned
                and r["role"] == ("parameter" if learned else "buffer"),
                "invalid radius owner",
            )
            layout = {
                k: {f: v for f, v in t.items() if f != "values"}
                for k, t in tensors.items()
                if k != "log_radius"
            }
            return recipe, layout, r

        recipe, layout, r = radius(site)
        src_recipe, src_layout, src_r = radius(donor_site)
        require(
            recipe == src_recipe and layout == src_layout,
            "incompatible recipient/donor recipes",
        )
        changed += digest(r["values"]) != digest(src_r["values"])
        r["values"] = list(src_r["values"])
    return out, changed


def divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def effects(scores):
    ff, fl, lf, ll = (scores[k] for k in ("FF", "FL", "LF", "LL"))
    return {
        "radius_on_fixed_recipient": fl - ff,
        "radius_on_learned_recipient": ll - lf,
        "other_at_fixed_radius": lf - ff,
        "other_at_learned_radius": ll - fl,
        "total_LL_minus_FF": ll - ff,
        "interaction": (ll - lf) - (fl - ff),
    }


def project(cases):
    require(
        set(cases) == set(ROSTER)
        and len(cases) * len(PROMPTS) == PLAN["budget"]["max_model_prompt_pairs"],
        "incomplete model matrix",
    )
    rows, seqs = {}, {}
    for key in ROSTER:
        case = cases[key]
        require(set(case["records"]) == set(PROMPTS), "incomplete prompt matrix")
        prompts, seqs[key] = {}, {}
        for pid in PROMPTS:
            row = case["records"][pid]
            ids = row["generated"]
            require(
                0 < len(ids) <= PLAN["generation"]["max_new_tokens"]
                and (
                    (row["finish"] == "eos" and ids[-1] == 50256)
                    or (
                        row["finish"] == "length"
                        and len(ids) == PLAN["generation"]["max_new_tokens"]
                        and 50256 not in ids
                    )
                ),
                "unfinished generation",
            )
            require(
                row["targets"] > 0 and math.isfinite(row["nll"]) and row["nll"] >= 0,
                "invalid target NLL",
            )
            seqs[key][pid] = ids
            prompts[pid] = {
                k: row[k]
                for k in ("nll", "targets", "data_id", "raw_sha256", "steps_sha256")
            }
            if key != "base":
                ref = rows["base"]["prompts"][pid]
                require(
                    (row["data_id"], row["targets"])
                    == (ref["data_id"], ref["targets"]),
                    "unpaired targets",
                )
        rows[key] = {
            "geometry_id": case["geometry_id"],
            "source_checkpoint_ids": case["source_checkpoint_ids"],
            "changed_radius_sites": case["changed_radius_sites"],
            "prompts": prompts,
            "target_weighted_nll": weighted(
                [(p["nll"], p["targets"]) for p in prompts.values()]
            ),
        }
    contrasts = {}
    for s in PLAN["seeds"]:
        scores = {
            c: rows[f"{s}-{c}"]["target_weighted_nll"] for c in PLAN["cases_per_seed"]
        }
        contrasts[str(s)] = {
            "nll": effects(scores),
            "first_free_generation_difference": {
                f"{left}-{right}": {
                    p: divergence(seqs[f"{s}-{left}"][p], seqs[f"{s}-{right}"][p])
                    for p in PROMPTS
                }
                for left, right in PAIRS
            },
        }
    return {
        "schema": "spiraltorch.native_book.radius_interchange_results.v1",
        "models": rows,
        "authored_targets_per_model": sum(
            p["targets"] for p in rows["base"]["prompts"].values()
        ),
        "generated_tokens": sum(len(ids) for d in seqs.values() for ids in d.values()),
        "paired": contrasts,
        "both_radius_deltas_negative_at_every_seed": all(
            x["nll"][k] < 0
            for x in contrasts.values()
            for k in ("radius_on_fixed_recipient", "radius_on_learned_recipient")
        ),
    }

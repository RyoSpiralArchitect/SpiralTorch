"""Crossover of two independently pinned adapter groups; never train or select a hybrid."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import (
    BOOKS,
    PLAN,
    SEEDS,
    guard_ml,
    read,
    require,
    save,
    sha,
    weighted,
)
from joint_worker import load_base, paths, wheels
from common import FORBIDDEN


def contrasts(v):
    aa, aw, wa, ww = (v[k] for k in ("AA", "AW", "WA", "WW"))
    return {
        "total": ww - aa,
        "lora": ((wa - aa) + (ww - aw)) / 2,
        "geometry": ((aw - aa) + (ww - wa)) / 2,
        "interaction": (ww - wa) - (aw - aa),
    }


def run(a):
    require(
        not a.receipt.exists()
        and sha(a.out / "results.json")
        == a.result_pin
        == read(a.out / "status.json")["results_sha256"],
        "requires the completed primary fit and a new output",
    )
    result = read(a.out / "results.json")
    decl = read(a.out / "declaration.json")
    require(
        sha(a.out / "declaration.json") == result["declaration_sha256"]
        and sha(a.data / "manifest.json") == decl["data_sha256"]
        and sha(a.data / "windows.json")
        == read(a.data / "manifest.json")["windows_sha256"],
        "changed primary inputs",
    )
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg["wheel_sha256"] == result["wheel_sha256"]
        and pkg["native_sha256"] == sha(st._rs.__file__) == result["native_sha256"],
        "not the primary scoring engine",
    )
    base = load_base(st, a.model)
    models, names, owners = [], [], {}
    for s in SEEDS:
        sources = {}
        for code, arm in (("A", "affine"), ("W", "wave")):
            name = f"{s}-{arm}"
            r = result["fits"][name]
            ck = (a.out / name / "checkpoint-512.json").read_text()
            sources[code] = st.NativeCausalLm.from_session_checkpoint_at(
                base,
                ck,
                r["checkpoint_id"],
                update=512,
                recorded_data_id=r["data_id"],
                post_lora_id=r["lora_id"],
                post_geometry_id=r["geometry_id"],
            )
        for l in ("A", "W"):
            for g in ("A", "W"):
                name = f"{s}-{l}{g}"
                model = base.with_lora_json(sources[l].lora_to_json()).with_geometry(
                    sources[g].geometry_artifact()
                )
                require(
                    model.lora_artifact_id == sources[l].lora_artifact_id
                    and model.geometry_artifact_id == sources[g].geometry_artifact_id,
                    "crossed wrong owners",
                )
                names.append(name)
                owners[name] = [model.lora_artifact_id, model.geometry_artifact_id]
                models.append(model)
    groups = read(a.data / "windows.json")
    chapters, means, components = {}, {}, {}
    for book in BOOKS:
        chapter = chapters[book] = {}
        for k, docs in groups[book].items():
            ev = st.NativeCausalLm.evaluate_variants(
                models, docs, 64, microbatch_windows=2
            )
            score = ev.evaluations
            require(
                ev.shared_prefix_layers == 8
                and len(score) == 12
                and len({x.data_id for x in score}) == 1
                and all(
                    x.target_tokens == 756
                    and x.window_count == 12
                    and x.skipped_windows == x.ignored_targets == 0
                    and math.isfinite(x.nll)
                    for x in score
                ),
                "crossed evaluation coverage",
            )
            values = chapter[k] = {n: x.nll for n, x in zip(names, score, strict=True)}
            for s in SEEDS:
                for label, arm in (("AA", "affine"), ("WW", "wave")):
                    require(
                        values[f"{s}-{label}"]
                        == result["section_nll"][book][k][f"{s}-{arm}"],
                        "diagonal did not reproduce primary score",
                    )
        by_state = means[book] = {
            n: weighted([(v[n], 756) for v in chapter.values()]) for n in names
        }
        components[book] = {}
        for s in SEEDS:
            c = contrasts({p: by_state[f"{s}-{p}"] for p in ("AA", "AW", "WA", "WW")})
            require(
                c["total"]
                == result["pairs"][book]["seeds"][str(s)]["wave_minus_affine"]
                and math.isclose(
                    c["geometry"] + c["lora"], c["total"], rel_tol=0, abs_tol=1e-14
                ),
                "contrast mismatch",
            )
            components[book][str(s)] = c
    require(
        not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules)
        and wheels.installed_identity(inv) == pkg,
        "runtime changed",
    )
    return {
        "schema": "spiraltorch.native_lm.attention_group_interchange.v1",
        "primary_result_sha256": a.result_pin,
        "plan_sha256": sha(HERE / "interchange_plan.json"),
        "wheel_sha256": result["wheel_sha256"],
        "owners_lora_then_geometry": owners,
        "section_nll": chapters,
        "book_nll": means,
        "contrasts": components,
        "diagonal_scores_reproduced": 108,
        "off_diagonal_scores": 108,
        "fit_updates": 0,
    }


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument("--result-pin", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    save(args.receipt, run(args))

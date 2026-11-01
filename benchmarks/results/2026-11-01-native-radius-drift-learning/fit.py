"""Isolated native workers. Each performs one preflight, one fit or final scoring."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import (  # noqa: E402
    KEYS,
    ORIGINAL,
    PLAN,
    PRIOR,
    STOPS,
    read,
    require,
    sha,
    save,
    train_windows,
)
from common import FORBIDDEN, guard_ml, ordering, weighted  # noqa: E402
from worker import load_base, logits_id  # noqa: E402
import byte_release_artifacts as wheels  # noqa: E402


def source_pins():
    return {
        p: sha(HERE / p)
        for p in ("plan.json", "protocol.py", "fit.py", "run_study.py", "audit.py")
    }


def initial_pin(initials):
    expected = read(PRIOR / "execution.json")["preflight_sha256"]
    require(sha(initials / "result.json") == expected, "original preflight changed")
    original = read(initials / "result.json")["initial"]
    return {a: original["fixed4" if a == "fixed" else "learned4"] for a in PLAN["arms"]}


def initial(st, base, a, arm, decl):
    name = "fixed4" if arm == "fixed" else "learned4"
    raw = (a.initials / (name + ".json")).read_bytes()
    return st.GeometryAdapterArtifact.from_json(
        raw.decode(), decl["initial_ids"][arm], base.base_model_identity
    )


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def radii(state):
    return {
        p["path"]: f32(
            next(t["values"][0] for t in p["tensors"] if t["name"] == "log_radius")
        )
        for p in state["placements"]
    }


def preflight(st, base, a, decl, train):
    artifacts = {arm: initial(st, base, a, arm, decl) for arm in PLAN["arms"]}
    require(
        artifacts["free"].artifact_id == artifacts["bounded"].artifact_id,
        "unmatched trainable initial artifacts",
    )
    traces = {}
    for arm, obj in artifacts.items():
        vals = radii(json.loads(obj.state_json()))
        require(
            list(vals) == [f"transformer.h.{layer}.mlp" for layer in ORIGINAL["layers"]]
            and all(
                math.isclose(math.exp(v), PLAN["initial_radius"], rel_tol=1e-7)
                for v in vals.values()
            ),
            "wrong initial radius/sites",
        )
        model = base.with_geometry(obj)
        traces[arm] = [logits_id(model, train[i]) for i in (0, 127, 255, 511)]
    require(
        traces["fixed"] == traces["free"] == traces["bounded"],
        "initial train probes differ",
    )
    return {
        "initial_ids": decl["initial_ids"],
        "matched_train_logits": traces["fixed"],
        "updates": 0,
    }


def train_case(st, base, a, decl, train):
    from spiraltorch.native_corpus_runs import step_and_save_observation

    case = f"{a.seed}-{a.arm}"
    require(case in KEYS, "unexpected fit")
    out = a.output / case
    out.mkdir(mode=0o700)
    art = initial(st, base, a, a.arm, decl)
    start = radii(json.loads(art.state_json()))
    options = {"max_gradient_norm": ORIGINAL["max_gradient_norm"]}
    if a.arm == "bounded":
        options["max_log_radius_drift"] = PLAN["max_log_radius_drift"]
    optimizer = st.NativeGeometryAdamW(art, **options)
    trainer = st.NativeCausalLmTrainer(
        base.with_geometry(art),
        optimizer,
        [train[i] for i in ordering(a.seed, len(train))],
        **ORIGINAL["recipe"],
    )
    require(
        trainer.total_updates == PLAN["updates"] and trainer.completed_updates == 0,
        "wrong training plan",
    )
    stops = []
    with (
        (out / "steps.jsonl").open("x", buffering=1) as log,
        (out / "observed-index.jsonl").open("x", buffering=1) as index,
    ):
        for u in range(1, trainer.total_updates + 1):
            if u in STOPS:
                step, entry = step_and_save_observation(trainer, out)
                current = radii(
                    json.loads(trainer.model.geometry_artifact().state_json())
                )
                maximum = max(abs(current[k] - v) for k, v in start.items())
                projections = trainer.optimizer.log_radius_projection_count
                if a.arm == "bounded":
                    require(
                        maximum <= trainer.optimizer.max_log_radius_drift
                        and projections is not None,
                        "observed radius escaped its original interval",
                    )
                else:
                    require(
                        projections is None and (a.arm != "fixed" or maximum == 0),
                        "unexpected projection/radius motion",
                    )
                info = {
                    **entry,
                    "projection_count": projections,
                    "max_abs_log_drift": maximum,
                }
                stops.append(info)
                index.write(json.dumps(info, allow_nan=False) + "\n")
            else:
                step = trainer.step()
            log.write(step.to_json() + "\n")
            require(
                step.update == u and step.target_tokens == 126,
                "wrong accepted update or target count",
            )
    require(
        trainer.exhausted and trainer.checkpoint_id == stops[-1]["checkpoint_id"],
        "incomplete trainer",
    )
    return {
        "case": case,
        "data_id": trainer.data_id,
        "initial_geometry_id": art.artifact_id,
        "geometry_id": trainer.model.geometry_artifact_id,
        "checkpoint_id": trainer.checkpoint_id,
        "steps_sha256": sha(out / "steps.jsonl"),
        "index_sha256": sha(out / "observed-index.jsonl"),
        "stops": stops,
    }


def score_all(st, base, a, decl):
    roster = read(a.output / "fitted.json")
    require(
        sha(a.output / "fitted.json") == a.fitted_pin and list(roster["order"]) == KEYS,
        "unsealed fit roster",
    )
    groups = read(a.data / "evaluation.json")
    require(
        sha(a.data / "evaluation.json") == decl["evaluation_sha256"],
        "evaluation bytes changed",
    )
    folder = a.output / "scores"
    folder.mkdir(mode=0o700)
    for name in ["base", "initial", *KEYS]:
        if name == "base":
            model = base
        elif name == "initial":
            model = base.with_geometry(initial(st, base, a, "fixed", decl))
        else:
            result = read(a.output / name / "result.json")
            require(
                sha(a.output / name / "result.json") == roster["cases"][name],
                "fit receipt changed",
            )
            last = result["stops"][-1]
            model = st.NativeCausalLm.from_training_checkpoint(
                base,
                (a.output / name / last["file"]).read_text(),
                result["checkpoint_id"],
            )
            require(
                model.geometry_artifact_id == result["geometry_id"],
                "wrong scoring model",
            )
        scores = {}
        for book, chapters in groups.items():
            rows = {}
            for k, docs in chapters.items():
                ev = model.evaluate_documents(docs, 64, microbatch_windows=2)
                require(
                    ev.target_tokens == 504
                    and ev.window_count == 8
                    and ev.skipped_windows == ev.ignored_targets == 0
                    and math.isfinite(ev.nll),
                    "wrong chapter evaluation",
                )
                rows[k] = {
                    "nll": ev.nll,
                    "targets": ev.target_tokens,
                    "data_id": ev.data_id,
                }
            scores[book] = {
                "chapters": rows,
                "targets": sum(x["targets"] for x in rows.values()),
                "nll": weighted([(r["nll"], r["targets"]) for r in rows.values()]),
            }
        save(
            folder / (name + ".json"),
            {"geometry_id": model.geometry_artifact_id, "scores": scores},
        )
    return {
        "cases": ["base", "initial", *KEYS],
        "files": {k: sha(folder / (k + ".json")) for k in ["base", "initial", *KEYS]},
    }


def work(a):
    decl = read(a.output / "declaration.json")
    require(
        sha(a.output / "declaration.json") == a.declaration_pin
        and decl["source"] == source_pins()
        and initial_pin(a.initials) == decl["initial_ids"]
        and sha(a.data / "manifest.json") == decl["manifest_sha256"],
        "worker inputs changed",
    )
    guard_ml()
    import spiraltorch as st

    inventory = wheels.wheel_archive_inventory(a.wheel)
    package = wheels.installed_identity(inventory)
    require(
        package["wheel_sha256"] == decl["wheel_sha256"]
        and package["native_sha256"] == decl["native_sha256"]
        and sha(st._rs.__file__) == package["native_sha256"]
        and package["wheel_sha256"] != ORIGINAL["wheel_sha256"],
        "wrong native wheel",
    )
    base = load_base(st, a.model)
    if a.task == "score":
        result = score_all(st, base, a, decl)
        path = a.output / "scored.json"
    else:
        train = train_windows(a.train)
        if a.task == "init":
            result = preflight(st, base, a, decl, train)
            path = a.output / "preflight.json"
        else:
            require(
                sha(a.output / "preflight.json") == a.preflight_pin, "unbound preflight"
            )
            result = train_case(st, base, a, decl, train)
            path = a.output / f"{a.seed}-{a.arm}" / "result.json"
    require(
        wheels.installed_identity(inventory) == package
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "package/runtime changed",
    )
    save(
        path,
        {
            **result,
            "declaration_sha256": a.declaration_pin,
            "package": package,
            "foreign_loaded": False,
        },
    )


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["init", "fit", "score"])
    for k in ("output", "train", "model", "wheel", "initials", "data"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--declaration-pin", required=True)
    p.add_argument("--preflight-pin")
    p.add_argument("--fitted-pin")
    p.add_argument("--seed", type=int, choices=PLAN["seeds"])
    p.add_argument("--arm", choices=PLAN["arms"])
    work(p.parse_args())

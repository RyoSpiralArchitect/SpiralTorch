"""Native workers and deterministic file checks for paired GPT-2 transfer."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sample import OLD, ORIGINAL, PLAN, read, require, save, sha
from common import FORBIDDEN, digest, guard_ml, ordering, weighted
from worker import artifact, load_base, logits_id, wheels

DRIFT = HERE.parent / "2026-11-01-native-radius-drift-learning"
C1, C2 = PLAN["cohorts"]["release"], PLAN["cohorts"]["development"]
ARMS = ("affine", "free")
TERMINAL = [f"{s}-{arm}" for s in C1 + C2 for arm in ARMS]
NAMES = ["base", "initial-affine", "initial-free", *TERMINAL]
BOOKS = list(PLAN["books"])


def source_pins():
    files = [
        HERE / n for n in ("plan.json", "sample.py", "transfer.py", "run_transfer.py")
    ]
    files += [
        OLD / n
        for n in (
            "common.py",
            "plan.json",
            "worker.py",
            "results.json",
            "execution.json",
        )
    ]
    files += [
        DRIFT / "results.json",
        HERE.parents[2] / "tools" / "byte_release_artifacts.py",
        HERE.parents[2] / "tools" / "owned_process.py",
    ]
    return {str(p.relative_to(HERE.parents[2])): sha(p) for p in files}


def initial_receipt(old):
    folder = old / "before" / "preflight"
    require(
        sha(folder / "result.json") == read(OLD / "execution.json")["preflight_sha256"],
        "initial receipt",
    )
    receipt = read(folder / "result.json")
    for name in ("affine", "learned4"):
        require(
            "sha256:" + sha(folder / (name + ".json")) == receipt["initial"][name],
            "initial bytes",
        )
    return receipt


def location(a, case):
    seed, arm = case.split("-")
    require(case in TERMINAL, "unexpected case")
    if int(seed) in C1:
        return (
            a.old
            / "run"
            / f"{seed}-{'affine' if arm == 'affine' else 'learned4'}"
            / "checkpoint.json"
        )
    folder = a.drift if arm == "free" else None
    return (folder / "run" if folder else a.out) / case / "checkpoint-000256.json"


def check_fit(path, record, start):
    pin = "sha256:" + sha(path)
    state = read(path)
    require(
        pin == record["checkpoint_id"]
        and state["updates"] == 256
        and state["config"] == ORIGINAL["recipe"]
        and state["base_model_identity"] == ORIGINAL["base_id"]
        and state["geometry_id"] == record["geometry_id"]
        and state["data_id"] == record["data_id"],
        "terminal state mismatch",
    )
    opt = json.loads(state["optimizer_json"])
    require(
        opt["schema"] == "spiraltorch.native_geometry_adamw.v1"
        and opt["revision"] == 256
        and opt["geometry_artifact_id"] == state["geometry_id"]
        and (opt["beta1"], opt["beta2"], opt["epsilon"], opt["max_gradient_norm"])
        == (0.9, 0.999, 1e-8, ORIGINAL["max_gradient_norm"]),
        "wrong AdamW policy, owner or revision",
    )
    log = path.parent / "steps.jsonl"
    require(sha(log) == record["steps_sha256"], "changed step stream")
    steps = [json.loads(line) for line in log.read_bytes().splitlines()]
    require(len(steps) == 256, "incomplete step stream")
    geometry = start
    for i, row in enumerate(steps):
        require(
            row["update"] == i + 1
            and row["pre_geometry_artifact_id"] == geometry
            and row["target_tokens"] == 126
            and row["microbatches"] == 1
            and row["window_start"] == 2 * i
            and row["window_end"] == 2 * i + 2
            and row["epoch"] == 0
            and math.isfinite(row["loss"]),
            "step lineage or budget",
        )
        geometry = row["post_geometry_artifact_id"]
    require(geometry == state["geometry_id"], "terminal step lineage")
    if path.name != "checkpoint.json":
        index = path.parent / "observed-index.jsonl"
        require(sha(index) == record["index_sha256"], "changed observation index")
        entries = [json.loads(line) for line in index.read_bytes().splitlines()]
        require(
            [r["updates"] for r in entries] == list(range(32, 257, 32)),
            "observation cadence",
        )
        for e in entries:
            u = e["updates"]
            require(
                e["file"] == f"checkpoint-{u:06}.json"
                and e["observation_file"] == f"observation-{u:06}.json"
                and "sha256:" + sha(path.parent / e["file"]) == e["checkpoint_id"]
                and "sha256:" + sha(path.parent / e["observation_file"])
                == e["observation_id"],
                "observed pair changed",
            )
        require(entries[-1]["checkpoint_id"] == pin, "wrong final observation")
    return {
        k: record[k]
        for k in ("checkpoint_id", "geometry_id", "data_id", "steps_sha256")
    }


def archived(a, initial):
    for folder, proof in [(a.old, OLD), (a.drift, DRIFT)]:
        require(
            sha(folder / "run" / "fitted.json")
            == read(proof / "results.json")["fitted_sha256"],
            "historical seal",
        )
    old = read(a.old / "run" / "fitted.json")["cases"]
    recent = read(a.drift / "run" / "fitted.json")["cases"]
    rows = {}
    for seed in C1 + C2:
        for arm, name in [("affine", "affine"), ("free", "learned4")]:
            case = f"{seed}-{arm}"
            if seed in C1:
                record = old[f"{seed}-{name}"]
                require(
                    record["seed"] == seed
                    and record["arm"] == name
                    and record["order_sha256"] == digest(ordering(seed, 512)),
                    "historical order",
                )
            elif arm == "free":
                receipt = a.drift / "run" / case / "result.json"
                require(sha(receipt) == recent[case], "archived free fit receipt")
                record = read(receipt)
                require(
                    record["case"] == case
                    and record["initial_geometry_id"] == initial[name]
                    and record["package"]["wheel_sha256"]
                    == PLAN["development_wheel_sha256"]
                    and record["package"]["native_sha256"]
                    == PLAN["development_native_sha256"],
                    "free fit provenance",
                )
            else:
                continue
            rows[case] = check_fit(location(a, case), record, initial[name])
    return rows


def arm(st, base, a, name, initial):
    return artifact(
        st, base, a.old / "before" / "preflight" / (name + ".json"), initial[name]
    )


def fit(st, base, a, decl):
    from spiraltorch.native_corpus_runs import step_and_save_observation

    case = a.stage
    seed = int(case.split("-")[0])
    path = location(a, case)
    path.parent.mkdir(mode=0o700, exist_ok=False)
    train_path = a.old / "data" / "train.json"
    require(sha(train_path) == PLAN["train_sha256"], "training windows changed")
    train = read(train_path)
    art = arm(st, base, a, "affine", decl["initial"])
    trainer = st.NativeCausalLmTrainer(
        base.with_geometry(art),
        st.NativeGeometryAdamW(art, max_gradient_norm=ORIGINAL["max_gradient_norm"]),
        [train[i] for i in ordering(seed, len(train))],
        **ORIGINAL["recipe"],
    )
    require(
        trainer.data_id == decl["archived"][f"{seed}-free"]["data_id"]
        and trainer.total_updates == 256
        and trainer.completed_updates == 0,
        "unmatched training data/order",
    )
    with (
        (path.parent / "steps.jsonl").open("x", buffering=1) as log,
        (path.parent / "observed-index.jsonl").open("x", buffering=1) as ix,
    ):
        for u in range(1, 257):
            if u % 32 == 0:
                step, entry = step_and_save_observation(trainer, path.parent)
                ix.write(json.dumps(entry, allow_nan=False) + "\n")
            else:
                step = trainer.step()
            require(step.update == u and step.target_tokens == 126, "wrong update")
            log.write(step.to_json() + "\n")
    require(trainer.exhausted, "trainer not exhausted")
    record = {
        "checkpoint_id": trainer.checkpoint_id,
        "data_id": trainer.data_id,
        "geometry_id": trainer.model.geometry_artifact_id,
        "steps_sha256": sha(path.parent / "steps.jsonl"),
        "index_sha256": sha(path.parent / "observed-index.jsonl"),
    }
    check_fit(path, record, art.artifact_id)
    return path.parent / "result.json", record


def score(st, base, a, decl, roster):
    groups = read(a.data / "windows.json")
    folder = a.out / "scores"
    folder.mkdir(mode=0o700, exist_ok=False)
    seals = {}
    first = None
    for name in NAMES:
        if name == "base":
            model = base
        elif name.startswith("initial-"):
            key = "affine" if name.endswith("affine") else "learned4"
            model = base.with_geometry(arm(st, base, a, key, decl["initial"]))
        else:
            pin = roster[name]
            model = st.NativeCausalLm.from_training_checkpoint(
                base, location(a, name).read_text(), pin["checkpoint_id"]
            )
            require(
                model.geometry_artifact_id == pin["geometry_id"],
                "wrong terminal geometry",
            )
        scores = {}
        for book in BOOKS:
            sections = {}
            for section, docs in groups[book].items():
                ev = model.evaluate_documents(docs, 64, microbatch_windows=2)
                require(
                    ev.target_tokens == 756
                    and ev.window_count == 12
                    and ev.skipped_windows == ev.ignored_targets == 0
                    and math.isfinite(ev.nll),
                    "invalid objective/denominator",
                )
                sections[section] = {
                    "nll": ev.nll,
                    "targets": ev.target_tokens,
                    "data_id": ev.data_id,
                }
            require(len(sections) == 6, "missing section")
            scores[book] = {
                "sections": sections,
                "targets": 4536,
                "nll": weighted([(r["nll"], r["targets"]) for r in sections.values()]),
            }
        if name == "initial-affine":
            first = scores
        if name == "initial-free":
            require(
                first == scores,
                "initial held-out functions differ; stop before terminal scoring",
            )
        path = folder / (name + ".json")
        save(
            path,
            {
                "model": name,
                "geometry_id": model.geometry_artifact_id,
                "scores": scores,
            },
        )
        seals[name] = sha(path)
    return a.out / "scored.json", {"files": seals}


def work(a):
    decl = read(a.out / "declaration.json")
    require(
        sha(a.out / "declaration.json") == a.pin
        and source_pins() == decl["source"]
        and sha(a.data / "windows.json") == decl["windows_sha256"],
        "unbound evaluation/declaration",
    )
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg["wheel_sha256"] == PLAN["development_wheel_sha256"]
        and pkg["native_sha256"] == PLAN["development_native_sha256"]
        and pkg["all_original_files_match"]
        and sha(st._rs.__file__) == pkg["native_sha256"],
        "wrong installed engine",
    )
    base = load_base(st, a.model)
    if a.stage == "preflight":
        receipt = initial_receipt(a.old)
        train = read(a.old / "data" / "train.json")
        ids = {}
        for key in ("affine", "learned4"):
            m = base.with_geometry(arm(st, base, a, key, decl["initial"]))
            ids[key] = [logits_id(m, train[i]) for i in (0, 170, 341, 511)]
        require(ids["affine"] == ids["learned4"], "unmatched initial logits")
        path, row = (
            a.out / "preflight.json",
            {
                "probes": ids,
                "equal_to_original_binary": ids["affine"]
                == receipt["train_probe_logits"]["affine"],
            },
        )
    elif a.stage == "score":
        require(sha(a.out / "fitted.json") == a.stage_pin, "fit roster changed")
        path, row = score(st, base, a, decl, read(a.out / "fitted.json")["cases"])
    else:
        require(
            sha(a.out / "preflight.json") == a.stage_pin,
            "initial matching preflight changed",
        )
        path, row = fit(st, base, a, decl)
    require(
        pkg == wheels.installed_identity(inv)
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "runtime changed",
    )
    save(path, {**row, "declaration_sha256": a.pin, "package": pkg})


def paths(p):
    for name in ("out", "data", "old", "drift", "model", "wheel"):
        p.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    p.add_argument(
        "stage", choices=["preflight", "score", *(f"{s}-affine" for s in C2)]
    )
    p.add_argument("--pin", required=True)
    p.add_argument("--stage-pin")
    work(p.parse_args())

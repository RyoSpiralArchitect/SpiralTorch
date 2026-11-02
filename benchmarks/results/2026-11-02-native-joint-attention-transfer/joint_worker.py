"""Isolated native SGD, pinned inference and greedy decode; no Python learning math."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import (
    ARMS,
    BOOKS,
    CASES,
    KEYS,
    OLD,
    ORIGINAL,
    PLAN,
    STATES,
    digest,
    guard_ml,
    ordering,
    read,
    require,
    save,
    sha,
    source_pins,
)
from common import FORBIDDEN
from worker import artifact, load_base, logits_id, put_payload, wheels


def inputs(a):
    decl = read(a.out / "declaration.json")
    require(
        sha(a.out / "declaration.json") == a.pin
        and decl["source"] == source_pins()
        and decl["data_sha256"] == sha(a.data / "manifest.json"),
        "declaration/source changed",
    )
    return decl


def initial_model(st, base, out, name, pins):
    m = base.with_lora(PLAN["layer"], PLAN["rank"], PLAN["alpha"])
    require(m.lora_artifact_id == pins["lora"], "wrong deterministic LoRA")
    if name != "lora":
        m = m.with_geometry(artifact(st, base, out / (name + ".json"), pins[name]))
    return m


def init(st, base, a, train):
    old = a.old / "before" / "preflight"
    require(
        sha(old / "result.json") == read(OLD / "execution.json")["preflight_sha256"],
        "wrong original calibration",
    )
    original = read(old / "result.json")["initial"]
    folder = a.out / "initial"
    folder.mkdir(mode=0o700)
    lora = base.with_lora(PLAN["layer"], PLAN["rank"], PLAN["alpha"])
    record = json.loads(lora.lora_to_json())
    require(
        len(record["a"]) == len(record["b"]) == 3072 and not any(record["b"]),
        "wrong LoRA initialization",
    )
    pins = {"lora": lora.lora_artifact_id}
    site_ids = {}
    for arm, key in [("affine", "affine"), ("wave", "learned4")]:
        src = artifact(st, base, old / (key + ".json"), original[key])
        doc = json.loads(src.state_json())
        places = [
            p
            for p in doc["placements"]
            if p["path"] == f"transformer.h.{PLAN['layer']}.mlp"
        ]
        require(len(places) == 1, "missing original calibrated site")
        place = places[0]
        place["path"] = PLAN["site"]
        doc["placements"] = [place]
        g = st.GeometryAdapterArtifact.create_json(json.dumps(doc))
        pins[arm] = g.artifact_id
        put_payload(folder / (arm + ".json"), g.to_json(), g.artifact_id)
        runtime = st.NativeGeometryRuntime(g)
        probe = [((i % 31) - 15) / 8 for i in range(2 * 768)]
        site_ids[arm] = hashlib.sha256(
            runtime.forward(PLAN["site"], probe, 2).output_buffer()
        ).hexdigest()
    states = {
        "base": base,
        **{k: initial_model(st, base, folder, k, pins) for k in ARMS},
    }
    probes = {
        k: [logits_id(m, train[i]) for i in (0, 170, 341, 511)]
        for k, m in states.items()
    }
    require(
        site_ids["affine"] == site_ids["wave"]
        and probes["lora"] == probes["base"]
        and probes["affine"] == probes["wave"]
        and all(x != y for x, y in zip(probes["lora"], probes["wave"])),
        "initial function controls failed",
    )
    return folder / "receipt.json", {
        "owners": pins,
        "site_hashes": site_ids,
        "logits": probes,
        "train_updates": 0,
        "parameter_hashes": {k: params(states[k]) for k in ARMS},
    }


def params(model):
    l = json.loads(model.lora_to_json())
    out = {k: digest(l[k]) for k in ("a", "b")}
    g = model.geometry_artifact()
    if g is not None:
        for t in json.loads(g.state_json())["placements"][0]["tensors"]:
            out[t["name"]] = hashlib.sha256(
                struct.pack("<" + "f" * len(t["values"]), *t["values"])
            ).hexdigest()
    return out


def boundary(session, path):
    payload, pin = session.checkpoint_with_id()
    put_payload(path, payload, pin)
    m = session.model
    return {
        "checkpoint_id": pin,
        "update": session.completed_updates,
        "recorded_data_id": session.data_id,
        "post_lora_id": m.lora_artifact_id,
        "post_geometry_id": m.geometry_artifact_id,
        "parameters": params(m),
    }


def checked_stage(out, case, stage, pin):
    p = out / case / (stage + ".json")
    require(sha(p) == pin, "unbound fit stage")
    r = read(p)
    require(
        r["case"] == case
        and r["stage"] == stage
        and sha(p.parent / (stage + ".jsonl")) == r["steps_sha256"],
        "wrong fit stage",
    )
    return r


def train_half(st, base, a, train):
    require(a.case in KEYS, "unknown fit")
    seed, key = a.case.split("-")
    init_file = a.out / "initial" / "receipt.json"
    require(sha(init_file) == a.init_pin, "unbound initial receipt")
    pins = read(init_file)["owners"]
    mode = "lora" if key == "lora" else "joint"
    order = ordering(int(seed), len(train))
    docs = [train[i] for i in order]
    folder = a.out / a.case
    if a.stage == "first":
        folder.mkdir(mode=0o700)
        model = initial_model(st, base, a.out / "initial", key, pins)
        session = st.NativeCausalLmTrainingSession(
            model,
            docs,
            PLAN["context"],
            learning_rate=PLAN["learning_rate"],
            mode=mode,
            epochs=1,
        )
        start, stop = 0, 256
    else:
        prev = checked_stage(a.out, a.case, "first", a.upstream)
        require(
            prev["declaration_sha256"] == a.pin and prev["boundary"]["update"] == 256,
            "wrong halfway ancestry",
        )
        session = st.NativeCausalLmTrainingSession.from_json(
            base,
            docs,
            (folder / "checkpoint-256.json").read_text(),
            prev["boundary"]["checkpoint_id"],
        )
        require(
            session.checkpoint_id == prev["boundary"]["checkpoint_id"]
            and session.data_id == prev["boundary"]["recorded_data_id"]
            and session.model.lora_artifact_id == prev["boundary"]["post_lora_id"]
            and session.model.geometry_artifact_id
            == prev["boundary"]["post_geometry_id"],
            "fresh-process resume changed",
        )
        start, stop = 256, 512
    require(
        session.completed_updates == start and session.total_updates == 512,
        "SGD update budget",
    )
    before = params(session.model)
    l, g = session.model.lora_artifact_id, session.model.geometry_artifact_id
    log = folder / (a.stage + ".jsonl")
    with log.open("x", buffering=1) as stream:
        for u in range(start + 1, stop + 1):
            report = session.step()
            raw = report.to_json()
            stream.write(raw + "\n")
            row = json.loads(raw)
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
                and row["pre_lora_artifact_id"] == l
                and row["pre_geometry_artifact_id"] == g
                and math.isfinite(row["loss"]),
                "step continuity",
            )
            l, g = row["post_lora_artifact_id"], row["post_geometry_artifact_id"]
    b = boundary(session, folder / f"checkpoint-{stop}.json")
    require(
        (b["update"], b["post_lora_id"], b["post_geometry_id"], session.exhausted)
        == (stop, l, g, stop == 512),
        "terminal owner/cursor",
    )
    return folder / (a.stage + ".json"), {
        "case": a.case,
        "stage": a.stage,
        "init_sha256": a.init_pin,
        "previous_sha256": a.upstream,
        "order_sha256": digest(order),
        "before_parameters": before,
        "steps_sha256": sha(log),
        "boundary": b,
    }


def score(st, base, a):
    require(sha(a.out / "fitted.json") == a.upstream, "unsealed fitted roster")
    roster = read(a.out / "fitted.json")
    init_file = a.out / "initial" / "receipt.json"
    require(
        roster["init_sha256"] == sha(init_file) and set(roster["cases"]) == set(KEYS),
        "fit roster",
    )
    pins = read(init_file)["owners"]
    models = [
        base,
        *[initial_model(st, base, a.out / "initial", k, pins) for k in ARMS],
    ]
    for key in KEYS:
        row = checked_stage(a.out, key, "second", roster["cases"][key]["second"])
        b = row["boundary"]
        require(
            b["update"] == 512 and row["declaration_sha256"] == a.pin,
            "nonterminal scoring owner",
        )
        model = st.NativeCausalLm.from_session_checkpoint_at(
            base,
            (a.out / key / "checkpoint-512.json").read_text(),
            b["checkpoint_id"],
            **{
                k: b[k]
                for k in (
                    "update",
                    "recorded_data_id",
                    "post_lora_id",
                    "post_geometry_id",
                )
            },
        )
        require(
            (model.geometry_artifact_id, model.lora_artifact_id)
            == (b["post_geometry_id"], b["post_lora_id"]),
            "restored model owners",
        )
        models.append(model)
    manifest = read(a.data / "manifest.json")
    require(
        manifest["windows_sha256"] == sha(a.data / "windows.json")
        and manifest["prompts_sha256"] == sha(a.data / "prompts.json"),
        "scoring inputs changed",
    )
    groups = read(a.data / "windows.json")
    folder = a.out / "scores"
    folder.mkdir(mode=0o700)
    sections = {}
    for book in BOOKS:
        for section in sorted(groups[book], key=int):
            docs = groups[book][section]
            pair = st.NativeCausalLm.evaluate_variants(
                models, docs, 64, microbatch_windows=2
            )
            ev = pair.evaluations
            require(
                pair.shared_prefix_layers == PLAN["layer"]
                and len(ev) == len(STATES)
                and len({v.data_id for v in ev}) == 1
                and all(
                    v.target_tokens == 756
                    and v.window_count == 12
                    and v.ignored_targets == v.skipped_windows == 0
                    and math.isfinite(v.nll)
                    for v in ev
                )
                and ev[0].nll == ev[1].nll
                and ev[2].nll == ev[3].nll,
                "held-out denominator/initial control/shared depth",
            )
            if not sections:
                require(
                    [v.to_json() for v in ev]
                    == [
                        m.evaluate_documents(docs, 64, microbatch_windows=2).to_json()
                        for m in models
                    ],
                    "first chapter scalar mismatch",
                )
            p = folder / f"{book}-{section}.json"
            save(p, json.loads(pair.to_json()))
            sections[p.stem] = sha(p)
    return a.out / "scored.json", {
        "fitted_sha256": a.upstream,
        "sections": sections,
        "states": STATES,
    }


def generate(st, base, a):
    require(sha(a.out / "scored.json") == a.upstream, "generation before scoring")
    scored = read(a.out / "scored.json")
    require(
        scored["fitted_sha256"] == sha(a.out / "fitted.json"),
        "scored fit roster changed",
    )
    roster = read(a.out / "fitted.json")
    require(
        roster["init_sha256"] == sha(a.out / "initial" / "receipt.json"),
        "unbound initial generation models",
    )
    pins = read(a.out / "initial" / "receipt.json")["owners"]
    require(
        sha(a.data / "prompts.json")
        == read(a.data / "manifest.json")["prompts_sha256"],
        "generation prompts changed",
    )
    prompts = read(a.data / "prompts.json")
    cases = {}
    for name in STATES:
        if name == "base":
            model = base
        elif name.startswith("initial-"):
            model = initial_model(st, base, a.out / "initial", name[8:], pins)
        else:
            r = checked_stage(a.out, name, "second", roster["cases"][name]["second"])
            b = r["boundary"]
            model = st.NativeCausalLm.from_session_checkpoint_at(
                base,
                (a.out / name / "checkpoint-512.json").read_text(),
                b["checkpoint_id"],
                **{
                    k: b[k]
                    for k in (
                        "update",
                        "recorded_data_id",
                        "post_lora_id",
                        "post_geometry_id",
                    )
                },
            )
        items = []
        for ids in prompts:
            gen = st.NativeCausalLmGenerator(model, ids, json.dumps(PLAN["generation"]))
            while not gen.finished:
                gen.step()
            items.append({"ids": list(gen.generated_ids), "reason": gen.finish_reason})
        cases[name] = items
    require(
        cases["base"] == cases["initial-lora"]
        and cases["initial-affine"] == cases["initial-wave"],
        "generation initial controls",
    )
    return a.out / "generated.json", {"cases": cases, "scored_sha256": a.upstream}


def paths(p):
    for k in ("out", "old", "model", "wheel", "data"):
        p.add_argument("--" + k, type=Path, required=True)


def work(a):
    decl = inputs(a)
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg["wheel_sha256"] == decl["wheel_sha256"]
        and pkg["native_sha256"] == decl["native_sha256"] == sha(st._rs.__file__)
        and pkg["all_original_files_match"],
        "installed engine differs",
    )
    base = load_base(st, a.model)
    if a.stage in ("score", "generate"):
        path, result = (score if a.stage == "score" else generate)(st, base, a)
    else:
        t = a.old / "data" / "train.json"
        require(sha(t) == PLAN["train_sha256"], "training input changed")
        train = read(t)
        path, result = (init if a.stage == "init" else train_half)(st, base, a, train)
    require(
        wheels.installed_identity(inv) == pkg
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "runtime changed",
    )
    save(path, {**result, "declaration_sha256": a.pin, "package": pkg})


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    p.add_argument("stage", choices=["init", "first", "second", "score", "generate"])
    p.add_argument("--case", choices=KEYS)
    p.add_argument("--pin", required=True)
    p.add_argument("--init-pin")
    p.add_argument("--upstream")
    work(p.parse_args())

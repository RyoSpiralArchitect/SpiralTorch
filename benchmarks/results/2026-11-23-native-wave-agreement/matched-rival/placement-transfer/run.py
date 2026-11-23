"""Prepare an immutable plan, or execute it once using the pinned CPU extension."""
import argparse
import base64
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import types

d = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
a = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("analyze.py"))))
need = d.need


def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("offline study")
    if event == "import" and args[0].split(".")[0] in {
            "torch", "numpy", "safetensors", "transformers", "huggingface_hub", "accelerate"}:
        raise RuntimeError("native-only study: " + args[0])


def models(st, arms):
    data = {}
    for name, pin in d.DATA.items():
        raw = (d.FIXTURES/name).read_bytes()
        need(d.digest(raw) == pin, "fixture drift: " + name)
        data[name] = json.loads(raw)
    fixture = data["native_causal_lm_cache_fixture.json"]
    case, = (c for c in fixture["cases"] if c["name"] == "ordinary/wave")
    bridge = data["native_line_owners_bridge.json"]
    archive = st.SafeTensorArchive.from_bytes(base64.b64decode(fixture["weights"], validate=True))
    base = st.NativeCausalLm.from_gpt2(case["config"], archive)
    need(base.base_model_identity == d.BASE and bridge["lora_id"] == d.LORA, "wrong base/LoRA")
    active = base.with_lora_json(bridge["lora"], expected_id=d.LORA)
    wave = active.with_geometry(st.GeometryAdapterArtifact.from_json(
        case["geometry"], case["geometry_id"], case["base_id"])).freeze_lora()
    need(wave.frozen_lora_to_json() == active.lora_to_json(), "LoRA freeze changed owner")
    art = json.loads(wave.geometry_artifact().to_json())
    placement, = art["placements"]
    old = placement["recipe"]
    placement["recipe"] = dict(schema="spiraltorch.channel_affine_adapter.v1", features=old["features"],
                               strength=old["strength"], max_values=old["kernel"]["max_values"])
    affine = wave.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(art)))
    origins = {}
    for arm in arms:
        if arm == d.REF:
            origins[arm] = active
        else:
            recipe, path = arm.split("@")
            origins[arm] = {"wave": wave, "affine": affine}[recipe].with_single_geometry_at(path)
    return origins


def selected(model, arm):
    if arm == d.REF:
        need(model.base_model_identity == d.BASE and model.geometry_artifact_id is None and
             model.frozen_lora_artifact_id is None, "LoRA companion drift")
        raw, pin = model.lora_to_json(), model.lora_artifact_id
    else:
        need(model.base_model_identity == d.FROZEN and model.lora_artifact_id is None and
             model.frozen_lora_artifact_id == d.LORA and
             d.owner_id(model.frozen_lora_to_json().encode(), d.REF) == d.LORA, "frozen LoRA/base drift")
        raw, pin = model.geometry_artifact().to_json(), model.geometry_artifact_id
    need(d.owner_id(raw.encode(), arm) == pin, "selected canonical identity")
    return raw, pin, d.coordinates(raw, arm)


def snapshot(model, spec, path, pin):
    rows = spec["train"] + spec["test"]
    result = []
    for row in rows:
        costs = model.score_lines(row["prompt"], d.CANDIDATES, stop_id=0)
        a.verify_costs(costs)
        result.append({"key": row["key"], "costs": costs})
    prompt = rows[0]["prompt"]
    logits = [model.forward(p, [1, len(p)], last_only=True).values()
              for p in (prompt, prompt+[7], prompt+[8])]
    a.verify_sentinel(logits, result[0]["costs"])
    d.save(path, dict(owner=pin, rows=result, sentinel=logits))


def perform(root, sha):
    spec = d.read_plan(root)
    need(d.digest((root/"spec.json").read_bytes()) == sha, "not the declared run")
    attempt = root/"attempt"
    attempt.mkdir(mode=0o700)
    for name in ("starts", "checkpoints", "scores"):
        (attempt/name).mkdir(mode=0o700)
    need(os.environ.get("SPIRALTORCH_BACKEND") == "cpu" and
         all(os.environ.get(k) == "1" for k in ("RAYON_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS")),
         "CPU environment not configured")
    sys.addaudithook(guard)
    import spiraltorch as st
    native = importlib.import_module("spiraltorch.spiraltorch")
    need(d.digest(Path(native.__file__).read_bytes()) == d.EXTENSION and st.NativeCausalLm is native.NativeCausalLm,
         "native extension identity")
    models_by_arm = models(st, spec["arms"])
    matched = None
    starts = {}
    for arm, model in models_by_arm.items():
        raw, pin, vector = selected(model, arm)
        if arm != d.REF:
            if matched is None:
                matched = vector
            need(matched == vector, "unmatched geometry initialization")
        else:
            need(pin == d.LORA, "LoRA reference initialization")
        starts[arm] = pin, vector
        d.save(attempt/"starts"/(arm+".json"), dict(weights=raw, id=pin, base=model.base_model_identity,
                                                    frozen_lora=model.frozen_lora_artifact_id))
        snapshot(model, spec, attempt/"scores"/(arm+".json"), pin)

    checkpoints = {}
    with (attempt/"steps.jsonl").open("xb") as log:
        for arm, role, seed in d.cells(spec):
            key = d.cell_key(arm, role, seed)
            raw_plan, ordering = d.session_plan(spec, arm, role, seed)
            s = st.NativeCausalLmLineTrainingSession(models_by_arm[arm], raw_plan)
            origin, before = starts[arm]
            previous = origin
            for i, row in enumerate(ordering*spec["epochs"]):
                receipt = s.step()
                raw, now, vector = selected(s.model, arm)
                record = dict(cell=key, key=row["key"], id=now, coordinates=vector, step=receipt)
                log.write(d.encoded(record))
                log.flush()
                a.verify_step(record, arm, role, s.plan_id, i, ordering, previous, before)
                previous, before = now, vector
            need(s.exhausted and s.completed_updates == len(ordering)*spec["epochs"], "missing updates")
            checkpoint, cp_id = s.checkpoint_with_id()
            data = checkpoint.encode("utf-8")
            with (attempt/"checkpoints"/(key+".json")).open("xb") as f:
                f.write(data)
            a.verify_checkpoint(data, cp_id, s.plan_id, arm, origin, now, i+1, vector)
            checkpoints[key] = cp_id
            snapshot(s.model, spec, attempt/"scores"/(key+".json"), now)
            need(selected(models_by_arm[arm], arm)[1] == origin, "origin was mutated")
    files = {p.relative_to(attempt).as_posix(): d.digest(p.read_bytes())
             for p in sorted(attempt.rglob("*")) if p.is_file()}
    d.save(root/"completed.json", dict(spec=sha, files=files, checkpoints=checkpoints))
    print("complete:", len(starts), "origins,", len(checkpoints), "cells", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    prep = action.add_parser("prepare")
    prep.add_argument("kind", choices=("rehearsal", "formal"))
    prep.add_argument("root", type=Path)
    run = action.add_parser("run")
    run.add_argument("root", type=Path)
    run.add_argument("spec_sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        args.root.mkdir(mode=0o700)
        print(d.save(args.root/"spec.json", dict(d.design(args.kind == "formal"), sources=d.sources())))
    else:
        perform(args.root, args.spec_sha256)


if __name__ == "__main__":
    main()

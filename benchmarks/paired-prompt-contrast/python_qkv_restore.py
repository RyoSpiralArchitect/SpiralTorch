"""Restore the sealed four-lane study through Python; compare bytes, never train."""
import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import struct
import sys
import time

PINS = {
    "plan": "72361929d059757822c9ce43b25ef10f953945d9279f69c5b3447696da5f8648",
    "training": "4b739c0c9dbea1890d6c7f1c93306b8e7921f9a148ae2cbe0456a13a468fb06f",
    "scoring": "e9067ee01c9ff6e15f8e502e2a802f7935741d2839717178d7864c5cf6e5cc38",
}
EXTENSION = "3497e4a24cd8df3fef1423c078dcf44c0bd5de45798f1a77bff7a5a5c27a2725"
LANES = ("lag0-ce", "lag0-smooth", "lag1-ce", "lag1-smooth")
FOREIGN = {"torch", "numpy", "transformers", "safetensors", "accelerate"}
POLICY = dict(kind="spiraltorch.python_qkv_restore.v1", rows=6144, updates=0,
              seconds=300, cache_values=1 << 20, quality_metrics=False, retries=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked(path, pin, maximum=16 << 20):
    with Path(path).open("rb") as f:
        raw = f.read(maximum + 1)
    require(len(raw) <= maximum, "byte budget: " + str(path))
    require(hashlib.sha256(raw).hexdigest() == pin, "changed input: " + str(path))
    return raw


def put(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    return sha(path)


def reference(row):
    bits = row["ordinary"]
    require(len(bits) == 64 and all(type(x) is int and 0 <= x < 2**32 for x in bits),
            "invalid reference vocabulary row")
    require(row["cached"] == bits, "original cache mismatch")
    return struct.pack("<64I", *bits)


def inputs(root):
    """Admit the entire saved roster and retain exact bytes before native loading."""
    watched = {}

    def get(path, pin, maximum=16 << 20):
        path = Path(path).resolve()
        raw = checked(path, pin, maximum)
        watched[str(path)] = pin
        return raw

    plan = json.loads(get(root / "plan.json", PINS["plan"]))
    train = json.loads(get(root / "training/complete.json", PINS["training"]))
    scores = json.loads(get(root / "scoring/complete.json", PINS["scoring"]))
    layout = json.loads(get(Path(plan["parent"]) / "plan.json", plan["parent_pins"]["plan"]))
    require(train["plan"] == scores["plan"] == PINS["plan"]
            and scores["training"] == PINS["training"], "broken saved study chain")
    require(set(train["lanes"]) == set(scores["states"]) == set(LANES), "wrong lanes")
    weights = [Path(p) for p in layout["inputs"] if Path(p).name == "model.safetensors"]
    require(len(weights) == 1, "ambiguous base archive")
    weights = weights[0]
    archive = get(weights, layout["inputs"][str(weights)], 32 << 20)
    models = {}
    for lag in (0, 1):
        owner = layout["routes"][str(lag)]
        files = {}
        for name in ("config.json", "lora.json", "geometry.json"):
            path = weights.parent / f"lag{lag}" / name
            files[name] = get(path, layout["inputs"][str(path)], 1 << 20).decode()
        models[lag] = dict(owner=owner, files=files)
    endpoints = {}
    for lane in LANES:
        state = train["lanes"][lane][-1]
        require(state["steps"] == 256 and state["id"].startswith("sha256:"), "wrong endpoint")
        raw = get(state["path"], state["id"][7:], 65536)
        endpoints[lane] = dict(record=state, raw=raw)
        require(set(scores["states"][lane]) == set(layout["evaluation"]), "missing saved cell")
    rows = branches = prefills = 0
    for key, cell in layout["evaluation"].items():
        prompts, cut = cell["prompts"], cell["prefix_len"]
        require(prompts and type(cut) is int and 0 < cut < min(map(len, prompts)), "invalid prefix roster")
        rows += len(prompts)
        branches += sum(len(q) - cut for q in prompts)
        prefills += len({tuple(q[:cut]) for q in prompts})
        for lane in LANES:
            saved = scores["states"][lane][key]
            require(saved["serving"] == endpoints[lane]["record"]["serving"]
                    and len(saved["rows"]) == len(prompts), "wrong saved owner/row count")
            for row in saved["rows"]:
                reference(row)
    require(rows * 4 == POLICY["rows"] and len(layout["evaluation"]) == 24, "changed scope")
    return dict(watched=watched, archive=archive, models=models, endpoints=endpoints,
                cells=layout["evaluation"], scores=scores["states"],
                calls=dict(ordinary=rows * 4, branches=branches * 4, prefills=prefills * 4))


def package_inventory(package):
    files = {}
    for path in sorted((package / "spiraltorch").rglob("*")):
        require(not path.is_symlink(), "unexpected package symlink")
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(package))] = sha(path)
    require(files.get("spiraltorch/spiraltorch.abi3.so") == EXTENSION, "wrong saved extension")
    return files


def sources():
    paths = [Path(__file__).resolve(), Path(__file__).with_name("test_python_qkv_restore.py").resolve()]
    return {str(p): sha(p) for p in paths}


def freeze(root, package, out):
    data = inputs(root)
    return put(out, dict(policy=POLICY, study=str(root), package=str(package),
                         inputs=data["watched"], calls=data["calls"], sources=sources(),
                         package_files=package_inventory(package)))


class RejectForeignML(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FOREIGN:
            self.attempts.append(fullname)
            raise RuntimeError("foreign ML import: " + fullname)


def restore(st, data, lane):
    spec = data["models"][int(lane[3])]
    owner, files = spec["owner"], spec["files"]
    model = st.NativeCausalLm.from_gpt2(files["config.json"], st.SafeTensorArchive.from_bytes(data["archive"]))
    model = model.with_readout_accumulator("f32_wide_backward").with_lora_json(files["lora.json"])
    model = model.with_geometry(st.GeometryAdapterArtifact.from_json(
        files["geometry.json"], owner["geometry"], owner["base"]))
    require((model.base_model_identity, model.lora_artifact_id, model.geometry_artifact_id)
            == (owner["base"], owner["lora"], owner["geometry"]), "foreign base/adapters")
    endpoint = data["endpoints"][lane]
    state = endpoint["record"]
    trainer = st.NativeCausalLmQkvAdamW.from_bytes(model, endpoint["raw"], state["id"])
    require(trainer.steps == 256 and trainer.layer == 0 and trainer.serving_id == state["serving"],
            "foreign restored state")
    require(trainer.checkpoint_with_id() == (endpoint["raw"], state["id"]), "checkpoint roundtrip mismatch")
    return trainer


def compare_cell(trainer, cell, saved, ordinary_file, cached_file, guard):
    require(len(saved["rows"]) == len(cell["prompts"]), "row count differs")
    owner = trainer.serving_id
    require(saved["serving"] == owner, "foreign saved serving state")
    prefixes = {}
    retained = branches = 0
    digest = hashlib.sha256()
    cut = cell["prefix_len"]
    for index, (prompt, row) in enumerate(zip(cell["prompts"], saved["rows"])):
        guard()
        key = tuple(prompt[:cut])
        if key not in prefixes:
            prefix = trainer.prefill(list(key), max_cache_values=POLICY["cache_values"])
            retained += prefix.cache_values
            require(retained <= POLICY["cache_values"], "prefix inventory budget")
            prefixes[key] = prefix
        prefix = prefixes[key]
        old_logits = prefix.logits_f32le()
        branch = prefix
        for token in prompt[cut:]:
            branch = branch.branch(token)
            branches += 1
        ordinary, cached = trainer.last_logits_f32le(prompt), branch.logits_f32le()
        ordinary_file.write(ordinary)
        cached_file.write(cached)
        expected = reference(row)
        require(ordinary == cached == expected,
                f"row {index} mismatch: expected={expected.hex()} ordinary={ordinary.hex()} cached={cached.hex()}")
        require(branch.serving_id == prefix.serving_id == owner and branch.token_ids == prompt
                and prefix.token_ids == list(key) and prefix.logits_f32le() == old_logits,
                "cache ownership or parent mutation")
        digest.update(ordinary)
    return dict(rows=len(saved["rows"]), prefills=len(prefixes), branches=branches,
                logits_sha256=digest.hexdigest())


def run(path, pin):
    plan = json.loads(checked(path, pin, 1 << 20))
    require(plan["policy"] == POLICY and plan["sources"] == sources(), "changed witness source/policy")
    data = inputs(Path(plan["study"]))
    package = Path(plan["package"])
    require(data["watched"] == plan["inputs"] and data["calls"] == plan["calls"], "changed study inputs")
    require(package_inventory(package) == plan["package_files"], "changed Python package")
    out = path.parent / "comparison"
    out.mkdir(mode=0o700)
    finder = RejectForeignML()
    require(not any(n.split(".")[0] in FOREIGN | {"spiraltorch"} for n in sys.modules), "preloaded runtime")
    sys.meta_path.insert(0, finder)
    begin = time.monotonic()
    results = {}

    def guard():
        require(time.monotonic() - begin < POLICY["seconds"], "CPU wall budget")
        require(not finder.attempts, "swallowed foreign import")

    try:
        sys.path.insert(0, str(package))
        st = importlib.import_module("spiraltorch")
        require(Path(st.__file__).resolve() == package / "spiraltorch/__init__.py"
                and Path(st._rs.__file__).resolve() == package / "spiraltorch/spiraltorch.abi3.so", "wrong loaded runtime")
        guard()
        for lane in LANES:
            trainer = restore(st, data, lane)
            checkpoint = trainer.checkpoint_with_id()
            cells = {}
            with (out / (lane + "-ordinary.f32le")).open("xb") as ordinary, (out / (lane + "-cached.f32le")).open("xb") as cached:
                for key in sorted(data["cells"]):
                    cells[key] = compare_cell(trainer, data["cells"][key], data["scores"][lane][key], ordinary, cached, guard)
                    put(out / (lane + "-" + key + ".json"), cells[key])
            require(trainer.checkpoint_with_id() == checkpoint, "inference changed optimizer")
            results[lane] = dict(serving=trainer.serving_id, checkpoint=checkpoint[1], cells=cells)
            print(lane, "all saved rows identical; checkpoint unchanged", flush=True)
        measured = {name: sum(c[field] for r in results.values() for c in r["cells"].values())
                    for name, field in (("ordinary", "rows"), ("prefills", "prefills"), ("branches", "branches"))}
        require(measured == plan["calls"] and measured["ordinary"] == POLICY["rows"], "incomplete comparison")
        require(all(sha(p) == h for p, h in plan["inputs"].items()), "source artifact changed during run")
        require(package_inventory(package) == plan["package_files"] and sources() == plan["sources"], "runtime source changed")
        guard()
        foreign = [n for n in sys.modules if n.split(".")[0] in FOREIGN]
        require(not foreign, "foreign runtime loaded")
        return put(out / "complete.json", dict(plan=pin, results=results, calls=measured,
            import_attempts=finder.attempts, foreign_modules=foreign, updates=0,
            wall_seconds=time.monotonic()-begin, raw_files={p.name: sha(p) for p in sorted(out.glob("*.f32le"))}))
    except Exception as error:
        put(out / "failure.json", dict(error=type(error).__name__, detail=str(error), completed_lanes=list(results),
                                      import_attempts=finder.attempts, wall_seconds=time.monotonic()-begin))
        raise


if __name__ == "__main__":
    os.umask(0o077)
    if len(sys.argv) == 5 and sys.argv[1] == "freeze":
        print(freeze(*(Path(p).resolve() for p in sys.argv[2:])))
    elif len(sys.argv) == 4 and sys.argv[1] == "run":
        print(run(Path(sys.argv[2]).resolve(), sys.argv[3]))
    else:
        raise SystemExit("freeze STUDY PACKAGE PLAN | run PLAN SHA256")

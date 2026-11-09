"""Exclusive, bounded native execution of the already-sealed layout protocol."""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import length_position_protocol as protocol

sha, require = protocol.sha, protocol.require


def put(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    return sha(path)


def read(path, pin):
    require(sha(path) == pin, "changed artifact: " + str(path))
    return json.loads(path.read_bytes())


def inputs(p):
    require(all(sha(k) == v for k, v in p["inputs"].items()), "changed frozen input")


def load(path, pin):
    p = read(path, pin)
    require(p["policy"] == protocol.POLICY, "changed experiment policy")
    inputs(p)
    parents = [Path(k).parent for k, v in p["inputs"].items() if v == protocol.PARENT]
    drivers = [Path(k) for k, v in p["inputs"].items() if v == protocol.DRIVER]
    require(len(parents) == len(drivers) == 1, "ambiguous native inputs")
    parent, driver = parents[0], drivers[0]
    fit = json.loads((parent / "fit.json").read_bytes())
    require(p["training"] == {view: [protocol.training(fit, view, i) for i in range(256)]
                              for view in ("fixed", "varied")}
            and p["evaluation"] == protocol.evaluation(fit), "changed batches or evaluation")
    return p, parent, driver


def sources():
    return {str(path): sha(path) for path in (Path(__file__).resolve(),
            Path(protocol.__file__).resolve(), Path(__file__).with_name("test_length_position_run.py").resolve())}


def seal(path, pin):
    load(path, pin)
    return put(path.parent / "execution.json", dict(plan=pin, sources=sources(),
        interpretation="Strict full-vocabulary top1: ties fail. No retries or scoring intermediate checkpoints.",
        transport="Each single-step call saves optimizer state for the next batch. All raw states retained; only step256 evaluated."))


def execution(path, pin):
    e = read(path.parent / "execution.json", pin)
    require(e["sources"] == sources(), "changed execution source")
    return load(path, e["plan"])


def floats(bits):
    require(len(bits) == 64 and all(type(x) is int and 0 <= x < 2**32 for x in bits), "invalid logits")
    out = struct.unpack("<64f", struct.pack("<64I", *bits))
    require(all(map(math.isfinite, out)), "nonfinite logits")
    return out


def checkpoint(path, pin, owner, step):
    require(pin == "sha256:" + sha(path), "checkpoint hash mismatch")
    raw = path.read_bytes()
    require(4 < len(raw) < 65536, "wrong checkpoint size")
    n, = struct.unpack_from("<I", raw)
    require(n <= 16384 and len(raw) == 4 + n + 3072 * 12, "wrong checkpoint payload")
    h = json.loads(raw[4:4+n])
    require(h["schema"] == "spiraltorch.qkv_adamw.v1" and h["width"] == 32 and h["layer"] == 0
            and h["steps"] == step and all(h[k] == owner[k] for k in ("base", "lora", "geometry")),
            "foreign checkpoint owner or optimizer clock")
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", raw[4+n:])), "nonfinite state")


class Native:
    def __init__(self, p, parent, driver, out, seconds):
        self.p, self.parent, self.driver, self.out = p, parent, driver, out
        self.end = time.monotonic() + seconds
        out.mkdir(mode=0o700)

    def call(self, name, lag, action, start=None):
        require(shutil.disk_usage(self.out).free >= 4 * 2**30, "disk floor")
        left = self.end - time.monotonic()
        require(left > 0, "phase deadline")
        lane = self.out / name
        lane.mkdir()
        owner = self.p["routes"][str(lag)]
        if start:
            checkpoint(Path(start["path"]), start["id"], owner, start["steps"])
        if action["kind"] == "fit":
            action = action | dict(out=str(lane / "state"))
        request = {k: owner[k] for k in ("base", "lora", "geometry")}
        request.update(layer=0, start={k: start[k] for k in ("path", "id")} if start else None, action=action)
        put(lane / "request.json", request)
        md = self.parent / f"lag{lag}"
        with (lane / "stdout").open("xb") as stdout, (lane / "stderr").open("xb") as stderr:
            r = subprocess.run([self.driver, md / "config.json", self.parent / "model.safetensors",
                                md / "lora.json", md / "geometry.json"],
                input=json.dumps(request).encode(), stdout=stdout, stderr=stderr,
                timeout=min(30, left), env=dict(os.environ, SPIRALTORCH_BACKEND="cpu", RAYON_NUM_THREADS="1"))
        put(lane / "exit.json", dict(code=r.returncode))
        require(r.returncode == 0, "native error: " + name)
        z = json.loads((lane / "stdout").read_bytes())
        require(all(z[k] == owner[k] for k in ("base", "lora", "geometry"))
                and z["key_lag"] == (dict(layer=0, lag=lag) if lag else None), "foreign reply")
        before = start["steps"] if start else 0
        if start:
            require(z["initial"] == start["serving"], "restored wrong serving state")
        if action["kind"] == "fit":
            step = before + 1
            require(z["steps"] == step and len(z["data"]["updates"]) == 1
                    and z["data"]["updates"][0]["steps"] == step, "wrong update count")
            report = z["data"]["updates"][0]
            require(math.isfinite(report["gold_nll"]) and all(map(math.isfinite, report["gradient_l2"])), "nonfinite update")
            pin = z["data"]["pins"][str(step)]
            require(pin["serving"] == z["end"], "wrong saved serving identity")
            state = lane / f"state/{step}.bin"
            checkpoint(state, pin["checkpoint"], owner, step)
            return dict(path=str(state), id=pin["checkpoint"], steps=step, serving=z["end"], report=report)
        require(z["steps"] == before and z["initial"] == z["end"], "score mutated model")
        rows = z["data"]["rows"]
        require(len(rows) == len(action["prompts"]), "missing score rows")
        for row in rows:
            require(row["ordinary"] == row["cached"], "cache mismatch")
            floats(row["ordinary"])
        return dict(rows=rows, serving=z["end"])


def steps(io, name, lag, batches, start=None, reference=None):
    state, records = start, []
    for batch in batches:
        number = state["steps"] + 1 if state else 1
        state = io.call(f"{name}-{number:03d}", lag,
                        batch | dict(kind="fit", rate=.01, updates=1, save_at=[1]), state)
        if reference is not None:
            expected = reference[number - 1]
            require(all(state[k] == expected[k] for k in ("id", "steps", "serving", "report")), "resume diverged")
        records.append(state)
    return records


def train(path, epin):
    p, parent, driver = execution(path, epin)
    io = Native(p, parent, driver, path.parent / "training", 600)
    lanes = {}
    for lag, view in p["policy"]["lanes"]:
        name = f"lag{lag}-{view}"
        lanes[name] = steps(io, name, lag, p["training"][view])
        print(name, "completed 256", flush=True)
    reference = lanes["lag1-varied"]
    resume = steps(io, "resume", 1, p["training"]["varied"][128:], reference[127], reference)
    inputs(p)
    return put(io.out / "complete.json", dict(execution=epin, lanes=lanes, resume=resume, updates=1152))


def score(path, epin, tpin):
    p, parent, driver = execution(path, epin)
    t = read(path.parent / "training/complete.json", tpin)
    require(t["execution"] == epin and t["updates"] == 1152, "unverified training")
    io = Native(p, parent, driver, path.parent / "scoring", 240)
    states = [(f"lag{k}-origin", k, None) for k in (0, 1)] + [
        (f"lag{k}-{view}", k, t["lanes"][f"lag{k}-{view}"][-1]) for k, view in p["policy"]["lanes"]]
    results = {}
    for name, lag, state in states:
        results[name] = {}
        serving = None
        for key, cell in p["evaluation"].items():
            z = io.call(name + "-" + key, lag, dict(kind="score", prompts=cell["prompts"], prefix_len=cell["prefix_len"]), state)
            require(serving is None or serving == z["serving"], "changing scoring state")
            serving = z["serving"]
            results[name][key] = z
        print(name, "scored", flush=True)
    inputs(p)
    return put(io.out / "complete.json", dict(execution=epin, training=tpin, states=results))


def metric(rows, labels):
    require(len(rows) == len(labels) and len(rows) > 0, "incomplete metric")
    correct, losses = 0, []
    for row, y in zip(rows, labels):
        require(row["ordinary"] == row["cached"] and type(y) is int and 0 <= y < 64, "invalid score pair")
        v = floats(row["ordinary"])
        correct += v[y] > max(x for i, x in enumerate(v) if i != y)
        hi = max(v)
        losses.append(hi + math.log(math.fsum(math.exp(x - hi) for x in v)) - v[y])
    return dict(correct=correct, count=len(rows), nll=math.fsum(losses) / len(rows))


def audit(path, epin, tpin, spin):
    p, _, _ = execution(path, epin)
    t = read(path.parent / "training/complete.json", tpin)
    s = read(path.parent / "scoring/complete.json", spin)
    require(t["execution"] == epin and s["execution"] == epin and s["training"] == tpin, "broken evidence chain")
    cells = {name: {key: metric(v["rows"], p["evaluation"][key]["gold"]) for key, v in scores.items()}
             for name, scores in s["states"].items()}
    primary = {}
    for name, by_cell in cells.items():
        selected = [x for k, x in by_cell.items() if k.startswith("f58-") and k.endswith("-untrained_pairs")]
        require(len(selected) == 6 and sum(x["count"] for x in selected) == 576, "incomplete primary cells")
        primary[name] = dict(correct=sum(x["correct"] for x in selected),
                             nll=math.fsum(x["nll"] for x in selected) / 6,
                             competent=all(x["correct"] >= 87 for x in selected))
    v, f = primary["lag1-varied"], primary["lag1-fixed"]
    interaction = (v["correct"] - f["correct"]) - (primary["lag0-varied"]["correct"] - primary["lag0-fixed"]["correct"])
    return put(path.parent / "audit.json", dict(execution=epin, training=tpin, scoring=spin,
        cells=cells, primary=primary, efficacy=v["correct"] > f["correct"] and v["nll"] <= f["nll"],
        competence=v["competent"], descriptive_interaction=interaction, quality_scope="exploratory synthetic layout comparison only"))


if __name__ == "__main__":
    os.umask(0o077)
    commands = {"seal": (seal, 2), "train": (train, 2), "score": (score, 3), "audit": (audit, 4)}
    require(len(sys.argv) >= 2 and sys.argv[1] in commands, "seal/train/score/audit PLAN PINS...")
    fn, count = commands[sys.argv[1]]
    require(len(sys.argv) == count + 2, "wrong phase arguments")
    print(fn(Path(sys.argv[2]).resolve(), *sys.argv[3:]))

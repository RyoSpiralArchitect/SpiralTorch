"""Fixed-coefficient, matched native-QKV smoothing comparison; no search."""
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import length_position_run as r

PINS = dict(plan="d87cf993b08086a2d9f0f322bdd8a870fbbaaa8c533a15fb09e09a7954876a46",
            training="26de6093e228893680f816e07e7fc4630d3dbf133bd33bffc76f30d78e79f9ad",
            scoring="c3230e455c4e31e6d255545410c0dd912df32d827deecb5eb0b10ece5b5b5b10")
POLICY = dict(
    kind="spiraltorch.qkv_smoothing_comparison.v1",
    coefficient=.1,
    lanes=[[lag, mode] for lag in (0, 1) for mode in ("ce", "smooth")],
    data="Reuse the previous sealed varied-layout batches and all evaluation cells. Already-consumed exploratory material, not fresh holdout.",
    matched="Same origin, ordering, 96 answers and 1056 tokens/step, cap4, lr .01, default AdamW, zero decay; block0 QKV only. LoRA/geometry and every other weight fixed.",
    updates="Each lane 256 updates. CE replay must match all previous 256 checkpoint digests and reports; stop on first mismatch. Smoothed lag1 has a prescribed exact 128->256 continuation (128 extra updates).",
    scoring="Four final models x 1536 rows = 6144 ordinary/cache pairs. No origin remeasurement or intermediate evaluation. CE final logits must reproduce saved originals exactly. Evaluate unsmoothed full-vocab NLL; strict top1, ties fail.",
    primary="Lag1 smooth must lower aggregate gold NLL without lowering strict correct count versus CE on six filler58 untrained-pair cells (576 rows). Competence separately requires >=87/96 in every cell. Report ordinary controls and all cells; filler59 cannot rescue primary.",
    bounds="CPU, 1152 updates total; train<=600s, score<=240s, disk>=4GiB, first failure stops, no retries, coefficient search, GPU, release or overwrites.",
)


def sources():
    files = [Path(__file__).resolve(), Path(r.__file__).resolve(), Path(r.protocol.__file__).resolve(),
             Path(__file__).with_name("test_qkv_smoothing_study.py").resolve()]
    return {str(p): r.sha(p) for p in files}


def baseline(parent):
    p, model, _ = r.load(parent / "plan.json", PINS["plan"])
    t = r.read(parent / "training/complete.json", PINS["training"])
    r.require(r.sha(parent / "scoring/complete.json") == PINS["scoring"], "changed original scores")
    return p, t, model


def freeze(parent, driver, path):
    baseline(parent)
    return r.put(path, dict(policy=POLICY, parent=str(parent), parent_pins=PINS,
        driver=str(driver), binary=r.sha(driver), sources=sources()))


def load(path, pin):
    p = r.read(path, pin)
    r.require(p["policy"] == POLICY and p["parent_pins"] == PINS and p["sources"] == sources(), "changed protocol")
    r.require(r.sha(p["driver"]) == p["binary"], "changed executable")
    previous, training, model = baseline(Path(p["parent"]))
    return p, previous, training, model


def batches(original, mode):
    r.require(mode in ("ce", "smooth"), "unknown objective")
    return [dict(x) if mode == "ce" else x | dict(label_smoothing=POLICY["coefficient"]) for x in original]


def train(path, pin):
    p, old, original, model = load(path, pin)
    io = r.Native(old, model, Path(p["driver"]), path.parent / "training", 600)
    lanes = {}
    for lag, mode in POLICY["lanes"]:
        name = f"lag{lag}-{mode}"
        reference = original["lanes"][f"lag{lag}-varied"] if mode == "ce" else None
        lanes[name] = r.steps(io, name, lag, batches(old["training"]["varied"], mode), reference=reference)
        print(name, "completed 256", flush=True)
    ref = lanes["lag1-smooth"]
    resume = r.steps(io, "resume", 1, batches(old["training"]["varied"], "smooth")[128:], ref[127], ref)
    load(path, pin)
    return r.put(io.out / "complete.json", dict(plan=pin, lanes=lanes, resume=resume, updates=1152))


def score(path, pin, training):
    p, old, _, model = load(path, pin)
    t = r.read(path.parent / "training/complete.json", training)
    r.require(t["plan"] == pin and t["updates"] == 1152, "incomplete training")
    original = r.read(Path(p["parent"]) / "scoring/complete.json", PINS["scoring"])
    io = r.Native(old, model, Path(p["driver"]), path.parent / "scoring", 240)
    results = {}
    for lag, mode in POLICY["lanes"]:
        name = f"lag{lag}-{mode}"
        state = t["lanes"][name][-1]
        r.require(state["steps"] == 256, "wrong final endpoint")
        results[name] = {}
        for key, cell in old["evaluation"].items():
            z = io.call(name + "-" + key, lag,
                        dict(kind="score", prompts=cell["prompts"], prefix_len=cell["prefix_len"]), state)
            if mode == "ce":
                r.require(z == original["states"][f"lag{lag}-varied"][key], "CE output replay mismatch")
            results[name][key] = z
        print(name, "scored", flush=True)
    load(path, pin)
    return r.put(io.out / "complete.json", dict(plan=pin, training=training, states=results))


def summarize(cells):
    primary = {}
    for name, by_cell in cells.items():
        chosen = [v for k, v in by_cell.items() if k.startswith("f58-") and k.endswith("-untrained_pairs")]
        r.require(len(chosen) == 6 and all(v["count"] == 96 for v in chosen), "incomplete primary")
        primary[name] = dict(correct=sum(v["correct"] for v in chosen),
                             nll=math.fsum(v["nll"] for v in chosen)/6,
                             competent=all(v["correct"] >= 87 for v in chosen))
    smooth, ce = primary["lag1-smooth"], primary["lag1-ce"]
    return dict(primary=primary, efficacy=smooth["nll"] < ce["nll"] and smooth["correct"] >= ce["correct"],
                competence=smooth["competent"])


def audit(path, pin, training, scoring):
    _, old, _, _ = load(path, pin)
    t = r.read(path.parent / "training/complete.json", training)
    s = r.read(path.parent / "scoring/complete.json", scoring)
    r.require(t["plan"] == pin and s["plan"] == pin and s["training"] == training, "broken receipt chain")
    r.require(set(s["states"]) == {f"lag{k}-{m}" for k, m in POLICY["lanes"]}, "missing lane")
    cells = {}
    for name, by_cell in s["states"].items():
        r.require(set(by_cell) == set(old["evaluation"]), "missing cell")
        cells[name] = {key: r.metric(v["rows"], old["evaluation"][key]["gold"]) for key, v in by_cell.items()}
    return r.put(path.parent / "audit.json", dict(plan=pin, training=training, scoring=scoring,
                 cells=cells, **summarize(cells)))


if __name__ == "__main__":
    os.umask(0o077)
    phases = {"freeze": (freeze, 3), "train": (train, 2), "score": (score, 3), "audit": (audit, 4)}
    r.require(len(sys.argv) > 1 and sys.argv[1] in phases, "freeze/train/score/audit")
    fn, count = phases[sys.argv[1]]
    r.require(len(sys.argv) == count + 2, "wrong arguments")
    args = [Path(x).resolve() for x in sys.argv[2:]] if sys.argv[1] == "freeze" else [Path(sys.argv[2]).resolve(), *sys.argv[3:]]
    print(fn(*args))

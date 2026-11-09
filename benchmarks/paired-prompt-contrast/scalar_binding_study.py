"""Train a native rank-one scalar on known pairs; unlock new pair combinations only after training."""

import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rank_one_projection as rank
from projection_reach import base, signature, swap, task
from shared_prefix_study import obj, put, read, record, require, sha, write

STATES = task.STATES
ARMS = ("hold", "aligned", "orthogonal")
STEPS = 48
RATE, MAX_STEP, MAX_ABS = 16.0, 0.08, 4.0
WALL = 180
KIND = "spiraltorch.entity_scalar_pair_learning.v1"
SETTINGS = dict(
    steps=STEPS,
    rate=RATE,
    max_step=MAX_STEP,
    max_abs=MAX_ABS,
    train="Offset 1. Every step uses one edge, four assignment-reversed pairs; "
    "edge order (5*t+3)%8. Mean pairwise logistic loss on the standby-on contrast. "
    "Both trainable arms use identical batches and SGD bounds, not matched endpoints.",
    fresh="Offset 2: eight pair combinations unused by the entity study, known identities. "
    "Offset 3 was previously scored; do not use it. Offset 4 remains unscored.",
    primary="Against BOTH no-update and trained perpendicular arm in each origin: fresh "
    "interaction +0.01, >=6/8 edge gains >5e-6, both order gains >5e-6. "
    "Report weaker resolved direction separately. Decision improvement must exceed "
    "both in strict correct choices; competence requires >=6/8 all-correct edges. "
    "Report whole-vocabulary strict gold top1 and mean gold NLL separately.",
    integrity="Every endpoint row must bit-match ordinary inference on a separately merged checkpoint. "
    "Hold must replay the archived two named training logits and unrestricted greedy ID.",
    updates=192,
    backwards=1536,
    readouts_per_route=768,
    attempts_per_stage=1,
    wall=WALL,
)


def code():
    return rank.source_hashes() | {
        n: signature(Path(__file__).with_name(n + ".py"))
        for n in ("scalar_binding_study", "scalar_binding_readback")
    }


def fresh():
    rows = []
    for edge, a in enumerate(task.ENTITIES):
        es = [a, task.ENTITIES[(edge + 2) % 8]]
        for order, idle, query in itertools.product(range(2), repeat=3):
            prompt = task.render(es, order, idle, query)
            rows.append(
                dict(
                    id=task.identifier(prompt),
                    prompt=prompt,
                    entities=es,
                    edge=edge,
                    order=order,
                    idle=idle,
                    query=query,
                    gold=task.IDLE if query == idle else task.ON,
                )
            )
    return sorted(rows, key=lambda r: r["id"])


def validate_splits(train, new):
    old = task.deck("test")  # Regenerate old input identities, not their scores.
    pairsets = [{tuple(sorted(r["entities"])) for r in rows} for rows in (train, old, new)]
    require(
        train == task.deck("train")
        and new == fresh()
        and all(len(s) == 8 for s in pairsets)
        and len(set.union(*pairsets)) == 24
        and len({r["id"] for group in (train, old, new) for r in group}) == 192,
        "reused edge or prompt",
    )


def pairs(rows, t):
    edge = (5 * t + 3) % 8
    batch = [r for r in rows if r["edge"] == edge]
    result = []
    for o, q in itertools.product(range(2), repeat=2):
        (pos,) = [r for r in batch if (r["order"], r["query"], r["idle"]) == (o, q, q)]
        (neg,) = [r for r in batch if (r["order"], r["query"], r["idle"]) == (o, q, 1 - q)]
        require(
            pos["gold"] == task.IDLE
            and neg["gold"] == task.ON
            and [i for i, (x, y) in enumerate(zip(pos["prompt"], neg["prompt"], strict=True)) if x != y]
            == [2, 5],
            "not an assignment-reversed pair",
        )
        result.append((pos, neg))
    return result


def factors(bank, state, arm):
    u, v = bank[state]
    if arm == "orthogonal":
        v = [x for j in range(0, len(v), 2) for x in (-v[j + 1], v[j])]
    return u, v


def create(model, bank, state, arm):
    return model.scalar_output_adapter(0, *(struct.pack("<32d", *v) for v in factors(bank, state, arm)))


def freeze(prior, prior_pin, measure_pin, audit_pin, roster, whl, whash, ext, out):
    prev = prior / "protocol"
    p = obj(prev / "plan.json", prior_pin)
    m = obj(prior / "measured/complete.json", measure_pin)
    a = obj(prior / "report/results.json", audit_pin, 2 << 20)
    require(
        p["kind"] == rank.KIND
        and p["code"] == rank.source_hashes()
        and p["settings"] == rank.PLAN
        and m["plan"] == a["plan"] == prior_pin
        and a["measurement"] == measure_pin
        and a["origins"]
        and a["endpoints"],
        "unqualified rank-one study",
    )
    r = obj(roster, p["roster"])
    axes = obj(prev / "axes.json", p["axes"], 1 << 20)
    require(
        r["directions"] == p["directions"] and r["model_calls"] == 0 and r["fit"] == p["folds"]["fit"],
        "wrong factors",
    )
    bank = {}
    for state in STATES:
        rec = r["roles"][state + "/origin"][rank.LABEL]
        bank[state] = [rec["left"], rec["right"]]
        for arm, old in (("aligned", "rank1"), ("orthogonal", "orthogonal")):
            u, v = factors(bank, state, arm)
            require(
                axes[state + "/origin"][old] == [0.0] * 32 + [x * y for x in u for y in v],
                "factor orientation changed",
            )
    folds = [obj(prev / (k + ".json"), p["folds"][k]) for k in ("fit", "check")]
    train = sorted(folds[0] + folds[1], key=lambda x: x["id"])
    new = fresh()
    validate_splits(train, new)
    inv = base.candidate_inventory(whl, whash, ext)
    require(ext != p["native"], "expected native scalar implementation")
    out.mkdir(mode=0o700)
    for sub in ("fit", "blind", "keys", "model"):
        (out / sub).mkdir(mode=0o700)
    origins = {s: p["roles"][s + "/origin"] for s in STATES}
    for name, h in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((v[7:] + ".json", v[7:]) for v in origins.values()),
    ]:
        write(out / "model" / name, read(prev / "model" / name, h, 100000))
    baseline = {}
    for line in read(prior / "measured/observations.jsonl", m["observations"], 2 << 20).splitlines():
        line = json.loads(line)
        if line["arm"] == "baseline" and line["role"].endswith("/origin"):
            state = line["role"].split("/")[0]
            baseline[state] = {x["id"]: y for x, y in zip(folds[0] + folds[1], line["values"], strict=True)}
    require(set(baseline) == set(STATES), "old baseline missing")
    decks = {
        k: dict(
            queries=put(out / "blind" / (k + ".json"), task.queries(rows)),
            labels=put(out / "keys" / (k + ".json"), task.key(rows)),
        )
        for k, rows in (("train", train), ("fresh", new))
    }
    return put(
        out / "plan.json",
        dict(
            kind=KIND,
            settings=SETTINGS,
            code=code(),
            prior=[prior_pin, measure_pin, audit_pin],
            old_wheel=p["wheel"],
            wheel=whash,
            native=ext,
            inventory=put(out / "wheel-inventory.json", inv),
            files=p["files"],
            base=p["base"],
            lora=p["lora"],
            origins=origins,
            bank=put(out / "factors.json", bank),
            baseline=put(out / "original-training-logits.json", baseline),
            train=put(out / "fit/rows.json", train),
            decks=decks,
        ),
    )


def stack(proto, p):
    inv = obj(proto / "wheel-inventory.json", p["inventory"], 2 << 20)
    require((inv["sha256"], inv["native"]["sha256"]) == (p["wheel"], p["native"]), "wrong installation")
    receipt = base.installed_identity(inv)
    st = base.native()
    require(
        st.NativeCausalLmScalarOutput is st._rs.NativeCausalLmScalarOutput
        and Path(st._rs.__file__).resolve()
        == Path(st.__file__).resolve().parent.parent / inv["native"]["path"],
        "foreign native adapter",
    )
    model = base.open_model(st, proto / "model", p["files"])
    require((model.base_model_identity, model.lora_artifact_id) == (p["base"], p["lora"]), "wrong decoder")
    return (
        st,
        receipt,
        {s: swap.attach(st, model, proto / "model" / (h[7:] + ".json"), h) for s, h in p["origins"].items()},
    )


def checkpoint(folder, s):
    raw = s.checkpoint_json().encode()
    h = sha(raw)
    path = folder / (h + ".json")
    if path.exists():
        read(path, h)
    else:
        write(path, raw)
    return h


def train(proto, p, pp, out):
    _, installed, models = stack(proto, p)
    bank = obj(proto / "factors.json", p["bank"])
    rows = obj(proto / "fit/rows.json", p["train"])
    folder = out / "checkpoints"
    folder.mkdir(mode=0o700)
    ends = {}
    updates = 0
    with (out / "steps.jsonl").open("xb") as log:
        for state in STATES:
            model = models[state]
            for arm in ARMS:
                name = state + "/" + arm
                active = create(model, bank, state, arm)
                h = checkpoint(folder, active)
                record(log, dict(event="start", name=name, checkpoint=h))
                if arm != "hold":
                    for t in range(STEPS):
                        group = pairs(rows, t)
                        pos, neg = zip(*group, strict=True)
                        active, report = active.step_pairs(
                            [r["prompt"] for r in pos],
                            [r["prompt"] for r in neg],
                            task.IDLE,
                            task.ON,
                            learning_rate=RATE,
                            max_step=MAX_STEP,
                            max_abs=MAX_ABS,
                        )
                        h = checkpoint(folder, active)
                        require(
                            (active.steps, active.coefficient) == (t + 1, report["after"])
                            and len(report["pairs"]) == 4,
                            "invalid native step",
                        )
                        record(
                            log,
                            dict(
                                event="step",
                                name=name,
                                ids=[[x["id"], y["id"]] for x, y in group],
                                report=report,
                                checkpoint=h,
                            ),
                        )
                        updates += 1
                        if t + 1 == STEPS // 2:
                            active = model.restore_scalar_output_adapter(
                                read(folder / (h + ".json"), h).decode(), h
                            )
                            require(
                                active.checkpoint_json().encode() == read(folder / (h + ".json"), h),
                                "inexact resume",
                            )
                ends[name] = h
    require(updates == SETTINGS["updates"], "wrong update budget")
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            installation=installed,
            steps=signature(out / "steps.jsonl"),
            ends=ends,
            updates=updates,
            backwards=8 * updates,
        ),
    )


def score(proto, p, pp, out, trained, tp):
    m = closed(proto, trained, pp, "train", None, tp)
    st, installed, models = stack(proto, p)
    bank = obj(proto / "factors.json", p["bank"])
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    config = read(proto / "model/config.json", p["files"]["config.json"]).decode()
    put(out / "consumed.json", dict(plan=pp, training=tp, cohort="offset2; not fresh for later runs"))
    decks = {k: obj(proto / "blind" / (k + ".json"), v["queries"]) for k, v in p["decks"].items()}
    require(
        all(
            len(qs) == 64
            and len({r["id"] for r in qs}) == 64
            and all(set(r) == {"id", "prompt"} and r["id"] == task.identifier(r["prompt"]) for r in qs)
            for qs in decks.values()
        ),
        "score input contains labels or duplicates",
    )
    calls = 0
    with (
        (out / "logits.f32le").open("xb") as f,
        (out / "ordinary.f32le").open("xb") as g,
        (out / "cases.jsonl").open("xb") as log,
    ):
        for state in STATES:
            model = models[state]
            for arm in ARMS:
                name = state + "/" + arm
                h = m["ends"][name]
                s = model.restore_scalar_output_adapter(
                    read(trained / "checkpoints" / (h + ".json"), h).decode(), h
                )
                u, v = factors(bank, state, arm)
                delta = rank.f32s([0.0] * 32 + [s.coefficient * (x * y) for x in u for y in v])
                changed, actual = rank.merge_bytes(raw, delta)
                independent = model if changed == raw else rank.rebind(st, config, changed, model)
                for split, qs in decks.items():
                    compact = bytearray()
                    ordinary = bytearray()
                    for q in qs:
                        ids = q["prompt"]
                        compact.extend(s.last_logits_f32le(ids))
                        ordinary.extend(
                            rank.f32s(independent.forward(ids, [1, len(ids)], last_only=True).values())
                        )
                        calls += 1
                    f.write(compact)
                    g.write(ordinary)
                    record(
                        log,
                        dict(
                            name=name,
                            split=split,
                            checkpoint=h,
                            weight=sha(changed),
                            norm=math.hypot(*actual),
                            scalar=sha(compact),
                            ordinary=sha(ordinary),
                        ),
                    )
    require(calls == SETTINGS["readouts_per_route"], "incomplete evaluation")
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            training=tp,
            installation=installed,
            readouts_per_route=calls,
            files={k: signature(out / k) for k in ("logits.f32le", "ordinary.f32le", "cases.jsonl")},
        ),
    )


def closed(proto, out, pp, phase, parent, pin):
    claim = dict(phase=phase, plan=pp, parent=parent, attempt=1, output=os.path.relpath(out, proto))
    require(
        obj(proto / (phase + "-attempt.json"), signature(out / "attempt.json")) == claim, "claim mismatch"
    )
    r = obj(out / "complete.json", pin)
    e = json.loads((out / "exit.json").read_bytes())
    require(
        r["plan"] == pp
        and e["exit"] == 0
        and 0 < e["elapsed"] <= WALL
        and read(out / "process.log", e["log"]).splitlines()[-1:] == [pin.encode()],
        "incomplete process",
    )
    return r


def run(phase, proto, pp, out, trained=None, tp=None, worker=False):
    p = obj(proto / "plan.json", pp)
    require(p["kind"] == KIND and p["settings"] == SETTINGS and p["code"] == code(), "stale scalar study")
    claim = dict(phase=phase, plan=pp, parent=tp, attempt=1, output=os.path.relpath(out, proto))
    if worker:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (WALL, WALL))
        swap.fence(proto, phase)

        def scope(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = Path(os.path.realpath(os.fsdecode(args[0])))
                if "validation" in path.parts and not any(
                    path.is_relative_to(d)
                    for d in [proto, out, Path(sys.prefix).resolve()]
                    + ([trained] if trained is not None else [])
                ):
                    raise PermissionError("external evidence not admitted")

        sys.addaudithook(scope)
        require(
            obj(proto / (phase + "-attempt.json"), signature(out / "attempt.json")) == claim, "unclaimed run"
        )
        return train(proto, p, pp, out) if phase == "train" else score(proto, p, pp, out, trained, tp)
    if phase == "score":
        from scalar_binding_readback import training_audit

        training_audit(proto, pp, trained, tp)
    require(not out.exists(), "occupied output")
    put(proto / (phase + "-attempt.json"), claim)
    out.mkdir(mode=0o700)
    put(out / "attempt.json", claim)
    env = dict(
        os.environ,
        SPIRALTORCH_BACKEND="cpu",
        HF_HUB_OFFLINE="1",
        RAYON_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"),
    )
    args = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_" + phase, str(proto), pp, str(out)]
    if phase == "score":
        args.extend([str(trained), tp])
    start = time.monotonic()
    with (out / "process.log").open("xb") as log:
        try:
            rc = subprocess.run(
                args, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=WALL
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    elapsed = time.monotonic() - start
    put(out / "exit.json", dict(exit=rc, elapsed=elapsed, log=signature(out / "process.log")))
    require(rc == 0 and elapsed <= WALL, "first failure retained; no retry")
    return signature(out / "complete.json")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "freeze" and len(args) == 9:
        print(freeze(Path(args[0]), *args[1:4], Path(args[4]), Path(args[5]), *args[6:8], Path(args[8])))
    elif cmd.lstrip("_") in ("train", "score") and len(args) == (3 if cmd.endswith("train") else 5):
        print(
            run(
                cmd.lstrip("_"),
                Path(args[0]).resolve(),
                args[1],
                Path(args[2]).resolve(),
                *([Path(args[3]).resolve(), args[4]] if len(args) == 5 else []),
                worker=cmd.startswith("_"),
            )
        )
    elif cmd == "audit" and len(args) == 7:
        from scalar_binding_readback import audit

        print(audit(Path(args[0]), args[1], Path(args[2]), args[3], Path(args[4]), args[5], Path(args[6])))
    else:
        raise SystemExit(
            "freeze PRIOR PLAN MEASUREMENT AUDIT ROSTER WHEEL WHEEL_PIN NATIVE_PIN PROTO | "
            "train PROTO PIN OUT | score PROTO PIN OUT TRAINED TRAIN_PIN | "
            "audit PROTO PIN TRAINED TRAIN_PIN SCORED SCORE_PIN OUT"
        )

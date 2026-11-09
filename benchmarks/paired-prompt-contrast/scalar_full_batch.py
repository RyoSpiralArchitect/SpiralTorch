"""Compare fixed and safeguarded scalar SGD on all 32 original training pairs; no evaluation deck."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scalar_binding_study as prior
from scalar_binding_readback import training_audit
from shared_prefix_study import obj, put, read, record, require, sha, write

KIND = "spiraltorch.training_only_scalar_full_batch.v1"
AXES = ("aligned", "orthogonal")
RULES = ("fixed", "guarded")
EARLY, LIMIT, TRIALS, FLOOR, WALL = 6, 24, 8, 1e-7, 180
SGD = dict(learning_rate=prior.RATE, max_step=prior.MAX_STEP, max_abs=prior.MAX_ABS)
SETTINGS = dict(
    early=EARLY,
    limit=LIMIT,
    sgd=SGD,
    max_trials=TRIALS,
    min_decrease=FLOOR,
    armijo=0.0001,
    axes=AXES,
    rules=RULES,
    wall=WALL,
    attempts=1,
    split="Only the previous 64 offset-one training prompts in 32 assignment-reversed pairs. "
    "Never stage prior blind decks, keys or scored results. Two starting geometries, not seeds.",
    comparison="Hold, saved minibatch endpoint, full-batch fixed and full-batch guarded SGD. "
    "Start both full rules at zero; same loss, examples, axis, order and initial proposal. "
    "Primary at 6 full-batch calls or earlier terminal refusal: at most the previous 192 "
    "pair derivatives per arm, not equal forward cost. Continue fixed through 24, "
    "guarded until 24 or first refusal. Never select an earlier minimum.",
    gate="Training only. Primary aligned rule in each origin: final pair loss lower than hold and "
    "mini by >1e-5; incremental guard benefit requires also beating full fixed by >1e-5. "
    "Report period two, later endpoint, native accepted/refused trials and cost, binary "
    "choices, paired ranking, interaction, full-vocab gold NLL separately. No unseen claim.",
)


def sources():
    return prior.code() | {
        n: prior.signature(Path(__file__).with_name(n + ".py"))
        for n in ("scalar_full_batch", "scalar_full_batch_audit")
    }


def settings():
    # Round-trip the sequence constants into their on-disk JSON form.
    return json.loads(json.dumps(SETTINGS))


def batch(rows):
    require(rows == prior.task.deck("train"), "not the original training cohort")
    ps = [pair for t in range(8) for pair in prior.pairs(rows, t)]
    require(
        len(ps) == 32 and {r["id"] for pair in ps for r in pair} == {r["id"] for r in rows},
        "incomplete full batch",
    )
    return ps


def freeze(previous, pp, tp, cp, wheel, wp, ext, out):
    old = previous / "protocol"
    trained = previous / "trained"
    p = obj(old / "plan.json", pp)
    mini = training_audit(old, pp, trained, tp)
    m = prior.closed(old, trained, pp, "train", None, tp)
    cy = obj(previous / "cycle-readback.json", cp)
    require(
        cy["kind"] == "training_only_frozen_state_cycle_readback"
        and cy["model_calls"] == 0
        and all(
            mini[s + "/aligned"]["clipped_steps"] == 48
            and mini[s + "/aligned"]["coefficient"] == 0
            and cy["results"][s]["at_zero_negative_edges"] == 8
            for s in prior.STATES
        ),
        "wrong prior experiment",
    )
    inv = prior.base.candidate_inventory(wheel, wp, ext)
    require(ext != p["native"], "old native extension")
    out.mkdir(mode=0o700)
    for k in ("model", "mini", "fit"):
        (out / k).mkdir(mode=0o700)
    for f, h in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((v[7:] + ".json", v[7:]) for v in p["origins"].values()),
    ]:
        write(out / "model" / f, read(old / "model" / f, h, 100000))
    for name in ("factors.json", "original-training-logits.json", "fit/rows.json"):
        key = {"factors.json": "bank", "original-training-logits.json": "baseline", "fit/rows.json": "train"}[
            name
        ]
        write(out / name, read(old / name, p[key]))
    ids = [[u["id"], v["id"]] for u, v in batch(obj(out / "fit/rows.json", p["train"]))]
    for h in set(m["ends"].values()):
        write(out / "mini" / (h + ".json"), read(trained / "checkpoints" / (h + ".json"), h))
    return put(
        out / "plan.json",
        {k: p[k] for k in ("base", "lora", "origins", "files", "bank", "baseline", "train")}
        | dict(
            kind=KIND,
            code=sources(),
            settings=settings(),
            parent=[pp, tp, cp],
            mini=m["ends"],
            pairs=ids,
            zero_gradient={s: cy["results"][s]["at_zero_full_gradient"] for s in prior.STATES},
            zero_loss={s: cy["results"][s]["prior_finite_full_losses"]["0"] for s in prior.STATES},
            wheel=wp,
            native=ext,
            inventory=put(out / "wheel-inventory.json", inv),
        ),
    )


def enforce(proto, out):
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (WALL, WALL))
    permitted = (proto, out, Path(sys.prefix).resolve())

    def guard(event, args):
        if event == "open" and isinstance(args[0], (bytes, str)):
            p = Path(os.path.realpath(os.fsdecode(args[0])))
            if (
                any(k in p.parts for k in ("blind", "keys", "scored"))
                or p.name in {"truth.json", "queries.json", "answer-key.json"}
                or ("validation" in p.parts and not any(p.is_relative_to(d) for d in permitted))
            ):
                raise PermissionError("not an admitted training input")

    sys.addaudithook(guard)


def worker(proto, pp, out):
    enforce(proto, out)
    p = obj(proto / "plan.json", pp)
    claim = dict(plan=pp, attempt=1, output=os.path.relpath(out, proto))
    require(
        p["kind"] == KIND
        and p["settings"] == settings()
        and p["code"] == sources()
        and obj(proto / "attempt.json", prior.signature(out / "attempt.json")) == claim,
        "unfrozen run",
    )
    st, installation, models = prior.stack(proto, p)
    bank = obj(proto / "factors.json", p["bank"])
    rows = obj(proto / "fit/rows.json", p["train"])
    ps = batch(rows)
    require([[x["id"], y["id"]] for x, y in ps] == p["pairs"], "pair order changed")
    args = ([x["prompt"] for x, y in ps], [y["prompt"] for x, y in ps], prior.task.IDLE, prior.task.ON)
    snaps = out / "checkpoints"
    snaps.mkdir(mode=0o700)
    labels = {}
    meta = {}
    dst = out / "steps.jsonl"
    counts = dict(gradient_prompts=0, objective_prompts=0, endpoint_prompts_per_route=0)
    with dst.open("xb") as f:
        for state in prior.STATES:
            for axis in AXES:
                key = state + "/" + axis
                model = models[state]
                original = prior.create(model, bank, state, axis)

                def seal(s):
                    h = prior.checkpoint(snaps, s)
                    meta[h] = (state, axis)
                    return h

                labels[key + "/hold"] = seal(original)
                mh = p["mini"][key]
                mini = model.restore_scalar_output_adapter(
                    read(proto / "mini" / (mh + ".json"), mh).decode(), mh
                )
                require(seal(mini) == mh, "prior endpoint did not restore identically")
                labels[key + "/mini"] = mh
                for rule in RULES:
                    name = key + "/" + rule
                    s = original
                    for t in range(1, LIMIT + 1):
                        if rule == "fixed":
                            nxt, r = s.step_pairs(*args, **SGD)
                            end = nxt.pair_objective(*args)
                            counts["objective_prompts"] += 64
                            stop = False
                        else:
                            nxt, r = s.step_pairs_backtracking(
                                *args, **SGD, max_trials=TRIALS, min_decrease=FLOOR
                            )
                            stop = r["status"] != "accepted"
                            end = r["baseline"] if stop else r["trials"][-1]["objective"]
                            counts["objective_prompts"] += 64 * (1 + len(r["trials"]))
                        counts["gradient_prompts"] += 64
                        before = prior.checkpoint(snaps, s)
                        s = nxt
                        h = seal(s)
                        record(
                            f,
                            dict(
                                name=name,
                                iteration=t,
                                before=before,
                                after=h,
                                step=s.steps,
                                report=r,
                                objective=end,
                            ),
                        )
                        if t == EARLY or stop and t < EARLY:
                            labels[name + "/early"] = h
                        if stop or t == LIMIT:
                            labels[name + "/end"] = h
                            break
    # Decode only the 64 training rows at named milestones, never a validation set.
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    config = read(proto / "model/config.json", p["files"]["config.json"]).decode()
    with (
        (out / "endpoints.jsonl").open("xb") as log,
        (out / "logits.f32le").open("xb") as sf,
        (out / "ordinary.f32le").open("xb") as of,
    ):
        for h in sorted(set(labels.values())):
            state, axis = meta[h]
            model = models[state]
            s = model.restore_scalar_output_adapter(read(snaps / (h + ".json"), h).decode(), h)
            u, v = prior.factors(bank, state, axis)
            edited, actual = prior.rank.merge_bytes(
                raw, prior.rank.f32s([0.0] * 32 + [s.coefficient * (a * b) for a in u for b in v])
            )
            ordinary = model if edited == raw else prior.rank.rebind(st, config, edited, model)
            x = b"".join(s.last_logits_f32le(r["prompt"]) for r in rows)
            y = b"".join(
                prior.rank.f32s(ordinary.forward(r["prompt"], [1, len(r["prompt"])], last_only=True).values())
                for r in rows
            )
            sf.write(x)
            of.write(y)
            record(
                log,
                dict(
                    checkpoint=h,
                    state=state,
                    axis=axis,
                    weight=sha(edited),
                    norm=math.hypot(*actual),
                    scalar=sha(x),
                    ordinary=sha(y),
                ),
            )
            counts["endpoint_prompts_per_route"] += 64
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            installation=installation,
            labels=labels,
            counts=counts,
            files={
                k: prior.signature(out / k)
                for k in ("steps.jsonl", "endpoints.jsonl", "logits.f32le", "ordinary.f32le")
            },
        ),
    )


def once(proto, pp, out):
    p = obj(proto / "plan.json", pp)
    require(
        p["kind"] == KIND and p["code"] == sources() and p["settings"] == settings() and not out.exists(),
        "stale plan or output",
    )
    claim = dict(plan=pp, attempt=1, output=os.path.relpath(out, proto))
    put(proto / "attempt.json", claim)
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
    start = time.monotonic()
    with (out / "process.log").open("xb") as f:
        try:
            rc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "_worker",
                    str(proto),
                    pp,
                    str(out),
                ],
                cwd=out,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=WALL,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    elapsed = time.monotonic() - start
    put(out / "exit.json", dict(exit=rc, elapsed=elapsed, log=prior.signature(out / "process.log")))
    require(rc == 0 and elapsed <= WALL, "first failure retained; no second attempt")
    return prior.signature(out / "complete.json")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 8:
        print(freeze(Path(a[0]), *a[1:4], Path(a[4]), *a[5:7], Path(a[7])))
    elif cmd in ("run", "_worker") and len(a) == 3:
        print((once if cmd == "run" else worker)(Path(a[0]).resolve(), a[1], Path(a[2]).resolve()))
    elif cmd == "audit" and len(a) == 5:
        from scalar_full_batch_audit import audit

        print(audit(Path(a[0]), a[1], Path(a[2]), a[3], Path(a[4])))
    else:
        raise SystemExit(
            "freeze PRIOR PLAN TRAINING CYCLE WHEEL WHEEL_PIN NATIVE_PIN OUT | run PROTO PIN OUT | audit PROTO PIN MEASURED PIN OUT"
        )

"""Recompute trial decisions, costs and endpoint metrics from the training-only worker's raw outputs."""

import io
import json
import math
import os
import struct

from scalar_full_batch import (
    AXES,
    EARLY,
    FLOOR,
    KIND,
    LIMIT,
    RULES,
    SGD,
    TRIALS,
    WALL,
    batch,
    prior,
    settings,
    sources,
)
from scalar_binding_readback import as_bits, install, near, pair_means
from shared_prefix_study import obj, put, read, require, sha


def objective(z):
    require(len(z["pairs"]) == 32 and all(len(v) == 2 for v in z["pairs"]), "wrong number of objective pairs")
    val = pair_means([[a, b, 0.0, 0.0] for a, b in z["pairs"]])[0]
    require(near(val, z["loss"]), "incorrect forward loss")
    return z["loss"]


def proposal(x, g):
    def clip(v, b):
        return max(-b, min(b, v))

    return clip(x + clip(-SGD["learning_rate"] * g, SGD["max_step"]), SGD["max_abs"])


def check_step(rule, x, step, r, end):
    rows = r["pairs"] if rule == "fixed" else r["derivatives"]
    require(len(rows) == 32 and r["before"] == x, "wrong state or derivative roster")
    loss, g = pair_means(rows)
    require(
        near(g, r["gradient"]) and near(loss, r["loss"] if rule == "fixed" else r["gradient_loss"]),
        "incorrect objective or gradient reduction",
    )
    first = proposal(x, r["gradient"])
    end_loss = objective(end)
    if rule == "fixed":
        require(r["after"] == first and r["step"] == step + 1, "wrong fixed update")
        return False, 0
    start = objective(r["baseline"])
    require(r["baseline"]["pairs"] == [v[:2] for v in rows], "native routes disagree on original row")
    trials = r["trials"]
    require(len(trials) <= TRIALS, "extra backtracks")
    candidate = first
    selected = None
    status = "trials_exhausted"
    for j, z in enumerate(trials):
        d = candidate - x
        threshold = max(FLOOR, 0.0001 * (-r["gradient"] * d))
        require(
            d != 0.0 and z["coefficient"] == candidate and near(threshold, z["required_decrease"]),
            "incorrect trial proposal",
        )
        drop = start - objective(z["objective"])
        if drop > 0.0 and drop >= z["required_decrease"]:
            require(j + 1 == len(trials), "continued after success")
            status = "accepted"
            selected = z
            break
        smaller = (x + candidate) * 0.5
        if smaller == candidate or smaller == x:
            require(j + 1 == len(trials), "continued after losing displacement")
            status = "no_displacement"
            break
        candidate = smaller
    if not trials and first == x:
        status = "zero_gradient" if r["gradient"] == 0.0 else "no_displacement"
    if status == "trials_exhausted":
        require(len(trials) == TRIALS, "stopped before trial budget")
    require(status == r["status"], "false status")
    if selected is not None:
        require(
            r["step"] == step + 1
            and r["after"] == selected["coefficient"]
            and selected["objective"] == end
            and end_loss < start,
            "false acceptance",
        )
    else:
        require(
            r["step"] == step and r["after"] == x and end == r["baseline"], "refusal advanced or changed loss"
        )
    return selected is None, len(trials)


def verify_snapshot(folder, h, p, state, axis, bank, cursor, scale):
    s = obj(folder / (h + ".json"), h)
    u, v = prior.factors(bank, state, axis)
    require(
        s
        == dict(
            schema="spiraltorch.causal_lm.scalar_output.v1",
            base=p["base"],
            lora=p["lora"],
            geometry=p["origins"][state],
            layer=0,
            readout="f32_wide_backward",
            steps=cursor,
            coefficient_bits=as_bits(scale),
            left_bits=list(map(as_bits, u)),
            right_bits=list(map(as_bits, v)),
        ),
        "changed factor, frozen owner, or checkpoint",
    )


def cycle(values):
    return (
        len(values) >= 4 and values[0] != values[1] and all(x == values[j % 2] for j, x in enumerate(values))
    )


def audit(proto, pp, out, mp, dst):
    p = obj(proto / "plan.json", pp)
    m = obj(out / "complete.json", mp)
    claim = dict(plan=pp, attempt=1, output=os.path.relpath(out, proto))
    ex = json.loads((out / "exit.json").read_bytes())
    require(
        p["kind"] == KIND
        and p["settings"] == settings()
        and p["code"] == sources()
        and m["plan"] == pp
        and obj(proto / "attempt.json", prior.signature(out / "attempt.json")) == claim
        and m["installation"] == install(proto, p)
        and ex["exit"] == 0
        and 0 < ex["elapsed"] <= WALL
        and read(out / "process.log", ex["log"]).splitlines()[-1:] == [mp.encode()],
        "invalid attempt, code or wheel",
    )
    rows = obj(proto / "fit/rows.json", p["train"])
    ps = batch(rows)
    require([[a["id"], b["id"]] for a, b in ps] == p["pairs"], "wrong data")
    fs = m["files"]
    logs = iter(map(json.loads, io.BytesIO(read(out / "steps.jsonl", fs["steps.jsonl"], 16 << 20))))
    snaps = out / "checkpoints"
    bank = obj(proto / "factors.json", p["bank"])
    labels = m["labels"]
    costs = dict(gradient_prompts=0, objective_prompts=0, endpoint_prompts_per_route=0)
    seen = set()
    summary = {}
    pair_obs = {}
    expected_labels = {}
    end_scale = {}
    end_owner = {}
    for state in prior.STATES:
        for axis in AXES:
            key = state + "/" + axis
            hold = labels[key + "/hold"]
            mini = labels[key + "/mini"]
            require(mini == p["mini"][key], "incorrect old checkpoint")
            mini_blob = read(proto / "mini" / (mini + ".json"), mini)
            old_c = struct.unpack("<d", struct.pack("<Q", json.loads(mini_blob)["coefficient_bits"]))[0]
            for h, k, c, n in ((mini, "mini", old_c, 48), (hold, "hold", 0.0, 0)):
                verify_snapshot(snaps, h, p, state, axis, bank, n, c)
                seen.add(h)
                expected_labels[key + "/" + k] = h
                end_scale[h] = c
                end_owner[h] = (state, axis)
            common = {}
            for rule in RULES:
                name = key + "/" + rule
                h = hold
                x = 0.0
                step = 0
                scales = [x]
                increases = halved = trials = 0
                last_loss = None
                for t in range(1, LIMIT + 1):
                    z = next(logs)
                    r = z["report"]
                    require((z["name"], z["iteration"], z["before"]) == (name, t, h), "bad trajectory")
                    stop, k = check_step(rule, x, step, r, z["objective"])
                    if rule == "fixed":
                        common[h] = (r["pairs"], r["after"], z["objective"])
                        start = r["loss"]
                    else:
                        start = r["baseline"]["loss"]
                        if h in common:
                            ds, c, measured = common[h]
                            require(
                                ds == r["derivatives"]
                                and (
                                    not k
                                    or (r["trials"][0]["coefficient"], r["trials"][0]["objective"])
                                    == (c, measured)
                                ),
                                "first proposal is not the ordinary fixed control",
                            )
                        if k > 1 and not stop:
                            halved += 1
                    if last_loss is not None:
                        require(near(start, last_loss), "lost current-state objective")
                    if t == 1 and axis == "aligned":
                        require(
                            near(r["gradient"], p["zero_gradient"][state])
                            and near(start, p["zero_loss"][state]),
                            "original all-edge adjoints did not replay",
                        )
                    h = z["after"]
                    x = r["after"]
                    step = r["step"]
                    require(z["step"] == step, "wrong step count")
                    end = z["objective"]
                    verify_snapshot(snaps, h, p, state, axis, bank, step, x)
                    if h in pair_obs:
                        require(pair_obs[h] == end, "same state, different observations")
                    pair_obs[h] = end
                    last_loss = end["loss"]
                    increases += last_loss > start
                    trials += k
                    seen.add(h)
                    scales.append(x)
                    costs["gradient_prompts"] += 64
                    costs["objective_prompts"] += 64 * (1 + k)
                    if t == EARLY or stop and t < EARLY:
                        expected_labels[name + "/early"] = h
                    if stop or t == LIMIT:
                        expected_labels[name + "/end"] = h
                        break
                for milestone in ("early", "end"):
                    tag = expected_labels[name + "/" + milestone]
                    s = obj(snaps / (tag + ".json"), tag)
                    end_scale[tag] = struct.unpack("<d", struct.pack("<Q", s["coefficient_bits"]))[0]
                    end_owner[tag] = (state, axis)
                summary[name] = dict(
                    calls=t,
                    accepted=step,
                    terminal=(r.get("status") if stop else "iteration_limit"),
                    trial_forwards=trials * 64,
                    updates_accepting_a_shorter_step=halved,
                    objective_increases=increases,
                    early_period_two=cycle(scales[: EARLY + 1]),
                    whole_period_two=cycle(scales),
                    coefficients=scales,
                )
    require(
        expected_labels == labels
        and next(logs, None) is None
        and {h + ".json" for h in seen} == {x.name for x in snaps.iterdir()},
        "roster or checkpoint mismatch",
    )
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    baseline = obj(proto / "original-training-logits.json", p["baseline"])
    bins = [io.BytesIO(read(out / k, fs[k], 24 * 64 * 64 * 4)) for k in ("logits.f32le", "ordinary.f32le")]
    events = iter(map(json.loads, io.BytesIO(read(out / "endpoints.jsonl", fs["endpoints.jsonl"]))))
    metrics = {}
    for h in sorted(set(labels.values())):
        z = next(events)
        state, axis = end_owner[h]
        c = end_scale[h]
        u, v = prior.factors(bank, state, axis)
        edited, d = prior.rank.merge_bytes(
            raw, prior.rank.f32s([0.0] * 32 + [c * (a * b) for a in u for b in v])
        )
        x, y = [s.read(64 * 64 * 4) for s in bins]
        require(
            len(x) == 64 * 64 * 4
            and x == y
            and (z["state"], z["axis"], z["checkpoint"]) == (state, axis, h)
            and sha(x) == z["scalar"] == z["ordinary"]
            and z["weight"] == sha(edited)
            and z["norm"] == math.hypot(*d),
            "physical readout or owner mismatch",
        )
        vals = {r["id"]: v for r, v in zip(rows, struct.iter_unpack("<64f", x), strict=True)}
        if c == 0:
            require(
                all(
                    [v[prior.task.IDLE], v[prior.task.ON], max(range(64), key=v.__getitem__)]
                    == baseline[state][k]
                    for k, v in vals.items()
                ),
                "wrong old baseline logits",
            )
        gaps = {k: v[prior.task.IDLE] - v[prior.task.ON] for k, v in vals.items()}
        ordered = [[gaps[a["id"]], gaps[b["id"]]] for a, b in ps]
        loss = pair_means([[a, b, 0.0, 0.0] for a, b in ordered])[0]
        if h in pair_obs:
            require(
                pair_obs[h]["pairs"] == ordered and near(pair_obs[h]["loss"], loss), "endpoint loss differs"
            )
        metrics[h] = prior.task.metrics(prior.task.key(rows), vals) | dict(
            pair_loss=loss,
            coefficient=c,
            correctly_ranked_pairs=sum(a - b > prior.task.EPS for a, b in ordered),
        )
        costs["endpoint_prompts_per_route"] += 64
    require(
        costs == m["counts"] and next(events, None) is None and all(not b.read(1) for b in bins),
        "unclosed evaluation",
    )
    by_name = {name: metrics[h] for name, h in labels.items()}
    gates = {}
    for s in prior.STATES:
        k = s + "/aligned/"

        def loss(what):
            return by_name[k + what]["pair_loss"]

        gates[s] = {r: all(loss(ref) - loss(r + "/early") > 1e-5 for ref in ("hold", "mini")) for r in RULES}
        gates[s]["guard_increment"] = (
            gates[s]["guarded"] and loss("fixed/early") - loss("guarded/early") > 1e-5
        )
    dst.mkdir(mode=0o700)
    return put(
        dst / "results.json",
        dict(
            plan=pp,
            measurement=mp,
            costs=costs,
            paths=summary,
            outcomes=by_name,
            gates=gates,
            scope="Training data only. No new generalization, language quality or equal-runtime claim.",
        ),
    )

"""Verify indexed fits, optimizer multiplicities and signed assignment/query effects without a model."""

import io
import itertools
import json
import math
import struct

from query_swap_binding import (
    CALLS,
    EPS,
    H,
    KIND,
    PLAN,
    RULES,
    SPACES,
    STATES,
    cases,
    closed,
    design,
    fit_request,
    obj,
    prev,
    put,
    rank,
    read,
    require,
    sha,
    sources,
    task,
)
from scalar_binding_readback import as_bits, near
from scalar_full_batch_audit import check_step
from slot_common_readback import archived, mean_vector, sub
from byte_release_artifacts import PACKAGE_ROOTS


def installation(proto, p, training):
    spec = p if training else p["reference"]
    inv = obj(
        proto / ("wheel-inventory.json" if training else "reference-wheel.json"), spec["inventory"], 2 << 20
    )
    require((inv["sha256"], inv["native"]["sha256"]) == (spec["wheel"], spec["native"]), "foreign wheel")
    return dict(
        version=inv["version"],
        wheel_sha256=spec["wheel"],
        native_sha256=spec["native"],
        package_files=sum(k.split("/", 1)[0] in PACKAGE_ROOTS for k in inv["files"]),
        all_original_files_match=True,
        python_bytecode_absent=True,
    )


def group_means(gs, pairs, labels):
    return [
        mean_vector([sub(gs[i], gs[j]) for (i, j), label in zip(pairs, labels, strict=True) if label == k])
        for k in range(8)
    ]


def verify_fit(proto, pp, fit, tp):
    p = obj(proto / "plan.json", pp)
    m = closed(proto, fit, pp, "train", None, tp)
    require(
        (p["kind"], p["settings"], p["code"]) == (KIND, PLAN, sources())
        and m["installation"] == installation(proto, p, True),
        "wrong study or wheel",
    )
    rows = obj(proto / "fit/rows.json", p["folds"]["fit"])
    require(rows == [r for r in task.deck("train") if not r["edge"] % 2], "not the fit edges")
    d = design(rows)
    fs = m["files"]
    bank = iter(map(json.loads, io.BytesIO(read(fit / "bank.jsonl", fs["bank.jsonl"], 1 << 20))))
    binary = io.BytesIO(read(fit / "gradients.f64le", fs["gradients.f64le"], 1 << 20))
    axes = obj(fit / "axes.json", m["axes"], 2 << 20)
    steps = iter(map(json.loads, io.BytesIO(read(fit / "steps.jsonl", fs["steps.jsonl"], 8 << 20))))
    factors = obj(proto / "factors.json", p["factors"])
    counts = dict.fromkeys(m["counts"], 0)
    ends = {}
    paths = {}
    seen = set()
    roster = {}
    for state, space in itertools.product(STATES, SPACES):
        t = next(bank)
        w = 32 * 32 if space == "dense" else 32
        b = binary.read(32 * w * 8)
        require(
            len(b) == 32 * w * 8
            and (t["state"], t["space"], t["sha"]) == (state, space, sha(b))
            and len(t["contrasts"]) == 32,
            "wrong bank rows",
        )
        gs = list(struct.iter_unpack(f"<{w}d", b))
        roster[state, space] = (gs, t["contrasts"])
        counts["bank_vjps"] += 32
        for rule in RULES:
            name = "/".join((state, space, rule))
            spec = axes[name]
            c = spec["certificate"]
            axis = spec["coordinates"]
            edges, labels, method = fit_request(d, rule)
            means = group_means(gs, edges, labels)
            norms = list(map(lambda x: math.hypot(*x), means))
            require(
                (
                    c["base"],
                    c["lora"],
                    c["geometry"],
                    c["readout"],
                    c["layer"],
                    c["width"],
                    c["target_id"],
                    c["alternative_id"],
                    c["fixed_left"],
                    c["rule"],
                )
                == (
                    p["base"],
                    p["lora"],
                    p["origins"][state],
                    "f32_wide_backward",
                    0,
                    32,
                    task.IDLE,
                    task.ON,
                    space == "left",
                    method,
                )
                and c["group_sizes"] == [labels.count(i) for i in range(8)]
                and all(near(x, y) for x, y in zip(norms, c["group_norms"], strict=True)),
                "wrong group partition or owner",
            )
            if axis is None:
                require(
                    c["refusal"] in ("zero_mean", "unresolved", "uncertified") and not c["slopes"],
                    "unexplained refusal",
                )
                continue
            require(
                c["refusal"] is None
                and near(math.hypot(*axis), 1.0)
                and all(near(prev.dot(g, axis), s) for g, s in zip(means, c["slopes"], strict=True)),
                "wrong raw slopes",
            )
            if method == "equal_mean":
                g = mean_vector(means)
                other = mean_vector(group_means(gs, d["Q"], d["groups"]))
                require(math.dist(g, other) < 1e-11, "balanced mean A != Q")
            else:
                a = c["unit_group_weights"]
                require(min(a) >= 0 and near(math.fsum(a), 1.0), "wrong hull weights")
                g = [
                    math.fsum(z * x / n for z, x, n in zip(a, col, norms, strict=True))
                    for col in zip(*means, strict=True)
                ]
                h = math.hypot(*g)
                margin = min(s / n for s, n in zip(c["slopes"], norms, strict=True))
                require(
                    margin > 0 and abs(c["hull_norm"] - h) < 1e-11 and abs(h - margin) < 1e-9,
                    "uncertified worst cosine",
                )
            gn = math.hypot(*g)
            require(
                gn > 0 and math.dist(axis, [x / gn for x in g]) < 1e-8,
                "direction not fitted exclusively on admitted rows",
            )
            if space == "dense":
                continue
            left = factors[state][0]
            step = 0
            x = 0.0
            initial = next(steps)
            require((initial["event"], initial["name"]) == ("start", name), "missing initial state")

            def snap(h):
                s = obj(fit / "checkpoints" / (h + ".json"), h)
                require(
                    s
                    == dict(
                        schema="spiraltorch.causal_lm.scalar_output.v1",
                        base=p["base"],
                        lora=p["lora"],
                        geometry=p["origins"][state],
                        readout="f32_wide_backward",
                        layer=0,
                        steps=step,
                        coefficient_bits=as_bits(x),
                        left_bits=list(map(as_bits, left)),
                        right_bits=list(map(as_bits, axis)),
                    ),
                    "foreign scalar checkpoint",
                )
                seen.add(h)

            snap(initial["checkpoint"])
            last = None
            for j in range(1, CALLS + 1):
                z = next(steps)
                r = z["report"]
                require(
                    (z["event"], z["name"], z["call"]) == ("step", name, j)
                    and r["gradient_prompt_evaluations"] == 32
                    and all(
                        o["prompt_evaluations"] == 32
                        for o in [r["baseline"], *(t["objective"] for t in r["trials"])]
                    ),
                    "extra decoder calls or wrong lane",
                )
                if j == 1:
                    for (i, k), v in zip(d["A"] + d["Q"], r["derivatives"], strict=True):
                        require(
                            (v[0], v[1])
                            == (
                                t["contrasts"][i]["target_minus_alternative"],
                                t["contrasts"][k]["target_minus_alternative"],
                            )
                            and near(v[2], prev.dot(gs[i], axis))
                            and near(v[3], prev.dot(gs[k], axis)),
                            "wrong native scalar seed",
                        )
                if last is not None:
                    require(last == r["baseline"], "current state did not replay")
                end = r["trials"][-1]["objective"] if r["status"] == "accepted" else r["baseline"]
                stop, n = check_step("guarded", x, step, r, end)
                step = r["step"]
                x = r["after"]
                last = end
                counts["optimizer_vjps"] += 32
                counts["optimizer_forwards"] += 32 * (1 + n)
                counts["updates"] += not stop
                snap(z["checkpoint"])
                if stop:
                    break
            ends[name] = z["checkpoint"]
            paths[name] = dict(calls=j, steps=step, coefficient=x)
    require(
        next(bank, None) is None
        and not binary.read(1)
        and next(steps, None) is None
        and counts == m["counts"]
        and ends == m["ends"]
        and paths == m["paths"]
        and {h + ".json" for h in seen} == {f.name for f in (fit / "checkpoints").iterdir()}
        and set(axes) == {"/".join(x) for x in itertools.product(STATES, SPACES, RULES)},
        "incomplete fit archive",
    )
    return roster


def cells(d, gaps):
    """Per edge, average the TWO orders at fixed identity. Do not collapse the two identities."""
    result = []
    for e in sorted({k[0] for k in d["keys"]}):
        vals = {}
        for kind in ("A", "Q"):
            vals[kind] = [
                prev.avg(
                    [
                        gaps[i] - gaps[j]
                        for (i, j), (edge, o, entity) in zip(d[kind], d["keys"], strict=True)
                        if edge == e and entity == k
                    ]
                )
                for k in (0, 1)
            ]
        require(abs(math.fsum(vals["A"]) - math.fsum(vals["Q"])) < 1e-12, "factorial interaction mismatch")
        result.append(vals["A"] + vals["Q"])
    return result


def delta_cells(a, b):
    return [sub(x, y) for x, y in zip(a, b, strict=True)]


def factorial(c):
    a, b, q, r = c
    common = (a + b) * 0.5
    state = (a - b) * 0.5
    query = (q - r) * 0.5
    require(
        abs(common - (q + r) * 0.5) < 1e-12 and abs(min(c) - (common - max(abs(state), abs(query)))) < 1e-12,
        "unmatched factorial cell",
    )
    return dict(shared=common, assignment_only=state, query_only=query, worst=min(c))


def metrics(rows, d, vs):
    gaps = [v[task.IDLE] - v[task.ON] for v in vs]
    correct = full = 0
    cost = []
    for r, v, gap in zip(rows, vs, gaps, strict=True):
        g = r["gold"]
        largest = max(v)
        cost.append(largest - v[g] + math.log(math.fsum(math.exp(x - largest) for x in v)))
        full += v[g] > max(x for j, x in enumerate(v) if j != g)
        correct += (gap if g == task.IDLE else -gap) > EPS
    out = {}
    for s in ("A", "Q"):
        ms = [gaps[i] - gaps[j] for i, j in d[s]]
        out[s] = dict(
            ranked=sum(x > EPS for x in ms),
            loss=prev.avg([max(-x, 0.0) + math.log1p(math.exp(-abs(x))) for x in ms]),
        )
    return out | dict(cells=cells(d, gaps), correct=correct, top1=full, nll=prev.avg(cost))


def audit(proto, pp, fit, tp, score, sp, archive, out):
    roster = verify_fit(proto, pp, fit, tp)
    p = obj(proto / "plan.json", pp)
    f = closed(proto, fit, pp, "train", None, tp)
    s = closed(proto, score, pp, "score", tp, sp)
    fs = s["files"]
    require(
        s["installation"] == installation(proto, p, False) and s["training"] == tp,
        "wrong independent score stage",
    )
    axes = obj(fit / "axes.json", f["axes"], 2 << 20)
    folds = {k: obj(proto / k / "rows.json", p["folds"][k]) for k in ("fit", "check")}
    rows = folds["fit"] + folds["check"]
    ds = {k: design(rr) for k, rr in folds.items()}
    require(
        rows == [r for i in (0, 1) for r in task.deck("train") if r["edge"] % 2 == i]
        and task.queries(rows) == obj(proto / "queries.json", p["queries"]),
        "foreign score cohort",
    )
    mats, gaps = archived(archive, p, folds)
    factors = obj(proto / "factors.json", p["factors"])
    replay = {}
    for (state, space), (gs, cs) in roster.items():
        err = 0.0
        for row, g, c in zip(folds["fit"], gs, cs, strict=True):
            old = mats[state][row["id"]]
            if space == "left":
                old = [prev.dot(factors[state][0], old[j::32]) for j in range(32)]
            err = max(err, math.dist(old, g))
            require(
                abs(gaps[state][row["id"]] - c["target_minus_alternative"]) < 2e-5
                and math.dist(old, g) <= 3e-6 + 3e-5 * math.hypot(*old),
                "foreign fit adjoint",
            )
        replay[state + "/" + space] = err
    logs = iter(map(json.loads, io.BytesIO(read(score / "cases.jsonl", fs["cases.jsonl"]))))
    bins = io.BytesIO(read(score / "full.f32le", fs["full.f32le"], 5000 * 256))
    overlays = io.BytesIO(read(score / "overlay.bin", fs["overlay.bin"], 5000 * 12))
    old = obj(proto / "baseline.json", p["baseline"])
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    counts = dict(ordinary=0, overlay=0, restored=0)
    hold = {}
    finite = {}
    learned = {}
    phys_metrics = {}
    route_errors = {}
    for state, name, kind, scale in cases(axes):
        z = next(logs)
        if kind == "learned":
            scale = f["paths"][name]["coefficient"]
        require(
            (z["state"], z["name"], z["kind"], z["scale"]) == (state, name, kind, scale), "incorrect case"
        )
        buf = bins.read(16384)
        require(len(buf) == 16384 and sha(buf) == z["full"], "bad whole-vocabulary rows")
        vs = list(struct.iter_unpack("<64f", buf))
        require(all(all(map(math.isfinite, v)) for v in vs), "nonfinite logits")
        counts["ordinary"] += 64
        measure = {
            k: metrics(rr, ds[k], vs[i * 32 : (i + 1) * 32]) for i, (k, rr) in enumerate(folds.items())
        }
        if kind == "baseline":
            require(
                z["weight"] == sha(raw)
                and z["displacement"] == 0.0
                and all(
                    [v[task.IDLE], v[task.ON], max(range(64), key=v.__getitem__)] == old[state][r["id"]]
                    for v, r in zip(vs, rows, strict=True)
                ),
                "old baseline changed",
            )
            hold[state] = measure
            continue
        u = axes[name]["coordinates"]
        if "/left/" in name:
            u = [x * y for x in factors[state][0] for y in u]
        require(near(math.hypot(*u), 1.0), "nonunit perturbation")
        edited, disp = rank.merge_bytes(raw, rank.f32s([0.0] * 32 + [scale * x for x in u]))
        require(
            z["weight"] == sha(edited) and near(math.hypot(*disp), z["displacement"]),
            "weights do not match fitted axis",
        )
        if kind == "learned":
            require(z["witness"] == z["full"], "restored endpoint differs")
            counts["restored"] += 64
            learned[name] = measure
            continue
        require(abs(z["displacement"] / abs(scale) - 1) < 1e-4, "unmatched radius")
        counts["overlay"] += 64
        b = overlays.read(768)
        require(len(b) == 768 and sha(b) == z["witness"], "wrong overlay")
        err = 0.0
        for v, (a, c, g) in zip(vs, struct.iter_unpack("<ffI", b), strict=True):
            require(math.isfinite(a - c) and 0 <= g < 64, "invalid overlay")
            err = max(
                err, abs(a - v[task.IDLE]), abs(c - v[task.ON]), abs((a - c) - (v[task.IDLE] - v[task.ON]))
            )
        route_errors[name] = max(route_errors.get(name, 0.0), err)
        for fold, rr in folds.items():
            grad_cells = cells(ds[fold], [prev.dot(mats[state][r["id"]], disp[32:]) for r in rr])
            phys_metrics[name, scale, fold] = dict(
                observed=delta_cells(measure[fold]["cells"], hold[state][fold]["cells"]), predicted=grad_cells
            )
    require(
        not bins.read(1) and not overlays.read(1) and next(logs, None) is None and counts == s["counts"],
        "incomplete native scores",
    )
    for name, v in axes.items():
        if v["coordinates"] is None:
            finite[name] = None
            continue
        data = {}
        agrees = True
        worst_err = 0.0
        for h, fold in itertools.product(H, folds):
            a, b = (phys_metrics[name, sg * h, fold] for sg in (1, -1))
            reg = []
            good = 0
            for pos, neg, dp, dn in zip(
                a["observed"], b["observed"], a["predicted"], b["predicted"], strict=True
            ):
                central = [x * 0.5 for x in sub(pos, neg)]
                pred = [x * 0.5 for x in sub(dp, dn)]
                err = sub(central, pred)
                worst_err = max(worst_err, *map(abs, err))
                agrees &= all(abs(z) <= 1e-6 + 0.1 * abs(x) for z, x in zip(err, pred, strict=True))
                good += min(*pos, *central, *pred) > EPS
                reg.append(dict(plus=pos, central=central, predicted=pred, components=factorial(pos)))
            data[str(h) + "/" + fold] = dict(edges=reg, all_four=good)
        finite[name] = dict(
            gate=agrees and route_errors[name] <= 8e-6 and data["0.02/check"]["all_four"] >= 3,
            agrees=agrees,
            max_error=worst_err,
            route_error=route_errors[name],
            data=data,
        )
    comparisons = {}
    for name, end in learned.items():
        state = name.split("/")[0]
        refs = {"hold": hold[state]}
        if state + "/left/mean" in learned:
            refs["mean"] = learned[state + "/left/mean"]
        detail = {}
        for label, ref in refs.items():
            changes = delta_cells(end["check"]["cells"], ref["check"]["cells"])
            detail[label] = dict(
                changes=changes,
                components=[factorial(c) for c in changes],
                all_four=sum(min(c) > EPS for c in changes),
                more_choices=end["check"]["correct"] > ref["check"]["correct"],
            )
        comparisons[name] = dict(
            vs=detail,
            gate=len(detail) == 2 and all(d["all_four"] >= 3 for d in detail.values()),
            more_choices=len(detail) == 2 and all(d["more_choices"] for d in detail.values()),
        )
    out.mkdir(mode=0o700)
    return put(
        out / "results.json",
        dict(
            plan=pp,
            training=tp,
            scoring=sp,
            counts={"train": f["counts"], "score": counts},
            paths=f["paths"],
            replay=replay,
            origins=hold,
            finite=finite,
            learned=learned,
            comparisons=comparisons,
            scope="Only offset-one training edges; no validation, pretrained model or broad language claim.",
        ),
    )

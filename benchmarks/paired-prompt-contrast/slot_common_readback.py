"""Model-free verification of fit provenance, dual certificates, physical displacements and slot effects."""

import io
import json
import math
import os
import struct

from scalar_binding_readback import install
from slot_common_output import (
    ARMS,
    EPS,
    H,
    KIND,
    SETTINGS,
    STATES,
    W,
    WALL,
    avg,
    dot,
    obj,
    paired,
    plane,
    put,
    rank,
    read,
    require,
    sha,
    signature,
    sources,
    task,
)


def mean_vector(rows):
    return [avg(col) for col in zip(*rows, strict=True)]


def sub(x, y):
    return [a - b for a, b in zip(x, y, strict=True)]


def archived(folder, p, folds):
    parent = obj(folder / "protocol/plan.json", p["pullback"])
    m = obj(folder / "measured/complete.json", p["measurement"])
    require(
        m["plan"] == p["pullback"] and parent["base"] == p["base"] and parent["lora"] == p["lora"],
        "wrong archived native model",
    )
    blocksize = 8 * (W + W * W) * 8 + 2 * 64 * 4
    binary = io.BytesIO(read(folder / "measured/pullbacks.bin", m["files"]["pullbacks.bin"], 256 * blocksize))
    mats = {s: {} for s in STATES}
    contrasts = {s: {} for s in STATES}
    wanted = {f: {r["id"] for r in rr} for f, rr in folds.items()}
    for line in read(folder / "measured/linear.jsonl", m["files"]["linear.jsonl"], 4 << 20).splitlines():
        t = json.loads(line)
        block = binary.read(blocksize)
        require(len(block) == blocksize and sha(block) == t["pullback"], "archived VJP checksum")
        if t["role"].endswith("/origin"):
            state = t["role"].split("/")[0]
            key = t["id"]
            require(
                parent["roles"][t["role"]] == p["origins"][state]
                and key in wanted[t["fold"]]
                and key not in mats[state],
                "foreign old row",
            )
            mats[state][key] = struct.unpack_from("<1024d", block, (3 * 1056 + 32) * 8)
            c = t["primal"]
            require((c["target_id"], c["alternative_id"]) == (task.IDLE, task.ON), "wrong old contrast")
            contrasts[state][key] = c["target_logit"] - c["alternative_logit"]
    require(not binary.read(1) and all(len(m) == 64 for m in mats.values()), "incomplete archived pullbacks")
    return mats, contrasts


def verify_fit(native, axes, pairs, archived_g, old_gap, bank):
    summaries = {}
    required = set()
    for state, space, r in native:
        require((state, space) not in required, "duplicate native bank")
        required.add((state, space))
        u, v = bank[state]
        gs = r["pair_gradients"]
        max_l2 = max_raw = 0.0
        for (a, b, slot, j), g, gap in zip(pairs, gs, r["contrasts"], strict=True):
            d = sub(archived_g[state][a["id"]], archived_g[state][b["id"]])
            if space == "left":
                d = [dot(u, d[k::W]) for k in range(W)]
            err = math.hypot(*sub(d, g))
            require(
                err <= 3e-6 + 3e-5 * math.hypot(*d), "native pair gradient differs from archived dense VJP"
            )
            max_l2 = max(max_l2, err)
            max_raw = max(
                max_raw, abs(gap[0] - old_gap[state][a["id"]]), abs(gap[1] - old_gap[state][b["id"]])
            )
        require(max_raw <= 2e-5, "native and archived fit contrasts differ")
        for rule, spec in r["directions"].items():
            labels = [j if rule == "edge_slot" else slot for _, _, slot, j in pairs]
            groups = [
                [g for label, g in zip(labels, gs, strict=True) if label == k] for k in range(1 + max(labels))
            ]
            means = list(map(mean_vector, groups))
            norms = [math.hypot(*g) for g in means]
            a = spec["coordinates"]
            name = space + "/" + rule
            require(
                spec["group_sizes"] == list(map(len, groups))
                and len(spec["group_norms"]) == len(norms)
                and all(
                    math.isclose(x, y, abs_tol=1e-14, rel_tol=1e-12)
                    for x, y in zip(spec["group_norms"], norms, strict=True)
                ),
                "wrong group scale",
            )
            if a is None:
                require(
                    spec["refusal"] in ("unresolved", "zero_mean", "uncertified")
                    and axes[state][name] is None,
                    "invalid refusal",
                )
                continue
            if rule == "mean":
                g = mean_vector(means)
            else:
                alpha = spec["unit_group_weights"]
                require(
                    len(alpha) == len(norms) and min(alpha) >= 0.0 and abs(math.fsum(alpha) - 1) < 1e-12,
                    "bad dual",
                )
                g = [
                    math.fsum(z * x / n for z, x, n in zip(alpha, col, norms, strict=True))
                    for col in zip(*means, strict=True)
                ]
                h = math.hypot(*g)
                margin = min(dot(x, a) / n for x, n in zip(means, norms, strict=True))
                require(
                    abs(h - spec["hull_norm"]) < 1e-11 and abs(h - margin) < 1e-9 and margin > 0,
                    "uncertified common slope",
                )
            gn = math.hypot(*g)
            require(
                gn > 0
                and math.hypot(*sub(a, [x / gn for x in g])) < 1e-8
                and all(
                    math.isclose(dot(m, a), s, rel_tol=1e-9, abs_tol=1e-12)
                    for m, s in zip(means, spec["slopes"], strict=True)
                ),
                "wrong fitted direction",
            )
            require(
                axes[state][name] == ([x * y for x in u for y in a] if space == "left" else a),
                "wrong expanded weights",
            )
        if space == "left":
            a = plane(gs, [s for _, _, s, j in pairs], v)
            require(
                axes[state]["plane/raw"] == ([x * y for x in u for y in a] if a else None),
                "wrong raw-plane control",
            )
        summaries[state + "/" + space] = dict(
            max_archived_gradient_l2_error=max_l2, max_archived_gap_error=max_raw
        )
    require(required == {(s, k) for s in STATES for k in ("dense", "left")}, "incomplete fit bank")
    return summaries


def gap(v):
    return v[task.IDLE] - v[task.ON]


def fold_metrics(rr, values):
    ps = paired(rr)
    margins = [gap(values[p["id"]]) - gap(values[n["id"]]) for p, n, s, j in ps]
    slots = [[m for m, (_, _, s, _) in zip(margins, ps, strict=True) if s == slot] for slot in (0, 1)]

    def correct(r):
        return (
            values[r["id"]][r["gold"]] - values[r["id"]][task.ON if r["gold"] == task.IDLE else task.IDLE]
        ) > EPS

    costs = []
    full = 0
    for r in rr:
        v = values[r["id"]]
        g = r["gold"]
        largest = max(v)
        costs.append(largest - v[g] + math.log(math.fsum(math.exp(x - largest) for x in v)))
        full += v[g] > max(x for i, x in enumerate(v) if i != g)

    def logistic(x):
        return max(-x, 0.0) + math.log1p(math.exp(-abs(x)))

    return dict(
        margins=margins,
        slot_means=[avg(s) for s in slots],
        slot_pair_loss=[avg([logistic(x) for x in s]) for s in slots],
        ranked_by_slot=[sum(x > EPS for x in s) for s in slots],
        two_choice=sum(map(correct, rr)),
        vocab_top1=full,
        gold_nll=avg(costs),
    )


def test_gate(table, agreement):
    big = table[str(H[-1])]["check"]
    return (
        agreement
        and all(min(big[k]) > EPS for k in ("plus", "central", "predicted"))
        and min(big["positive_edges"]) >= 3
        and all(cell["agrees"] for scale in table.values() for cell in scale.values())
    )


def audit(proto, pp, out, mp, archive, dst):
    p = obj(proto / "plan.json", pp)
    m = obj(out / "complete.json", mp)
    require(
        p["kind"] == KIND
        and p["settings"] == SETTINGS
        and p["code"] == sources()
        and m["plan"] == pp
        and m["installation"] == install(proto, p),
        "wrong plan or wheel",
    )
    require(
        obj(proto / "attempt.json", signature(out / "attempt.json"))
        == dict(plan=pp, attempt=1, output=os.path.relpath(out, proto)),
        "wrong attempt",
    )
    for phase in ("native", "physical"):
        x = json.loads((out / (phase + "-exit.json")).read_bytes())
        log = read(out / (phase + ".log"), x["log"])
        require(
            x["exit"] == 0
            and 0 < x["cumulative_seconds"] <= WALL
            and (phase == "native" or log.splitlines()[-1:] == [mp.encode()]),
            "unsuccessful physical phase",
        )
    sealed = obj(out / "axes.json", m["axes"], 2 << 20)
    axes = sealed["weights"]
    require(
        (sealed["plan"], sealed["native_fit"], sealed["fit"]) == (pp, m["native_fit"], p["folds"]["fit"]),
        "unfrozen physical axes",
    )
    native = [json.loads(line) for line in read(out / "native.jsonl", m["native_fit"], 4 << 20).splitlines()]
    folds = {k: obj(proto / (k + ".json"), p["folds"][k]) for k in ("fit", "check")}
    rows = folds["fit"] + folds["check"]
    require(
        rows == [r for i in (0, 1) for r in task.deck("train") if r["edge"] % 2 == i],
        "not the expected training folds",
    )
    mats, old_gap = archived(archive, p, folds)
    witness = verify_fit(
        native, axes, paired(folds["fit"]), mats, old_gap, obj(proto / "factors.json", p["factors"])
    )
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    counts = dict(original=0, ordinary=0, overlay=0)
    fs = m["files"]
    logs = iter(map(json.loads, io.BytesIO(read(out / "cases.jsonl", fs["cases.jsonl"]))))
    binary = io.BytesIO(read(out / "rows.f32le", fs["rows.f32le"], (128 + len(ARMS) * 512) * 256))
    oracle = io.BytesIO(read(out / "overlay.bin", fs["overlay.bin"], len(ARMS) * 512 * 12))
    old = obj(proto / "baseline.json", p["baseline"])
    results = {}
    starts = {}
    gates = {}

    def read_rows(t):
        b = binary.read(64 * 256)
        require(len(b) == 64 * 256 and t["values"] == sha(b), "bad logit block")
        vs = {r["id"]: v for r, v in zip(rows, struct.iter_unpack("<64f", b), strict=True)}
        require(all(all(map(math.isfinite, v)) for v in vs.values()), "nonfinite ordinary logits")
        return vs

    for state in STATES:
        t = next(logs)
        require((t["state"], t["arm"]) == (state, "origin"), "bad baseline")
        vs = read_rows(t)
        counts["original"] += 64
        require(
            all(
                [v[task.IDLE], v[task.ON], max(range(64), key=v.__getitem__)] == old[state][k]
                for k, v in vs.items()
            ),
            "old training baseline changed",
        )
        initial = {k: fold_metrics(rr, vs) for k, rr in folds.items()}
        starts[state] = initial
        results[state] = {}
        gates[state] = {}
        for arm in ARMS:
            unit = axes[state][arm]
            table = {}
            max_error = 0.0
            max_pred_error = 0.0
            greedy_diffs = 0
            if unit is None:
                results[state][arm] = None
                gates[state][arm] = False
                continue
            for h in H:
                signs = {}
                cut = {}
                for side in (-1, 1):
                    z = next(logs)
                    require(
                        (z["state"], z["arm"], z["radius"], z["side"]) == (state, arm, h, side), "wrong trial"
                    )
                    d = rank.f32s([0.0] * W + [side * h * x for x in unit])
                    changed, disp = rank.merge_bytes(raw, d)
                    require(
                        sha(d) == z["delta"]
                        and sha(changed) == z["weight"]
                        and abs(math.hypot(*disp) / h - 1) < 1e-4
                        and math.hypot(*disp) == z["norm"],
                        "incorrect physical displacement",
                    )
                    signs[side] = disp[W:]
                    v = read_rows(z)
                    triplets = oracle.read(64 * 12)
                    require(len(triplets) == 64 * 12 and sha(triplets) == z["overlay"], "wrong overlay")
                    for ids, (a, b, g) in zip(rows, struct.iter_unpack("<ffI", triplets), strict=True):
                        w = v[ids["id"]]
                        require(math.isfinite(a - b) and g < 64, "invalid overlay")
                        max_error = max(
                            max_error, abs(a - w[task.IDLE]), abs(b - w[task.ON]), abs(a - b - gap(w))
                        )
                        greedy_diffs += g != max(range(64), key=w.__getitem__)
                    cut[side] = {k: fold_metrics(rr, v) for k, rr in folds.items()}
                    counts["ordinary"] += 64
                    counts["overlay"] += 64
                details = {}
                half = [(a - b) / 2 for a, b in zip(signs[1], signs[-1], strict=True)]
                for fold, rr in folds.items():
                    o = initial[fold]
                    pos, neg = (cut[s][fold] for s in (1, -1))
                    plus = sub(pos["slot_means"], o["slot_means"])
                    central = [(a - b) / 2 for a, b in zip(pos["slot_means"], neg["slot_means"], strict=True)]
                    ps = paired(rr)
                    pred = []
                    edges = []
                    for slot in (0, 1):
                        grad = mean_vector(
                            [
                                sub(mats[state][a["id"]], mats[state][b["id"]])
                                for a, b, s, j in ps
                                if s == slot
                            ]
                        )
                        pred.append(dot(grad, half))
                        gain = sub(pos["margins"], o["margins"])
                        edges.append(
                            [
                                avg(
                                    [
                                        v
                                        for (a, b, s, j), v in zip(ps, gain, strict=True)
                                        if s == slot and a["edge"] == e
                                    ]
                                )
                                for e in sorted({r["edge"] for r in rr})
                            ]
                        )
                    errs = list(map(abs, sub(central, pred)))
                    max_pred_error = max(max_pred_error, *errs)
                    details[fold] = dict(
                        plus=plus,
                        central=central,
                        predicted=pred,
                        agrees=all(e <= 1e-6 + 0.10 * abs(m) for e, m in zip(errs, pred, strict=True)),
                        by_edge=edges,
                        positive_edges=[sum(x > EPS for x in ee) for ee in edges],
                        plus_metrics={k: v for k, v in pos.items() if k != "margins"},
                    )
                table[str(h)] = details
            results[state][arm] = dict(
                radii=table,
                max_row_error=max_error,
                max_prediction_error=max_pred_error,
                greedy_disagreements=greedy_diffs,
            )
            gates[state][arm] = test_gate(table, max_error <= 8e-6)
    require(
        next(logs, None) is None and not binary.read(1) and not oracle.read(1) and counts == m["counts"],
        "incomplete observations",
    )
    dst.mkdir(mode=0o700)
    return put(
        dst / "results.json",
        dict(
            plan=pp,
            measurement=mp,
            counts=counts,
            starts=starts,
            fit_replay=witness,
            results=results,
            gates=gates,
            both_origins={a: all(gates[s][a] for s in STATES) for a in ARMS},
            scope="Prior training edges only. Positive finite changes are not new-task accuracy or learned endpoints.",
        ),
    )

"""Model-free readback of base/adapter provenance, archived dense replay, and finite rank-one changes."""

import io
import json
import math
import os
import struct

from rank_one_projection import (
    ARMS,
    KIND,
    LABEL,
    PLAN,
    ROUTES,
    W,
    axes,
    dot,
    f32s,
    merge_bytes,
    obj,
    pr,
    put,
    read,
    require,
    sha,
    source_hashes,
    vals,
)
from byte_release_artifacts import PACKAGE_ROOTS


def mean(xs):
    return math.fsum(xs) / len(xs)


def pin_rows(folder, filename, pins, limit):
    return io.BytesIO(read(folder / filename, pins[filename], limit))


def reference(parent, p, folds):
    src = parent / "measured"
    old = obj(parent / "protocol/plan.json", p["parent"])
    pr.admit(parent / "protocol", old)
    result = obj(src / "complete.json", p["measurement"])
    require(
        result["plan"] == p["parent"]
        and result["counts"] == pr.MAX
        and all(
            old[k] == p[k]
            for k in ("base", "lora", "roles", "wheel", "native", "inventory", "files", "folds")
        ),
        "wrong parent",
    )
    d = obj(src / "directions.json", result["directions"], 4 << 20)
    r = obj(parent / "rank-one-candidates.json", p["roster"])
    require(
        d["plan"] == r["plan"] == p["parent"]
        and d["fit"] == r["fit"] == p["folds"]["fit"]
        and result["directions"] == p["directions"] == r["directions"]
        and r["model_calls"] == 0,
        "directions or roster do not descend from fit rows",
    )
    pins = result["files"]
    lines = iter(pin_rows(src, "linear.jsonl", pins, 4 << 20))
    width = W + W * W
    size = 8 * width * 8 + 2 * pr.VECTOR.size
    binary = pin_rows(src, "pullbacks.bin", pins, size * 256)
    primals = {role: [] for role in pr.ROLES}
    gradients = {}
    construction = {}
    for fold, rows in folds.items():
        for role in pr.ROLES:
            gs = []
            for row in rows:
                meta = json.loads(next(lines))
                block = binary.read(size)
                require(
                    len(block) == size
                    and (meta["role"], meta["id"], meta["fold"], meta["pullback"])
                    == (role, row["id"], fold, sha(block)),
                    "wrong archived row",
                )
                v = meta["primal"]
                primals[role].append([v["target_logit"], v["alternative_logit"], v["greedy_id"]])
                g = struct.unpack_from(f"<{width}d", block, 3 * width * 8)
                gs.append([pr.sign(row) * x for x in g])
            gradient = [mean(col) for col in zip(*gs, strict=True)]
            gradients[role, fold] = gradient
            if fold == "fit":
                norm = math.hypot(*gradient)
                require(d["roles"][role][LABEL] == [x / norm for x in gradient], "direction not fit-only")
                construction[role] = axes(d["roles"][role][LABEL], r["roles"][role][LABEL])
    require(next(lines, None) is None and binary.read(1) == b"", "trailing derivative evidence")
    binary = pin_rows(src, "contrasts.bin", pins, pr.MAX["finite"] * pr.VALUE.size)
    dense = {}
    for line in pin_rows(src, "finite.jsonl", pins, 4 << 20):
        meta = json.loads(line)
        block = binary.read(64 * pr.VALUE.size)
        require(len(block) == 64 * pr.VALUE.size and meta["values"] == sha(block), "wrong finite evidence")
        if meta["label"] == LABEL:
            key = (meta["role"], meta["radius"], meta["sign"])
            require(key not in dense, "repeated dense reference")
            dense[key] = block
    require(not binary.read(1) and len(dense) == 16, "missing dense comparison")
    return construction, primals, gradients, dense


def metrics(rows, baseline, minus, plus, predicted):
    require(len(rows) == len(baseline) == len(minus) == len(plus) == 32, "incomplete fold")

    def gap(v):
        return v[0] - v[1]

    def choice(x):
        return (x > 0) - (x < 0)

    c, dp, flips, greedy, ties = [], [], [], [], []
    for r, b, n, p in zip(rows, baseline, minus, plus, strict=True):
        sg = pr.sign(r)
        c.append(sg * (gap(p) - gap(n)) / 2)
        dp.append(sg * (gap(p) - gap(b)))
        flips.append([choice(gap(z)) != choice(gap(b)) for z in (n, p)])
        greedy.append([z[2] != b[2] for z in (n, p)])
        ties.append([gap(z) == 0 for z in (n, p)])
    central = mean(c)

    def by_sign(flags):
        return [sum(v) for v in zip(*flags, strict=True)]

    return dict(
        central=central,
        predicted=predicted,
        prediction_ok=abs(central - predicted) <= 1e-6 + 0.10 * abs(predicted),
        plus=mean(dp),
        by_edge={
            str(e): mean([v for v, r in zip(c, rows, strict=True) if r["edge"] == e])
            for e in sorted({r["edge"] for r in rows})
        },
        by_order=[mean([v for v, r in zip(c, rows, strict=True) if r["order"] == o]) for o in (0, 1)],
        two_choice_flips=by_sign(flips),
        full_greedy_flips=by_sign(greedy),
        ties=by_sign(ties),
    )


def effect_gate(dense, rank1, control, h):
    d, v, c = (x["central"] for x in (dense, rank1, control))
    return (
        d > 0
        and v >= 0.90 * d
        and v > max(pr.RESOLVE, 2 * abs(c))
        and (
            h != pr.RADII[-1]
            or (
                rank1["plus"] > 0
                and min(rank1["by_order"]) > 0
                and sum(x > pr.RESOLVE for x in rank1["by_edge"].values()) >= 3
            )
        )
    )


def packed(rows):
    require(
        len(rows) == 64 and all(0 <= g < 64 and math.isfinite(a - b) for a, b, g in rows),
        "bad physical values",
    )
    return b"".join(pr.VALUE.pack(*r) for r in rows)


def adapter_record(folder, owner, kind):
    require(
        isinstance(owner, str)
        and owner.startswith("sha256:")
        and len(owner) == 71
        and all(c in "0123456789abcdef" for c in owner[7:]),
        "invalid adapter identifier",
    )
    with (folder / (owner[7:] + ".json")).open("rb") as f:
        raw = f.read(100001)
    prefix = b"spiraltorch.native_lora.v1\0" if kind == "lora" else b""
    require(len(raw) <= 100000 and sha(prefix + raw) == owner[7:], "wrong adapter bytes")
    return json.loads(raw)


def audit(parent, proto, pp, out, mp, dest):
    p = obj(proto / "plan.json", pp)
    require(p["kind"] == KIND and p["settings"] == PLAN and p["code"] == source_hashes(), "unfrozen audit")
    m = obj(out / "complete.json", mp)
    ex = json.loads((out / "exit.json").read_bytes())
    claim = dict(plan=pp, output=os.path.relpath(out, proto), attempt=1)
    inv = obj(proto / "wheel-inventory.json", p["inventory"], 2 << 20)
    require(
        m["plan"] == pp
        and claim == obj(proto / "attempt.json", pr.signature(out / "attempt.json"))
        and m["counts"] == PLAN["calls"]
        and ex["exit"] == 0
        and 0 < ex["elapsed"] <= PLAN["wall_seconds"]
        and read(out / "process.log", ex["log"]).splitlines()[-1:] == [mp.encode()]
        and m["installed"]
        == dict(
            version=inv["version"],
            wheel_sha256=p["wheel"],
            native_sha256=p["native"],
            all_original_files_match=True,
            python_bytecode_absent=True,
            package_files=sum(k.split("/", 1)[0] in PACKAGE_ROOTS for k in inv["files"]),
        ),
        "invalid run identity",
    )
    folds = {fold: obj(proto / (fold + ".json"), p["folds"][fold]) for fold in ("fit", "check")}
    for parity, rows in enumerate(folds.values()):
        pr.fold_rows(rows, parity)
    units, primals, gradients, dense_ref = reference(parent, p, folds)
    require(obj(proto / "axes.json", p["axes"], 1 << 20) == units, "construction mismatch")
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    seq = iter(map(json.loads, io.BytesIO(read(out / "observations.jsonl", m["observations"], 2 << 20))))
    counts = dict.fromkeys(PLAN["calls"], 0)
    scores, checks, gates = {}, {}, {}
    archive_names = set()
    adapter_names = {p["lora"][7:] + ".json"}
    lo = adapter_record(out / "adapters", p["lora"], "lora")
    require(lo["base_model_identity"] == p["base"] and lo["layer"] == 1, "wrong original LoRA")
    for role in pr.ROLES:
        geo = adapter_record(proto / "model", p["roles"][role], "geometry")
        require(
            geo["base_model_identity"] == p["base"]
            and [s["path"] for s in geo["placements"]] == ["transformer.h.1.attn"],
            "wrong original geometry",
        )
        b = next(seq)
        require(
            (b["role"], b["arm"]) == (role, "baseline") and packed(b["values"]) == packed(primals[role]),
            "original logits no longer byte-replay",
        )
        counts["baseline"] += len(b["values"])
        result = {r: {} for r in ROUTES}
        max_row, max_fold, greedy_disagreements = 0.0, 0.0, 0
        for arm in ARMS:
            for route in ROUTES:
                result[route][arm] = {}
            for h in pr.RADII:
                observations, vectors = {}, {}
                for s in (-1, 1):
                    z = next(seq)
                    require(
                        (z["role"], z["arm"], z["radius"], z["sign"]) == (role, arm, h, s),
                        "wrong trial order",
                    )
                    delta = f32s([s * h * x for x in units[role][arm]])
                    changed, actual = merge_bytes(raw, delta)
                    archive = sha(changed) + ".safetensors"
                    require(
                        z["weights"] + ".safetensors" == archive
                        and read(out / "weights" / archive, z["weights"], 100000) == changed
                        and z["delta"] == sha(delta)
                        and z["displacement"] == math.hypot(*actual)
                        and abs(z["displacement"] / h - 1) < 1e-4
                        and len(z["owners"]) == 3
                        and all(
                            isinstance(x, str) and x.startswith("sha256:") and len(x) == 71
                            for x in z["owners"]
                        )
                        and all(
                            x != y
                            for x, y in zip(
                                z["owners"], (p["base"], p["lora"], p["roles"][role]), strict=True
                            )
                        ),
                        "not the specified reimported displacement",
                    )
                    archive_names.add(archive)
                    new_base, new_lo, new_geo = z["owners"]
                    for kind, owner, original in (("lora", new_lo, lo), ("geometry", new_geo, geo)):
                        require(
                            adapter_record(out / "adapters", owner, kind)
                            == original | {"base_model_identity": new_base},
                            "rebound adapter parameters differ",
                        )
                        adapter_names.add(owner[7:] + ".json")
                    for route, v in (("overlay", vals(delta)), ("merged", actual)):
                        block = packed(z[route])
                        if arm == "dense" and route == "overlay":
                            require(block == dense_ref[role, h, s], "original dense does not byte-replay")
                        observations[route, s] = z[route]
                        vectors[route, s] = v
                        counts[route] += len(z[route])
                    for (a, b1, g), (x, y, g2) in zip(z["overlay"], z["merged"], strict=True):
                        max_row = max(max_row, abs(a - x), abs(b1 - y), abs((a - b1) - (x - y)))
                        greedy_disagreements += g != g2
                for route in ROUTES:
                    delta = [(a - b) / 2 for a, b in zip(vectors[route, 1], vectors[route, -1], strict=True)]
                    result[route][arm][str(h)] = {}
                    for i, (fold, rows) in enumerate(folds.items()):
                        sl = slice(i * 32, (i + 1) * 32)
                        result[route][arm][str(h)][fold] = metrics(
                            rows,
                            primals[role][sl],
                            observations[route, -1][sl],
                            observations[route, 1][sl],
                            dot(gradients[role, fold], delta),
                        )
                for fold in folds:
                    x, y = (result[r][arm][str(h)][fold]["central"] for r in ROUTES)
                    max_fold = max(max_fold, abs(x - y))
        checks[role] = dict(
            max_row=max_row,
            max_fold=max_fold,
            greedy_disagreements=greedy_disagreements,
            routes_agree=max_row <= 8e-6 and max_fold <= 1e-6,
        )
        all_pred = all(
            result[r][a][str(h)][f]["prediction_ok"]
            for r in ROUTES
            for a in ARMS
            for h in pr.RADII
            for f in folds
        )
        gates[role] = dict(
            predictions=all_pred,
            routes=checks[role]["routes_agree"],
            effect=all(
                effect_gate(*(result[r][a][str(h)]["check"] for a in ARMS), h)
                for r in ROUTES
                for h in pr.RADII
            ),
        )
        scores[role] = result
    require(
        next(seq, None) is None
        and counts == PLAN["calls"]
        and archive_names == {x.name for x in (out / "weights").iterdir()}
        and adapter_names == {x.name for x in (out / "adapters").iterdir()},
        "incomplete or extra observation",
    )
    dest.mkdir(mode=0o700)
    return put(
        dest / "results.json",
        dict(
            plan=pp,
            measurement=mp,
            counts=counts,
            scores=scores,
            agreement=checks,
            gates=gates,
            origins=all(all(gates[s + "/origin"].values()) for s in pr.task.STATES),
            endpoints=all(all(gates[s + "/direct_matched"].values()) for s in pr.task.STATES),
            scope=PLAN["scope"],
        ),
    )

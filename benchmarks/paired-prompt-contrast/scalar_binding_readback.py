"""No-model verification of all scalar updates and label join after a committed first score."""

import io
import json
import math
import struct

from scalar_binding_study import (
    ARMS,
    KIND,
    MAX_ABS,
    MAX_STEP,
    RATE,
    SETTINGS,
    STATES,
    STEPS,
    closed,
    code,
    factors,
    fresh,
    obj,
    pairs,
    put,
    rank,
    read,
    require,
    sha,
    task,
    validate_splits,
)
from byte_release_artifacts import PACKAGE_ROOTS


def as_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def near(x, y):
    return math.isclose(x, y, rel_tol=2e-13, abs_tol=1e-14)


def gap(row):
    return row[task.IDLE] - row[task.ON]


def pair_means(observations):
    losses = []
    grads = []
    for a, b, da, db in observations:
        require(all(map(math.isfinite, (a, b, da, db))), "nonfinite pair")
        x = b - a
        exp = math.exp(-abs(x))
        losses.append(max(0.0, x) + math.log1p(exp))
        grads.append((db - da) * (1.0 if x > 0 else exp) / (1.0 + exp))
    return math.fsum(losses) / len(losses), math.fsum(grads) / len(grads)


def install(proto, p):
    inv = obj(proto / "wheel-inventory.json", p["inventory"], 2 << 20)
    require((inv["sha256"], inv["native"]["sha256"]) == (p["wheel"], p["native"]), "wrong wheel")
    return dict(
        version=inv["version"],
        wheel_sha256=p["wheel"],
        native_sha256=p["native"],
        package_files=sum(k.split("/", 1)[0] in PACKAGE_ROOTS for k in inv["files"]),
        all_original_files_match=True,
        python_bytecode_absent=True,
    )


def training_audit(proto, pp, trained, tp):
    p = obj(proto / "plan.json", pp)
    require(p["kind"] == KIND and p["settings"] == SETTINGS and p["code"] == code(), "wrong plan")
    m = closed(proto, trained, pp, "train", None, tp)
    rows = obj(proto / "fit/rows.json", p["train"])
    require(
        rows == task.deck("train") and m["installation"] == install(proto, p), "wrong training data or wheel"
    )
    bank = obj(proto / "factors.json", p["bank"])
    seen = set()
    stats = {}
    ends = {}
    logs = iter(map(json.loads, io.BytesIO(read(trained / "steps.jsonl", m["steps"], 4 << 20))))
    for state in STATES:
        for arm in ARMS:
            name = state + "/" + arm
            u, v = factors(bank, state, arm)
            n = 0 if arm == "hold" else STEPS
            scalar = 0.0
            means = []
            clipped = 0
            for t in range(n + 1):
                z = next(logs)
                require(
                    z["name"] == name and z["event"] == ("start" if t == 0 else "step"),
                    "wrong training event",
                )
                if t:
                    group = pairs(rows, t - 1)
                    r = z["report"]
                    require(
                        z["ids"] == [[x["id"], y["id"]] for x, y in group]
                        and r["step"] == t
                        and len(r["pairs"]) == 4
                        and scalar == r["before"],
                        "wrong batch or old coordinate",
                    )
                    loss, g = pair_means(r["pairs"])
                    require(near(g, r["gradient"]) and near(loss, r["loss"]), "wrong logistic derivative")
                    proposed = -RATE * r["gradient"]
                    clipped += abs(proposed) > MAX_STEP
                    scalar = max(-MAX_ABS, min(MAX_ABS, scalar + max(-MAX_STEP, min(MAX_STEP, proposed))))
                    require(r["after"] == scalar, "not one clipped SGD step")
                    means.append(loss)
                h = z["checkpoint"]
                seen.add(h)
                snap = obj(trained / "checkpoints" / (h + ".json"), h)
                require(
                    snap
                    == dict(
                        schema="spiraltorch.causal_lm.scalar_output.v1",
                        base=p["base"],
                        lora=p["lora"],
                        geometry=p["origins"][state],
                        readout="f32_wide_backward",
                        layer=0,
                        steps=t,
                        coefficient_bits=as_bits(scalar),
                        left_bits=list(map(as_bits, u)),
                        right_bits=list(map(as_bits, v)),
                    ),
                    "checkpoint owner, floating-point bits, axis or counter changed",
                )
            ends[name] = h
            stats[name] = dict(
                coefficient=scalar,
                steps=n,
                clipped_steps=clipped,
                first_epoch_online_loss=(math.fsum(means[:8]) / 8 if means else None),
                last_epoch_online_loss=(math.fsum(means[-8:]) / 8 if means else None),
            )
    require(
        next(logs, None) is None
        and {h + ".json" for h in seen} == {x.name for x in (trained / "checkpoints").iterdir()}
        and m["ends"] == ends
        and (m["updates"], m["backwards"]) == (SETTINGS["updates"], SETTINGS["backwards"]),
        "incomplete steps or unexpected checkpoint",
    )
    return stats


def decision(by_arm):
    aligned = by_arm["aligned"]
    compare = {k: task.advantage(aligned, by_arm[k]) for k in ("hold", "orthogonal")}
    return dict(
        comparisons=compare,
        practical=all(v["passed"] for v in compare.values()),
        resolved=all(
            v["delta"] > task.EPS and v["positive_edges"] >= 6 and min(v["by_order"]) > task.EPS
            for v in compare.values()
        ),
        more_correct=all(aligned["correct_choices"] > by_arm[k]["correct_choices"] for k in compare),
        competence=aligned["complete_edges"] >= 6
        and all(aligned["complete_edges"] > by_arm[k]["complete_edges"] for k in compare),
    )


def audit(proto, pp, trained, tp, scored, sp, out):
    stats = training_audit(proto, pp, trained, tp)
    p = obj(proto / "plan.json", pp)
    m = closed(proto, scored, pp, "score", tp, sp)
    train = closed(proto, trained, pp, "train", None, tp)
    require(
        m["installation"] == train["installation"]
        and m["training"] == tp
        and m["readouts_per_route"] == SETTINGS["readouts_per_route"],
        "wrong score stage",
    )
    decks = {"train": task.deck("train"), "fresh": fresh()}
    validate_splits(decks["train"], decks["fresh"])
    for split, rows in decks.items():
        require(
            obj(proto / "keys" / (split + ".json"), p["decks"][split]["labels"]) == task.key(rows)
            and obj(proto / "blind" / (split + ".json"), p["decks"][split]["queries"]) == task.queries(rows),
            "key mismatch",
        )
    logs = iter(map(json.loads, io.BytesIO(read(scored / "cases.jsonl", m["files"]["cases.jsonl"]))))
    total = 64 * 4 * SETTINGS["readouts_per_route"]
    streams = [io.BytesIO(read(scored / k, m["files"][k], total)) for k in ("logits.f32le", "ordinary.f32le")]
    bank = obj(proto / "factors.json", p["bank"])
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    old = obj(proto / "original-training-logits.json", p["baseline"])
    scores = {}
    gates = {}
    changes = {}
    replayed = 0
    for state in STATES:
        scores[state] = {k: {} for k in decks}
        changes[state] = {}
        gates[state] = {}
        choices = {}
        for arm in ARMS:
            name = state + "/" + arm
            u, v = factors(bank, state, arm)
            c = stats[name]["coefficient"]
            edited, actual = rank.merge_bytes(
                raw, rank.f32s([0.0] * 32 + [c * (x * y) for x in u for y in v])
            )
            for split in sorted(decks):
                rows = decks[split]
                z = next(logs)
                x, y = [s.read(64 * 64 * 4) for s in streams]
                require(
                    (z["name"], z["split"], z["checkpoint"]) == (name, split, train["ends"][name])
                    and len(x) == 64 * 64 * 4
                    and x == y
                    and sha(x) == z["scalar"] == z["ordinary"]
                    and sha(edited) == z["weight"]
                    and z["norm"] == math.hypot(*actual),
                    "physical checkpoint or full decoder differs",
                )
                values = {r["id"]: v for r, v in zip(rows, struct.iter_unpack("<64f", x), strict=True)}
                metric = task.metrics(task.key(rows), values)
                margins = []
                pred = []
                for r in rows:
                    v = values[r["id"]]
                    difference = gap(v)
                    greedy = max(range(len(v)), key=v.__getitem__)
                    if arm == "hold" and split == "train":
                        require(
                            [v[task.IDLE], v[task.ON], greedy] == old[state][r["id"]],
                            "old named logits did not replay",
                        )
                        replayed += 1
                    pred.append((int(difference > task.EPS) - int(difference < -task.EPS), greedy))
                for edge in range(8):
                    for pos, neg in pairs(rows, edge):
                        margins.append(gap(values[pos["id"]]) - gap(values[neg["id"]]))
                metric.update(
                    pair_logistic_loss=pair_means([[m, 0.0, 0.0, 0.0] for m in margins])[0],
                    correctly_ranked_pairs=sum(x > task.EPS for x in margins),
                    pairs=len(margins),
                )
                scores[state][split][arm] = metric
                if arm == "hold":
                    choices[split] = pred
                else:
                    changes[state][split + "/" + arm] = dict(
                        two_choice=sum(a[0] != b[0] for a, b in zip(choices[split], pred, strict=True)),
                        unrestricted=sum(a[1] != b[1] for a, b in zip(choices[split], pred, strict=True)),
                    )
        for split in decks:
            gates[state][split] = decision(scores[state][split])
    require(
        replayed == 128 and next(logs, None) is None and all(not s.read(1) for s in streams),
        "incomplete readback",
    )
    out.mkdir(mode=0o700)
    return put(
        out / "results.json",
        dict(
            plan=pp,
            training=tp,
            scoring=sp,
            states=stats,
            scores=scores,
            gates=gates,
            changes=changes,
            replayed_training_rows=replayed,
            scope="Known entities, eight new pair combinations, two related starting geometries, no pretrained model.",
        ),
    )

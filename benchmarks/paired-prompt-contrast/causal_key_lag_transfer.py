"""Read-only transfer and matched-distance test of already-fixed native QKV endpoints."""

from collections import Counter
import itertools
import math
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_key_lag_fit as capacity
from shared_prefix_study import obj, put, read, require, sha, write

ROW = struct.Struct("<64f")
EPS = 5e-6
FILLERS = (58, 59)
KINDS = ("origin", "aligned", "reflected")
PLAN = dict(
    kind="spiraltorch.causal_key_lag_transfer.v1",
    endpoints="Only marker-last training: original state and fixed final-256 aligned/reflected checkpoints at each lag 0/1/2. Nine fixed models; no fitting, selection or checkpoint interpolation.",
    data="All four fit edges and all twelve other pairings of the same four left/four right identifiers; eight assignment/query/order views each. Untrained by these QKV endpoints, NOT globally fresh: earlier studies already consumed some of these pairings. No previous held scores or other held decks are read.",
    positions="Original length 9 plus length 13 with two filler IDs per fact. For each filler 58/59, place entity at distance 1, 2 or 3 before its value. Value, separator, query and answer-marker positions are fixed across the three distances; sequence length and token multiset match. Fillers are not assumed neutral and were absent from fitting. Padding and position changes can harm behavior independently of distance.",
    execution="Same pinned native executable, model parameters and frozen geometry; cached and ordinary scoring together. All 9 x 7 x 128 = 8064 question/model rows, one CPU attempt <=240 seconds, first error stops. Prefix length is exactly prompt length minus two. Old-fit baseline must reproduce stored original bytes for each model.",
    primary="On original-length twelve untrained pairs: lag1 aligned requires >=87/96 unrestricted strict top1 and >=22/24 ranked quartets; strictly more correct than ordinary aligned, lag2 aligned, own origin and own reflected. Count this as within-vocabulary recombination only, not novel-entity or natural-language generalization.",
    distance="Within each padded length/filler: compare matching lag1/distance1 and lag2/distance2 to their two other distances. Both matching routes need >=87/96 top1 and >=22/24 ranked quartets, >=16/96 top1 gain, >=10/12 positive per-edge interaction changes (>5e-6) and mean interaction gain >.01 against EACH nonmatching distance. Require both fillers for the full double-match claim. If the padded matching route fails competence, flag its distance test as inconclusive rather than blaming distance alone.",
    interpretation="Fit rows, untrained rows, original layout, padded-adjacent controls and distance variants stay separate. All rows become consumed. No pretrained GPT-2, extra learning, GPU, new wheel, publication or automatic retries.",
)


def sources():
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__), Path(__file__).with_name("test_causal_key_lag_transfer.py"))
    return capacity.sources() | {str(p.relative_to(root)): capacity.prior.signature(p) for p in paths}


def variants():
    yield "original", None, 1
    for filler, distance in itertools.product(FILLERS, (1, 2, 3)):
        yield f"f{filler}.d{distance}", filler, distance


def prompt(pair, order, idle, query, filler=None, distance=1):
    require(len(pair) == 2 and pair[0] != pair[1], "invalid identity pair")
    require(all(x in (0, 1) for x in (order, idle, query)), "invalid factorial view")
    require(distance in (1, 2, 3) and (filler in FILLERS or filler is None), "invalid separation")
    if filler is None:
        require(distance == 1, "unfilled distance must be one")
        return capacity.pp.task.render(pair, order, idle, query)

    def fact(which):
        before = [filler] * 3
        before[3 - distance] = pair[which]
        return before + [26 if which == idle else 24]

    return [60, *fact(order), 46, *fact(1 - order), 47, pair[query], 23]


def decks(training):
    require(len(training) == 32 and len({tuple(q) for q in training}) == 32, "wrong fitting input")
    edges = []
    for first in range(0, 32, 8):
        q = training[first]
        pair = (q[1], q[4])
        expected = [prompt(pair, *view) for view in itertools.product(range(2), repeat=3)]
        require(training[first : first + 8] == expected, "foreign training order or assignment")
        edges.append(pair)
    left, right = tuple(p[0] for p in edges), tuple(p[1] for p in edges)
    require(len(set(left + right)) == 8 and not set(left + right) & set(FILLERS), "overlapping roles")
    unseen = [(x, y) for x in left for y in right if (x, y) not in edges]
    require(
        len(unseen) == 12 and not set(FILLERS) & set(sum(training, [])), "wrong new-edge count or fillers"
    )
    result = {}
    for name, filler, distance in variants():
        rows = []
        for cohort, pairs in (("fit", edges), ("untrained", unseen)):
            for edge, pair in enumerate(pairs):
                for order, idle, query in itertools.product(range(2), repeat=3):
                    q = prompt(pair, order, idle, query, filler, distance)
                    rows.append(
                        dict(
                            prompt=q,
                            id=sha(bytes(q)),
                            cohort=cohort,
                            edge=edge,
                            pair=list(pair),
                            order=order,
                            idle=idle,
                            query=query,
                            gold=16 if idle == query else 2,
                        )
                    )
        require(len({r["id"] for r in rows}) == 128, "duplicate questions")
        result[name] = rows
    return result


def measures(rows, raw):
    require(len(raw) == len(rows) * ROW.size and rows, "incomplete logits")
    counts = Counter((r["edge"], r["order"], r["idle"], r["query"]) for r in rows)
    size = len({r["edge"] for r in rows})
    require(
        counts
        == Counter(
            {(e, o, a, q): 1 for e in range(size) for o, a, q in itertools.product(range(2), repeat=3)}
        ),
        "incomplete quartet roster",
    )
    cells, nll, correct, binary = {}, [], 0, 0
    for r, z in zip(rows, ROW.iter_unpack(raw), strict=True):
        require(all(map(math.isfinite, z)), "nonfinite output")
        gold = 16 if r["idle"] == r["query"] else 2
        require(r["gold"] == gold, "foreign answer label")
        gap = z[16] - z[2]
        cells[r["edge"], r["order"], r["idle"], r["query"]] = gap
        correct += z[gold] > max(x for i, x in enumerate(z) if i != gold)
        binary += (gap if gold == 16 else -gap) > EPS
        nll.append(capacity.a.nll(z, gold)[0])
    interactions, ranked = [], 0
    for edge in range(size):
        views = []
        for order in (0, 1):
            aa, ab, ba, bb = [cells[edge, order, a, q] for a, q in itertools.product(range(2), repeat=2)]
            ranked += min(aa, bb) - max(ab, ba) > EPS
            views.append(math.fsum((aa, -ab, -ba, bb)) / 4)
        interactions.append(math.fsum(views) / 2)
    return dict(
        rows=len(rows),
        edges=size,
        full_correct=correct,
        binary_correct=binary,
        ranked=ranked,
        nll=math.fsum(nll) / len(nll),
        by_edge=interactions,
        interaction=math.fsum(interactions) / size,
    )


def competent(x):
    require(x["rows"] == 96 and x["edges"] == 12, "wrong competence population")
    return x["full_correct"] >= 87 and x["ranked"] >= 22


def seal(reply, qs):
    data = reply["data"]
    require(data["prefixes"] == len({tuple(q[:-2]) for q in qs}), "wrong factual-prefix inventory")
    require(len(data["rows"]) == len(qs), "incomplete model reply")
    full, cached, maximum = [], [], 0.0
    for row in data["rows"]:
        a, b = (struct.pack("<64I", *row[k]) for k in ("ordinary", "cached"))
        maximum = max(maximum, capacity.prev.cx.agree(a, b))
        full.append(a)
        cached.append(b)
    return b"".join(full), b"".join(cached), maximum


def decisions(records):
    def metric(model, variant):
        return records[model][variant]["untrained"]

    target = metric("1.aligned", "original")
    references = {n: metric(n, "original") for n in ("0.aligned", "2.aligned", "1.origin", "1.reflected")}
    primary = dict(
        competent=competent(target),
        correct_gains={k: target["full_correct"] - v["full_correct"] for k, v in references.items()},
    )
    primary["passed"] = primary["competent"] and all(x > 0 for x in primary["correct_gains"].values())
    distance = {}
    for lag, filler in itertools.product((1, 2), FILLERS):
        match = metric(f"{lag}.aligned", f"f{filler}.d{lag}")
        controls = {}
        for d in (1, 2, 3):
            if d == lag:
                continue
            other = metric(f"{lag}.aligned", f"f{filler}.d{d}")
            changes = [a - b for a, b in zip(match["by_edge"], other["by_edge"], strict=True)]
            controls[str(d)] = dict(
                correct_gain=match["full_correct"] - other["full_correct"],
                positive_edges=sum(x > EPS for x in changes),
                mean_gain=math.fsum(changes) / len(changes),
            )
        gate = competent(match)
        distance[f"lag{lag}.f{filler}"] = dict(
            matching_competent=gate,
            inconclusive_due_to_matching_failure=not gate,
            controls=controls,
            passed=gate
            and all(
                c["correct_gain"] >= 16 and c["positive_edges"] >= 10 and c["mean_gain"] > 0.01
                for c in controls.values()
            ),
        )
    return dict(
        primary=primary, distance=distance, double_match_passed=all(x["passed"] for x in distance.values())
    )


def checked_parent(parent, pins):
    p = obj(parent / "protocol/plan.json", pins["plan"])
    t = obj(parent / "trained/complete.json", pins["training"])
    s = obj(parent / "scored/complete.json", pins["serving"])
    require(
        p["kind"] == "spiraltorch.causal_key_lag_fit.v2"
        and p["schedule"] == capacity.PLAN
        and p["source"] == capacity.sources()
        and t["plan"] == s["plan"] == pins["plan"]
        and s["training"] == pins["training"]
        and t["updates"] == 3072,
        "foreign parent or changed source",
    )
    return p, t, s


def freeze(root, parent, pins, driver):
    p, t, s = checked_parent(parent, pins)
    require(capacity.prior.signature(driver) == p["binary"], "wrong saved executable")
    rows = obj(parent / "protocol/fit.json", p["fit"])
    roster = decks(rows)
    models = {}
    for lag, kind in itertools.product((0, 1, 2), KINDS):
        lane = f"{lag}.marker.{kind if kind != 'origin' else 'aligned'}"
        run = obj(parent / "trained" / (lane + ".json"), t["records"][lane])
        step = 0 if kind == "origin" else 256
        item = run["data"]["pins"]["256"] if step else None
        if step:
            raw = read(parent / "trained" / lane / "256.bin", item["checkpoint"][7:], 100000)
            capacity.checkpoint(raw, p["routes"][str(lag)], step)
        models[f"{lag}.{kind}"] = dict(
            lag=lag,
            lane=lane,
            step=step,
            checkpoint=item["checkpoint"] if step else None,
            owner=item["serving"] if step else run["initial"],
            baseline=s["records"][lane][str(step)]["full"],
        )
    root.mkdir(mode=0o700)
    return put(
        root / "plan.json",
        dict(
            schedule=PLAN,
            source=sources(),
            parent_path=str(parent),
            parent=pins,
            binary=p["binary"],
            models=models,
            decks={k: put(root / (k + ".json"), v) for k, v in roster.items()},
            columns=dict(fit=32, untrained=96),
            rows=8064,
            updates=0,
        ),
    )


def run(root, p, pin, out, driver):
    parent = Path(p["parent_path"])
    pp, _, _ = checked_parent(parent, p["parent"])
    require(
        os.environ.get("SPIRALTORCH_BACKEND") == "cpu" and capacity.prior.signature(driver) == p["binary"],
        "wrong CPU runtime",
    )
    records = {}
    for model, metadata in p["models"].items():
        lag, step = metadata["lag"], metadata["step"]
        start = None
        if step:
            checkpoint = parent / "trained" / metadata["lane"] / "256.bin"
            read(checkpoint, metadata["checkpoint"][7:], 100000)
            start = dict(path=str(checkpoint), id=metadata["checkpoint"])
        items = {}
        for variant, _, _ in variants():
            rows = obj(root / (variant + ".json"), p["decks"][variant])
            qs = [r["prompt"] for r in rows]
            prefix_len = len(qs[0]) - 2
            z = capacity.call(
                driver,
                parent / "protocol",
                pp,
                lag,
                dict(kind="score", prompts=qs, prefix_len=prefix_len),
                start,
            )
            require(
                z["initial"] == z["end"] == metadata["owner"] and z["steps"] == step,
                "scoring mutated or substituted weights",
            )
            raw, cached, maximum = seal(z, qs)
            name = f"{model}.{variant}"
            items[variant] = dict(
                raw=write(out / (name + ".full"), raw),
                cached=write(out / (name + ".cached"), cached),
                max_error=maximum,
            )
            if variant == "original":
                baseline = read(
                    parent / "scored" / f"{metadata['lane']}.{step}.full", metadata["baseline"], 100000
                )
                require(raw[: 32 * ROW.size] == baseline, "old fit replay changed")
        if step:
            read(checkpoint, metadata["checkpoint"][7:], 100000)
        records[model] = items
    return put(out / "complete.json", dict(plan=pin, records=records, rows=8064, updates=0))


def audit(root, p, pin, out, measurement_pin):
    measured = obj(root.parent / "measured/complete.json", measurement_pin)
    require(
        measured["plan"] == pin
        and measured["rows"] == 8064
        and measured["updates"] == 0
        and set(measured["records"]) == set(p["models"]),
        "incomplete measurements",
    )
    parent = Path(p["parent_path"])
    pp, _, _ = checked_parent(parent, p["parent"])
    expected = decks(obj(parent / "protocol/fit.json", pp["fit"]))
    records = {}
    for model, values in measured["records"].items():
        require(set(values) == set(p["decks"]), "incomplete model roster")
        items = {}
        for variant, entry in values.items():
            rows = obj(root / (variant + ".json"), p["decks"][variant])
            require(rows == expected[variant], "changed factorial roster")
            name = f"{model}.{variant}"
            raw = read(root.parent / "measured" / (name + ".full"), entry["raw"])
            cached = read(root.parent / "measured" / (name + ".cached"), entry["cached"])
            require(len(raw) == len(cached) == 128 * ROW.size, "incomplete stored rows")
            error = max(
                capacity.prev.cx.agree(raw[i : i + ROW.size], cached[i : i + ROW.size])
                for i in range(0, len(raw), ROW.size)
            )
            require(error == entry["max_error"], "cache-error mismatch")
            if variant == "original":
                owner = p["models"][model]
                old = read(parent / "scored" / f"{owner['lane']}.{owner['step']}.full", owner["baseline"])
                require(raw[: 32 * ROW.size] == old, "baseline differs on independent readback")
            items[variant] = dict(
                fit=measures(rows[:32], raw[: 32 * ROW.size]),
                untrained=measures(rows[32:], raw[32 * ROW.size :]),
            )
        records[model] = items
    return put(
        out / "summary.json",
        dict(
            plan=pin,
            measurements=measurement_pin,
            records=records,
            decisions=decisions(records),
            dataset_consumed=True,
            updates=0,
        ),
    )


def main(argv):
    phase, folder, *args = argv
    os.umask(0o077)
    root = Path(folder).resolve()
    if phase == "freeze":
        parent, plan, training, serving, driver = args
        return freeze(
            root, Path(parent).resolve(), dict(plan=plan, training=training, serving=serving), Path(driver)
        )
    pin, extra = args
    p = obj(root / "plan.json", pin)
    require(p["schedule"] == PLAN and p["source"] == sources(), "source or plan drift")
    out = root.parent / {"run": "measured", "audit": "readback"}[phase]
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, attempt=1, phase=phase))
    return run(root, p, pin, out, Path(extra)) if phase == "run" else audit(root, p, pin, out, extra)


if __name__ == "__main__":
    print(main(sys.argv[1:]))

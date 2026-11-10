"""Train with forward-checked CE protection, and independently replay every proposed endpoint."""

from collections import Counter
import hashlib
import itertools
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_quartet_learning as a
from shared_prefix_study import obj, put, read, require

cx, pp, old, last, prior = a.cx, a.pp, a.old, a.last, a.prior
ARMS = ("answer", "scaled", "anchor", "mixed", "correct", "reflected")
CORRECTIONS = dict(anchor="anchor_only", mixed="mixed", correct="interaction", reflected="interaction")
CONFIG = dict(
    length=old.LENGTH, max_trials=5, retain_answer=0.75, min_answer_drop=1e-8, min_interaction_gain=1e-8
)
PLAN = dict(
    data="Last 16 unused A/B cross-pool pairs: upper four A with each B half at shift 0 for fit, shift 1 for held. All B and those four A are familiar on held. Transfer only: new identity roles (40,41,42,43) x (44,45,56,57), shifts 0 and 1.",
    site="Same L0 value site, Walsh bit 2 and two frozen geometry starts. 32 free right coordinates, 64 alternating search opportunities, AB/BA. No validation prompt or label in a training guard.",
    arms="Unprotected CE and train-scaled mixture; finite CE anchor only; guarded corrections toward the same fixed mixture, pure interaction or query-reflected pure interaction. In every guard, protect the other training microbatch as well. Same two component VJPs and opportunities, NOT equal forward-call counts.",
    search=CONFIG,
    acceptance="CE anchor halves from .08. Correction interpolates the safe anchor toward a radius-.08 proposal, halving its fraction. Strict active CE drop, protected CE nonincrease, retain >=75% of this anchor's real CE progress. Correction must beat both origin and anchor on its own interaction by >1e-8. Fall back to anchor or hold. Stop on two distinct held batches at unchanged weights.",
    primary="For guarded pure interaction vs unprotected CE, guarded CE anchor and reflected guarded interaction: >=6/8 shared-margin gains (>5e-6) against EACH. Separate behavior: more binary choices AND strictly ranked quartets than each, full-vocab top1 no lower, NLL <= CE-only + .01. Both origins and both orders required. Held and transfer are separate.",
    controls="Also compare guarded mixed to the unprotected calibrated mixture on the identical roster. A good CE receipt is not evidence of better endpoint CE than CE-only, or better binding.",
    replay="Reconstruct every trial recipe and re-evaluate its full-vocab CE on BOTH fitting batches, and active aligned/reflected interaction, on the existing CPU wheel after ordinary safetensors reimport. Verify all dispositions, stalled lanes, checkpoint IDs, ordinary/cached endpoint rows, and a prespecified split continuation of initial/AB/correct. Identical owners may share observations.",
    budget="One CPU attempt per phase (240 seconds each), no pretrained model, previous held scores, GPU, wheel rebuild, publication or retries. Store first failure separately. Optimizer backtracking is part of the frozen policy.",
)


def sources():
    rust = Path(__file__).resolve().parents[2] / "crates/st-nn/src/causal_lm"
    return a.sources() | {
        p.name: prior.signature(p)
        for p in (Path(__file__), Path(a.__file__), rust / "quartet_projection/answer/guard.rs")
    }


def decks():
    aa, bb = pp.task.ENTITIES, pp.IDS
    seen = {
        tuple(sorted((q[1], q[4])))
        for roster in (cx.decks(), a.decks())
        for deck in roster.values()
        for q in deck
    }
    require(len(seen) == 48, "wrong previously consumed pairs")
    x, y = (40, 41, 42, 43), (44, 45, 56, 57)
    require(not set(x + y) & set(aa + bb + (60, 46, 47, 23, 24, 26, 16, 2)), "transfer role collision")
    out = {}
    for fold, (left, right, shift) in zip(
        cx.FOLDS,
        (
            (aa[4:], bb[:4], 0),
            (aa[4:], bb[4:], 0),
            (aa[4:], bb[:4], 1),
            (aa[4:], bb[4:], 1),
            (x, y, 0),
            (x, y, 1),
        ),
        strict=True,
    ):
        pairs = [(v, right[(i + shift) % 4]) for i, v in enumerate(left)]
        edges = {tuple(sorted(e)) for e in pairs}
        require(len(edges) == 4 and not edges & seen, "reused pair")
        seen.update(edges)
        out[fold] = [
            pp.task.render(e, o, t, q) for e in pairs for o, t, q in itertools.product(range(2), repeat=3)
        ]
    require(len({tuple(sorted((u, v))) for u in aa for v in bb} - seen) == 0, "cross-pool graph not closed")
    return out


def lanes():
    return itertools.product(old.STATES, ("ab", "ba"), ARMS)


def lookup(root, p):
    return [obj(root / (f + ".json"), p["folds"][f]) for f in cx.FOLDS[:2]]


def freeze(root, parent, ppin, ah, binary):
    p = obj(parent / "protocol/plan.json", ppin)
    m = obj(parent / "readback-v2/summary.json", ah, 2 << 20)
    require(
        p["kind"] == "spiraltorch.full_answer_quartets.v1"
        and p["schedule"] == a.PLAN
        and m["plan"] == ppin
        and p["dependencies"] == prior.sources(),
        "wrong parent",
    )
    return a.freeze_projection_plan(
        root,
        parent,
        p,
        dict(
            kind="spiraltorch.answer_guard.v1",
            parent=[ppin, ah],
            schedule=PLAN,
            source=sources(),
            binary=prior.signature(binary),
        ),
        decks(),
    )


def train(root, p, pin, out, binary, cp):
    c = obj(root.parent / "calibration/complete.json", cp)
    require(c["plan"] == pin, "wrong calibration")
    qs = lookup(root, p)
    records = {}
    continuation = None
    opportunities = updates = 0
    for state, order, arm in lanes():
        idx = ([0, 1] if order == "ab" else [1, 0]) * 32
        weight = (
            c["scales"][state]["mix"]
            if arm in ("scaled", "mixed")
            else (0.0 if arm in ("answer", "anchor") else 1.0)
        )
        action = a.action([a.batch(q, arm == "reflected") for q in qs], idx, weight, True)
        if arm in CORRECTIONS:
            action["update"] = dict(
                mode="guarded", protect_other=True, config=CONFIG | dict(correction=CORRECTIONS[arm])
            )
        packet = cx.invoke(binary, root, p, state, *a.SITE, action)
        tag = ".".join((state, order, arm))
        records[tag] = put(out / (tag + ".json"), packet)
        steps = packet["steps"]
        require(
            0 < len(steps) <= 64
            and [s["batch"] for s in steps] == idx[: len(steps)]
            and packet["initial"] == cx.owner_id(p, state, *a.SITE, [0] * 32),
            "wrong schedule or starting point",
        )
        before = packet["initial"]
        held = set()
        for j, step in enumerate(steps):
            owner = step["owner"]
            require(owner == cx.owner_id(p, state, *a.SITE, step["right_bits"]), "foreign reported owner")
            if arm in CORRECTIONS:
                require((step["report"]["status"] == "held") == (before == owner), "wrong mutation status")
            if owner == before:
                held.add(step["batch"])
            else:
                held.clear()
                updates += 1
            require(
                arm not in CORRECTIONS or len(held) < 2 or j == len(steps) - 1,
                "retried a deterministic stall",
            )
            before = owner
        require(
            packet["owner"] == before
            and (len(steps) == 64 or packet.get("stalled"))
            and (arm not in CORRECTIONS or packet["stalled"] == (len(held) == 2)),
            "incorrect terminal state",
        )
        opportunities += len(steps)
        if (state, order, arm) == ("initial", "ab", "correct"):
            # Cut just after an accepted step: no outstanding held-batch history crosses the boundary.
            cuts = [j + 1 for j, s in enumerate(steps[:-1]) if j < 32 and s["report"]["status"] != "held"]
            cut = max(cuts, default=0)
            start = (
                {k: steps[cut - 1][k] for k in ("owner", "right_bits")}
                if cut
                else dict(owner=packet["initial"], right_bits=[0] * 32)
            )
            replay = cx.invoke(binary, root, p, state, *a.SITE, action | dict(order=idx[cut:], start=start))
            rh = put(out / "resume.json", replay)
            continuation = dict(hash=rh, cut=cut, steps=len(replay["steps"]))
            require(
                replay["initial"] == start["owner"]
                and replay["steps"] == steps[cut:]
                and replay["owner"] == before
                and replay["stalled"] == packet["stalled"],
                "resumed search diverged",
            )
    return put(
        out / "complete.json",
        dict(
            plan=pin,
            calibration=cp,
            ends=records,
            opportunities=opportunities,
            updates=updates,
            continuation=continuation,
        ),
    )


def packets(root, t):
    require(set(t["ends"]) == {".".join(v) for v in lanes()}, "incomplete endpoints")
    for ln in lanes():
        tag = ".".join(ln)
        yield ln, tag, obj(root.parent / "trained" / (tag + ".json"), t["ends"][tag], 4 << 20)


def snapshots(root, p, t):
    for state in old.STATES:
        yield state + ".hold", state, [0] * 32
    for (state, _, _), tag, packet in packets(root, t):
        b = packet["steps"][-1]["right_bits"]
        require(packet["owner"] == cx.owner_id(p, state, *a.SITE, b), "wrong endpoint")
        yield tag, state, b


def close(x, y):
    return math.isclose(x, y, abs_tol=2e-12, rel_tol=2e-12)


def candidate(pre, grad, length):
    return [x - length * (g / grad["norm"]) for x, g in zip(pre, grad["gradient"], strict=True)]


def accepted(before, point, required, anchor, correction):
    drop = before["answer"] - point["answer"]
    return (
        drop > 0
        and drop >= required
        and point["protected_answer"] <= before["protected_answer"]
        and (
            not correction
            or min(before["interaction"], anchor["interaction"]) - point["interaction"]
            > CONFIG["min_interaction_gain"]
        )
    )


class WheelReference:
    def __init__(self, root, p, st, models):
        self.cfg = read(root / "model/config.json", p["files"]["config.json"]).decode()
        self.arch = cx.LayerArchive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
        start, stride, _ = self.arch.layout[0, "query"]
        self.arch.layout[0, "value"] = (start, stride, 2 * pp.W)
        self.p, self.st, self.models, self.qs = p, st, models, lookup(root, p)
        self.cache = {}

    def values(self, state, bits):
        owner = cx.owner_id(self.p, state, *a.SITE, bits)
        if owner not in self.cache:
            delta = [u * v for u in pp.FACTORS[a.SITE[2]] for v in last.doubles(bits)]
            m, _ = old.snapshot(self.st, self.models[state], self.arch, self.cfg, {(0, "value"): delta})
            folds = []
            digest = hashlib.sha256()
            for qs in self.qs:
                costs = []
                gaps = []
                for q in qs:
                    row = m.forward(q, [1, 9], last_only=True).values()
                    digest.update(old.ROW.pack(*row))
                    costs.append(a.nll(row, a.gold(q))[0])
                    gaps.append(row[16] - row[2])
                folds.append(
                    dict(
                        ce=math.fsum(costs) / len(qs),
                        aligned=last.reference_seed(gaps, "interaction", "aligned")[0],
                        reflected=last.reference_seed(gaps, "interaction", "symmetric")[0],
                    )
                )
            self.cache[owner] = dict(logit_sha=digest.hexdigest(), folds=folds)
        return self.cache[owner]["folds"]


def certify(root, p, pin, out, st, models, cp, tp):
    c = obj(root.parent / "calibration/complete.json", cp)
    t = obj(root.parent / "trained/complete.json", tp)
    require(c["plan"] == t["plan"] == pin and t["calibration"] == cp, "wrong trial chain")
    ref = WheelReference(root, p, st, models)
    stats = {}
    for (state, _, arm), tag, packet in packets(root, t):
        counts = Counter()
        previous = [0] * 32
        max_ce_rise = 0.0
        counts["opportunities"] = len(packet["steps"])
        for step in packet["steps"]:
            pre = last.doubles(previous)
            i = step["batch"]
            report = step["report"]
            d = report["derivatives"] if arm in CORRECTIONS else report
            require(
                (d["prompt_forwards"], d["projection_pullbacks"]) == (32, 64), "different gradient budget"
            )
            mix = (
                c["scales"][state]["mix"]
                if arm in ("scaled", "mixed")
                else (0.0 if arm in ("answer", "anchor") else 1.0)
            )
            require(
                d["mix"] == mix
                and all(
                    close((1 - mix) * x + mix * y, z)
                    for x, y, z in zip(
                        d["answer"]["gradient"],
                        d["interaction"]["gradient"],
                        d["combined"]["gradient"],
                        strict=True,
                    )
                ),
                "misassigned gradient mixture",
            )

            def physical(bits):
                folds = ref.values(state, bits)
                return dict(
                    answer=folds[i]["ce"],
                    protected_answer=folds[1 - i]["ce"],
                    interaction=folds[i]["reflected" if arm == "reflected" else "aligned"],
                )

            def check_point(actual, stored):
                require(
                    all(close(v, stored[k]) for k, v in actual.items()), "factual wheel objective mismatch"
                )

            origin = physical(previous)
            if arm not in CORRECTIONS:
                grad = d["combined"]
                new = last.bits(candidate(pre, grad, old.LENGTH)) if grad["norm"] else previous
                require(new == step["right_bits"], "unguarded proposal differs")
                final = physical(new)
                counts["unprotected"] += 1
            else:
                before = report["before"]
                check_point(origin, before)
                chosen = previous
                anchor = list(pre)
                limit = before
                last_point = before
                status = "held"
                seen = Counter()
                finished = set()
                for row in report["trials"]:
                    cor = row["correction"]
                    f = row["fraction"]
                    j = seen[cor]
                    seen[cor] += 1
                    require(
                        j < CONFIG["max_trials"]
                        and f == 0.5**j
                        and cor not in finished
                        and (cor or not seen[True]),
                        "bad trial order",
                    )
                    if not cor:
                        right = candidate(pre, d["answer"], old.LENGTH * f)
                    else:
                        require(arm != "anchor", "unexpected correction")
                        proposal = candidate(
                            pre, d["combined"] if arm == "mixed" else d["interaction"], old.LENGTH
                        )
                        right = (
                            proposal
                            if f == 1.0
                            else [(1 - f) * x + f * y for x, y in zip(anchor, proposal, strict=True)]
                        )
                    bits = last.bits(right)
                    owner = cx.owner_id(p, state, *a.SITE, bits)
                    require(
                        owner == row["owner"] and close(math.dist(right, pre), row["distance"]),
                        "trial weight reconstruction differs",
                    )
                    point = row["point"]
                    check_point(physical(bits), point)
                    pred = math.fsum(
                        g * (x - y) for g, x, y in zip(d["answer"]["gradient"], pre, right, strict=True)
                    )
                    floor = max(
                        CONFIG["min_answer_drop"],
                        1e-4 * max(0.0, pred),
                        CONFIG["retain_answer"] * (before["answer"] - limit["answer"]) if cor else 0.0,
                    )
                    require(
                        close(row["required_answer_drop"], floor)
                        and row["eligible"]
                        == accepted(before, point, row["required_answer_drop"], limit, cor),
                        "incorrect acceptance",
                    )
                    counts["trials"] += 1
                    counts["rejected_trials"] += not row["eligible"]
                    if row["eligible"]:
                        finished.add(cor)
                        chosen = bits
                        last_point = point
                        status = "corrected" if cor else "anchor"
                        if not cor:
                            anchor = right
                            limit = point
                require(
                    status == report["status"]
                    and chosen == step["right_bits"]
                    and last_point == report["after"],
                    "published the wrong trial",
                )
                require(report["objective_forwards"] == 64 * (1 + sum(seen.values())), "wrong measured work")
                counts["objective_forwards"] += report["objective_forwards"]
                counts[status] += 1
                final = physical(chosen)
            max_ce_rise = max(
                max_ce_rise,
                final["answer"] - origin["answer"],
                final["protected_answer"] - origin["protected_answer"],
            )
            previous = step["right_bits"]
        require(
            arm not in CORRECTIONS or max_ce_rise < 2e-12,
            "a complete training batch worsened on guarded update",
        )
        stats[tag] = dict(
            counts=counts, max_training_ce_rise=max_ce_rise, stalled=packet.get("stalled", False)
        )
    h = put(out / "owners.json", ref.cache)
    return put(
        out / "complete.json",
        dict(
            plan=pin,
            training=tp,
            owners=h,
            distinct_owners=len(ref.cache),
            wheel_rows=64 * len(ref.cache),
            stats=stats,
        ),
    )


def audit(root, p, pin, out, tp, wp, sp):
    t = obj(root.parent / "trained/complete.json", tp)
    s = obj(root.parent / "scored/complete.json", sp)
    w = obj(root.parent / "witness/complete.json", wp)
    require(
        t["plan"] == s["plan"] == w["plan"] == pin and w["training"] == s["train"] == tp, "foreign receipts"
    )
    owners = obj(root.parent / "witness/owners.json", w["owners"], 16 << 20)
    ds = decks()
    require(all(obj(root / (f + ".json"), p["folds"][f]) == qs for f, qs in ds.items()), "wrong decks")
    scores = {}
    for tag, state, right in snapshots(root, p, t):
        r = s["records"][tag]
        folder = root.parent / "scored"
        full = read(folder / (tag + ".full"), r["full"])
        cache = read(folder / (tag + ".cache"), r["cache"])
        require(
            len(full) == len(cache) == 192 * 256
            and max(cx.agree(full[j : j + 256], cache[j : j + 256]) for j in range(0, len(full), 256))
            == r["max_error"],
            "invalid cached replay",
        )
        owner = cx.owner_id(p, state, *a.SITE, right)
        require(
            hashlib.sha256(full[: 64 * 256]).hexdigest() == owners[owner]["logit_sha"],
            "training witness and endpoint disagree",
        )
        scores[tag] = {f: a.metrics(ds[f], full[j * 8192 : (j + 1) * 8192]) for j, f in enumerate(cx.FOLDS)}
    require(set(scores) == set(s["records"]) and s["rows"] == len(scores) * 192, "wrong serving count")
    continuation = t["continuation"]
    folder = root.parent / "trained"
    lane = "initial.ab.correct"
    full = obj(folder / (lane + ".json"), t["ends"][lane], 4 << 20)
    replay = obj(folder / "resume.json", continuation["hash"], 4 << 20)
    cut = continuation["cut"]
    require(
        replay["initial"] == (full["steps"][cut - 1]["owner"] if cut else full["initial"])
        and replay["steps"] == full["steps"][cut:]
        and replay["owner"] == full["owner"]
        and replay["stalled"] == full["stalled"],
        "saved continuation differs",
    )
    return put(
        out / "summary.json",
        dict(plan=pin, training=tp, witness=wp, serving=sp, scores=scores, gates=endpoint_gates(scores)),
    )


def endpoint_gates(scores):
    gates = {}
    for state, order, subset in itertools.product(old.STATES, ("ab", "ba"), ("held", "transfer")):
        target = ".".join((state, order, "correct"))
        folds = [f for f in cx.FOLDS if f.startswith(subset)]

        def total(tag, k):
            return sum(scores[tag][f][k] for f in folds)

        vs = {}
        for arm in ("answer", "anchor", "reflected"):
            ref = ".".join((state, order, arm))
            delta = [
                math.fsum(x - y for x, y in zip(a, b, strict=True)) / 4
                for f in folds
                for a, b in zip(scores[target][f]["edges"], scores[ref][f]["edges"], strict=True)
            ]
            vs[arm] = dict(
                edges=sum(x > pp.EPS for x in delta),
                delta=math.fsum(delta) / 8,
                correct_gain=total(target, "correct") - total(ref, "correct"),
                ranked_gain=total(target, "ranked") - total(ref, "ranked"),
                full_gain=total(target, "full_correct") - total(ref, "full_correct"),
                nll_delta=(total(target, "answer_nll") - total(ref, "answer_nll")) / 2,
            )
        gates[target + "." + subset] = dict(
            vs=vs,
            shared=all(v["edges"] >= 6 for v in vs.values()),
            behavior=vs["answer"]["nll_delta"] <= 0.01
            and all(
                v["correct_gain"] > 0 and v["ranked_gain"] > 0 and v["full_gain"] >= 0 for v in vs.values()
            ),
        )
    return gates


def main(argv):
    os.umask(0o077)
    phase, r, *args = argv
    root = Path(r).resolve()
    if phase == "freeze":
        parent, pin, h, b = args
        return freeze(root, Path(parent), pin, h, Path(b))
    pin = args[0]
    p = obj(root / "plan.json", pin)
    require(
        p["schedule"] == PLAN and p["source"] == sources() and p["dependencies"] == prior.sources(),
        "plan or source drift",
    )
    out = (
        root.parent
        / {
            "calibrate": "calibration",
            "train": "trained",
            "witness": "witness",
            "serve": "scored",
            "audit": "readback",
        }[phase]
    )
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, phase=phase, attempt=1))
    if phase == "audit":
        return audit(root, p, pin, out, *args[1:])
    binary = Path(args[1]).resolve()
    require(
        prior.signature(binary) == p["binary"] and os.environ.get("SPIRALTORCH_BACKEND") == "cpu",
        "wrong backend or driver",
    )
    if phase in ("calibrate", "train", "witness"):

        def fence(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).name in {k + ".json" for k in cx.FOLDS[2:]}
            ):
                raise PermissionError("evaluation roster sealed")

        sys.addaudithook(fence)
    if phase == "train":
        return train(root, p, pin, out, binary, *args[2:])
    st, installed, models = prior.stack(root, p, True)
    put(out / "installation.json", installed)
    if phase == "calibrate":
        return a.calibrate(root, p, pin, out, binary, models)
    if phase == "witness":
        return certify(root, p, pin, out, st, models, *args[2:])
    return a.serve(root, p, pin, out, binary, st, models, *args[2:], endpoint_rows=snapshots)


if __name__ == "__main__":
    print(main(sys.argv[1:]))

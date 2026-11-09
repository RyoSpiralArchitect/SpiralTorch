"""Train one native V projection with full-vocabulary answer CE and optional quartet supervision."""

import itertools
import math
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contextual_key_learning as cx
from shared_prefix_study import obj, put, read, require, write

pp, old, prior, last = cx.pp, cx.old, cx.prior, cx.last
SITE = (0, "value", 1)
POLICIES = ("answer", "interaction", "half", "half_control", "fit_scale", "fit_control")
STEPS = 64
PLAN = dict(
    design="Primary site fixed throughout: first-block V, second Walsh left factor, 32 free right coordinates. Same two frozen geometry starts, alternating batch orders AB and BA. No comparison to prior K lanes as if data were equal.",
    roster="Cross first 4 A identities with last 4 B: shifts 0/1 fit and 2/3 held. Last 4 A with first 4 B: shifts 2/3 out-of-fit transfer only. All 24 pairs disjoint from the preceding 24 and both within-pool histories. Same gold labels in every arm.",
    loss="(1-mix)*full-vocabulary gold next-token CE + mix*per-quartet interaction logistic. CE only, interaction only, 1/2 aligned/symmetric, train-calibrated aligned/symmetric. Symmetry reflects only the quartet, NEVER gold. Each prompt computes both VJPs from one forward in every measured arm.",
    scale="Before fitting, observe the two gradient vectors on all 64 fit prompts at zero right. Freeze mix = norm(CE)/(norm(CE)+norm(aligned interaction)) per origin, unchanged across updates/orders and both calibrated arms. Stop if either norm <=1e-12 or the ratio is not representable. No eval-based tuning.",
    updates="24 lanes x 64 normalized steps of 0.08, 32 prompts and 16 equally weighted quartets per step. Only endpoints see held/transfer. One additional prespecified continuation: initial/AB/fit_scale resumes from step 32 for 32 steps, exact next reports required.",
    verification="Old wheel reconstructs CE gradient by summing every non-gold token contrast weighted by its softmax probability, and interaction via the named-pair adjoint. CE loss <=2e-6 abs and component error <=2e-6+1e-4 relative; interaction <=1e-9+1e-8 relative. All endpoint ordinary Rust vs independently merged wheel bits must match; cached inference retains prior numeric/greedy gates.",
    gates="For half and fit_scale separately against answer alone AND their own CE+reflected control: >=6/8 edges gain mean binding margin >5e-6. Behavior separately: strictly more binary choices AND completely ranked quartets than both, full-vocab top1 no lower and gold NLL <= CE-only + 0.01 nats. Require both geometry starts and both orders, assess held and transfer separately. Also report hold and interaction-only without endpoint selection.",
    budget="CPU only, one attempt per physical phase <=240 seconds, stop on first error, no GPT-2, GPU, further cache cleanup, wheel construction or release.",
)


def sources():
    rust = Path(__file__).resolve().parents[2] / "crates/st-nn/src/causal_lm"
    return cx.sources() | {
        p.name: prior.signature(p)
        for p in (
            Path(__file__),
            Path(cx.__file__),
            rust / "quartet_projection/answer.rs",
            rust / "learning.rs",
        )
    }


def decks():
    a, b = pp.task.ENTITIES, pp.IDS
    used = {tuple(sorted((q[1], q[4]))) for deck in cx.decks().values() for q in deck}
    out = {}
    for fold, (ia, ib, shift) in zip(
        cx.FOLDS, ((0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (4, 0, 2), (4, 0, 3)), strict=True
    ):
        pairs = [(a[ia + i], b[ib + (i + shift) % 4]) for i in range(4)]
        block = {tuple(sorted(e)) for e in pairs}
        require(
            len(block) == 4 and block.isdisjoint(used) and all(e[0] in a and e[1] in b for e in pairs),
            "partner overlap",
        )
        used.update(block)
        out[fold] = [
            pp.task.render(e, o, t, q) for e in pairs for o, t, q in itertools.product(range(2), repeat=3)
        ]
    require(
        len(used) == 48 and len({tuple(q) for qs in out.values() for q in qs}) == 192,
        "invalid depth/answer split",
    )
    return out


def gold(q):
    require(len(q) == 9 and q[7] in (q[1], q[4]) and {q[2], q[5]} == {24, 26}, "invalid question")
    return 16 if q[2 if q[1] == q[7] else 5] == 26 else 2


def batch(qs, reflected):
    require(len(qs) % 4 == 0, "incomplete quartet")
    return dict(
        prompts=qs,
        gold=[gold(q) for q in qs],
        groups=[
            [i + j for j in order]
            for i in range(0, len(qs), 4)
            for order in ((0, 1, 2, 3), (1, 0, 3, 2) if reflected else (0, 1, 2, 3))
        ],
    )


def mixture(policy, scale):
    return {
        "answer": 0.0,
        "interaction": 1.0,
        "half": 0.5,
        "half_control": 0.5,
        "fit_scale": scale,
        "fit_control": scale,
    }[policy]


def nll(z, y):
    mx = max(z)
    exps = [math.exp(x - mx) for x in z]
    den = math.fsum(exps)
    return mx - z[y] + math.log(den), [e / den for e in exps]


def freeze(root, parent, pin, h, binary):
    p = obj(parent / "protocol/plan.json", pin)
    m = obj(parent / "readback/summary.json", h, 2 << 20)
    require(
        p["kind"] == "spiraltorch.contextual_key_depth.v1"
        and p["schedule"] == cx.PLAN
        and p["dependencies"] == prior.sources()
        and m["plan"] == pin,
        "wrong previous run",
    )
    return freeze_projection_plan(
        root,
        parent,
        p,
        dict(
            kind="spiraltorch.full_answer_quartets.v1",
            parent=[pin, h],
            schedule=PLAN,
            source=sources(),
            binary=prior.signature(binary),
        ),
        decks(),
    )


def freeze_projection_plan(root, parent, p, spec, folds):
    root.mkdir(mode=0o700)
    md = root / "model"
    md.mkdir(mode=0o700)
    for f, d in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((s[7:] + ".json", s[7:]) for s in p["origins"].values()),
    ]:
        write(md / f, read(parent / "protocol/model" / f, d, 100000))
    write(
        root / "wheel-inventory.json", read(parent / "protocol/wheel-inventory.json", p["inventory"], 2 << 20)
    )
    _, _, models = prior.stack(root, p, True)
    prior.rank.retain_adapter(md, models["initial"], "lora")
    return put(
        root / "plan.json",
        {
            k: p[k]
            for k in ("base", "lora", "origins", "files", "wheel", "native", "inventory", "dependencies")
        }
        | spec
        | dict(folds={k: put(root / (k + ".json"), v) for k, v in folds.items()}),
    )


def action(batches, order, weight, descend, start=None):
    return dict(
        kind="answers",
        batches=batches,
        order=order,
        update=dict(mode="descent", length=old.LENGTH) if descend else dict(mode="observe"),
        binding=dict(ids=[16, 2], mix_bits=last.bits([weight])[0]),
        start=start,
    )


def dense_reference(model, rows):
    """CE derivative = sum_{k != gold} p_k * derivative(logit[k] - logit[gold])."""
    ans = []
    losses = []
    gaps = []
    gap_grads = []
    outside = []
    u = pp.FACTORS[SITE[2]]
    for q in rows:
        y = gold(q)
        z = model.forward(q, [1, 9], last_only=True).values()
        cost, ps = nll(z, y)
        losses.append(cost)
        outside.append(1.0 - ps[16] - ps[2])
        terms = []
        for k, pw in enumerate(ps):
            if k == y:
                continue
            r = model.projection_contrast_pullback(q, k, y, site=SITE[1], layer=SITE[0], basis="dense")
            require(
                (
                    r["contrast"]["target_minus_alternative"],
                    r["site"],
                    r["layer"],
                    r["base_model_identity"],
                    r["geometry_artifact_id"],
                    r["lora_artifact_id"],
                )
                == (
                    z[k] - z[y],
                    SITE[1],
                    SITE[0],
                    model.base_model_identity,
                    model.geometry_artifact_id,
                    model.lora_artifact_id,
                ),
                "foreign dense reference",
            )
            w = struct.unpack("<1024d", r["weight_f64le"])
            g = [math.sumprod(u, w[j::32]) for j in range(32)]
            terms.append([pw * x for x in g])
            if k in (16, 2):
                gaps.append(z[16] - z[2])
                gap_grads.append([x if k == 16 else -x for x in g])
        ans.append([math.fsum(v) for v in zip(*terms, strict=True)])
    ce = [math.fsum(col) / len(rows) for col in zip(*ans, strict=True)]
    # The reference solver uses 8 quartets per call; equal-size microbatches carry equal weight.
    igrads = []
    iloss = []
    for n in range(0, len(rows), 32):
        loss, alpha = last.reference_seed(gaps[n : n + 32], "interaction", "aligned")
        iloss.append(loss)
        igrads.append([math.sumprod(alpha, col) for col in zip(*gap_grads[n : n + 32], strict=True)])
    return dict(
        answer=[math.fsum(losses) / len(rows), *ce],
        interaction=[
            math.fsum(iloss) / len(iloss),
            *[math.fsum(v) / len(iloss) for v in zip(*igrads, strict=True)],
        ],
        mean_other_probability=math.fsum(outside) / len(rows),
        dense_vjps=len(rows) * 63,
    )


def calibrate(root, p, pin, out, binary, models):
    qs = [q for k in cx.FOLDS[:2] for q in obj(root / (k + ".json"), p["folds"][k])]
    scales = {}
    for state in old.STATES:
        packet = cx.invoke(binary, root, p, state, *SITE, action([batch(qs, False)], [0], 0.5, False))
        ph = put(out / (state + ".json"), packet)
        (step,) = packet["steps"]
        z = step["report"]
        require(
            step["right_bits"] == [0] * 32
            and packet["initial"] == packet["owner"] == cx.owner_id(p, state, *SITE, [0] * 32)
            and (z["prompt_forwards"], z["projection_pullbacks"]) == (64, 128),
            "not a zero-update observation",
        )
        ref = dense_reference(models[state], qs)
        rh = put(out / (state + ".reference.json"), ref)
        for key, atol, rtol in (("answer", 2e-6, 1e-4), ("interaction", 1e-9, 1e-8)):
            v = z[key]
            require(
                abs(v["loss"] - ref[key][0]) <= atol
                and all(
                    math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
                    for a, b in zip(v["gradient"], ref[key][1:], strict=True)
                ),
                "objective differs from independent dense wheel: " + key,
            )
        ce, inter = z["answer"]["norm"], z["interaction"]["norm"]
        require(min(ce, inter) > 1e-12 and math.isfinite(ce + inter), "no bounded calibration")
        scale = ce / (ce + inter)
        require(0 < scale < 1, "unrepresentable mixture")
        scales[state] = dict(
            mix=scale,
            ce_norm=ce,
            interaction_norm=inter,
            cosine=math.sumprod(z["answer"]["gradient"], z["interaction"]["gradient"]) / (ce * inter),
            native=ph,
            reference=rh,
        )
    return put(out / "complete.json", dict(plan=pin, scales=scales, weight_updates=0))


def lanes():
    return itertools.product(old.STATES, ("ab", "ba"), POLICIES)


def train(root, p, pin, out, binary, cp):
    c = obj(root.parent / "calibration/complete.json", cp)
    require(c["plan"] == pin, "foreign scale")
    qs = [obj(root / (k + ".json"), p["folds"][k]) for k in cx.FOLDS[:2]]
    records = {}
    first_ce = {}
    resume = None
    for state, order, policy in lanes():
        scale = mixture(policy, c["scales"][state]["mix"])
        idx = ([0, 1] if order == "ab" else [1, 0]) * (STEPS // 2)
        batches = [batch(rows, policy.endswith("_control")) for rows in qs]
        packet = cx.invoke(binary, root, p, state, *SITE, action(batches, idx, scale, True))
        tag = ".".join((state, order, policy))
        records[tag] = put(out / (tag + ".json"), packet)
        steps = packet["steps"]
        oldbits = [0] * 32
        require(
            packet["initial"] == cx.owner_id(p, state, *SITE, oldbits) and len(steps) == STEPS,
            "foreign training origin",
        )
        for j, (step, i) in enumerate(zip(steps, idx, strict=True)):
            r = step["report"]
            ce = r["answer"]
            inter = r["interaction"]
            combined = r["combined"]
            require(
                (step["batch"], r["mix"], r["prompt_forwards"], r["projection_pullbacks"])
                == (i, scale, 32, 64),
                "different exposure budget",
            )
            if j == 0:
                require(first_ce.setdefault((state, i), ce) == ce, "control changed gold CE at origin")
            expected = [
                (1 - scale) * a + scale * b for a, b in zip(ce["gradient"], inter["gradient"], strict=True)
            ]
            n = math.hypot(*expected)
            require(
                math.dist(expected, combined["gradient"]) < 1e-12
                and math.isclose(n, combined["norm"], rel_tol=1e-12)
                and abs((1 - scale) * ce["loss"] + scale * inter["loss"] - combined["loss"]) < 1e-12,
                "wrong objective mixture",
            )
            before = last.doubles(oldbits)
            expected = (
                [x - old.LENGTH * (g / n) for x, g in zip(before, expected, strict=True)] if n else before
            )
            oldbits = step["right_bits"]
            require(
                math.dist(expected, last.doubles(oldbits)) < 1e-12
                and step["owner"] == cx.owner_id(p, state, *SITE, oldbits),
                "wrong trained state",
            )
        require(packet["owner"] == steps[-1]["owner"], "end owner mismatch")
        if (state, order, policy) == ("initial", "ab", "fit_scale"):
            cut = STEPS // 2
            start = {k: steps[cut - 1][k] for k in ("owner", "right_bits")}
            replay = cx.invoke(binary, root, p, state, *SITE, action(batches, idx[cut:], scale, True, start))
            resume = put(out / "resume.json", replay)
            require(
                replay["initial"] == start["owner"]
                and replay["steps"] == steps[cut:]
                and replay["owner"] == packet["owner"],
                "resume diverged",
            )
    return put(
        out / "complete.json",
        dict(
            plan=pin,
            calibration=cp,
            ends=records,
            resume=resume,
            resume_steps=32,
            updates=STEPS * len(records),
        ),
    )


def snapshots(root, p, t):
    require(set(t["ends"]) == {".".join(v) for v in lanes()} and t["updates"] == 1536, "incomplete learning")
    for state in old.STATES:
        yield state + ".hold", state, [0] * 32
    for state, order, policy in lanes():
        tag = ".".join((state, order, policy))
        packet = obj(root.parent / "trained" / (tag + ".json"), t["ends"][tag], 2 << 20)
        right = packet["steps"][-1]["right_bits"]
        require(packet["owner"] == cx.owner_id(p, state, *SITE, right), "mismatched saved endpoint")
        yield tag, state, right


def serve(root, p, pin, out, binary, st, models, tp, *, endpoint_rows=snapshots):
    t = obj(root.parent / "trained/complete.json", tp)
    require(t["plan"] == pin, "wrong fit")
    ds = [obj(root / (k + ".json"), p["folds"][k]) for k in cx.FOLDS]
    arch = cx.LayerArchive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
    layer, site, i = SITE
    start, stride, _ = arch.layout[layer, "query"]
    arch.layout[layer, site] = (start, stride, 2 * pp.W)
    cfg = read(root / "model/config.json", p["files"]["config.json"]).decode()
    records = {}
    for tag, state, right in endpoint_rows(root, p, t):
        delta = [u * v for u in pp.FACTORS[i] for v in last.doubles(right)]
        m, _ = old.snapshot(st, models[state], arch, cfg, {(layer, site): delta})
        a = []
        b = []
        max_error = 0.0
        for off in (0, 2, 4):
            qs = ds[off] + ds[off + 1]
            ps, branches = cx.paths(qs)
            z = cx.invoke(
                binary,
                root,
                p,
                state,
                *SITE,
                dict(kind="branches", right_bits=right, prefixes=ps, paths=branches),
            )
            require(z["prefills"] == 32, "incorrect common-prefix count")
            for q, r in zip(qs, z["rows"], strict=True):
                ordinary, cached = cx.rowbytes(r["ordinary"]), cx.rowbytes(r["cached"])
                reference = old.ROW.pack(*m.forward(q, [1, 9], last_only=True).values())
                if ordinary != reference:
                    write(out / (tag + ".mismatch"), ordinary + reference)
                    raise ValueError("physical value-column replay differs: " + tag)
                max_error = max(max_error, cx.agree(ordinary, cached))
                a.append(ordinary)
                b.append(cached)
        records[tag] = dict(
            full=write(out / (tag + ".full"), b"".join(a)),
            cache=write(out / (tag + ".cache"), b"".join(b)),
            max_error=max_error,
        )
    return put(out / "complete.json", dict(plan=pin, train=tp, rows=len(records) * 192, records=records))


def metrics(qs, raw):
    v = cx.from_rows(qs, raw)
    costs = []
    gaps = []
    for q, z in zip(qs, old.ROW.iter_unpack(raw), strict=True):
        costs.append(nll(z, gold(q))[0])
        gaps.append(z[16] - z[2])
    return v | dict(
        answer_nll=math.fsum(costs) / len(costs),
        ranked=sum(
            min(gaps[j], gaps[j + 3]) - max(gaps[j + 1], gaps[j + 2]) > pp.EPS for j in range(0, 32, 4)
        ),
        prefer_standby=sum(x > 0 for x in gaps),
    )


def audit(root, p, pin, out, cp, tp, sp):
    c = obj(root.parent / "calibration/complete.json", cp)
    t = obj(root.parent / "trained/complete.json", tp)
    s = obj(root.parent / "scored/complete.json", sp)
    require(
        t["plan"] == s["plan"] == c["plan"] == pin and t["calibration"] == cp and s["train"] == tp,
        "broken lineage",
    )
    witness = "initial.ab.fit_scale"
    uninterrupted = obj(root.parent / "trained" / (witness + ".json"), t["ends"][witness], 2 << 20)
    replay = obj(root.parent / "trained/resume.json", t["resume"], 2 << 20)
    require(
        replay["initial"] == uninterrupted["steps"][31]["owner"]
        and replay["steps"] == uninterrupted["steps"][32:]
        and replay["owner"] == uninterrupted["owner"],
        "saved continuation does not match",
    )
    ds = decks()
    require(all(obj(root / (k + ".json"), p["folds"][k]) == v for k, v in ds.items()), "changed fold")
    scores = {}
    gates = {}
    for tag, _, _ in snapshots(root, p, t):
        rec = s["records"][tag]
        folder = root.parent / "scored"
        a = read(folder / (tag + ".full"), rec["full"])
        b = read(folder / (tag + ".cache"), rec["cache"])
        require(
            len(a) == len(b) == 192 * 256
            and max(cx.agree(a[j : j + 256], b[j : j + 256]) for j in range(0, len(a), 256))
            == rec["max_error"],
            "serving mismatch",
        )
        scores[tag] = {k: metrics(ds[k], a[j * 8192 : (j + 1) * 8192]) for j, k in enumerate(cx.FOLDS)}
    require(
        len(scores) == len(s["records"]) == 26 and s["rows"] == 192 * len(scores), "incomplete serving roster"
    )
    for state, order, policy, split in itertools.product(
        old.STATES, ("ab", "ba"), ("half", "fit_scale"), ("held", "transfer")
    ):
        k = ".".join((state, order, policy))
        ce = ".".join((state, order, "answer"))
        ctrl = ".".join((state, order, {"half": "half_control", "fit_scale": "fit_control"}[policy]))
        fs = [f for f in cx.FOLDS if f.startswith(split)]

        def val(k, x):
            return sum(scores[k][f][x] for f in fs)

        vs = {}
        for ref in (ce, ctrl):
            diff = [
                math.fsum(x - y for x, y in zip(a, b, strict=True)) / 4
                for f in fs
                for a, b in zip(scores[k][f]["edges"], scores[ref][f]["edges"], strict=True)
            ]
            vs[ref] = dict(
                shared_delta=math.fsum(diff) / 8,
                edges=sum(d > pp.EPS for d in diff),
                correct_gain=val(k, "correct") - val(ref, "correct"),
                ranked_gain=val(k, "ranked") - val(ref, "ranked"),
                full_gain=val(k, "full_correct") - val(ref, "full_correct"),
            )
        nll_delta = (val(k, "answer_nll") - val(ce, "answer_nll")) / 2
        gates[k + "." + split] = dict(
            vs=vs,
            nll_delta=nll_delta,
            shared_gate=all(v["edges"] >= 6 for v in vs.values()),
            behavior_gate=nll_delta <= 0.01
            and all(
                v["correct_gain"] > 0 and v["ranked_gain"] > 0 and v["full_gain"] >= 0 for v in vs.values()
            ),
        )
    return put(
        out / "summary.json",
        dict(plan=pin, calibration=cp, training=tp, serving=sp, scores=scores, gates=gates),
    )


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
        "source or plan drift",
    )
    out = (
        root.parent
        / {"calibrate": "calibration", "train": "trained", "serve": "scored", "audit": "readback"}[phase]
    )
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, phase=phase, attempt=1))
    if phase == "audit":
        return audit(root, p, pin, out, *args[1:])
    binary = Path(args[1]).resolve()
    require(
        prior.signature(binary) == p["binary"] and os.environ.get("SPIRALTORCH_BACKEND") == "cpu",
        "wrong runtime",
    )
    if phase in ("calibrate", "train"):

        def fence(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).name in {k + ".json" for k in cx.FOLDS[2:]}
            ):
                raise PermissionError("non-fit roster sealed")

        sys.addaudithook(fence)
    if phase == "train":
        return train(root, p, pin, out, binary, *args[2:])
    st, installed, models = prior.stack(root, p, True)
    put(out / "installation.json", installed)
    return (
        calibrate(root, p, pin, out, binary, models)
        if phase == "calibrate"
        else serve(root, p, pin, out, binary, st, models, *args[2:])
    )


if __name__ == "__main__":
    print(main(sys.argv[1:]))

"""Fit at two depths; serve ordinary and shared-prefix predictions; intervene on one causal key."""

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_site_binding as last
from partner_routing_audit import from_rows
from shared_prefix_study import obj, pack, put, read, require, write

pp, old, prior = last.pp, last.old, last.prior
FOLDS = ("train0", "train1", "held0", "held1", "transfer0", "transfer1")
OBJECTIVE = "interaction"
PLAN = dict(
    data="Cross two previously separate identity pools. First four of each pool: shifts 0/1 fit, 2/3 held. Last four of each pool: shifts 2/3 specificity transfer only. All 24 pairs new. No previous task's held examples.",
    arms="2 related geometry origins x Q/K x layers 0/1 x Walsh bits 1/2 x aligned/symmetric. Same training cycle, 8 native steps of length .08. Only the 32 right coordinates train.",
    primary="Within the eight held pairs: aligned layer-1 K with original K factor (bit 2) vs hold, aligned layer-0 K, aligned layer-1 Q and symmetric layer-1 K. In both origins: mean over four A/Q margins gains >5e-6 on >=6/8 edges against EACH reference. Report bit 1 and eight out-of-fit pairs separately.",
    behavior="Separately require >=6/8 edges improving ALL FOUR margins vs each reference, more binary correct than each, and more strictly ranked quartets than hold. No endpoint selection.",
    serving="Native and independent merged-wheel ordinary logits must bit-match. Prefill seven factual tokens once per distinct prefix, branch through question + readout. All cached logits within 5e-5 + 5e-6*abs(reference), greedy unchanged if well-separated.",
    probe="Within held edges only, fixed snapshots: hold, bits-2 aligned K0/K1 and symmetric K1, both starts. In each prompt transfer ONLY position-2 K from a three-token donor [60,other entity,same cue]. Score actual remaining six tokens. No-op self must bit-match, same-token K0 must bit-match and have no effect. K1 differences or effects are measurements, not required. Compare difference of factual and patched binding margins to hold; not a whole pathway ablation.",
    effort="CPU only. 240 seconds per physical phase, one run per phase, retain failures. No GPT-2, GPU, wheel rebuild, cache cleanup, publishing or model downloads.",
)


def sources():
    return last.source_pins() | {
        p.name: prior.signature(p)
        for p in (
            Path(__file__),
            Path(__file__).resolve().parents[2] / "crates/st-nn/src/causal_lm/context.rs",
            Path(last.__file__),
        )
    }


def decks():
    a, b = pp.task.ENTITIES, pp.IDS
    require(len(a) == len(b) == 8 and not set(a) & set(b), "identity pool collision")
    out = {}
    seen = set()
    for fold, (start, shift) in zip(FOLDS, ((0, 0), (0, 1), (0, 2), (0, 3), (4, 2), (4, 3)), strict=True):
        pairs = [(a[start + i], b[start + (i + shift) % 4]) for i in range(4)]
        require(len(set(pairs)) == 4 and not seen & set(pairs), "pair leakage")
        seen.update(pairs)
        out[fold] = [
            pp.task.render(e, o, t, q) for e in pairs for o, t, q in itertools.product(range(2), repeat=3)
        ]
    require(len({tuple(q) for v in out.values() for q in v}) == 192, "duplicate prompt")
    return out


def lanes():
    return itertools.product(old.STATES, (0, 1), last.SITES, (0, 1), last.VIEWS)


def name(state, layer, site, factor, view):
    return f"{state}.L{layer}.{site}.{factor}.{view}"


def owner_id(p, state, layer, site, i, right):
    digest = hashlib.sha256(b"spiraltorch.rank_one_projection_prefix.v1\0")
    for field in (p["base"], p["lora"], p["origins"][state], "f32_wide_backward", site):
        s = field.encode()
        digest.update(struct.pack("<Q", len(s)) + s)
    digest.update(struct.pack("<QQ", layer, pp.W))
    for values in (last.bits(pp.FACTORS[i]), right):
        digest.update(struct.pack("<32Q", *values))
    return "sha256:" + digest.hexdigest()


class LayerArchive(pp.Archive):
    """Use ordinary HF input-major fused QKV storage, independently of native packing."""

    def __init__(self, raw):
        super().__init__(raw)
        front = 8 + struct.unpack_from("<Q", raw)[0]
        header = json.loads(raw[8:front])
        self.layout = {}
        for layer in (0, 1):
            tensor = header[f"h.{layer}.attn.c_attn.weight"]
            require(tensor["shape"] == [pp.W, 3 * pp.W], "not the fixture's QKV matrix")
            for part, site in enumerate(("query", "key")):
                self.layout[layer, site] = (front + tensor["data_offsets"][0], 3 * pp.W, part * pp.W)


def paths(rows):
    prefixes = []
    ix = {}
    branches = []
    for q in rows:
        tag = tuple(q[:7])
        key = ix.setdefault(tag, len(ix))
        if key == len(prefixes):
            prefixes.append(list(tag))
        branches.append(dict(prefix=key, suffix=q[7:]))
    require(
        len(prefixes) * 2 == len(rows) and all(len(b["suffix"]) == 2 for b in branches), "unbalanced prefix"
    )
    return prefixes, branches


def probe_cases(rows):
    return [dict(recipient=q[:3], donor=[q[0], q[4], q[2]], position=2, suffix=q[3:]) for q in rows]


def freeze(root, parent, plan_pin, summary_pin, binary):
    p = obj(parent / "protocol/plan.json", plan_pin)
    m = obj(parent / "readback/summary.json", summary_pin, 1 << 20)
    require(
        p["kind"] == "spiraltorch.single_projection_quartet.v1"
        and p["plan"] == last.PLAN
        and m["plan"] == plan_pin
        and m["updates"] == 256
        and p["dependencies"] == prior.sources(),
        "wrong predecessor",
    )
    root.mkdir(mode=0o700)
    md = root / "model"
    md.mkdir(mode=0o700)
    for f, h in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((s[7:] + ".json", s[7:]) for s in p["origins"].values()),
    ]:
        write(md / f, read(parent / "protocol/model" / f, h, 100000))
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
        | dict(
            kind="spiraltorch.contextual_key_depth.v1",
            parent=[plan_pin, summary_pin],
            schedule=PLAN,
            source=sources(),
            binary=prior.signature(binary),
            folds={k: put(root / (k + ".json"), v) for k, v in decks().items()},
        ),
    )


def invoke(binary, root, p, state, layer, site, i, action):
    md = root / "model"
    request = dict(
        base=p["base"],
        lora=p["lora"],
        geometry=p["origins"][state],
        layer=layer,
        site=site,
        left_bits=last.bits(pp.FACTORS[i]),
        action=action,
    )
    proc = subprocess.run(
        [
            binary,
            md / "config.json",
            md / "model.safetensors",
            md / (p["lora"][7:] + ".json"),
            md / (request["geometry"][7:] + ".json"),
        ],
        input=pack(request),
        stdout=subprocess.PIPE,
        check=True,
        cwd=md,
        timeout=180,
    )
    result = json.loads(proc.stdout)
    require(all(result[k] == request[k] for k in ("base", "lora", "geometry")), "foreign process result")
    payload = result["payload"]
    if "right_bits" in action:
        require(
            payload["owner"] == owner_id(p, state, layer, site, i, action["right_bits"]),
            "serving snapshot mismatch",
        )
    return payload


def train(root, p, pin, out, binary, models):
    qs = [obj(root / (f + ".json"), p["folds"][f]) for f in FOLDS[:2]]
    ref = {}
    for state, layer, site in itertools.product(old.STATES, (0, 1), last.SITES):
        rows = [
            models[state].projection_contrast_pullback(q, 16, 2, site=site, layer=layer, basis="dense")
            for q in qs[0]
        ]
        require(
            all(
                (
                    r["base_model_identity"],
                    r["geometry_artifact_id"],
                    r["lora_artifact_id"],
                    r["layer"],
                    r["site"],
                )
                == (p["base"], p["origins"][state], p["lora"], layer, site)
                for r in rows
            ),
            "foreign dense gradient",
        )
        weights = [struct.unpack("<1024d", r["weight_f64le"]) for r in rows]
        for i, u in enumerate(pp.FACTORS):
            ref[state, layer, site, i] = (
                [r["contrast"]["target_minus_alternative"] for r in rows],
                [[math.sumprod(u, w[j::32]) for j in range(32)] for w in weights],
            )
    ends = {}
    for args in lanes():
        state, layer, site, i, view = args
        packet = invoke(
            binary,
            root,
            p,
            state,
            layer,
            site,
            i,
            dict(
                kind="train",
                ids=[16, 2],
                batches=qs,
                groups=last.groups(view),
                order=last.ORDER,
                objective=OBJECTIVE,
                length=old.LENGTH,
            ),
        )
        lane = name(*args)
        ends[lane] = put(out / (lane + ".json"), packet)
        steps = packet["steps"]
        require([z["batch"] for z in steps] == last.ORDER, "wrong step order")
        g, r = ref[state, layer, site, i]
        loss, cs = last.reference_seed(g, OBJECTIVE, view)
        seed = [loss, *[math.sumprod(cs, v) for v in zip(*r, strict=True)]]
        s = steps[0]["report"]
        require(
            all(
                math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-8)
                for x, y in zip(seed, [s["loss"], *s["gradient"]], strict=True)
            ),
            "compact gradient differs from dense reference",
        )
        right = [0.0] * 32
        for z in steps:
            r = z["report"]
            grad = r["gradient"]
            norm = r["gradient_norm"]
            require(
                math.isfinite(r["loss"])
                and math.isclose(math.hypot(*grad), norm, rel_tol=1e-12)
                and (r["quartets"], r["prompt_evaluations"]) == (16, 32),
                "invalid step",
            )
            expected = (
                [x - old.LENGTH * (y / norm) for x, y in zip(right, grad, strict=True)] if norm else right
            )
            right = last.doubles(z["right_bits"])
            require(math.dist(expected, right) < 1e-12, "step mismatch")
    return put(out / "complete.json", dict(plan=pin, ends=ends, updates=256, dense_reference_rows=256))


def snapshots(root, t, p):
    require(set(t["ends"]) == {name(*x) for x in lanes()} and t["updates"] == 256, "missing endpoints")
    for state in old.STATES:
        yield state + ".hold", (state, 0, "key", 1), [0] * 32
    for args in lanes():
        lane = name(*args)
        steps = obj(root.parent / "trained" / (lane + ".json"), t["ends"][lane])["steps"]
        yield lane, args[:4], steps[-1]["right_bits"]


def rowbytes(row):
    return struct.pack("<64I", *row)


def agree(factual, cached):
    expected = old.ROW.unpack(factual)
    other = old.ROW.unpack(cached)
    require(
        all(
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 5e-5 + 5e-6 * abs(a)
            for a, b in zip(expected, other, strict=True)
        ),
        "cached/full numerical mismatch",
    )
    j = max(range(64), key=expected.__getitem__)
    if expected[j] - max(expected[:j] + expected[j + 1 :]) > 2 * (5e-5 + 5e-6 * max(map(abs, expected))):
        require(max(range(64), key=other.__getitem__) == j, "well-separated greedy decision changed")
    return max(abs(a - b) for a, b in zip(expected, other, strict=True))


def serve(root, p, pin, out, binary, st, models, tp):
    t = obj(root.parent / "trained/complete.json", tp)
    require(t["plan"] == pin, "foreign training")
    ds = {k: obj(root / (k + ".json"), p["folds"][k]) for k in FOLDS}
    arch = LayerArchive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
    cfg = read(root / "model/config.json", p["files"]["config.json"]).decode()
    records = {}
    for lane, args, right in snapshots(root, t, p):
        state, layer, site, i = args
        m, _ = old.snapshot(
            st,
            models[state],
            arch,
            cfg,
            {(layer, site): [u * v for u in pp.FACTORS[i] for v in last.doubles(right)]},
        )
        maxerr = 0.0
        count = 0
        with (out / (lane + ".full")).open("xb") as full, (out / (lane + ".cache")).open("xb") as cache:
            for off in (0, 2, 4):
                qs = ds[FOLDS[off]] + ds[FOLDS[off + 1]]
                prefixes, branches = paths(qs)
                z = invoke(
                    binary,
                    root,
                    p,
                    *args,
                    dict(kind="branches", right_bits=right, prefixes=prefixes, paths=branches),
                )
                require(z["prefills"] == len(prefixes), "wrong prefill count")
                count += z["prefills"]
                for q, r in zip(qs, z["rows"], strict=True):
                    a, b = rowbytes(r["ordinary"]), rowbytes(r["cached"])
                    full.write(a)
                    cache.write(b)
                    ref = old.ROW.pack(*m.forward(q, [1, 9], last_only=True).values())
                    if ref != a:
                        write(out / (lane + ".first-reference"), ref)
                        raise ValueError("independent merged wheel differs: " + lane)
                    maxerr = max(maxerr, agree(a, b))
        records[lane] = dict(
            ordinary=prior.signature(out / (lane + ".full")),
            cache=prior.signature(out / (lane + ".cache")),
            max_error=maxerr,
            prefills=count,
        )
    return put(out / "complete.json", dict(plan=pin, train=tp, rows=len(records) * 192, records=records))


def probe(root, p, pin, out, binary, tp, sp):
    t = obj(root.parent / "trained/complete.json", tp)
    s = obj(root.parent / "scored/complete.json", sp)
    require(t["plan"] == s["plan"] == pin and s["train"] == tp, "unbound endpoints")
    qs = [q for f in FOLDS[2:4] for q in obj(root / (f + ".json"), p["folds"][f])]
    results = {}
    for lane, args, right in snapshots(root, t, p):
        if not (
            lane.endswith(".hold") or (args[2:] == ("key", 1) and (lane.endswith("aligned") or args[1] == 1))
        ):
            continue
        rec = s["records"][lane]
        scored = root.parent / "scored"
        ordinary = read(scored / (lane + ".full"), rec["ordinary"])[2 * 8192 : 4 * 8192]
        z = invoke(
            binary,
            root,
            p,
            *args,
            dict(kind="key_probe", right_bits=right, cases=probe_cases(qs), layers=[0, 1]),
        )
        h = put(out / (lane + ".json"), z)
        kdiff = []
        gd = []
        seen = []
        factual = []
        swapped = []
        for j, r in enumerate(z["rows"]):
            fact = rowbytes(r["factual"])
            agree(ordinary[j * 256 : (j + 1) * 256], fact)
            lo, hi = r["probes"]
            require(
                (lo["layer"], hi["layer"]) == (0, 1)
                and all(x["self_logits"] == r["factual"] for x in (lo, hi))
                and lo["recipient_key"] == lo["donor_key"]
                and lo["donor_logits"] == r["factual"],
                "failed null substitution",
            )

            def vec(bits):
                return struct.unpack("<32f", struct.pack("<32I", *bits))

            kdiff.append(math.dist(vec(hi["recipient_key"]), vec(hi["donor_key"])))
            changed = rowbytes(hi["donor_logits"])
            v = old.ROW.unpack(fact)
            w = old.ROW.unpack(changed)
            require(all(map(math.isfinite, (*v, *w, kdiff[-1]))), "nonfinite key probe")
            gd.append((v[16] - v[2]) - (w[16] - w[2]))
            seen.append(fact != changed)
            factual.append(fact)
            swapped.append(changed)
        require(len(gd) == 64, "incomplete key experiment")

        def metric(rows):
            return sum(from_rows(qs[j : j + 32], b"".join(rows[j : j + 32]))["shared"] for j in (0, 32)) / 2

        results[lane] = dict(
            raw=h,
            mean_key_l2=sum(kdiff) / 64,
            changed_full=sum(seen),
            changed_gap=sum(abs(x) > pp.EPS for x in gd),
            mean_abs_gap=sum(map(abs, gd)) / 64,
            factual_shared=metric(factual),
            swapped_shared=metric(swapped),
            shared_effect=metric(factual) - metric(swapped),
        )
    require(len(results) == 8, "missing probe snapshots")
    return put(out / "complete.json", dict(plan=pin, train=tp, scoring=sp, probes=results))


def audit(root, p, pin, out, tp, sp, kp):
    s = obj(root.parent / "scored/complete.json", sp)
    t = obj(root.parent / "trained/complete.json", tp)
    probe = obj(root.parent / "probes/complete.json", kp)
    require(
        s["plan"] == t["plan"] == probe["plan"] == pin
        and s["train"] == probe["train"] == tp
        and probe["scoring"] == sp,
        "broken lineage",
    )
    ds = decks()
    require(all(obj(root / (k + ".json"), p["folds"][k]) == v for k, v in ds.items()), "wrong decks")
    metrics = {}
    noops = 0
    for lane, args, right in snapshots(root, t, p):
        rec = s["records"][lane]
        folder = root.parent / "scored"
        a = read(folder / (lane + ".full"), rec["ordinary"])
        b = read(folder / (lane + ".cache"), rec["cache"])
        require(
            len(a) == len(b) == 192 * 256
            and rec["prefills"] == 96
            and max(agree(a[i : i + 256], b[i : i + 256]) for i in range(0, len(a), 256)) == rec["max_error"],
            "bad serving inventory",
        )
        metrics[lane] = {}
        for n, k in enumerate(FOLDS):
            block = a[n * 8192 : (n + 1) * 8192]
            v = from_rows(ds[k], block)
            g = [r[16] - r[2] for r in old.ROW.iter_unpack(block)]
            v.update(
                ranked=sum(min(g[j], g[j + 3]) - max(g[j + 1], g[j + 2]) > pp.EPS for j in range(0, 32, 4)),
                prefer_standby=sum(x > 0 for x in g),
            )
            metrics[lane][k] = v
        if lane in probe["probes"]:
            z = obj(root.parent / "probes" / (lane + ".json"), probe["probes"][lane]["raw"], 2 << 20)
            require(z["owner"] == owner_id(p, *args, right) and len(z["rows"]) == 64, "bad key evidence")
            actual = []
            patched = []
            for j, r in enumerate(z["rows"]):
                require(
                    all(v["self_logits"] == r["factual"] for v in r["probes"])
                    and r["probes"][0]["donor_logits"] == r["factual"]
                    and r["probes"][0]["recipient_key"] == r["probes"][0]["donor_key"],
                    "failed negative control",
                )
                f = rowbytes(r["factual"])
                agree(a[2 * 8192 + j * 256 : 2 * 8192 + (j + 1) * 256], f)
                normal = old.ROW.unpack(f)
                swapped = old.ROW.unpack(rowbytes(r["probes"][1]["donor_logits"]))
                actual.append(normal[16] - normal[2])
                patched.append(swapped[16] - swapped[2])
                noops += 3

            def interaction(v):
                return math.fsum(x if i % 4 in (0, 3) else -x for i, x in enumerate(v)) / 32

            recorded = probe["probes"][lane]
            require(
                all(
                    abs(interaction(v) - recorded[k]) < 1e-12
                    for v, k in ((actual, "factual_shared"), (patched, "swapped_shared"))
                )
                and sum(abs(x - y) > pp.EPS for x, y in zip(actual, patched, strict=True))
                == recorded["changed_gap"],
                "substitution effect does not match raw logits",
            )
    decisions = {}
    for state, i, subset in itertools.product(old.STATES, (0, 1), ("held", "transfer")):
        folds = [k for k in FOLDS if k.startswith(subset)]
        target = name(state, 1, "key", i, "aligned")

        def count(k, metric):
            return sum(metrics[k][f][metric] for f in folds)

        refs = [
            state + ".hold",
            name(state, 0, "key", i, "aligned"),
            name(state, 1, "query", i, "aligned"),
            name(state, 1, "key", i, "symmetric"),
        ]
        vs = {}
        for ref in refs:
            dx = [
                [a - b for a, b in zip(e1, e0, strict=True)]
                for f in folds
                for e1, e0 in zip(metrics[target][f]["edges"], metrics[ref][f]["edges"], strict=True)
            ]
            ms = [math.fsum(e) / 4 for e in dx]
            vs[ref] = dict(
                delta=math.fsum(ms) / 8,
                positive=sum(x > pp.EPS for x in ms),
                all_four=sum(min(e) > pp.EPS for e in dx),
                correct_gain=count(target, "correct") - count(ref, "correct"),
            )
        decisions[target + "." + subset] = dict(
            vs=vs,
            correct=count(target, "correct"),
            ranked=count(target, "ranked"),
            shared_gate=all(v["positive"] >= 6 for v in vs.values()),
            behavior_gate=all(v["all_four"] >= 6 and v["correct_gain"] > 0 for v in vs.values())
            and count(target, "ranked") > count(refs[0], "ranked"),
        )
    return put(
        out / "summary.json",
        dict(
            plan=pin, training=tp, serving=sp, probe=kp, scores=metrics, decisions=decisions, no_op_rows=noops
        ),
    )


def main(argv):
    os.umask(0o077)
    phase, root, *args = argv
    root = Path(root).resolve()
    if phase == "freeze":
        parent, pin, h, b = args
        return freeze(root, Path(parent), pin, h, Path(b))
    p = obj(root / "plan.json", args[0])
    pin = args[0]
    require(
        p["schedule"] == PLAN and p["source"] == sources() and p["dependencies"] == prior.sources(),
        "source or plan changed",
    )
    if phase == "audit":
        out = root.parent / "readback"
        out.mkdir(mode=0o700)
        return audit(root, p, pin, out, *args[1:])
    binary = Path(args[1]).resolve()
    require(
        prior.signature(binary) == p["binary"] and os.environ.get("SPIRALTORCH_BACKEND") == "cpu",
        "foreign driver or backend",
    )
    out = root.parent / {"train": "trained", "serve": "scored", "probe": "probes"}[phase]
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, phase=phase, attempt=1))
    if phase == "probe":
        return probe(root, p, pin, out, binary, *args[2:])
    if phase == "train":

        def fence(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).name in {k + ".json" for k in FOLDS[2:]}
            ):
                raise PermissionError("non-fit roster is sealed")

        sys.addaudithook(fence)
    st, installed, models = prior.stack(root, p, True)
    put(out / "installation.json", installed)
    return (
        train(root, p, pin, out, binary, models)
        if phase == "train"
        else serve(root, p, pin, out, binary, st, models, *args[2:])
    )


if __name__ == "__main__":
    print(main(sys.argv[1:]))

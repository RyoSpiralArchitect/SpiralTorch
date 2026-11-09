"""Prospective K-only vs Q-only learning. Rust owns every optimizer step; the wheel is a reference."""

from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partner_routing_study as old
from partner_routing_audit import from_rows
from shared_prefix_study import obj, pack, put, read, require, write

pp, prior = old.pp, old.prior
FOLDS = ("train_a", "train_b", "held_a", "held_b")
SITES = ("key", "query")
OBJECTIVES = ("pairwise", "interaction")
VIEWS = ("aligned", "symmetric")
ORDER = [0, 1] * 4
PLAN = dict(
    steps=8,
    length=0.08,
    directions=["Walsh bit 1", "Walsh bit 2"],
    coordinates=32,
    populations="The remaining 16 unordered pairs of the previous eight identities. Two offset-3 matchings for fitting; two offset-2 matchings (8 edges) for scoring. No reuse of the previous 12 edges.",
    controls="Cross both sites with BOTH old fixed left factors, both objectives and label-reflection control. Start all rights at zero, same 32 prompts and 16 equally weighted indexed quartets each step. Two dependent frozen geometry states.",
    primary="K only, old K factor (bit 2), pairwise. In BOTH states, >=6/8 held edges with mean-margin gains >5e-6 over hold, matched Q and symmetric K separately. Call site-wide only if bit 1 also passes.",
    alternative="Same gates for interaction, additionally vs aligned pairwise K at the same factor.",
    behavior="Separate: >=6/8 improve EVERY order-averaged A/Q margin vs each designated reference, plus more correct binary choices than every reference. No arm selection.",
    ranking="Also count strict, calibration-free quartet rankings: both positive cells must outrank both negative cells within the same edge and presentation. Report, do not substitute for the behavior gate.",
    compatibility="Before accepting the fit, compare first-batch native losses and all compact gradients with the old dense wheel (1e-9 absolute + 1e-8 relative). Compare ALL ordinary raw logits bitwise on native vs independent safetensors replay.",
    bounds="One attempt each, CPU only, 240s per phase. No pretrained checkpoint, unpublished older evaluation, GPU, wheel build or public release.",
)


def source_pins():
    bench = Path(__file__).resolve().parent
    crate = bench.parents[1] / "crates/st-nn"
    paths = [
        bench / x
        for x in (
            "single_site_binding.py",
            "partner_routing_audit.py",
            "partner_routing_study.py",
            "partner_projection.py",
        )
    ]
    paths += [
        crate / x
        for x in (
            "examples/native_quartet_projection.rs",
            "src/causal_lm.rs",
            "src/causal_lm/quartet_projection.rs",
            "src/causal_lm/output_margin.rs",
            "src/causal_lm/scalar_output.rs",
            "src/causal_lm/projection_probe.rs",
        )
    ]
    return {p.name: prior.signature(p) for p in paths}


def decks():
    ids = pp.IDS
    edges = {
        FOLDS[0]: [(i, (i + 3) % 8) for i in range(0, 8, 2)],
        FOLDS[1]: [(i, (i + 3) % 8) for i in range(1, 8, 2)],
        FOLDS[2]: [(0, 2), (4, 6), (1, 3), (5, 7)],
        FOLDS[3]: [(2, 4), (6, 0), (3, 5), (7, 1)],
    }
    seen = {tuple(sorted((p[1], p[4]))) for fold in pp.roster().values() for p in fold}
    require(len(seen) == 12, "unexpected previous graph")
    out = {}
    for k, es in edges.items():
        pairs = [(ids[a], ids[b]) for a, b in es]
        block = {tuple(sorted(e)) for e in pairs}
        require(not (block & seen) and Counter(x for e in pairs for x in e) == Counter(ids), "partner reuse")
        seen.update(block)
        out[k] = [
            pp.task.render(e, o, a, q) for e in pairs for o, a, q in itertools.product(range(2), repeat=3)
        ]
    require(
        len(seen) == 28 and len({tuple(q) for qs in out.values() for q in qs}) == 128, "wrong edge coverage"
    )
    return out


def groups(view):
    require(view in VIEWS, "unknown supervision")
    return [
        [j + i for i in order]
        for j in range(0, 32, 4)
        for order in ((0, 1, 2, 3), (0, 1, 2, 3) if view == "aligned" else (1, 0, 3, 2))
    ]


def bits(xs):
    return list(struct.unpack(f"<{len(xs)}Q", struct.pack(f"<{len(xs)}d", *xs)))


def doubles(values):
    return list(struct.unpack(f"<{len(values)}d", struct.pack(f"<{len(values)}Q", *values)))


def lanes():
    return itertools.product(old.STATES, SITES, (0, 1), OBJECTIVES, VIEWS)


def tag(state, site, index, objname, view):
    return f"{state}.{site}.{index}.{objname}.{view}"


def freeze(root, parent, parent_pin, audit_pin, binary):
    p = obj(parent / "protocol/plan.json", parent_pin)
    audit = obj(parent / "audit.json", audit_pin)
    require(
        audit["plan"] == parent_pin
        and audit["nonzero_steps"] == 128
        and p["schedule"] == old.SCHEDULE
        and p["code"] == old.code()
        and p["dependencies"] == prior.sources(),
        "wrong parent",
    )
    root.mkdir(mode=0o700)
    folder = root / "model"
    folder.mkdir(mode=0o700)
    for name, pin in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((h[7:] + ".json", h[7:]) for h in p["origins"].values()),
    ]:
        write(folder / name, read(parent / "protocol/model" / name, pin, 100000))
    write(
        root / "wheel-inventory.json", read(parent / "protocol/wheel-inventory.json", p["inventory"], 2 << 20)
    )
    _, _, origins = prior.stack(root, p, True)
    for model in origins.values():
        prior.rank.retain_adapter(folder, model, "lora")
    files = {k: put(root / (k + ".json"), v) for k, v in decks().items()}
    return put(
        root / "plan.json",
        {
            k: p[k]
            for k in ("base", "lora", "origins", "wheel", "native", "inventory", "files", "dependencies")
        }
        | dict(
            kind="spiraltorch.single_projection_quartet.v1",
            plan=PLAN,
            source=source_pins(),
            parent=[parent_pin, audit_pin],
            binary=prior.signature(binary),
            folds=files,
        ),
    )


def drive(binary, root, p, state, site, index, action):
    args = [
        binary,
        root / "model/config.json",
        root / "model/model.safetensors",
        root / "model" / (p["lora"][7:] + ".json"),
        root / "model" / (p["origins"][state][7:] + ".json"),
    ]
    req = dict(
        base=p["base"],
        lora=p["lora"],
        geometry=p["origins"][state],
        layer=0,
        site=site,
        left_bits=bits(pp.FACTORS[index]),
        action=action,
    )
    done = subprocess.run(
        args, input=pack(req), stdout=subprocess.PIPE, check=True, cwd=root / "model", timeout=180
    )
    result = json.loads(done.stdout)
    require(all(result[k] == req[k] for k in ("base", "lora", "geometry")), "Rust owner mismatch")
    return result["payload"]


def reference_seed(gaps, objective, view):
    if objective == "pairwise":
        return pp.coefficients(gaps, view == "symmetric")
    # Independent of Rust's quartet loop: reflect the sign of the diagonal contrast.
    coeff = [0.0] * len(gaps)
    loss = 0.0
    for j in range(0, len(gaps), 4):
        v = (gaps[j] + gaps[j + 3] - gaps[j + 1] - gaps[j + 2]) / 2
        for sign in (1, 1 if view == "aligned" else -1):
            m = sign * v
            loss += (max(0.0, -m) + math.log1p(math.exp(-abs(m)))) / 16
            e = math.exp(-abs(m))
            g = -sign * (e if m >= 0 else 1.0) / (32 * (1 + e))
            for i, polarity in enumerate((1, -1, -1, 1)):
                coeff[j + i] += g * polarity
    return loss, coeff


def fit(root, p, pin, out, binary, models):
    train = [obj(root / (k + ".json"), p["folds"][k]) for k in FOLDS[:2]]
    refs = {}
    for state, site in itertools.product(old.STATES, SITES):
        gaps, grad = [], []
        for q in train[0]:
            r = models[state].projection_contrast_pullback(q, 16, 2, site=site, layer=0, basis="dense")
            require(
                (
                    r["base_model_identity"],
                    r["geometry_artifact_id"],
                    r["lora_artifact_id"],
                    r["site"],
                    r["width"],
                )
                == (p["base"], p["origins"][state], p["lora"], site, 32),
                "foreign reference pullback",
            )
            gaps.append(r["contrast"]["target_minus_alternative"])
            grad.append(struct.unpack("<1024d", r["weight_f64le"]))
        for i, factor in enumerate(pp.FACTORS):
            refs[state, site, i] = (gaps, [[math.sumprod(factor, m[j::32]) for j in range(32)] for m in grad])
    records = {}
    for args in lanes():
        state, site, i, objective, view = args
        packet = drive(
            binary,
            root,
            p,
            state,
            site,
            i,
            dict(
                kind="train",
                ids=[16, 2],
                batches=train,
                order=ORDER,
                groups=groups(view),
                objective=objective,
                length=old.LENGTH,
            ),
        )
        steps = packet["steps"]
        # Seal before comparisons, including any rejected first result.
        name = tag(*args)
        records[name] = put(out / (name + ".json"), packet)
        require([x["batch"] for x in steps] == ORDER, "wrong update order")
        gaps, dense = refs[state, site, i]
        loss, c = reference_seed(gaps, objective, view)
        g = [math.sumprod(c, col) for col in zip(*dense, strict=True)]
        first = steps[0]["report"]
        require(
            all(
                math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9)
                for a, b in zip([loss, *g], [first["loss"], *first["gradient"]], strict=True)
            ),
            "native first gradient disagrees with dense wheel",
        )
        prev = [0.0] * 32
        for step in steps:
            r = step["report"]
            d = doubles(step["right_bits"])
            require(
                r["prompt_evaluations"] == 32
                and r["quartets"] == 16
                and math.isfinite(r["loss"])
                and math.isclose(math.hypot(*r["gradient"]), r["gradient_norm"], rel_tol=1e-12),
                "bad step",
            )
            expected = (
                [x - old.LENGTH * y / r["gradient_norm"] for x, y in zip(prev, r["gradient"], strict=True)]
                if r["gradient_norm"]
                else prev
            )
            require(math.dist(expected, d) < 1e-12, "update did not apply its gradient")
            prev = d
    return put(
        out / "complete.json",
        dict(plan=pin, records=records, updates=len(records) * len(ORDER), dense_reference_rows=128),
    )


def score(root, p, pin, out, binary, models, st, tp):
    t = obj(root.parent / "trained/complete.json", tp)
    require(t["plan"] == pin and set(t["records"]) == {tag(*x) for x in lanes()}, "incomplete fit")
    queries = [q for k in FOLDS for q in obj(root / (k + ".json"), p["folds"][k])]
    cfg = read(root / "model/config.json", p["files"]["config.json"]).decode()
    arch = pp.Archive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
    files = {}

    def one(state, name, site, i, right):
        rust = drive(binary, root, p, state, site, i, dict(kind="score", right_bits=right, prompts=queries))[
            "logit_bits"
        ]
        raw = b"".join(struct.pack("<64I", *r) for r in rust)
        rh = write(out / (name + ".f32le"), raw)
        vs = doubles(right)
        m, _ = old.snapshot(st, models[state], arch, cfg, {site: [u * v for u in pp.FACTORS[i] for v in vs]})
        other = b"".join(old.ROW.pack(*m.forward(q, [1, 9], last_only=True).values()) for q in queries)
        if other != raw:
            write(out / (name + ".reference.f32le"), other)
            raise ValueError("native and independent physical logits differ: " + name)
        files[name] = rh

    for state in old.STATES:
        one(state, state + ".hold", "key", 1, [0] * 32)
    for args in lanes():
        state, site, i, _, _ = args
        name = tag(*args)
        right = obj(root.parent / "trained" / (name + ".json"), t["records"][name])["steps"][-1]["right_bits"]
        one(state, name, site, i, right)
    return put(
        out / "complete.json",
        dict(plan=pin, training=tp, raw=files, rows_per_runtime=len(queries) * len(files)),
    )


def readback(root, p, pin, tp, sp, out):
    s = obj(root.parent / "scored/complete.json", sp)
    t = obj(root.parent / "trained/complete.json", tp)
    require(
        t["plan"] == s["plan"] == pin
        and s["training"] == tp
        and set(s["raw"]) == set(t["records"]) | {k + ".hold" for k in old.STATES},
        "wrong run",
    )
    ds = {k: obj(root / (k + ".json"), p["folds"][k]) for k in FOLDS}
    require(ds == decks(), "changed roster")
    scores = {}
    for name, h in s["raw"].items():
        raw = read(root.parent / "scored" / (name + ".f32le"), h)
        require(len(raw) == 32768, "missing rows")
        scores[name] = {}
        for j, k in enumerate(FOLDS):
            block = raw[j * 8192 : (j + 1) * 8192]
            metrics = from_rows(ds[k], block)
            gaps = [v[16] - v[2] for v in old.ROW.iter_unpack(block)]
            ranked = sum(
                min(gaps[i], gaps[i + 3]) - max(gaps[i + 1], gaps[i + 2]) > pp.EPS for i in range(0, 32, 4)
            )
            scores[name][k] = metrics | dict(ranked_quartets=ranked)

    def compare(k, ref):
        a, b = scores[k], scores[ref]
        ds = [
            [a - b for a, b in zip(ae, be, strict=True)]
            for f in FOLDS[2:]
            for ae, be in zip(a[f]["edges"], b[f]["edges"], strict=True)
        ]
        ms = [math.fsum(x) / 4 for x in ds]
        return dict(
            mean=math.fsum(ms) / len(ms),
            positive=sum(x > pp.EPS for x in ms),
            all_four=sum(min(x) > pp.EPS for x in ds),
            correct_delta=sum(a[f]["correct"] - b[f]["correct"] for f in FOLDS[2:]),
        )

    decisions = {}
    for state, i, objname in itertools.product(old.STATES, (0, 1), OBJECTIVES):
        target = tag(state, "key", i, objname, "aligned")
        refs = [
            state + ".hold",
            tag(state, "query", i, objname, "aligned"),
            tag(state, "key", i, objname, "symmetric"),
        ]
        if objname == "interaction":
            refs.append(tag(state, "key", i, "pairwise", "aligned"))
        vs = {r: compare(target, r) for r in refs}
        decisions[target] = dict(
            shared_gate=all(v["positive"] >= 6 for v in vs.values()),
            behavior_gate=all(v["all_four"] >= 6 and v["correct_delta"] > 0 for v in vs.values()),
            contrasts=vs,
        )
    return put(
        out / "summary.json",
        dict(plan=pin, training=tp, scoring=sp, updates=t["updates"], scores=scores, decisions=decisions),
    )


def main(argv):
    os.umask(0o077)
    phase, r, *args = argv
    root = Path(r).resolve()
    if phase == "freeze":
        parent, ppin, apin, b = args
        return freeze(root, Path(parent), ppin, apin, Path(b).resolve())
    p = obj(root / "plan.json", args[0])
    require(
        p["plan"] == PLAN and p["source"] == source_pins() and p["dependencies"] == prior.sources(),
        "stale protocol",
    )
    if phase == "audit":
        out = root.parent / "readback"
        out.mkdir(mode=0o700)
        return readback(root, p, *args, out)
    require(phase in ("train", "score") and len(args) == (2 if phase == "train" else 3), "invalid phase")
    pin, b, *rest = args
    binary = Path(b).resolve()
    require(
        prior.signature(binary) == p["binary"] and os.environ.get("SPIRALTORCH_BACKEND") == "cpu",
        "changed executable or backend",
    )
    out = root.parent / ("trained" if phase == "train" else "scored")
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, phase=phase, attempt=1))
    if phase == "train":

        def fence(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).name in {"held_a.json", "held_b.json"}
            ):
                raise PermissionError("held partner rows are sealed")

        sys.addaudithook(fence)
    st, installed, models = prior.stack(root, p, True)
    put(out / "installation.json", installed)
    return (
        fit(root, p, pin, out, binary, models)
        if phase == "train"
        else score(root, p, pin, out, binary, models, st, *rest)
    )


if __name__ == "__main__":
    print(main(sys.argv[1:]))

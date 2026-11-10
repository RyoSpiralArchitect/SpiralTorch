"""Capacity first: ordinary vs lagged Q/K/V training on the same 32 consumed fit questions."""

import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guarded_answer_learning as prev
from shared_prefix_study import obj, pack, put, read, require, sha, write

a, pp, old, prior = prev.a, prev.pp, prev.old, prev.prior
MODEL_TYPE = "spiraltorch_gpt2_key_lag_v1"
SAVES = (16, 64, 128, 256)
MICROBATCH = 8
PLAN = dict(
    amendment="v1 stopped before any update: a 32x9x32 tensor exceeded the frozen geometry limit of 4096 values. Preserve v1 plan 9ea159b5229e3871d4f5e81fa7e279cb633419c110b0f4d0ba0f2c481b61eb95 and failure log bc21e49b1d64eb50d9cc52cf5cc303e21336bcfc49d9fee9574ab14d65eaa26b. v2 keeps whole sequences in microbatches of eight (2304 values), weights each mean gradient by sample count and applies AdamW once per logical batch. At the ordinary marker origin, cross-check the 32-row gradient with microbatches of four (absolute tolerance 2e-6).",
    microbatch_rows=MICROBATCH,
    data="Only the 32 original train0 prompts of the guard study, four already-consumed fit edges. No other training batch, held/transfer data or pretrained model. Marker-last is primary. Query-last exchanges only the final two tokens, keeping labels, prompt length, vocabulary and token multiset.",
    routes="Primary first block at lag 1 vs ordinary GPT-2 and causal lag 2. At value slot j use raw K[max(0,j-lag)]; Q, V, score biases and visibility stay at their original slots. Route is in the base ID. All initial weight bytes are identical; only the frozen adapter base-ID fields are explicitly rebound.",
    fit="All 3072 fused Q/K/V weight entries in block 0 train; QKV bias, other block, rank-2 output LoRA, geometry, embeddings and lm_head freeze. One starting geometry (initial). AdamW beta=.9/.999 epsilon=1e-8, lr=.01, decay=0, 256 steps. Two copies of 32 prompts each step. Negative control flips only gold in the second copy. No optimizer or stopping selection based on scores.",
    gate="At FINAL step 256: ordinary vocabulary top-1 correct 32/32 and all 8 edge/order quartets strictly ranked with margin >5e-6. Report NLL<=0.1 separately. Lag-specific endpoint claim additionally requires both other aligned routes and reflected lag-1 control NOT to solve on the same layout. The easier layout cannot rescue the primary. Report 0/16/64/128/256 milestones; this is fit capacity, not transfer.",
    checks="Explicit-gather/scatter kernel control and constructed content retrieval are separate from learned answers. Check all 3 Q/K/V gradient parts by physical +/-0.01 Frobenius edits under each route/layout before fitting. Full-vocab finite difference <=3e-5 + 2% relative error. Ordinary and prefix-cache logits must agree (5e-5 + 5e-6 relative). Reimport ALL final QKV arrays from fresh safetensors copies, keep other bytes fixed; check all logits bitwise. Old wheel must replay ordinary and refuse tagged lagged routes. Predetermined lag1/marker/aligned continuation 128 -> 256 must be exact.",
    bounds="CPU only, each physical phase <=240 wall seconds, one attempt, stop on first execution error. No PyPI, wheel rebuild, push or GPU.",
)


def sources():
    w = Path(__file__).resolve().parents[2]
    rs = w / "crates/st-nn/src/causal_lm"
    paths = [
        Path(__file__),
        w / "crates/st-nn/examples/native_key_lag_fit.rs",
        rs.parent / "causal_lm.rs",
        *(
            rs / f
            for f in ("qkv_training.rs", "key_lag.rs", "gpt2.rs", "context.rs", "learning.rs", "decoder.rs")
        ),
        w / "crates/st-kernel-contracts/src/attention.rs",
        w / "crates/st-kernel-contracts/src/attention/backward.rs",
        w / "crates/st-kernel-contracts/src/adamw.rs",
        Path(a.__file__),
        Path(prev.cx.__file__),
        Path(old.__file__),
        Path(pp.__file__),
        Path(__file__).with_name("partner_routing_audit.py"),
        Path(__file__).with_name("shared_prefix_study.py"),
    ]
    return prior.sources() | {str(x.relative_to(w)): prior.signature(x) for x in paths}


def present(q, layout):
    require(
        layout in ("marker", "query") and len(q) == 9 and q[-1] == 23 and q[-2] in (q[1], q[4]),
        "foreign fit prompt",
    )
    return q.copy() if layout == "marker" else q[:-2] + [q[-1], q[-2]]


def golds(rows, view):
    ys = [a.gold(q) for q in rows]
    require(view in ("aligned", "reflected") and ys.count(16) == ys.count(2) == 16, "invalid fit task")
    return ys + (ys if view == "aligned" else [18 - y for y in ys])


def base_id(ordinary, lag):
    if lag == 0:
        return ordinary
    return "sha256:" + sha(
        b"spiraltorch.native_causal_lm.gpt2.key_lag.v1\0" + ordinary.encode() + struct.pack("<QQ", 0, lag)
    )


def route(md, cfg, ordinary, lag, old_base, lora, geometry):
    md.mkdir(mode=0o700)
    new = base_id(ordinary, lag)

    def reowner(raw):
        require(
            raw.count(old_base.encode()) == 1 and json.loads(raw)["base_model_identity"] == old_base,
            "unbound adapter",
        )
        res = raw.replace(old_base.encode(), new.encode())
        require(
            json.loads(res) == json.loads(raw) | {"base_model_identity": new}, "non-owner parameter changed"
        )
        return res

    lo, ge = reowner(lora), reowner(geometry)
    expect = cfg if not lag else cfg | dict(model_type=MODEL_TYPE, spiraltorch_key_lag=dict(layer=0, lag=lag))
    return dict(
        base=new,
        lora="sha256:" + sha(b"spiraltorch.native_lora.v1\0" + lo),
        geometry="sha256:" + sha(ge),
        config=put(md / "config.json", expect),
        lo=write(md / "lora.json", lo),
        ge=write(md / "geometry.json", ge),
    )


def freeze(root, parent, pin, training, driver):
    p = obj(parent / "protocol/plan.json", pin)
    t = obj(parent / "trained/complete.json", training)
    require(
        p["kind"] == "spiraltorch.answer_guard.v1" and t["plan"] == pin and p["schedule"] == prev.PLAN,
        "wrong previous fit",
    )
    root.mkdir(mode=0o700)
    model = parent / "protocol/model"
    cfg = obj(model / "config.json", p["files"]["config.json"])
    require(
        (cfg["model_type"], cfg["n_layer"], cfg["n_embd"], cfg["vocab_size"]) == ("gpt2", 2, 32, 64),
        "wrong decoder",
    )
    raw = read(model / "model.safetensors", p["files"]["weights"], 100000)
    pp.Archive(raw)
    wp = write(root / "model.safetensors", raw)
    geo = p["origins"]["initial"]
    geom = read(model / (geo[7:] + ".json"), geo[7:])
    lp = model / (p["lora"][7:] + ".json")
    lora = read(lp, prior.signature(lp), 100000)
    require("sha256:" + sha(b"spiraltorch.native_lora.v1\0" + lora) == p["lora"], "foreign LoRA")
    qs = obj(parent / "protocol/train0.json", p["folds"]["train0"])
    golds(qs, "aligned")
    require(len({tuple(q) for q in qs}) == len(qs) == 32, "wrong training batch")
    prep = {str(k): route(root / f"lag{k}", cfg, p["base"], k, p["base"], lora, geom) for k in (0, 1, 2)}
    require(prep["0"]["lora"] == p["lora"] and prep["0"]["geometry"] == geo, "ordinary owner changed")
    return put(
        root / "plan.json",
        dict(
            kind="spiraltorch.causal_key_lag_fit.v2",
            schedule=PLAN,
            source=sources(),
            parent=[pin, training],
            binary=prior.signature(driver),
            routes=prep,
            weights=wp,
            fit=put(root / "fit.json", qs),
            inventory=write(
                root / "wheel-inventory.json",
                read(parent / "protocol/wheel-inventory.json", p["inventory"], 2 << 20),
            ),
            wheel=p["wheel"],
            native=p["native"],
        ),
    )


def invoke(driver, md, weights, variant, action, start=None):
    for k, f in (("config", "config.json"), ("lo", "lora.json"), ("ge", "geometry.json")):
        require(prior.signature(md / f) == variant[k], "changed model input")
    req = {k: variant[k] for k in ("base", "lora", "geometry")} | dict(layer=0, action=action, start=start)
    result = subprocess.run(
        [driver, md / "config.json", weights, md / "lora.json", md / "geometry.json"],
        input=pack(req),
        stdout=subprocess.PIPE,
        check=True,
        cwd=md,
        timeout=180,
    )
    z = json.loads(result.stdout)
    require(
        all(z[k] == req[k] for k in ("base", "lora", "geometry"))
        and z["key_lag"] == json.loads((md / "config.json").read_bytes()).get("spiraltorch_key_lag"),
        "foreign driver result",
    )
    return z


def call(driver, root, p, lag, act, start=None):
    return invoke(driver, root / f"lag{lag}", root / "model.safetensors", p["routes"][str(lag)], act, start)


def install(root, p):
    inv = obj(root / "wheel-inventory.json", p["inventory"], 2 << 20)
    require((inv["sha256"], inv["native"]["sha256"]) == (p["wheel"], p["native"]), "reference wheel mismatch")
    identity = prior.base.installed_identity(inv)
    return prior.base.native(), identity


def merge(raw, packed):
    """Output-major Q/K/V -> input-major Conv1D; splice only the single weight span."""
    pp.Archive(raw)
    require(len(packed) == 3072 and all(map(math.isfinite, f32s(packed))), "invalid QKV matrix")
    front = 8 + struct.unpack_from("<Q", raw)[0]
    h = json.loads(raw[8:front])["h.0.attn.c_attn.weight"]
    require(h["shape"] == [32, 96] and h["dtype"] == "F32", "foreign matrix")
    begin, end = [front + i for i in h["data_offsets"]]
    buf = struct.pack("<3072I", *(packed[j * 32 + i] for i in range(32) for j in range(96)))
    require(len(buf) == end - begin, "incorrect edit length")
    return raw[:begin] + buf + raw[end:]


def weight_bits(raw):
    archive = pp.Archive(raw)
    start, stride, _ = archive.layout["query"]
    return [
        struct.unpack_from("<I", raw, start + 4 * (i * stride + j))[0] for j in range(96) for i in range(32)
    ]


def f32s(bits):
    return struct.unpack(f"<{len(bits)}f", struct.pack(f"<{len(bits)}I", *bits))


def physical(st, root, p, out, lag, packed):
    raw = read(root / "model.safetensors", p["weights"], 100000)
    changed = merge(raw, packed)
    old = root / f"lag{lag}"
    ordinary = (root / "lag0/config.json").read_text()
    m = st.NativeCausalLm.from_gpt2(
        ordinary, st.SafeTensorArchive.from_bytes(changed)
    ).with_readout_accumulator("f32_wide_backward")
    v = p["routes"][str(lag)]
    spec = route(
        out,
        json.loads(ordinary),
        m.base_model_identity,
        lag,
        v["base"],
        read(old / "lora.json", v["lo"]),
        read(old / "geometry.json", v["ge"]),
    )
    write(out / "model.safetensors", changed)
    if lag == 0:
        m = m.with_lora_json((out / "lora.json").read_text()).with_geometry(
            st.GeometryAdapterArtifact.create_json((out / "geometry.json").read_text())
        )
        require(
            (m.base_model_identity, m.lora_artifact_id, m.geometry_artifact_id)
            == (spec["base"], spec["lora"], spec["geometry"]),
            "wrong independent wheel owners",
        )
    return spec, m if not lag else None


def scoring(qs):
    return dict(kind="score", prompts=qs, prefix_len=7)


def seal_rows(reply, qs, wheel=None):
    data = reply["data"]
    require(data["prefixes"] == len({tuple(q[:7]) for q in qs}), "wrong shared-prefix count")
    full = []
    cache = []
    error = 0.0
    for r, q in zip(data["rows"], qs, strict=True):
        x, y = (struct.pack("<64I", *r[k]) for k in ("ordinary", "cached"))
        error = max(error, prev.cx.agree(x, y))
        full.append(x)
        cache.append(y)
        if wheel is not None:
            require(
                old.ROW.pack(*wheel.forward(q, [1, len(q)], last_only=True).values()) == x,
                "old ordinary wheel differs",
            )
    return b"".join(full), b"".join(cache), error


def preflight(root, p, pin, out, driver, st):
    rows = obj(root / "fit.json", p["fit"])
    raw = read(root / "model.safetensors", p["weights"], 100000)
    orig = weight_bits(raw)
    require(merge(raw, orig) == raw, "original weights do not round-trip")
    w = f32s(orig)
    truth = [a.gold(q) for q in rows]
    report = {}
    for lag in (1, 2):
        try:
            st.NativeCausalLm.from_gpt2(
                (root / f"lag{lag}/config.json").read_text(), st.SafeTensorArchive.from_bytes(raw)
            )
        except (ValueError, RuntimeError) as e:
            require("unsupported GPT-2" in str(e), "unexpected old-wheel error")
        else:
            raise ValueError("old wheel silently admitted the new architecture")
    for lag, layout in itertools.product((0, 1, 2), ("marker", "query")):
        qs = [present(q, layout) for q in rows]
        lane = f"{lag}.{layout}"
        action = dict(kind="gradient", prompts=qs, gold=truth, microbatch=MICROBATCH)
        g = call(driver, root, p, lag, action)["data"]
        b = struct.pack("<3072I", *g["packed_bits"])
        h = write(out / (lane + ".gradient"), b)
        grad = struct.unpack("<3072f", b)
        repartition = None
        if (lag, layout) == (0, "marker"):
            alternate = call(driver, root, p, lag, action | dict(microbatch=4))["data"]
            grad_error = max(abs(x - y) for x, y in zip(grad, f32s(alternate["packed_bits"]), strict=True))
            loss_error = abs(g["loss"] - alternate["loss"])
            require(max(grad_error, loss_error) <= 2e-6, "microbatch repartition changed gradient or loss")
            repartition = dict(rows=4, max_grad_error=grad_error, loss_error=loss_error)
        z = call(driver, root, p, lag, scoring(qs))
        factual, _, _ = seal_rows(z, qs)

        def loss(blob):
            return (
                math.fsum(a.nll(v, y)[0] for v, y in zip(old.ROW.iter_unpack(blob), truth, strict=True)) / 32
            )

        require(abs(g["loss"] - loss(factual)) < 3e-6, "gold CE / ordinary logits differ")
        probes = []
        for part in range(3):
            start = part * 1024
            v = grad[start : start + 1024]
            norm = math.hypot(*v)
            require(norm > 0.0 and math.isfinite(norm), "zero or invalid QKV block gradient")
            costs = []
            for sgn in (-1, 1):
                vals = list(w)
                vals[start : start + 1024] = [
                    x + sgn * 0.01 * d / norm for x, d in zip(w[start : start + 1024], v, strict=True)
                ]
                bits = struct.unpack("<3072I", struct.pack("<3072f", *vals))
                md = out / f"{lane}.{part}.{sgn}"
                spec, model = physical(st, root, p, md, lag, bits)
                res = invoke(driver, md, md / "model.safetensors", spec, scoring(qs))
                logits, _, _ = seal_rows(res, qs, model)
                costs.append(loss(logits))
            central = (costs[1] - costs[0]) / 0.02
            require(
                abs(central - norm) <= 3e-5 + 0.02 * norm,
                "physical directional derivative differs: " + lane + f"/{part}",
            )
            probes.append(dict(norm=norm, central=central, losses=costs))
        report[lane] = dict(seed=h, gold_nll=g["loss"], checks=probes, repartition=repartition)
    return put(out / "complete.json", dict(plan=pin, old_wheel_refusals=2, checks=report))


def lanes():
    return itertools.product((0, 1, 2), ("marker", "query"), ("aligned", "reflected"))


def fit(root, p, pin, out, driver, fp):
    require(obj(root.parent / "preflight/complete.json", fp)["plan"] == pin, "missing finite gate")
    rows = obj(root / "fit.json", p["fit"])
    records = {}
    resume = None
    for lag, layout, mode in lanes():
        lane = f"{lag}.{layout}.{mode}"
        qs = [present(q, layout) for q in rows]
        folder = out / lane
        action = dict(
            kind="fit",
            prompts=2 * qs,
            gold=golds(rows, mode),
            microbatch=MICROBATCH,
            rate=0.01,
            updates=256,
            save_at=list(SAVES),
            out=str(folder),
        )
        z = call(driver, root, p, lag, action)
        records[lane] = put(out / (lane + ".json"), z)
        require(
            z["steps"] == 256
            and [x["steps"] for x in z["data"]["updates"]] == list(range(1, 257))
            and all(
                math.isfinite(x["gold_nll"]) and all(map(math.isfinite, x["gradient_l2"]))
                for x in z["data"]["updates"]
            ),
            "incomplete native steps",
        )
        pins = z["data"]["pins"]
        require(set(pins) == set(map(str, SAVES)), "unexpected snapshots")
        for n in SAVES:
            read(folder / f"{n}.bin", pins[str(n)]["checkpoint"][7:], 100000)
        if (lag, layout, mode) == (1, "marker", "aligned"):
            start = dict(path=str(folder / "128.bin"), id=pins["128"]["checkpoint"])
            replay = call(
                driver,
                root,
                p,
                lag,
                action | dict(updates=128, save_at=[128], out=str(out / "resume")),
                start,
            )
            resume = put(out / "resume.json", replay)
            require(
                replay["initial"] == pins["128"]["serving"]
                and replay["data"]["updates"] == z["data"]["updates"][128:]
                and replay["data"]["pins"]["256"] == pins["256"]
                and (out / "resume/256.bin").read_bytes() == (folder / "256.bin").read_bytes(),
                "resume disagrees",
            )
    return put(
        out / "complete.json",
        dict(plan=pin, preflight=fp, records=records, updates=3072, resume=resume, replay_updates=128),
    )


def checkpoint(blob, variant, steps):
    require(len(blob) >= 4, "truncated checkpoint")
    n = struct.unpack_from("<I", blob)[0]
    require(n <= 16384 and len(blob) == 4 + n + 3 * 4 * 3072, "wrong checkpoint shape")
    h = json.loads(blob[4 : 4 + n])
    require(
        h["schema"] == "spiraltorch.qkv_adamw.v1"
        and h["adam_bits"] == list(struct.unpack("<3I", struct.pack("<3f", 0.9, 0.999, 1e-8)))
        and (h["steps"], h["layer"], h["width"]) == (steps, 0, 32)
        and (h["base"], h["lora"], h["geometry"], h["readout"])
        == (variant["base"], variant["lora"], variant["geometry"], "f32_wide_backward"),
        "foreign checkpoint",
    )
    values = struct.unpack_from("<9216f", blob, 4 + n)
    require(all(map(math.isfinite, values)) and min(values[6144:]) >= 0.0, "invalid optimizer state")
    return struct.unpack_from("<3072I", blob, 4 + n)


def score(root, p, pin, out, driver, st, tp):
    t = obj(root.parent / "trained/complete.json", tp)
    require(t["plan"] == pin and t["updates"] == 3072 and len(t["records"]) == 12, "wrong fit")
    rows = obj(root / "fit.json", p["fit"])
    records = {}
    ordinary = weight_bits(read(root / "model.safetensors", p["weights"], 100000))
    for lag, layout, mode in lanes():
        lane = f"{lag}.{layout}.{mode}"
        qs = [present(q, layout) for q in rows]
        base = root.parent / "trained"
        run = obj(base / (lane + ".json"), t["records"][lane])
        items = {}
        for n in (0, *SAVES):
            start = None
            if n:
                item = run["data"]["pins"][str(n)]
                h = item["checkpoint"]
                f = base / lane / f"{n}.bin"
                bits = checkpoint(read(f, h[7:], 100000), p["routes"][str(lag)], n)
                start = dict(path=str(f), id=h)
            else:
                bits = ordinary
            z = call(driver, root, p, lag, scoring(qs), start)
            require(
                z["steps"] == n and z["end"] == (item["serving"] if n else run["initial"]),
                "wrong scoring state",
            )
            x, y, err = seal_rows(z, qs)
            name = f"{lane}.{n}"
            full = write(out / (name + ".full"), x)
            cached = write(out / (name + ".cached"), y)
            reimport = None
            if n in (0, 256):
                md = out / (name + ".merged")
                spec, model = physical(st, root, p, md, lag, bits)
                rep = invoke(driver, md, md / "model.safetensors", spec, scoring(qs))
                xx, yy, _ = seal_rows(rep, qs, model)
                require(xx == x and yy == y, "reimported trained weights differ")
                reimport = dict(weights=prior.signature(md / "model.safetensors"), source=spec)
            items[str(n)] = dict(full=full, cached=cached, max_error=err, reimport=reimport)
        records[lane] = items
    return put(
        out / "complete.json", dict(plan=pin, training=tp, records=records, checkpoint_rows=12 * 5 * 32)
    )


def audit(root, p, pin, out, tp, sp):
    t = obj(root.parent / "trained/complete.json", tp)
    s = obj(root.parent / "scored/complete.json", sp)
    require(t["plan"] == s["plan"] == pin and s["training"] == tp, "broken lineage")
    qs = obj(root / "fit.json", p["fit"])
    records = {}
    decisions = {}
    original = read(root / "model.safetensors", p["weights"], 100000)
    trained = root.parent / "trained"
    witness = obj(trained / "1.marker.aligned.json", t["records"]["1.marker.aligned"])
    resumed = obj(trained / "resume.json", t["resume"])
    require(
        resumed["data"]["updates"] == witness["data"]["updates"][128:]
        and resumed["end"] == witness["end"]
        and read(trained / "resume/256.bin", witness["data"]["pins"]["256"]["checkpoint"][7:], 100000)
        == (trained / "1.marker.aligned/256.bin").read_bytes(),
        "wrong saved continuation",
    )
    for lag, layout, mode in lanes():
        lane = f"{lag}.{layout}.{mode}"
        scores = {}
        log = obj(trained / (lane + ".json"), t["records"][lane])
        for n in (0, *SAVES):
            rec = s["records"][lane][str(n)]
            md = root.parent / "scored"
            name = f"{lane}.{n}"
            x = read(md / (name + ".full"), rec["full"])
            y = read(md / (name + ".cached"), rec["cached"])
            require(
                len(x) == len(y) == 8192
                and max(prev.cx.agree(x[j : j + 256], y[j : j + 256]) for j in range(0, len(x), 256))
                == rec["max_error"],
                "bad served rows",
            )
            if n:
                h = log["data"]["pins"][str(n)]["checkpoint"][7:]
                bits = checkpoint(read(trained / lane / f"{n}.bin", h, 100000), p["routes"][str(lag)], n)
            else:
                bits = weight_bits(original)
            if n in (0, 256):
                merged = read(
                    md / (name + ".merged") / "model.safetensors", rec["reimport"]["weights"], 100000
                )
                require(merged == merge(original, bits), "wrong physical reimport bytes")
            m = a.metrics(qs, x)
            scores[str(n)] = dict(
                nll=m["answer_nll"],
                correct=m["correct"],
                full=m["full_correct"],
                ranked=m["ranked"],
                shared=m["shared"],
                solved=m["full_correct"] == 32 and m["ranked"] == 8,
            )
        records[lane] = scores
    for layout in ("marker", "query"):
        good = records[f"1.{layout}.aligned"]["256"]["solved"]
        refs = {
            f"{lag}.{view}": records[f"{lag}.{layout}.{view}"]["256"]["solved"]
            for lag, view in ((0, "aligned"), (2, "aligned"), (1, "reflected"))
        }
        decisions[layout] = dict(lag1_fit=good, controls=refs, lag1_specific=good and not any(refs.values()))
    return put(
        out / "summary.json", dict(plan=pin, training=tp, serving=sp, records=records, decisions=decisions)
    )


def main(argv):
    phase, folder, *args = argv
    root = Path(folder).resolve()
    os.umask(0o077)
    if phase == "freeze":
        parent, pin, tp, b = args
        return freeze(root, Path(parent), pin, tp, Path(b))
    pin = args[0]
    p = obj(root / "plan.json", pin)
    require(p["schedule"] == PLAN and p["source"] == sources(), "source or plan drift")
    out = (
        root.parent / {"check": "preflight", "fit": "trained", "score": "scored", "audit": "readback"}[phase]
    )
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(plan=pin, phase=phase, attempt=1))
    if phase == "audit":
        return audit(root, p, pin, out, *args[1:])
    driver = Path(args[1]).resolve()
    require(
        prior.signature(driver) == p["binary"]
        and os.environ.get("SPIRALTORCH_BACKEND") == "cpu"
        and prior.signature(root / "model.safetensors") == p["weights"],
        "foreign physical runtime",
    )
    if phase == "fit":
        return fit(root, p, pin, out, driver, *args[2:])
    st, identity = install(root, p)
    put(out / "installation.json", identity)
    return (
        preflight(root, p, pin, out, driver, st)
        if phase == "check"
        else score(root, p, pin, out, driver, st, *args[2:])
    )


if __name__ == "__main__":
    print(main(sys.argv[1:]))

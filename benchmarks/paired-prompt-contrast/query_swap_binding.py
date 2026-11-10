"""Fit by queried identity; compare a one-token query swap with a fixed-query assignment swap."""

import itertools
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slot_common_output as prev
from scalar_binding_study import closed, checkpoint
from shared_prefix_study import obj, put, read, record, require, sha, write

task, rank, base, signature = prev.task, prev.rank, prev.base, prev.signature
STATES = prev.STATES
RULES = ("mean", "assignment_id", "query_id", "joint_id")
SPACES = ("dense", "left")
H = prev.H
EPS = prev.EPS
WALL = 180
CALLS = 6
SGD = dict(learning_rate=16.0, max_step=0.08, max_abs=4.0, max_trials=8, min_decrease=1e-7)
KIND = "spiraltorch.query_assignment_identity.v1"
PLAN = dict(
    rules=list(RULES),
    spaces=list(SPACES),
    radii=list(H),
    optimizer=SGD,
    calls=CALLS,
    wall=WALL,
    fit="Even training edges only, 32 unique prompts per native bank. A changes cues only, Q changes "
    "queried identity only. Both have exactly the same positive and negative multisets. Index groups "
    "by the standby-target prompt's queried token ID (8 identities), average both orders. "
    "Equal mean A and Q must agree; no duplicate control. Joint shares groups, NOT 16 separate constraints.",
    learn="Fit every resolved fixed-left axis (including mean) on the SAME 16 A + 16 Q pairs. Start "
    "coefficient zero. Up to six same-batch guarded SGD calls; first refusal ends the arm. "
    "Never train on check, select by check, choose a best checkpoint, or replace a refused axis.",
    physical="Independent prior wheel on original, signed 0.002/0.02 unit Frobenius displacements and "
    "all compact learned endpoints. Same 4 odd offset-one check edges, familiar identities with "
    "different partners. Count each edge once. No old held-out, fresh evaluation, or pretrained LM.",
    gate="At +.02 on each check edge BOTH positive identities and BOTH swap types improve by >5e-6 "
    "averaged over orders. For the same four cells also require central and linear predicted changes "
    ">5e-6. At least 3/4 edges in each origin; per-cell central error <=1e-6 + 10% at both radii. "
    "Learned endpoints reported separately against BOTH hold and learned mean: >=3/4 check edges "
    "improve all four margins by >5e-6; separately test strictly more correct choices than both. "
    "Every finite trial replays ordinary and overlay logits/gaps within 8e-6; scalar endpoints must "
    "bit-match ordinary full-vocab rows. Preserve every arm, refusals, nulls and any first error.",
)


def sources():
    return prev.sources() | {
        s: signature(Path(__file__).with_name(s + ".py"))
        for s in ("query_swap_binding", "query_swap_readback")
    }


def design(rows):
    cell = {(r["edge"], r["order"], r["idle"], r["query"]): j for j, r in enumerate(rows)}
    require(len(cell) == len(rows) == 32 and len({r["id"] for r in rows}) == 32, "not a four-edge fold")
    a = []
    q = []
    groups = []
    keys = []
    for e in sorted({r["edge"] for r in rows}):
        for o, k in itertools.product((0, 1), repeat=2):
            pos, na, nq = (cell[x] for x in ((e, o, k, k), (e, o, 1 - k, k), (e, o, k, 1 - k)))
            r, x, y = (rows[j] for j in (pos, na, nq))

            def diff(z):
                return [i for i, (u, v) in enumerate(zip(r["prompt"], z["prompt"], strict=True)) if u != v]

            require(
                r["gold"] == task.IDLE
                and x["gold"] == y["gold"] == task.ON
                and diff(x) == [2, 5]
                and diff(y) == [7]
                and r["prompt"][7] == r["entities"][k],
                "unmatched counterfactual",
            )
            a.append([pos, na])
            q.append([pos, nq])
            groups.append(task.ENTITIES.index(r["prompt"][7]))
            keys.append([e, o, k])
    require(
        sorted(n for _, n in a) == sorted(n for _, n in q)
        and len({n for _, n in a}) == 16
        and all(groups.count(j) == 2 for j in range(8)),
        "unbalanced identities or negative reuse",
    )
    return dict(A=a, Q=q, groups=groups, keys=keys)


def fit_request(d, r):
    if r in ("mean", "assignment_id"):
        return d["A"], d["groups"], "equal_mean" if r == "mean" else "common_cosine"
    if r == "query_id":
        return d["Q"], d["groups"], "common_cosine"
    require(r == "joint_id", "unknown rule")
    return d["A"] + d["Q"], 2 * d["groups"], "common_cosine"


def freeze(parent, pp, mp, wheel, wp, ext, out):
    old = parent / "protocol"
    p = obj(old / "plan.json", pp)
    m = obj(parent / "measured/complete.json", mp)
    require(
        p["kind"] == prev.KIND
        and p["code"] == prev.sources()
        and p["settings"] == prev.SETTINGS
        and m["plan"] == pp,
        "wrong prior physical source",
    )
    inv = base.candidate_inventory(wheel, wp, ext)
    require(ext != p["native"], "expected indexed native implementation")
    out.mkdir(mode=0o700)
    for k in ("model", "fit", "check"):
        (out / k).mkdir(mode=0o700)
    for f, h in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((v[7:] + ".json", v[7:]) for v in p["origins"].values()),
    ]:
        write(out / "model" / f, read(old / "model" / f, h, 100000))
    folds = {k: obj(old / (k + ".json"), v) for k, v in p["folds"].items()}
    queries = []
    for i, k in enumerate(("fit", "check")):
        rows = folds[k]
        require(rows == [r for r in task.deck("train") if r["edge"] % 2 == i], "foreign cohort")
        design(rows)
        require(put(out / k / "rows.json", rows) == p["folds"][k], "fold encoding changed")
        queries.extend(task.queries(rows))
    write(out / "factors.json", read(old / "factors.json", p["factors"]))
    write(out / "baseline.json", read(old / "baseline.json", p["baseline"]))
    write(out / "reference-wheel.json", read(old / "wheel-inventory.json", p["inventory"], 2 << 20))
    return put(
        out / "plan.json",
        {
            k: p[k]
            for k in (
                "base",
                "lora",
                "origins",
                "files",
                "factors",
                "folds",
                "baseline",
                "pullback",
                "measurement",
            )
        }
        | dict(
            kind=KIND,
            settings=PLAN,
            code=sources(),
            parent=[pp, mp],
            wheel=wp,
            native=ext,
            inventory=put(out / "wheel-inventory.json", inv),
            reference=dict(wheel=p["wheel"], native=p["native"], inventory=p["inventory"]),
            queries=put(out / "queries.json", queries),
        ),
    )


def stack(proto, p, is_fit):
    spec = p if is_fit else p["reference"]
    inv = obj(
        proto / ("wheel-inventory.json" if is_fit else "reference-wheel.json"), spec["inventory"], 2 << 20
    )
    require((inv["sha256"], inv["native"]["sha256"]) == (spec["wheel"], spec["native"]), "wrong wheel")
    installed = base.installed_identity(inv)
    st = base.native()
    require(
        st.NativeCausalLm is st._rs.NativeCausalLm
        and Path(st._rs.__file__).resolve()
        == Path(st.__file__).resolve().parent.parent / inv["native"]["path"],
        "foreign decoder",
    )
    m = base.open_model(st, proto / "model", p["files"])
    require((m.base_model_identity, m.lora_artifact_id) == (p["base"], p["lora"]), "wrong frozen owners")
    return (
        st,
        installed,
        {
            s: prev.previous.prior.swap.attach(st, m, proto / "model" / (v[7:] + ".json"), v)
            for s, v in p["origins"].items()
        },
    )


def doubles(b):
    return [v for (v,) in struct.iter_unpack("<d", b)]


def train(proto, p, pp, out):
    _, installation, models = stack(proto, p, True)
    rows = obj(proto / "fit/rows.json", p["folds"]["fit"])
    d = design(rows)
    prompts = [r["prompt"] for r in rows]
    factors = obj(proto / "factors.json", p["factors"])
    axes = {}
    ends = {}
    summary = {}
    snaps = out / "checkpoints"
    snaps.mkdir(mode=0o700)
    both = d["A"] + d["Q"]
    args = ([prompts[i] for i, j in both], [prompts[j] for i, j in both], task.IDLE, task.ON)
    counts = dict(bank_vjps=0, optimizer_vjps=0, optimizer_forwards=0, updates=0)
    with (
        (out / "bank.jsonl").open("xb") as log,
        (out / "gradients.f64le").open("xb") as raw,
        (out / "steps.jsonl").open("xb") as journal,
    ):
        for state in STATES:
            model = models[state]
            u, _ = factors[state]
            for space in SPACES:
                left = struct.pack("<32d", *u) if space == "left" else None
                bank = model.output_contrast_bank(prompts, task.IDLE, task.ON, layer=0, left_f64le=left)
                require(len(bank) == 32, "wrong native roster")
                b = b"".join(bank.gradient_f64le(i) for i in range(32))
                raw.write(b)
                record(log, dict(state=state, space=space, sha=sha(b), contrasts=bank.contrasts()))
                counts["bank_vjps"] += len(bank)
                for rule in RULES:
                    name = "/".join((state, space, rule))
                    pairs, labels, method = fit_request(d, rule)
                    fitted = bank.direction(pairs, labels, rule=method)
                    cert = fitted.certificate()
                    axis = fitted.coordinates_f64le()
                    require(
                        fitted.left_f64le() == left and (axis is None) == (cert["refusal"] is not None),
                        "invalid direction",
                    )
                    coords = doubles(axis) if axis is not None else None
                    axes[name] = dict(certificate=cert, coordinates=coords)
                    if rule == "mean":
                        alt = bank.direction(d["Q"], labels, rule="equal_mean").coordinates_f64le()
                        require(
                            (axis is None and alt is None)
                            or (
                                axis is not None
                                and alt is not None
                                and math.dist(coords, doubles(alt)) < 1e-10
                            ),
                            "mean A and Q did not coincide",
                        )
                    if space != "left" or axis is None:
                        continue
                    active = fitted.scalar_adapter(model)
                    h = checkpoint(snaps, active)
                    record(journal, dict(name=name, event="start", checkpoint=h))
                    for j in range(1, CALLS + 1):
                        next_, r = active.step_pairs_backtracking(*args, **SGD)
                        stop = r["status"] != "accepted"
                        require(
                            r["gradient_prompt_evaluations"] == r["baseline"]["prompt_evaluations"] == 32
                            and all(t["objective"]["prompt_evaluations"] == 32 for t in r["trials"]),
                            "shared prompt replay count changed",
                        )
                        counts["optimizer_vjps"] += 32
                        counts["optimizer_forwards"] += 32 * (1 + len(r["trials"]))
                        counts["updates"] += not stop
                        nxt = checkpoint(snaps, next_)
                        require(not stop or nxt == h, "refusal changed weights or cursor")
                        record(journal, dict(name=name, event="step", call=j, checkpoint=nxt, report=r))
                        active = next_
                        h = nxt
                        if stop:
                            break
                        if j == 3:
                            active = model.restore_scalar_output_adapter(
                                read(snaps / (h + ".json"), h).decode(), h
                            )
                    ends[name] = h
                    summary[name] = dict(calls=j, steps=active.steps, coefficient=active.coefficient)
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            installation=installation,
            counts=counts,
            axes=put(out / "axes.json", axes),
            ends=ends,
            paths=summary,
            files={k: signature(out / k) for k in ("bank.jsonl", "gradients.f64le", "steps.jsonl")},
        ),
    )


def cases(axes):
    for state in STATES:
        yield state, "origin", "baseline", 0.0
        for space, rule in itertools.product(SPACES, RULES):
            name = "/".join((state, space, rule))
            if axes[name]["coordinates"] is None:
                continue
            for h, sg in itertools.product(H, (-1, 1)):
                yield state, name, "finite", sg * h
            if space == "left":
                yield state, name, "learned", None


def score(proto, p, pp, out, fit, tp):
    m = closed(proto, fit, pp, "train", None, tp)
    st, installation, models = stack(proto, p, False)
    axes = obj(fit / "axes.json", m["axes"], 2 << 20)
    qs = obj(proto / "queries.json", p["queries"])
    require(
        len(qs) == 64
        and all(set(r) == {"id", "prompt"} and r["id"] == task.identifier(r["prompt"]) for r in qs),
        "score deck has labels or wrong keys",
    )
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    cfg = read(proto / "model/config.json", p["files"]["config.json"]).decode()
    factors = obj(proto / "factors.json", p["factors"])
    counts = dict(ordinary=0, overlay=0, restored=0)
    with (
        (out / "cases.jsonl").open("xb") as journal,
        (out / "full.f32le").open("xb") as buf,
        (out / "overlay.bin").open("xb") as probe,
    ):
        for state, name, kind, scale in cases(axes):
            model = models[state]
            s = None
            if kind == "baseline":
                changed = raw
                dn = 0.0
                other = model
            else:
                coord = axes[name]["coordinates"]
                unit = coord if "/dense/" in name else [a * b for a in factors[state][0] for b in coord]
                if kind == "learned":
                    h = m["ends"][name]
                    s = model.restore_scalar_output_adapter(
                        read(fit / "checkpoints" / (h + ".json"), h).decode(), h
                    )
                    scale = s.coefficient
                delta = rank.f32s([0.0] * 32 + [scale * x for x in unit])
                changed, disp = rank.merge_bytes(raw, delta)
                dn = math.hypot(*disp)
                other = model if changed == raw else rank.rebind(st, cfg, changed, model)
            v = b"".join(rank.f32s(other.forward(r["prompt"], [1, 9], last_only=True).values()) for r in qs)
            buf.write(v)
            counts["ordinary"] += 64
            witness = None
            if kind == "finite":
                vv = b"".join(
                    struct.pack(
                        "<ffI",
                        *rank.triplet(
                            model.projection_perturbed_contrast(
                                r["prompt"],
                                task.IDLE,
                                task.ON,
                                layer=0,
                                site="output",
                                basis="dense",
                                bias_f32le=delta[:128],
                                weight_f32le=delta[128:],
                            )
                        ),
                    )
                    for r in qs
                )
                probe.write(vv)
                witness = sha(vv)
                counts["overlay"] += 64
            if s is not None:
                deployed = b"".join(s.last_logits_f32le(r["prompt"]) for r in qs)
                require(deployed == v, "trained scalar disagrees with ordinary reimport")
                witness = sha(deployed)
                counts["restored"] += 64
            record(
                journal,
                dict(
                    state=state,
                    name=name,
                    kind=kind,
                    scale=scale,
                    displacement=dn,
                    weight=sha(changed),
                    full=sha(v),
                    witness=witness,
                ),
            )
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            training=tp,
            installation=installation,
            counts=counts,
            files={k: signature(out / k) for k in ("cases.jsonl", "full.f32le", "overlay.bin")},
        ),
    )


def stage(phase, proto, pp, out, fit=None, tp=None, worker=False):
    p = obj(proto / "plan.json", pp)
    require((p["kind"], p["settings"], p["code"]) == (KIND, PLAN, sources()), "unfrozen study")
    claim = dict(phase=phase, plan=pp, parent=tp, attempt=1, output=os.path.relpath(out, proto))
    if worker:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (WALL, WALL))
        paths = [proto, out, Path(sys.prefix).resolve()] + ([fit] if fit else [])

        def fence(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = Path(os.path.realpath(os.fsdecode(args[0])))
                if (
                    ("validation" in path.parts and not any(path.is_relative_to(d) for d in paths))
                    or any(k in path.parts for k in ("keys", "blind"))
                    or (
                        path.is_relative_to(proto / "check")
                        or (phase == "train" and path in (proto / "queries.json", proto / "baseline.json"))
                        or (phase == "score" and path.is_relative_to(proto / "fit"))
                    )
                ):
                    raise PermissionError("not an admitted input for " + phase)

        sys.addaudithook(fence)
        require(
            obj(proto / (phase + "-attempt.json"), signature(out / "attempt.json")) == claim,
            "unclaimed stage",
        )
        return train(proto, p, pp, out) if phase == "train" else score(proto, p, pp, out, fit, tp)
    if phase == "score":
        from query_swap_readback import verify_fit

        verify_fit(proto, pp, fit, tp)
    require(not out.exists(), "occupied destination")
    put(proto / (phase + "-attempt.json"), claim)
    out.mkdir(mode=0o700)
    put(out / "attempt.json", claim)
    env = dict(
        os.environ,
        SPIRALTORCH_BACKEND="cpu",
        HF_HUB_OFFLINE="1",
        RAYON_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"),
    )
    args = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_" + phase, str(proto), pp, str(out)]
    if fit:
        args.extend([str(fit), tp])
    start = time.monotonic()
    with (out / "process.log").open("xb") as log:
        try:
            rc = subprocess.run(
                args, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=WALL
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    elapsed = time.monotonic() - start
    put(out / "exit.json", dict(exit=rc, elapsed=elapsed, log=signature(out / "process.log")))
    require(rc == 0 and elapsed <= WALL, "first failure retained")
    return signature(out / "complete.json")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 7:
        print(freeze(Path(a[0]), a[1], a[2], Path(a[3]), *a[4:6], Path(a[6])))
    elif cmd.lstrip("_") in ("train", "score") and len(a) == (3 if cmd.endswith("train") else 5):
        print(
            stage(
                cmd.lstrip("_"),
                Path(a[0]).resolve(),
                a[1],
                Path(a[2]).resolve(),
                *([Path(a[3]).resolve(), a[4]] if len(a) == 5 else []),
                worker=cmd.startswith("_"),
            )
        )
    elif cmd == "audit" and len(a) == 8:
        from query_swap_readback import audit

        print(audit(Path(a[0]), a[1], Path(a[2]), a[3], Path(a[4]), a[5], Path(a[6]), Path(a[7])))
    else:
        raise SystemExit(
            "freeze SOURCE SOURCE_PLAN SOURCE_MEASURE WHEEL WHEEL_PIN NATIVE_PIN OUT | train PROTO PIN OUT | "
            "score PROTO PIN OUT TRAINED TRAIN_PIN | audit PROTO PIN TRAINED TRAIN_PIN SCORED SCORE_PIN ARCHIVE OUT"
        )

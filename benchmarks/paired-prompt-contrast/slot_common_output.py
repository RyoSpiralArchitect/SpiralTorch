"""Native output-margin fit on four training edges, independent physical check on four others."""

import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scalar_full_batch as previous
from scalar_binding_study import STATES, base, rank, task
from projection_reach import signature
from shared_prefix_study import obj, put, read, record, require, sha, write

KIND = "spiraltorch.slot_common_output.v1"
H = (0.002, 0.02)
EPS = 5e-6
WALL = 180
W = 32
ARMS = tuple(s + "/" + r for s in ("left", "dense") for r in ("mean", "slot", "edge_slot")) + ("plane/raw",)
SETTINGS = dict(
    axes=list(ARMS),
    radii=list(H),
    wall=WALL,
    attempts=1,
    updates=0,
    construction="Only four even offset-one training edges. Native dense weights (zero bias) or "
    "previous unit left factor and unrestricted right. Equal-slot mean, normalized "
    "common ascent of 2 slot means or all 8 edge-by-slot means. Refusals get no substitute. "
    "Plane control maximizes raw minimum margin in the prior leading/rotated right plane.",
    test="Both geometry origins, fit and odd training edges, both signs and radii, no new eval. "
    "At .02, on odd edges BOTH slots' positive, central and predicted changes >5e-6, "
    "positive changes >5e-6 on at least 3/4 edges per slot. Both radii and folds: central "
    "within 1e-6 + 10% of predicted for both slots. Overlay/ordinary named logits and gaps "
    "within 8e-6 per prompt. Report both origins and every arm; small-radius resolution separate.",
    integrity="Native fit differences vs archived dense or left-contracted fit differences: "
    "L2 error <=3e-6+3e-5*reference L2; raw named contrasts within 2e-5. "
    "Independently certify every resolved unit hull direction. Ordinary baseline replays old wheel.",
)


def sources():
    return {
        **previous.sources(),
        **{
            s: signature(Path(__file__).with_name(s + ".py"))
            for s in ("slot_common_output", "slot_common_readback")
        },
    }


def paired(rows):
    out = []
    for j, e in enumerate(sorted({r["edge"] for r in rows})):
        for o, q in itertools.product((0, 1), repeat=2):
            cell = [r for r in rows if (r["edge"], r["order"], r["query"]) == (e, o, q)]
            (pos,) = [r for r in cell if r["gold"] == task.IDLE]
            (neg,) = [r for r in cell if r["gold"] == task.ON]
            require(
                [i for i, (a, b) in enumerate(zip(pos["prompt"], neg["prompt"], strict=True)) if a != b]
                == [2, 5],
                "not an assignment reversed pair",
            )
            out.append((pos, neg, q ^ o, 2 * j + (q ^ o)))
    require(len(out) == 16 and len({r["id"] for a, b, _, _ in out for r in (a, b)}) == 32, "wrong split")
    return out


def freeze(last, prior_pin, pullback, pull_pin, measure_pin, lora, exe, wheel, out):
    old = last / "protocol"
    p = obj(old / "plan.json", prior_pin)
    require(
        p["kind"] == previous.KIND
        and p["code"] == previous.sources()
        and p["settings"] == previous.settings(),
        "wrong scalar source",
    )
    q = obj(pullback / "protocol/plan.json", pull_pin)
    m = obj(pullback / "measured/complete.json", measure_pin)
    require(
        m["plan"] == pull_pin
        and q["base"] == p["base"]
        and q["lora"] == p["lora"]
        and all(q["roles"][s + "/origin"] == p["origins"][s] for s in STATES),
        "incompatible archived pullbacks",
    )
    inv = base.candidate_inventory(wheel, p["wheel"], p["native"])
    data = obj(old / "fit/rows.json", p["train"])
    require(data == task.deck("train"), "non-training cohort")
    bank = obj(old / "factors.json", p["bank"])
    out.mkdir(mode=0o700)
    packet = out / "model"
    packet.mkdir(mode=0o700)
    for f, h in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((v[7:] + ".json", v[7:]) for v in p["origins"].values()),
    ]:
        write(packet / f, read(old / "model" / f, h, 100000))
    with lora.open("rb") as stream:
        lo = stream.read(100001)
    require(
        len(lo) <= 100000 and "sha256:" + sha(b"spiraltorch.native_lora.v1\0" + lo) == p["lora"], "wrong LoRA"
    )
    write(packet / "lora.json", lo)
    folds = {f: [r for r in data if r["edge"] % 2 == i] for i, f in enumerate(("fit", "check"))}
    pairs = paired(folds["fit"])
    deck = dict(
        base=p["base"],
        lora=p["lora"],
        layer=0,
        target=task.IDLE,
        alternative=task.ON,
        positive=[a["prompt"] for a, b, s, j in pairs],
        negative=[b["prompt"] for a, b, s, j in pairs],
        slot=[s for a, b, s, j in pairs],
        edge_slot=[j for a, b, s, j in pairs],
        states={
            s: dict(
                geometry=p["origins"][s],
                left_bits=list(struct.unpack("<32Q", struct.pack("<32d", *bank[s][0]))),
            )
            for s in STATES
        },
    )
    put(packet / "deck.json", deck)
    baseline = write(out / "baseline.json", read(old / "original-training-logits.json", p["baseline"]))
    return put(
        out / "plan.json",
        dict(
            kind=KIND,
            settings=SETTINGS,
            code=sources(),
            parent=prior_pin,
            pullback=pull_pin,
            measurement=measure_pin,
            files=p["files"],
            origins=p["origins"],
            base=p["base"],
            lora=p["lora"],
            baseline=baseline,
            native=p["native"],
            wheel=p["wheel"],
            inventory=put(out / "wheel-inventory.json", inv),
            exe=signature(exe),
            packet={x.name: signature(x) for x in packet.iterdir()},
            factors=put(out / "factors.json", bank),
            folds={k: put(out / (k + ".json"), v) for k, v in folds.items()},
        ),
    )


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def avg(x):
    return math.fsum(x) / len(x)


def plane(gs, slots, v):
    rot = [x for j in range(0, W, 2) for x in (-v[j + 1], v[j])]
    vecs = [
        [avg([dot(g, r) for g, k in zip(gs, slots, strict=True) if k == s]) for r in (v, rot)] for s in (0, 1)
    ]
    g, h = vecs
    d = [x - y for x, y in zip(g, h, strict=True)]
    den = dot(d, d)
    a = max(0.0, min(1.0, -dot(h, d) / den)) if den else 0.0
    lo = [y + a * x for x, y in zip(d, h, strict=True)]
    norm = math.hypot(*lo)
    if norm <= 1e-12:
        return None
    return [(lo[0] * x + lo[1] * y) / norm for x, y in zip(v, rot, strict=True)]


def directions(proto, p, out, native_pin):
    factors = obj(proto / "factors.json", p["factors"])
    d = obj(proto / "model/deck.json", p["packet"]["deck.json"])
    lines = list(map(json.loads, read(out / "native.jsonl", native_pin, 4 << 20).splitlines()))
    axes = {s: {} for s in STATES}
    require(
        [(s, k) for s, k, r in lines] == [(s, k) for s in sorted(STATES) for k in ("dense", "left")],
        "incomplete native fit",
    )
    for state, space, r in lines:
        u, v = factors[state]
        for rule in ("mean", "slot", "edge_slot"):
            spec = r["directions"][rule]
            axis = spec["coordinates"]
            require(
                (
                    spec["base"],
                    spec["lora"],
                    spec["geometry"],
                    spec["layer"],
                    spec["width"],
                    spec["target_id"],
                    spec["alternative_id"],
                    spec["readout"],
                    spec["left"],
                )
                == (
                    p["base"],
                    p["lora"],
                    p["origins"][state],
                    0,
                    W,
                    task.IDLE,
                    task.ON,
                    "f32_wide_backward",
                    u if space == "left" else None,
                ),
                "direction belongs to another stack",
            )
            require(
                (axis is None) == (spec["refusal"] is not None)
                and (axis is None or abs(math.hypot(*axis) - 1.0) < 1e-9),
                "invalid native direction",
            )
            axes[state][space + "/" + rule] = (
                None if axis is None else ([x * y for x in u for y in axis] if space == "left" else axis)
            )
        if space == "left":
            # A two-coordinate control from the same fit gradients, not from check-fold scores.
            rv = plane(r["pair_gradients"], d["slot"], v)
            axes[state]["plane/raw"] = None if rv is None else [x * y for x in u for y in rv]
            require(
                dot(r["directions"]["mean"]["coordinates"], v) > 1.0 - 1e-8,
                "leading-axis control no longer replays",
            )
    return put(
        out / "axes.json",
        dict(plan=signature(proto / "plan.json"), native_fit=native_pin, fit=p["folds"]["fit"], weights=axes),
    )


def measure(proto, p, pp, out, axes_pin):
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (WALL, WALL))

    def fence(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.path.realpath(os.fsdecode(args[0])))
            if any(k in path.parts for k in ("keys", "blind", "scored")) or (
                "validation" in path.parts
                and not any(path.is_relative_to(x) for x in (proto, out, Path(sys.prefix).resolve()))
            ):
                raise PermissionError("not an admitted physical input")

    sys.addaudithook(fence)
    axes = obj(out / "axes.json", axes_pin, 2 << 20)
    require(
        axes["plan"] == pp
        and axes["fit"] == p["folds"]["fit"]
        and obj(proto / "attempt.json", signature(out / "attempt.json"))
        == dict(plan=pp, attempt=1, output=os.path.relpath(out, proto)),
        "unclaimed or unsealed run",
    )
    st, installed, models = previous.prior.stack(proto, p)
    folds = [obj(proto / (f + ".json"), p["folds"][f]) for f in ("fit", "check")]
    rows = folds[0] + folds[1]
    baseline = obj(proto / "baseline.json", p["baseline"])
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    cfg = read(proto / "model/config.json", p["files"]["config.json"]).decode()
    counts = dict(original=0, ordinary=0, overlay=0)
    with (
        (out / "rows.f32le").open("xb") as binary,
        (out / "overlay.bin").open("xb") as extra,
        (out / "cases.jsonl").open("xb") as log,
    ):
        for state in STATES:
            model = models[state]

            def observe(current):
                return b"".join(
                    rank.f32s(current.forward(r["prompt"], [1, 9], last_only=True).values()) for r in rows
                )

            b = observe(model)
            for r, v in zip(rows, struct.iter_unpack("<64f", b), strict=True):
                require(
                    [v[task.IDLE], v[task.ON], max(range(64), key=v.__getitem__)] == baseline[state][r["id"]],
                    "baseline no longer byte-replays",
                )
            binary.write(b)
            record(log, dict(state=state, arm="origin", values=sha(b)))
            counts["original"] += 64
            for name in ARMS:
                unit = axes["weights"][state][name]
                if unit is None:
                    continue
                require(len(unit) == W * W and abs(math.hypot(*unit) - 1) < 1e-9, "wrong physical norm")
                for h in H:
                    for side in (-1, 1):
                        delta = rank.f32s([0.0] * W + [side * h * x for x in unit])
                        changed, actual = rank.merge_bytes(raw, delta)
                        edited = rank.rebind(st, cfg, changed, model)
                        b = observe(edited)
                        v = bytearray()
                        for r in rows:
                            z = model.projection_perturbed_contrast(
                                r["prompt"],
                                task.IDLE,
                                task.ON,
                                layer=0,
                                site="output",
                                basis="dense",
                                bias_f32le=delta[: 4 * W],
                                weight_f32le=delta[4 * W :],
                            )
                            v.extend(struct.pack("<ffI", *rank.triplet(z)))
                        binary.write(b)
                        extra.write(v)
                        record(
                            log,
                            dict(
                                state=state,
                                arm=name,
                                radius=h,
                                side=side,
                                delta=sha(delta),
                                weight=sha(changed),
                                norm=math.hypot(*actual),
                                values=sha(b),
                                overlay=sha(v),
                            ),
                        )
                        counts["ordinary"] += 64
                        counts["overlay"] += 64
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            axes=axes_pin,
            native_fit=axes["native_fit"],
            installation=installed,
            counts=counts,
            files={k: signature(out / k) for k in ("rows.f32le", "overlay.bin", "cases.jsonl")},
        ),
    )


def run(proto, pp, exe, out):
    p = obj(proto / "plan.json", pp)
    require(
        p["kind"] == KIND
        and p["settings"] == SETTINGS
        and p["code"] == sources()
        and signature(exe) == p["exe"]
        and {f.name: signature(f) for f in (proto / "model").iterdir()} == p["packet"]
        and not out.exists(),
        "stale fit packet or executable",
    )
    claim = dict(plan=pp, attempt=1, output=os.path.relpath(out, proto))
    put(proto / "attempt.json", claim)
    out.mkdir(mode=0o700)
    put(out / "attempt.json", claim)
    start = time.monotonic()
    env = dict(
        os.environ,
        SPIRALTORCH_BACKEND="cpu",
        RAYON_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        VECLIB_MAXIMUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"),
    )

    def call(label, args):
        with (out / (label + ".log")).open("xb") as f:
            try:
                rc = subprocess.run(
                    args,
                    cwd=out,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.01, WALL - (time.monotonic() - start)),
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        elapsed = time.monotonic() - start
        put(
            out / (label + "-exit.json"),
            dict(exit=rc, cumulative_seconds=elapsed, log=signature(out / (label + ".log"))),
        )
        require(rc == 0 and elapsed <= WALL, "first failure retained in " + label)

    call("native", [str(exe), str(proto / "model"), str(out / "native.jsonl")])
    ax = directions(proto, p, out, signature(out / "native.jsonl"))
    call(
        "physical",
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_measure", str(proto), pp, str(out), ax],
    )
    return signature(out / "complete.json")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 9:
        print(freeze(Path(a[0]), a[1], Path(a[2]), *a[3:5], *(Path(x) for x in a[5:])))
    elif cmd == "run" and len(a) == 4:
        print(run(Path(a[0]).resolve(), a[1], Path(a[2]).resolve(), Path(a[3]).resolve()))
    elif cmd == "_measure" and len(a) == 4:
        root = Path(a[0]).resolve()
        p = obj(root / "plan.json", a[1])
        print(measure(root, p, a[1], Path(a[2]).resolve(), a[3]))
    elif cmd == "audit" and len(a) == 6:
        from slot_common_readback import audit

        print(audit(Path(a[0]), a[1], Path(a[2]), a[3], Path(a[4]), Path(a[5])))
    else:
        raise SystemExit(
            "freeze PRIOR PRIOR_PLAN ARCHIVE ARCHIVE_PLAN ARCHIVE_MEASURE LORA_FILE EXE WHEEL OUT | run PROTO PIN EXE OUT | audit PROTO PIN MEASURED PIN ARCHIVE OUT"
        )

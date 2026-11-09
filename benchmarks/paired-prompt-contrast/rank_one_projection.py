"""Physical first-block output intervention on the pinned entity-binding training fixture."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import projection_reach as pr
from shared_prefix_study import obj, put, read, record, require, sha, write

KIND = "spiraltorch.entity_rank_one_projection.v1"
LABEL = "0/output/dense"
ARMS = ("dense", "rank1", "orthogonal")
ROUTES = ("overlay", "merged")
W = pr.W
WALL = 180
F32 = struct.Struct("<f")
PLAN = dict(
    layer=0,
    site="output",
    basis="dense",
    arms=list(ARMS),
    routes=list(ROUTES),
    radii=list(pr.RADII),
    wall_seconds=WALL,
    retries=0,
    updates=0,
    calls={"baseline": 256, "overlay": 3072, "merged": 3072},
    construction="Even training edges only. Unit dense with bias; zero-bias top outer product; "
    "same left factor and right factor rotated (-v1,v0) in each channel pair.",
    comparison="At both radii on odd training edges, on each physical route: signed central "
    "rank1 change >=90% of positive dense change, >2x abs(control), >5e-6. "
    "At .02 positive step and both order means positive, >=3/4 edge means >5e-6. "
    "Fold central must match its own realized-delta VJP within 1e-6+10% on every arm. "
    "Named logits and gaps between routes within 8e-6 per row and central changes "
    "within 1e-6 per fold. Replay archived dense overlay and baseline bytes exactly.",
    scope="Two construction and two endpoint snapshots, not independent seeds. Prior training "
    "prompts only; already inspected, not a generalization test, training or speed result.",
)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def f32s(values):
    return struct.pack(f"<{len(values)}f", *values)


def vals(data):
    return [x for (x,) in F32.iter_unpack(data)]


def axes(dense, rec):
    require(len(dense) == W + W * W and abs(math.hypot(*dense) - 1) < 1e-9, "wrong dense axis")
    u, v = rec["left"], rec["right"]
    require(all(len(x) == W and abs(math.hypot(*x) - 1) < 1e-9 for x in (u, v)), "bad factors")
    rows = [dense[i : i + W] for i in range(W, len(dense), W)]
    av = [dot(row, v) for row in rows]
    atu = [dot(col, u) for col in zip(*rows, strict=True)]
    s = rec["fraction_dense_fit_norm"]
    # Certify this is the leading singular pair without selecting a factor on check edges.
    require(
        s > math.hypot(*dense[W:]) / math.sqrt(2)
        and max(math.hypot(*(x - s * y for x, y in zip(a, b, strict=True))) for a, b in ((av, u), (atu, v)))
        < 1e-9,
        "not the leading fit singular pair",
    )
    rot = [z for j in range(0, W, 2) for z in (-v[j + 1], v[j])]
    require(abs(dot(v, rot)) < 1e-12, "control not perpendicular")

    def outer(right):
        return [0.0] * W + [x * y for x in u for y in right]

    return dict(dense=dense, rank1=outer(v), orthogonal=outer(rot))


def merge_bytes(raw, delta):
    """Change only first-block c_proj in an admitted synthetic F32 safetensors copy."""
    require(len(delta) == 4 * (W + W * W) and 8 < len(raw) <= 100000, "wrong delta or archive length")
    (header_size,) = struct.unpack_from("<Q", raw)
    start = 8 + header_size
    require(8 < start < len(raw), "invalid header")
    header = json.loads(raw[8:start])
    spans = []
    for name, entry in header.items():
        a, b = entry["data_offsets"]
        require(
            entry["dtype"] == "F32"
            and 0 <= a < b <= len(raw) - start
            and b - a == 4 * math.prod(entry["shape"]),
            "invalid tensor range",
        )
        spans.append((a, b))
    ordered = sorted(spans)
    require(
        ordered[0][0] == 0
        and ordered[-1][1] == len(raw) - start
        and all(left[1] == right[0] for left, right in zip(ordered, ordered[1:])),
        "aliased or incomplete archive",
    )
    out = bytearray(raw)
    actual = []
    for suffix, shape, block in (("bias", [W], delta[: 4 * W]), ("weight", [W, W], delta[4 * W :])):
        t = header["h.0.attn.c_proj." + suffix]
        require(t["shape"] == shape, "unexpected c_proj layout")
        a, b = (start + i for i in t["data_offsets"])
        orig, increment = vals(raw[a:b]), vals(block)
        require(all(map(math.isfinite, orig + increment)), "nonfinite edit")
        if any(x != 0.0 for x in increment):
            out[a:b] = f32s([x + d if d != 0.0 else x for x, d in zip(orig, increment, strict=True)])
        actual.extend(y - x for x, y in zip(orig, vals(out[a:b]), strict=True))
    return bytes(out), actual


def source_hashes():
    here = Path(__file__)
    return pr.sources() | {
        n: pr.signature(here.with_name(n + ".py"))
        for n in ("rank_one_projection", "rank_one_projection_audit", "rank_one_roster")
    }


def freeze(parent, parent_pin, measurement_pin, roster_pin, dest):
    old = parent / "protocol"
    measured = parent / "measured"
    p = obj(old / "plan.json", parent_pin)
    pr.admit(old, p)
    m = obj(measured / "complete.json", measurement_pin)
    require(m["plan"] == parent_pin and m["counts"] == pr.MAX, "incomplete source measurement")
    d = obj(measured / "directions.json", m["directions"], 4 << 20)
    r = obj(parent / "rank-one-candidates.json", roster_pin)
    require(
        r["kind"] == "fit_only_rank_one_no_bias.v1"
        and r["model_calls"] == 0
        and (d["plan"], d["fit"]) == (parent_pin, p["folds"]["fit"]) == (r["plan"], r["fit"])
        and r["directions"] == m["directions"]
        and set(d["roles"]) == set(r["roles"]) == set(pr.ROLES),
        "wrong factor provenance",
    )
    dest.mkdir(mode=0o700)
    chosen = {role: axes(d["roles"][role][LABEL], r["roles"][role][LABEL]) for role in pr.ROLES}
    directions = put(dest / "axes.json", chosen)
    refs = {k: p[k] for k in ("base", "lora", "roles", "wheel", "native", "inventory", "files", "folds")}
    (dest / "model").mkdir(mode=0o700)
    for name, pin in [
        ("config.json", p["files"]["config.json"]),
        ("model.safetensors", p["files"]["weights"]),
        *((v[7:] + ".json", v[7:]) for v in p["roles"].values()),
    ]:
        write(dest / "model" / name, read(old / "model" / name, pin, 100000))
    write(dest / "wheel-inventory.json", read(old / "wheel-inventory.json", p["inventory"], 2 << 20))
    for parity, fold in enumerate(("fit", "check")):
        rows = obj(old / (fold + ".json"), p["folds"][fold])
        require(
            rows == [r for r in pr.task.deck("train") if r["edge"] % 2 == parity],
            "not original training data",
        )
        require(put(dest / (fold + ".json"), rows) == p["folds"][fold], "row encoding changed")
    return put(
        dest / "plan.json",
        refs
        | dict(
            kind=KIND,
            settings=PLAN,
            code=source_hashes(),
            parent=parent_pin,
            measurement=measurement_pin,
            roster=roster_pin,
            directions=m["directions"],
            axes=directions,
        ),
    )


def rebind(st, config, changed, original):
    """The store identity changes on merge. Only that identity may differ in last-block adapters."""
    other = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(changed))
    other = other.with_readout_accumulator("f32_wide_backward")
    old, new = original.base_model_identity, other.base_model_identity
    require(old != new, "no physical weight change")

    def rebound(raw):
        spec = json.loads(raw)
        require(spec["base_model_identity"] == old, "foreign adapter witness")
        return spec | {"base_model_identity": new}

    lo = rebound(original.lora_to_json())
    geom = rebound(original.geometry_artifact().to_json())
    require(
        lo["layer"] == 1 and [s["path"] for s in geom["placements"]] == ["transformer.h.1.attn"],
        "reference adapters overlap edited site",
    )
    other = other.with_lora_json(json.dumps(lo)).with_geometry(
        st.GeometryAdapterArtifact.create_json(json.dumps(geom))
    )
    require(
        json.loads(other.lora_to_json()) == lo and json.loads(other.geometry_artifact().to_json()) == geom,
        "last-block parameters changed on reload",
    )
    return other


def triplet(v):
    a, b, g = v["target_logit"], v["alternative_logit"], v["greedy_id"]
    require(
        (v["target_id"], v["alternative_id"], v["prompt_tokens"]) == (pr.task.IDLE, pr.task.ON, 9)
        and math.isfinite(a - b)
        and v["target_minus_alternative"] == a - b
        and 0 <= g < 64,
        "invalid native row",
    )
    return (a, b, g)


def retain_adapter(folder, model, kind):
    raw = (model.lora_to_json() if kind == "lora" else model.geometry_artifact().to_json()).encode()
    owner = model.lora_artifact_id if kind == "lora" else model.geometry_artifact_id
    prefix = b"spiraltorch.native_lora.v1\0" if kind == "lora" else b""
    require(owner == "sha256:" + sha(prefix + raw), "adapter serialization does not match its owner")
    path = folder / (owner[7:] + ".json")
    if path.exists():
        read(path, sha(raw), 100000)
    else:
        write(path, raw)
    return owner


def run_worker(proto, pp, out):
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (WALL, WALL))
    require(
        obj(proto / "attempt.json", pr.signature(out / "attempt.json"))
        == dict(plan=pp, output=os.path.relpath(out, proto), attempt=1),
        "unclaimed worker",
    )

    def fence(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.path.realpath(os.fsdecode(args[0])))
            if (
                path.name in {"truth.json", "queries.json", "answer-key.json"}
                or any(
                    x in path.parts for x in ("keys", "blind", "scored", "pullbacks.bin", "directions.json")
                )
                or (
                    "validation" in path.parts
                    and not path.is_relative_to(proto)
                    and not path.is_relative_to(out)
                    and not path.is_relative_to(Path(sys.prefix).resolve())
                )
            ):
                raise PermissionError("outside frozen measurement inputs")

    sys.addaudithook(fence)
    p = obj(proto / "plan.json", pp)
    require(
        p["kind"] == KIND and p["settings"] == PLAN and p["code"] == source_hashes(), "unfrozen procedure"
    )
    inv = obj(proto / "wheel-inventory.json", p["inventory"], 2 << 20)
    require((inv["sha256"], inv["native"]["sha256"]) == (p["wheel"], p["native"]), "foreign inventory")
    installation = pr.base.installed_identity(inv)
    st = pr.base.native()
    require(
        st.NativeCausalLm is st._rs.NativeCausalLm
        and Path(st._rs.__file__).resolve()
        == Path(st.__file__).resolve().parent.parent / inv["native"]["path"],
        "foreign decoder",
    )
    raw = read(proto / "model/model.safetensors", p["files"]["weights"], 100000)
    config = read(proto / "model/config.json", p["files"]["config.json"]).decode()
    original = pr.base.open_model(st, proto / "model", p["files"])
    require((original.base_model_identity, original.lora_artifact_id) == (p["base"], p["lora"]), "wrong base")
    axes_by_role = obj(proto / "axes.json", p["axes"], 1 << 20)
    rows = [r for fold in ("fit", "check") for r in obj(proto / (fold + ".json"), p["folds"][fold])]
    ids = [r["prompt"] for r in rows]
    counts = dict.fromkeys(PLAN["calls"], 0)
    pair = (pr.task.IDLE, pr.task.ON)
    (out / "weights").mkdir(mode=0o700)
    (out / "adapters").mkdir(mode=0o700)
    retain_adapter(out / "adapters", original, "lora")
    with (out / "observations.jsonl").open("xb") as log:
        for role in pr.ROLES:
            owner = p["roles"][role]
            model = pr.swap.attach(st, original, proto / "model" / (owner[7:] + ".json"), owner)
            baseline = [triplet(model.next_token_contrast(prompt, *pair)) for prompt in ids]
            record(log, dict(role=role, arm="baseline", values=baseline))
            counts["baseline"] += len(baseline)
            for arm in ARMS:
                unit = axes_by_role[role][arm]
                require(abs(math.hypot(*unit) - 1) < 1e-9, "wrong norm")
                for h in pr.RADII:
                    for sign in (-1, 1):
                        delta = f32s([sign * h * x for x in unit])
                        changed, actual = merge_bytes(raw, delta)
                        require(abs(math.hypot(*actual) / h - 1) < 1e-4, "physical norm drift")
                        pin = sha(changed)
                        archive = out / "weights" / (pin + ".safetensors")
                        if not archive.exists():
                            write(archive, changed)
                        edited = rebind(st, config, read(archive, pin, 100000), model)
                        args = dict(
                            layer=0,
                            site="output",
                            basis="dense",
                            bias_f32le=delta[: 4 * W],
                            weight_f32le=delta[4 * W :],
                        )
                        overlay = [
                            triplet(model.projection_perturbed_contrast(prompt, *pair, **args))
                            for prompt in ids
                        ]
                        merged = [triplet(edited.next_token_contrast(prompt, *pair)) for prompt in ids]
                        for route in ROUTES:
                            counts[route] += len(ids)
                        record(
                            log,
                            dict(
                                role=role,
                                arm=arm,
                                radius=h,
                                sign=sign,
                                delta=sha(delta),
                                weights=pin,
                                displacement=math.hypot(*actual),
                                owners=[
                                    edited.base_model_identity,
                                    retain_adapter(out / "adapters", edited, "lora"),
                                    retain_adapter(out / "adapters", edited, "geometry"),
                                ],
                                overlay=overlay,
                                merged=merged,
                            ),
                        )
            require(
                (model.base_model_identity, model.lora_artifact_id, model.geometry_artifact_id)
                == (p["base"], p["lora"], owner),
                "source state changed",
            )
    require(counts == PLAN["calls"], "incomplete measurement")
    return put(
        out / "complete.json",
        dict(
            plan=pp,
            installed=installation,
            counts=counts,
            observations=pr.signature(out / "observations.jsonl"),
        ),
    )


def once(proto, pp, out):
    p = obj(proto / "plan.json", pp)
    require(
        p["settings"] == PLAN and p["code"] == source_hashes() and not out.exists(), "stale plan or output"
    )
    claim = dict(plan=pp, output=os.path.relpath(out, proto), attempt=1)
    put(proto / "attempt.json", claim)
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
    start = time.monotonic()
    with (out / "process.log").open("xb") as f:
        try:
            rc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "_worker",
                    str(proto),
                    pp,
                    str(out),
                ],
                env=env,
                cwd=out,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=WALL,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    elapsed = time.monotonic() - start
    put(out / "exit.json", dict(exit=rc, elapsed=elapsed, log=pr.signature(out / "process.log")))
    require(rc == 0 and elapsed <= WALL, "first physical failure retained; no retry")
    return pr.signature(out / "complete.json")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 5:
        print(freeze(Path(a[0]), *a[1:4], Path(a[4])))
    elif cmd in ("run", "_worker") and len(a) == 3:
        print((once if cmd == "run" else run_worker)(Path(a[0]).resolve(), a[1], Path(a[2]).resolve()))
    elif cmd == "audit" and len(a) == 6:
        from rank_one_projection_audit import audit

        print(audit(Path(a[0]), Path(a[1]), a[2], Path(a[3]), a[4], Path(a[5])))
    else:
        raise SystemExit(
            "freeze PARENT PARENT_PLAN PARENT_MEASUREMENT ROSTER OUT | run PROTOCOL PIN OUT | "
            "audit PARENT PROTOCOL PIN MEASURED PIN OUT"
        )

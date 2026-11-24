"""Replay captured site rows with Rust geometry only; verify outputs without an ML runtime."""
import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import struct
import sys


HERE = Path(__file__).resolve().parent
ARCHIVE = "180c2c60b9dc5ee7c871fda336946af498324fec5199fa9235789025265a1201"
SUBDIR = {"rehearsal": "rehearsal-pinned-head", "gpt2": "gpt2"}
RADII = (1, 0.5, 2, 4, 8, 16)
HOSTS = ("affine", "both")
BLOCKS = ("gate", "bias")
RULE = {"min_vs_affine": 0.5, "min_vs_one": 2.0, "min_cosine": 0.99}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def unpack(s):
    return base64.b64decode(s, validate=True)


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def vector(raw, width):
    need(sys.byteorder == "little" and len(raw) == width * 4, "incorrect f32 buffer")
    v = memoryview(raw).cast("f")
    need(all(map(math.isfinite, v)), "nonfinite f32 buffer")
    return v


def inputs(root, kind):
    manifest = (root / "SHA256SUMS").read_bytes()
    need(sha(manifest) == ARCHIVE, "wrong source archive manifest")
    pins = {p: h for h, p in (s.split("  ", 1) for s in manifest.decode().splitlines())}
    src = {}
    for name in ("plan.json", "attempt/rows.jsonl", "attempt/result.json", "attempt/receipt.json"):
        rel = SUBDIR[kind] + "/" + name
        raw = (root / rel).read_bytes()
        need(sha(raw) == pins[rel], "changed source " + rel)
        src[name] = raw
    p = json.loads(src["plan.json"])
    result = json.loads(src["attempt/result.json"])
    receipt = json.loads(src["attempt/receipt.json"])
    want = 24 if kind == "gpt2" else 4
    need(receipt["success"] is True and receipt["retry"] is False and receipt["exit_code"] == 0
         and result["plan"] == sha(src["plan.json"])
         and result["raw_sha256"] == sha(src["attempt/rows.jsonl"]), "unsealed parent")
    need(p["schema"] == "spiraltorch.wave_margin_study.v1" and p["kind"] == kind
         and p["calls"] == result["calls"] == 2 * want and result["maps"] == 8 * want
         and p["config"] == dict(curvature=-1., saturation=1., porosity=.05, log_radius=0.)
         and p["strength"] == .25, "wrong scope")
    cases = {c["key"]: c for c in p["train"]}
    schedule = list(map(tuple, p["schedule"]))
    rows = [json.loads(s) for s in src["attempt/rows.jsonl"].splitlines()]
    need(len(cases) == len(p["train"]) == want and len(schedule) == p["calls"]
         and set(schedule) == {(k, h) for k in cases for h in HOSTS}
         and schedule == [(r["key"], r["host"]) for r in rows], "unpaired source schedule")
    width = p["shape"][1]
    shared = {}
    for (key, host), row in zip(schedule, rows):
        r, c = row["receipt"], cases[key]
        need(r["schema"] == "spiraltorch.wave_margin_parameter_pullbacks.v1"
             and r["objective"] == "target_logit_minus_alternative_logit"
             and r["encoding"] == "little_endian_f32" and r["readout_accumulator"] == "f32_wide_backward"
             and r["base_model_identity"] == p["base"] and r["lora_artifact_id"] is None
             and r["site"] == p["site"] and r["strength"] == p["strength"] and r["factual"] == host
             and r["geometry_artifact_id"] == p["geometry"][host]["pin"]
             and {k: f32(v) for k,v in p["config"].items()} == r["config"]
             and [r["contrast"][k] for k in ("target_id", "alternative_id", "prompt_tokens")]
             == [c["gold"], c["rival"], len(c["prefix"])], "wrong source row")
        need(shared.setdefault(key, r["site_input"]) == r["site_input"], "unmatched site inputs")
        for val in (r["site_input"], r["upstream"]):
            vector(unpack(val), width)
        for arm in ("affine", "both"):
            need(set(r["arms"][arm]) == set(BLOCKS), "incomplete source gradient")
            for val in r["arms"][arm].values():
                vector(unpack(val), width)
    return p, rows, {k: sha(v) for k, v in src.items()}


def spec(root, kind):
    p, rows, pins = inputs(root, kind)
    return dict(schema="spiraltorch.captured_wave_radius.v1", kind=kind, source_manifest=ARCHIVE,
                inputs=pins, sources={n: sha((HERE / n).read_bytes()) for n in
                ("sweep.py", "README.md", "../bounded.py")}, extension=p["extension"],
                parent_adapter=p["geometry"]["both"]["pin"], radii=RADII, rule=RULE,
                min_per_host=(3 * len(rows) + 7) // 8, rows=len(rows), seconds=90,
                model_calls=0, examples_scored=0, updates=0)


def tensor_at(record, name):
    site, = record["placements"]
    tensor, = (t for t in site["tensors"] if t["name"] == name)
    return tensor


def log_buffer(record):
    t = tensor_at(record, "log_radius")
    need(t["shape"] == [] and t["role"] == "buffer" and t["requires_grad"] is False
         and len(t["values"]) == 1, "radius must be a fixed scalar")
    return t


def radius_copy(record, radius):
    need(math.isfinite(radius) and radius > 0 and log_buffer(record)["values"] == [0.0],
         "invalid radius or non-unit parent")
    out = copy.deepcopy(record)
    log_buffer(out)["values"] = [f32(math.log(radius))]
    return out


def check_variant(record, variant, radius):
    normalized = copy.deepcopy(variant)
    want = f32(math.log(radius))
    need(f32(log_buffer(normalized)["values"][0]) == want, "wrong saved log radius")
    log_buffer(normalized)["values"] = [0.0]
    need(normalized == record, "something other than the fixed radius changed")


def raw_record(p, host):
    raw = unpack(p["geometry"][host]["bytes"])
    need("sha256:" + sha(raw) == p["geometry"][host]["pin"], "wrong base adapter pin")
    return raw


def check_parent(p):
    rec = json.loads(raw_record(p, "both"))
    affine = json.loads(raw_record(p, "affine"))
    site, = rec["placements"]
    recipe = site["recipe"]
    need(site["path"] == p["site"] and recipe["schema"] == "spiraltorch.wave_gate_adapter.v2"
         and recipe["learnable_radius"] is False and recipe["features"] == p["shape"][1]
         and recipe["strength"] == p["strength"] and log_buffer(rec)["values"] == [0.0]
         and all(recipe["kernel"][k] == p["config"][k] for k in ("curvature","saturation","porosity"))
         and all(tensor_at(rec, k) == tensor_at(affine, k) for k in BLOCKS), "unmatched base adapters")
    return rec


def norm(x):
    return math.sqrt(math.fsum(a*a for a in x))


def ratio(a, b):
    return a / b if b else None


def direction(x, y):
    return ratio(math.fsum(a*b for a,b in zip(x,y)), norm(x)*norm(y))


def length_diff(x, y):
    return norm(a-b for a,b in zip(x,y))


def passes(m):
    return all(m[k] is not None and m[k] >= v for k, v in RULE.items())


def compare(row, base, obs, width):
    r = row["receipt"]
    current = [unpack(obs[k]) for k in BLOCKS]
    before = [unpack(base[k]) for k in BLOCKS]
    affine = [unpack(r["arms"]["affine"][k]) for k in BLOCKS]
    def metrics(now, one, control, n):
        a,b,c = (vector(raw, n) for raw in (now, one, control))
        return dict(min_vs_affine=ratio(norm(a),norm(c)), min_vs_one=ratio(norm(a),norm(b)),
                    min_cosine=direction(a,b))
    parts = {k: metrics(a,b,c,width) for k,a,b,c in zip(BLOCKS,current,before,affine)}
    m = metrics(b"".join(current),b"".join(before),b"".join(affine),2*width)
    x, y1, yr = (vector(unpack(a), width) for a in (r["site_input"],base["y"],obs["y"]))
    m.update(passed=passes(m), blocks=parts, forward_shift=ratio(length_diff(yr,y1),length_diff(y1,x)),
             forward_shift_vs_input=ratio(length_diff(yr,y1),norm(x)))
    return m


def stats(values):
    valid = [x for x in values if x is not None]
    return dict(defined=len(valid), median=median(valid) if valid else None, low=min(valid) if valid else None)


def aggregate(by_host, min_rows):
    report, selected = {}, None
    for radius in sorted(RADII):
        group = {}
        for host in HOSTS:
            rows = by_host[(radius, host)]
            group[host] = {k: stats([m[k] for m in rows]) for k in (*RULE, "forward_shift", "forward_shift_vs_input")}
            group[host].update(rows=len(rows), passes=sum(m["passed"] for m in rows),
                               blocks={b:{k:stats([m["blocks"][b][k] for m in rows]) for k in RULE} for b in BLOCKS})
        report[str(radius)] = group
        if selected is None and radius >= 1 and all(group[h]["passes"] >= min_rows for h in HOSTS):
            selected = radius
    return report, selected


def verify_artifacts(p, out):
    rec = check_parent(p)
    pins = {}
    for radius in RADII:
        raw = (out / ("radius-%s.json" % radius)).read_bytes()
        check_variant(rec, json.loads(raw), radius)
        pins[radius] = "sha256:" + sha(raw)
    need(len(set(pins.values())) == len(RADII) and pins[1] == p["geometry"]["both"]["pin"], "wrong artifact lineage")
    return pins


def assert_anchor(obs, row):
    for k in BLOCKS:
        need(obs[k] == row["receipt"]["arms"]["both"][k], "R=1 mismatch: %s/%s/%s" % (row["key"], row["host"], k))


def replay(p, rows, out, frozen):
    pins = verify_artifacts(p, out)
    by_host = {(r,h): [] for r in RADII for h in HOSTS}
    raw = (out / "local.jsonl").read_bytes()
    lines = raw.splitlines()
    need(len(lines) == len(RADII)*len(rows), "missing or extra local rows")
    base = {}
    for i, line in enumerate(lines):
        obs = json.loads(line)
        radius, row = RADII[i // len(rows)], rows[i % len(rows)]
        key = (row["key"], row["host"])
        need((obs["key"], obs["host"]) == key and obs["radius"] == radius and obs["artifact"] == pins[radius],
             "wrong local row order or artifact")
        if radius == 1:
            assert_anchor(obs, row)
            base[key] = obs
        by_host[radius, key[1]].append(compare(row, base[key], obs, p["shape"][1]))
    table, selected = aggregate(by_host, frozen["min_per_host"])
    return dict(protocol=sha(encode(frozen)), local_sha256=sha(raw), baseline_exact=len(base),
                local_calls=len(lines), model_calls=0, updates=0, examples_scored=0, results=table,
                selected=selected, selected_artifact=pins.get(selected), parent_artifact=pins[1])


def measure(p, rows, out, frozen):
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","RAYON_NUM_THREADS",
              "MKL_NUM_THREADS","BLIS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.update(SPIRALTORCH_BACKEND="cpu", SPIRALTORCH_AUTOTUNE="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("numpy","torch","safetensors","transformers"):
        need(name not in sys.modules, "unexpected ML runtime: " + name)
        sys.modules[name] = None
    from spiraltorch import GeometryAdapterArtifact, NativeGeometryRuntime, _rs
    need(sha(Path(_rs.__file__).read_bytes()) == frozen["extension"], "wrong loaded CPU extension")
    rec = check_parent(p)
    parent = GeometryAdapterArtifact.from_json(raw_record(p, "both").decode(), frozen["parent_adapter"], p["base"])
    with (out / "local.jsonl").open("xb", buffering=0) as f:
        for radius in RADII:
            art = parent if radius == 1 else GeometryAdapterArtifact.create_json(json.dumps(radius_copy(rec, radius)))
            payload = art.to_json().encode()
            check_variant(rec, json.loads(payload), radius)
            need("sha256:" + sha(payload) == art.artifact_id, "native artifact content differs from ID")
            write(out / ("radius-%s.json" % radius), payload)
            rt = NativeGeometryRuntime(art)
            need(rt.execution_backend == "rust_f32_cpu", "not native CPU geometry")
            for row in rows:
                r, width = row["receipt"], p["shape"][1]
                batch = rt.forward_buffer(p["site"], vector(unpack(r["site_input"]), width), 1)
                need(batch.shape == (1,width) and batch.source_artifact_id == art.artifact_id, "wrong forward batch")
                vjp = batch.vjp_parameters_buffer(vector(unpack(r["upstream"]), width))
                need(set(vjp) == set(BLOCKS), "not a fixed-radius gate/bias VJP")
                buffers = dict(vjp, y=batch.output_buffer())
                for raw in buffers.values():
                    vector(raw, width)
                obs = dict(key=row["key"], host=row["host"], radius=radius, artifact=art.artifact_id,
                           **{k:b64(v) for k,v in buffers.items()})
                f.write(encode(obs))
                if radius == 1:
                    assert_anchor(obs, row)
            need(rt.artifact().artifact_id == art.artifact_id == "sha256:" + sha(payload), "runtime mutated")
    result = replay(p, rows, out, frozen)
    write(out / "summary.json", encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("archive", type=Path)
    freeze.add_argument("kind", choices=SUBDIR)
    freeze.add_argument("protocol", type=Path)
    for name in ("measure", "verify"):
        s = sub.add_parser(name)
        for k in ("archive", "protocol", "out"):
            s.add_argument(k, type=Path)
    a = parser.parse_args()
    if a.cmd == "freeze":
        raw = encode(spec(a.archive, a.kind))
        write(a.protocol, raw)
        print(sha(raw))
        return
    raw = a.protocol.read_bytes()
    frozen = json.loads(raw)
    need(raw == encode(spec(a.archive, frozen["kind"])), "changed frozen protocol")
    p, rows, _ = inputs(a.archive, frozen["kind"])
    if a.cmd == "measure":
        result = measure(p, rows, a.out, frozen)
    else:
        receipt = json.loads((a.out / "receipt.json").read_bytes())
        need(receipt["success"] is True and receipt["retry"] is False and receipt["exit_code"] == 0
             and receipt["seconds_limit"] == frozen["seconds"] and receipt["elapsed"] <= frozen["seconds"],
             "unsuccessful bounded attempt")
        result = replay(p, rows, a.out, frozen)
        need((a.out / "summary.json").read_bytes() == encode(result), "offline summary differs")
    print(json.dumps({k:result[k] for k in ("baseline_exact","local_calls","model_calls","selected","local_sha256")}))


if __name__ == "__main__":
    main()

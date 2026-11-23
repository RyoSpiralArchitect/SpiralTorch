"""Four immutable offsets on a generated Rust decoder, or replay its logits without a model."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import struct
import sys

FIXTURES = Path(__file__).resolve().parents[5]/"bindings/st-wasm/tests"
SITE, H, T = "transformer.h.0.attn", 1/256, 1/65536
OFFSETS = (0., -H, H, -2*H, 2*H)
PROMPTS = [[1, 2], [2, 4], [2, 1, 3]]
BASE = "sha256:9f9c4fa145a8df64e56d864d65484617dacdc653ae8f0383d9271e0207146df5"
LORA = "sha256:2d5e02f430376e9476bc4bddbaaaec7db5ac72b1d568f373132e867136976337"
GEOMETRY = "sha256:44046c11d0e6e8f91640bd7fe6db923960ea3690a2979cba655e3676fd0c0d31"
EXTENSION = "ca72f2436471f8d754d02057bccb5a13b89383d6ccd64f9c76e88c9cc70f0d5f"
PREVIOUS = "682daccccc95e7dc01c5f02776210fea52958fd3f58ddf6ac9e3909eee677a49"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bias(record):
    placement, = record["placements"]
    need(placement["path"] == SITE, "wrong site")
    tensor, = [t for t in placement["tensors"] if t["name"] == "bias" and t["requires_grad"]]
    return tensor["values"]


def kl(reference, candidate):
    need(len(reference) == len(candidate) > 0 and all(map(math.isfinite, (*reference, *candidate))),
         "invalid logit row")
    z = [math.exp(x-max(reference)) for x in reference]
    total = math.fsum(z)
    p = [x/total for x in z]
    delta = [y-x for x, y in zip(reference, candidate)]
    mean = math.fsum(q*x for q, x in zip(p, delta))
    centered = [x-mean for x in delta]
    # log E_p exp(d) - E_p d, shifting d to avoid cancellation near identical rows.
    value = math.log1p(math.fsum(q*math.expm1(x) for q, x in zip(p, centered)))
    value -= math.fsum(q*x for q, x in zip(p, centered))
    need(math.isfinite(value) and value >= 0., "invalid KL")
    return value


def historical(path):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == PREVIOUS, "wrong original sensitivity journal")
    attempt, = [r["attempt"] for r in map(json.loads, data.splitlines())
                if r["case"] == "0/geometry/coordinate/envelope" and r["cursor"] == 0]
    axis, = [a for a in attempt["refusal"]["sensitivity"]["axes"]
             if (a["path"], a["tensor"], a["coordinate"]) == (SITE, "bias", 0)]
    return axis["probe_per_anchor_kl"]


def journal(old, out):
    historical(old)
    need(not out.exists(), "output already exists")
    fixture = runpy.run_path(str(FIXTURES/"native_line_owners_fixture.py"))
    import spiraltorch.spiraltorch as extension
    need(sha256(Path(extension.__file__).read_bytes()).hexdigest() == EXTENSION
         and fixture["st"].NativeCausalLm is extension.NativeCausalLm, "wrong native extension")
    origin, _ = fixture["host"]("ordinary/wave")
    geometry = origin.geometry_artifact().to_json()
    lora = origin.lora_to_json()
    need((origin.base_model_identity, origin.lora_artifact_id, origin.geometry_artifact_id)
         == (BASE, LORA, GEOMETRY) and sha256(lora.encode()).hexdigest() == LORA[7:]
         and sha256(geometry.encode()).hexdigest() == GEOMETRY[7:], "wrong composed origin")
    record = json.loads(geometry)
    before = f32(bias(record)[0])
    with out.open("x") as log:
        for offset in OFFSETS:
            model = origin
            if offset:
                edited = json.loads(geometry)
                bias(edited)[0] = f32(before+offset)
                artifact = fixture["st"].GeometryAdapterArtifact.create_json(json.dumps(edited, allow_nan=False))
                model = origin.with_geometry(artifact)
            canonical = model.geometry_artifact().to_json()
            check = json.loads(canonical)
            value = f32(bias(check)[0])
            bias(check)[0] = bias(record)[0]
            need(check == record and value-before == offset and model.base_model_identity == BASE
                 and model.lora_to_json() == lora and model.lora_artifact_id == LORA
                 and model.geometry_artifact_id == "sha256:"+sha256(canonical.encode()).hexdigest(),
                 "not the declared single-coordinate variant")
            logits = [model.forward(ids, [1, len(ids)], last_only=True).values() for ids in PROMPTS]
            row = dict(offset=offset, value=value, prompts=PROMPTS, logits=logits, base=BASE, lora=LORA,
                       geometry=model.geometry_artifact_id, extension=EXTENSION)
            log.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
            log.flush()
            need(origin.geometry_artifact().to_json() == geometry and origin.lora_to_json() == lora,
                 "origin mutated")


def summary(old, path):
    reference_kl = historical(old)
    blob = path.read_bytes()
    records = list(map(json.loads, blob.splitlines()))
    need(tuple(r["offset"] for r in records) == OFFSETS and records[0]["geometry"] == GEOMETRY
         and len({r["geometry"] for r in records}) == len(OFFSETS), "wrong variant set")
    for row in records:
        need((row["base"], row["lora"], row["extension"]) == (BASE, LORA, EXTENSION)
             and row["prompts"] == PROMPTS and row["value"]-records[0]["value"] == row["offset"]
             and len(row["logits"]) == len(PROMPTS) and all(len(z) == 11 for z in row["logits"]),
             "invalid readout")
    variants, totals, matches = [], {}, []
    for r in records[1:]:
        values = [kl(a, b) for a, b in zip(records[0]["logits"], r["logits"])]
        total = math.fsum(values)
        totals[r["offset"]] = total
        if abs(r["offset"]) == H and all(math.isclose(x, y, abs_tol=3e-15, rel_tol=1e-10)
                                         for x, y in zip(values, reference_kl)):
            matches.append(r["offset"])
        share = values[1]/total if total else None
        variants.append(dict(offset=r["offset"], kl_over_T=[x/T for x in values], middle_share=share,
                             concentrated=share is not None and share >= .85 and values[1] > max(values[::2])))
    need(len(matches) == 1, "neither or both signs reproduce the one-sided reference")
    return dict(schema="spiraltorch.synthetic_signed_bias_response.v1", variants=variants,
                two_scale_total_ratio={str(sign): totals[sign*2*H]/totals[sign*H] if totals[sign*H] else None
                                       for sign in (-1, 1)},
                persistent=all(r["concentrated"] for r in variants), historical_offset=matches[0],
                pretrained_calls=0, training_updates=0, synthetic_prompt_readouts=15,
                journal_sha256=sha256(blob).hexdigest(), previous_sha256=PREVIOUS)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "readback"))
    parser.add_argument("old", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "measure":
        journal(args.old, args.output)
    print(json.dumps(summary(args.old, args.output), indent=2, sort_keys=True, allow_nan=False))

"""Train-only local metric intervention on archived adjoints. No decoder or weight update."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys


HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE.parent / "training_anchors.py"))
M = runpy.run_path(str(HERE / "metric.py"))
S, E = A["S"], A["E"]
need, enc, sha = E["need"], E["enc"], E["sha"]
unpack, b64 = S["unpack"], S["b64"]
OBJECTIVES = ("full", "rival")
STAGES = ("affine", "saturation", "projection", "both")
PINS = {"gpt2": "18e3c788fa7275ebf57d10e4b11c81d1be4342280c693599a5890c7d1d198905",
        "rehearsal": "4e45bd042ab4c5c037375251c8199b7fbf9ac19b0d9c939f560394487098e06e"}


def parents(capture, matched, anchors, kind):
    raw = anchors.read_bytes()
    need(sha(raw) == PINS[kind], "wrong prior same-state anchor report")
    old = json.loads(raw)
    p, rows, pins = S["inputs"](capture, kind)
    need("evaluation" not in p and old["kind"] == kind and old["capture_inputs"] == pins
         and old["source_training_plan"] == p["source_plan"] and old["training_manifest"] == E["ROOTS"][2]
         and old["capture_manifest"] == S["ARCHIVE"], "incompatible train-only sources")
    S["check_parent"](p)
    need(p["site"] == "transformer.h.%d.attn" % (p["shape"][0]-1), "not the final attention output")
    read, opened = A["training_reader"](matched, kind, p["source_plan"])
    seeds = (17,31,73) if kind == "gpt2" else (11,)
    width = p["shape"][1]
    records = {(r["seed"],r["host"]):r for r in old["rows"]}
    receipts = {(r["key"],r["host"]):r["receipt"] for r in rows}
    selected = []
    for seed in seeds:
        case = min(p["train"], key=lambda c:sha(("relative/%s/%s"%(seed,c["key"])).encode()))
        key = case["key"]
        factual = {}
        for host, family in A["HOSTS"]:
            r = records[seed, host]
            need(r["key"] == key and r["origin"] == p["geometry"][host]["pin"], "wrong initial snapshot or order")
            raw_pair = []
            for obj in OBJECTIVES:
                stage = "%s-%s_%s" % (seed,family,obj)
                journal, tape = read(stage+".jsonl").splitlines(), read(stage+".g32")
                need(len(journal) == len(p["train"]) and len(tape) == 8*width*len(journal), "wrong training tape length")
                first, g = json.loads(journal[0]), tape[:8*width]
                A["check_join"](case, first, obj, r["origin"])
                need(sha(g) == first["grad_sha256"] == r["gradient_hashes"][obj], "stale first-step gradient")
                raw_pair.append(g)
            factual[host] = raw_pair
        receipt = receipts[key,"affine"]
        x = unpack(receipt["site_input"])
        incoming = [M["recover"](x, g, p["strength"]) for g in factual["affine"]]
        anchor = p["anchors"]["affine"][key]
        need(unpack(anchor["gradient"]) == factual["affine"][1], "unpaired independent choice gradient")
        q = anchor["q"]
        error = M["relative"](M["values"](incoming[1]), [-q*v for v in M["values"](unpack(receipt["upstream"]))])
        need(error <= p["anchor_tolerance"], "independent incoming margin does not close")
        selected.append(dict(key=key, seed=seed, x=x, incoming=incoming, factual=factual,
                             receipt=receipt, q=q, margin_error=error))
    need(len({r["key"] for r in selected}) == len(seeds)
         and all(old["training_inputs"][k] == v for k,v in opened.items()), "wrong archive member or duplicate case")
    return p, selected, dict(anchors=PINS[kind], training_members=opened, capture_members=pins)


def name(stage, radius):
    return stage if stage in ("affine","saturation") else "%s_r%d" % (stage,radius)


def spec(kind, extension, sources):
    need(re.fullmatch("[0-9a-f]{64}", extension), "expected installed extension SHA-256")
    return dict(schema="spiraltorch.archived_local_objective_metric.v1", kind=kind, extension=extension,
                radii=[1,4], objective_order=list(OBJECTIVES), input_sources=sources,
                analysis={n:sha((HERE/n).read_bytes()) for n in
                          ("metric.py","study.py","PROTOCOL.md","../training_anchors.py","../verify_training_anchors.py")},
                new_decoder_calls=0, weight_updates=0, scored_predictions=0)


def measure(p, rows, extension, out):
    os.environ.update(E["OLD"]["ENV"], SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    for k in ("numpy","torch","transformers","safetensors"):
        need(k not in sys.modules, "unexpected ML runtime")
        sys.modules[k] = None
    from spiraltorch import GeometryAdapterArtifact, NativeGeometryRuntime, _rs
    need(E["P"]["file_hash"](Path(_rs.__file__)) == extension, "wrong native extension")
    raw = S["raw_record"](p, "affine")
    art = GeometryAdapterArtifact.from_json(raw.decode(), p["geometry"]["affine"]["pin"], p["base"])
    rt = NativeGeometryRuntime(art)
    need(rt.execution_backend == "rust_f32_cpu", "non-CPU geometry")
    with (out/"rows.jsonl").open("xb", buffering=0) as sink:
        for row in rows:
            batch = rt.forward_buffer(p["site"], memoryview(row["x"]).cast("f"), 1)
            need(batch.shape == (1,p["shape"][1]) and batch.source_artifact_id == art.artifact_id, "wrong batch")
            dirs = [memoryview(d).cast("f") for d in row["incoming"]]
            maps = {}
            for radius in (1,4):
                cfg = dict(p["config"], log_radius=S["f32"](math.log(radius)))
                pair = batch.wave_parameter_pullback_pair_buffer(*dirs, config=cfg)
                need(len(pair) == 2 and all(set(arm)==set(STAGES) for arm in pair), "incomplete local pair")
                for stage in STAGES:
                    vs = [v[stage] for v in pair]
                    need(all(set(v)=={"gate","bias"} for v in vs), "unexpected channel coordinates")
                    flat = [bytes(v["gate"]+v["bias"]) for v in vs]
                    if stage == "affine":
                        need(flat == row["factual"]["affine"], "affine gate/bias do not reproduce the training tapes")
                    k = name(stage, radius)
                    encoded = [b64(v) for v in flat]
                    need(maps.setdefault(k, encoded) == encoded, "radius changed a radius-independent control")
            sink.write(enc(dict(key=row["key"], seed=row["seed"], site_input_sha=sha(row["x"]),
                                incoming=[b64(v) for v in row["incoming"]], maps=maps)))
    need(rt.artifact().artifact_id == art.artifact_id, "source artifact mutated")


def replay(p, rows, out, protocol):
    raw = (out/"rows.jsonl").read_bytes()
    saved = [json.loads(line) for line in raw.splitlines()]
    need(len(rows) == len(saved), "incomplete local comparison")
    width = p["shape"][1]
    result = []
    for row, obs in zip(rows, saved):
        need((row["key"],row["seed"],sha(row["x"])) == (obs["key"],obs["seed"],obs["site_input_sha"])
             and [unpack(v) for v in obs["incoming"]] == row["incoming"], "changed row or direction")
        maps = {k:[unpack(v) for v in pair] for k,pair in obs["maps"].items()}
        need(set(maps)=={name(s,r) for s in STAGES for r in (1,4)} and maps["affine"] == row["factual"]["affine"],
             "incomplete maps or wrong factual affine")
        maps["factual_wave_r1"] = row["factual"]["both"]
        need(all(len(pair) == 2 and all(len(v)==8*width for v in pair) for pair in maps.values()), "wrong gate/bias width")
        metrics = {k:M["alignment"](*pair) for k,pair in maps.items()}
        # Different backward seeds: diagnostic relative error, not an exact identity.
        errs = {}
        for stage in STAGES:
            unit = b"".join(unpack(row["receipt"]["arms"][stage][block]) for block in S["BLOCKS"])
            errs[stage] = M["relative"](M["values"](maps[name(stage,1)][1]), [-row["q"]*v for v in M["values"](unit)])
        result.append(dict(key=row["key"],seed=row["seed"],arms=metrics,
                           incoming_margin_relative_error=row["margin_error"],parameter_margin_relative_error=errs))
    return dict(protocol=sha(enc(protocol)), raw_sha256=sha(raw), rows=result, independent_prefixes=len(rows),
                paired_kernel_calls=2*len(rows), parameter_vjps=16*len(rows), new_decoder_calls=0, updates=0, scores=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("measure","verify"))
    ap.add_argument("kind",choices=tuple(PINS))
    for k in ("capture","matched","anchors","out"):
        ap.add_argument(k,type=Path)
    ap.add_argument("extension")
    args = ap.parse_args()
    p, rows, sources = parents(args.capture,args.matched,args.anchors,args.kind)
    protocol = spec(args.kind,args.extension,sources)
    out = args.out
    if args.mode=="measure":
        out.mkdir(mode=0o700,exist_ok=False)
        S["write"](out/"protocol.json",enc(protocol))
        measure(p,rows,args.extension,out)
        S["write"](out/"result.json",enc(replay(p,rows,out,protocol)))
    else:
        need((out/"protocol.json").read_bytes()==enc(protocol) and
             (out/"result.json").read_bytes()==enc(replay(p,rows,out,protocol)), "offline replay differs")
    print("%s: %d training anchors, %d local VJPs, zero decoder calls" % (args.mode,len(rows),16*len(rows)))


if __name__=="__main__":
    main()

"""Paired CPU wheel benchmark of finite-checked steps on an artificial WaveGate decoder."""
import argparse
from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import statistics
import struct
import subprocess
import sys
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
OLD = "ac7120c27d914423c4496fefb2d2adf0110ea0d904f7e0f5692a9842c06a32e7"
LAYERS, WIDTH, VOCAB, STEPS = 12, 64, 512, 4


def require(ok, why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def fixture(st):
    helper = HERE.parents[4]/"bindings/st-py/tests/test_native_causal_lm.py"
    config, raw = runpy.run_path(str(helper))["fixture"](width=WIDTH, heads=4, positions=32, layers=LAYERS,
                          vocabulary=VOCAB, constant_logits=[0.]*VOCAB, raw_weights=True)
    length, = struct.unpack_from("<Q", raw)
    start = 8+length
    header = json.loads(raw[8:start])
    data = bytearray(raw)
    rng = random.Random(20261123)
    require(sys.byteorder=="little" and array("f").itemsize==4, "requires little-endian binary32")
    for name, spec in sorted(header.items()):
        a,b = spec["data_offsets"]
        count = (b-a)//4
        if name.endswith(".bias"):
            v = array("f",[0.])*count
        elif ".ln_" in name or name.startswith("ln_f."):
            v = array("f",[1.])*count
        else:
            v = array("f", ((rng.getrandbits(16)-32768)/262144. for _ in range(count)))
        data[start+a:start+b] = v.tobytes()
    base = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(bytes(data)))
    return base.with_readout_accumulator("f32_wide_backward").with_forward_readout_storage("original_packed"), sha(data)


def wave(st, base, site):
    tensors = [dict(name=k,role="parameter",requires_grad=True,shape=[WIDTH],
                    values=[((i%n)-n//2)/div for i in range(WIDTH)])
               for k,n,div in (("gate",11,16.),("bias",7,128.))]
    tensors.append(dict(name="log_radius",role="buffer",requires_grad=False,shape=[],values=[0.]))
    record = dict(schema="spiraltorch.geometry_adapter_artifact.v1", policy="full_prefix_f32_no_cache.v1",
                  base_model_identity=base.base_model_identity, training=True,
                  placements=[dict(path="transformer.h.%s.attn"%site, training=True, tensors=tensors,
                      recipe=dict(schema="spiraltorch.wave_gate_adapter.v2",features=WIDTH,strength=.25,
                                  learnable_radius=False,kernel=dict(curvature=-1.,saturation=1.,porosity=.05,
                                                                    max_values=65536)))])
    return base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))


def batches():
    out = []
    for step in range(STEPS):
        ids = [(31*t+53*b+73*step+t*t)%VOCAB for b in range(2) for t in range(17)]
        labels, plan = [-100]*len(ids), []
        for pos in (16,33):
            labels[pos] = ids[pos]
            plan.append((pos, ids[pos], [(ids[pos]+79)%VOCAB]))
        out.append((ids, dict(labels=labels, contrasts=plan, target_l2=1./256)))
    return out


def chain(origin, data):
    m, trace, times = origin, [], []
    for ids, kw in data:
        before = m.geometry_artifact_id
        t = perf_counter_ns()
        nxt, receipt = m.step_declared_set_geometry_at_distance(ids, [2,17], **kw)
        times.append(perf_counter_ns()-t)
        guard = receipt["proposal_guard"]
        require(guard["pre_geometry_id"]==before and nxt.geometry_artifact_id==receipt["result_geometry_id"]
                ==(guard["post_geometry_id"] if receipt["accepted"] else before), "broken training lineage")
        trace.append(receipt)
        m = nxt
    audit_logits = bytes(m.forward(data[-1][0][:16], [1,16], last_only=True).values_buffer())
    return dict(trace=trace, final=m.geometry_artifact_id, logits_sha256=sha(audit_logits)), times


def worker(args):
    sys.meta_path.insert(0, runpy.run_path(str(HERE/"synthetic_step.py"))["NativeOnly"]())
    import spiraltorch as st
    extension = sha(Path(importlib.util.find_spec("spiraltorch.spiraltorch").origin).read_bytes())
    require(extension==args.extension, "wrong installed native extension")
    base, weights = fixture(st)
    data, sites = batches(), {}
    for site in args.sites:
        initial = wave(st,base,site)
        reference, _ = chain(initial,data)
        times = []
        for _ in range(3):
            result, ns = chain(initial,data)
            require(encoded(result)==encoded(reference), "different predictions in repeated chain")
            times.append(ns)
        sites[site] = dict(reference=reference, ns=times)
    save(args.output, dict(extension=extension, python=sys.version, base=base.base_model_identity, weights=weights,
                           batches=sha(encoded(data)), sites=sites))


def compare(args):
    root = args.output
    root.mkdir(mode=0o700,exist_ok=False)
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    interpreters = (args.old,args.new)
    pins = (OLD,args.extension)
    require(pins[0]!=pins[1], "same two builds")
    ref = None
    times = {str(i):[] for i in (0,LAYERS-1)}
    for block, order in enumerate([(0,1),(1,0),(1,0),(0,1)]):
        pair = {}
        for build in order:
            output = root/("%s-%s.json"%(block,("old","new")[build]))
            sites = [0,LAYERS-1] if block%2==0 else [LAYERS-1,0]
            subprocess.run([str(interpreters[build]), "-IB", str(Path(__file__).resolve()), "worker",
                            pins[build],str(output),*map(str,sites)],check=True,env=env)
            observed = json.loads(output.read_bytes())
            semantic = dict(python=observed["python"], base=observed["base"], weights=observed["weights"], batches=observed["batches"],
                            sites={k:r["reference"] for k,r in observed["sites"].items()})
            if ref is None:
                ref = semantic
            require(encoded(ref)==encoded(semantic), "wheels differ; no timing comparison")
            pair[build] = {k:statistics.median(sum(ns)/STEPS for ns in r["ns"])
                           for k,r in observed["sites"].items()}
        for site, rows in times.items():
            old, new = pair[0][site], pair[1][site]
            rows.append(dict(old_ns=old,new_ns=new,speedup=old/new))
    summary = {s:dict(pairs=rs,median_speedup=statistics.median(r["speedup"] for r in rs),
                      accepted=sum(r["accepted"] for r in ref["sites"][s]["trace"])) for s,rs in times.items()}
    save(root/"summary.json",dict(schema="spiraltorch.declared_set_suffix_bench.v1",paired=summary,
                                 reference=OLD,candidate=args.extension,semantic_sha256=sha(encoded(ref)),
                                 pretrained_calls=0,heldout_scores=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    c=actions.add_parser("compare")
    for key in ("old","new"):
        c.add_argument(key,type=Path)
    w=actions.add_parser("worker")
    for sub in (c,w):
        sub.add_argument("extension")
        sub.add_argument("output",type=Path)
    w.add_argument("sites",nargs=2,type=int)
    a=parser.parse_args()
    (compare if a.action=="compare" else worker)(a)

"""One exclusive CPU attempt; the predecessor is read-only."""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

p = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
d, k, s, u, need = p.d, p.k, p.s, p.u, p.need


def current(m, host, turn, owner):
    raw = m.geometry_artifact().to_json().encode()
    ident, v = p.state(raw, host, turn, owner)
    need(m.geometry_artifact_id == ident and m.base_model_identity == host["base"]
         and m.lora_artifact_id is None and m.frozen_lora_artifact_id == host["frozen_lora"]
         and m.frozen_lora_to_json() == host["lora_weights"] and m.readout_accumulator == "f32",
         "weights, companion or arithmetic changed")
    return ident, v, raw


def snapshot(out, name, m, host, turn, owner, rows):
    ident, vec, raw = current(m, host, turn, owner)
    k.native.raw_new(out/"weights"/(name+".json"), raw)
    z = u.logits(m, rows)
    for v in z.values():
        u.bits(v)
    d.save(out/"logits"/(name+".json"), dict(owner=ident, rows=z))
    return ident, vec, z


def same_logits(z, reference):
    need(z.keys() == reference.keys() and all(u.bits(v)==u.bits(reference[key]) for key,v in z.items()),
         "full logits do not coincide bitwise")


def native():
    need(os.environ.get("SPIRALTORCH_BACKEND") == "cpu" and all(os.environ.get(name) == "1" for name in
         ("OMP_NUM_THREADS","RAYON_NUM_THREADS","VECLIB_MAXIMUM_THREADS")), "CPU constraints")
    def offline(event, args):
        if event in ("socket.connect","socket.getaddrinfo") or (event == "import" and args[0].split(".")[0] in
                {"torch","numpy","safetensors","transformers","accelerate","huggingface_hub"}):
            raise ValueError("offline native runner")
    sys.addaudithook(offline)
    import spiraltorch as st
    ext = importlib.import_module("spiraltorch.spiraltorch")
    need(st.NativeCausalLm is ext.NativeCausalLm and d.digest(Path(ext.__file__).read_bytes()) == p.EXTENSION,
         "wrong native extension")
    return st


def run(root, previous, pin):
    raw = (root/"spec.json").read_bytes()
    sp = json.loads(raw)
    need(d.digest(raw) == pin and raw == d.encoded(p.plan(previous,sp["kind"])), "plan drift")
    out = root/"attempt"
    out.mkdir(mode=0o700)
    for folder in ("weights","logits"):
        (out/folder).mkdir()
    st = native()
    hosts = k.read(previous/"attempt/hosts.json")
    anchors = p.anchor_rows(previous)
    rows = sp["train"] + sp["test"]
    starts = {}
    for b in sp["bases"]:
        host = hosts[str(b)]
        data = (previous/("base-%d.safetensors"%b)).read_bytes()
        ref = k.read(previous/"attempt"/("frozen-%d.json"%b))
        case = s.fixtures_cached()[0]
        base = st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
        need(base.base_model_identity == host["original_base"], "restored wrong base")
        frozen = base.with_lora_json(host["lora_weights"]).freeze_lora()
        for turn in sp["turns"]:
            for owner in sp["owners"]:
                rec = p.artifact(host["base"],turn,owner,[0.]*6)
                m = frozen.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(rec)))
                name = s.key(b,turn,owner)
                ident, vec, z = snapshot(out,name,m,host,turn,owner,rows)
                same_logits(z,ref)
                starts[name] = m, ident, vec, z
    control_steps, control_logits = {}, {}
    with (out/"steps.jsonl").open("xb") as log:
        for b,turn,owner,teacher,order in p.cells(sp):
            name = s.key(b,turn,owner,teacher,order)
            original, ident, vec, z = starts[s.key(b,turn,owner)]
            m = original
            host = hosts[str(b)]
            anchor = anchors[s.key(b,turn,teacher,order)]
            for i,pair in enumerate(k.schedule(sp["train"],order,"pair")*sp["epochs"]):
                handles, members = zip(*(u.old.handle(m,r,r[teacher],1.) for r in pair))
                rec = dict(cell=name,i=i,keys=[r["key"] for r in pair],members=members)
                p.check_members(rec,pair,teacher,z,anchor)
                m, report = m.step_geometry_cohort_gradient_at_distance(handles,target_l2=2*d.D)
                after, v, _ = current(m,host,turn,owner)
                rec.update(id=after,values=v,report=report)
                log.write(d.encoded(rec))
                log.flush()
                p.check_motion(rec,vec,ident,host,owner)
                if owner == "bias":
                    key = (b,teacher,order,i)
                    sig = p.bias_signature(rec)
                    if turn == "identity":
                        control_steps[key] = sig
                    else:
                        need(sig == control_steps[key], "rotation changed bias-only update")
                ident, vec = after, v
            end, _, logits = snapshot(out,name,m,host,turn,owner,rows)
            need(end == ident and original.geometry_artifact_id == starts[s.key(b,turn,owner)][1],
                 "endpoint or origin drift")
            if owner == "bias":
                key = (b,teacher,order)
                if turn == "identity":
                    control_logits[key] = logits
                else:
                    same_logits(logits,control_logits[key])
    d.save(root/"complete.json", dict(spec=pin,files=k.identities(out)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command",required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("kind",choices=list(p.PARENT))
    for parser in (pre, sub.add_parser("run")):
        parser.add_argument("previous",type=Path)
        parser.add_argument("root",type=Path)
    sub.choices["run"].add_argument("pin")
    q = cli.parse_args()
    if q.command == "prepare":
        plan = p.plan(q.previous,q.kind)
        q.root.mkdir(mode=0o700)
        print(d.save(q.root/"spec.json",plan))
    else:
        run(q.root,q.previous,q.pin)

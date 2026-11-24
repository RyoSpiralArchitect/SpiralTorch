"""Pinned training-prefix study. The model is called in exactly one place."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import runpy
from statistics import median
import sys
import zipfile

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE.parent/"design.py"))
P, N = D["P"], D["N"]
need, encoded, digest, f32 = (D[k] for k in ("need", "encoded", "digest", "f32"))
write = D["OLD"]["write"]
HOSTS = ("affine", "both")
ARMS = ("affine", "saturation", "projection", "both")
CFG = dict(curvature=-1., saturation=1., porosity=.05, log_radius=0.)
ARCHIVE = "2e60ffdf268a9a9e78be20f7816f2b94a2e32ddf3b41e49d7ee24696b3641be3"


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def un64(s):
    return base64.b64decode(s,validate=True)


def sources():
    return {k:P["file_hash"](HERE/k) for k in ("README.md","bounded.py","study.py","../design.py",
                                               "../../native.py","../../probe.py","../../protocol.py",
                                               "../../../../../crates/st-nn/src/causal_lm.rs",
                                               "../../../../../crates/st-nn/src/causal_lm/gpt2.rs",
                                               "../../../../../crates/st-nn/src/causal_lm/readout_policy.rs")}


def prepare(old, kind, directory, model, wheel):
    sums=(old/"SHA256SUMS").read_bytes()
    need(digest(sums)==ARCHIVE,"wrong completed study")
    pins={file:pin for pin,file in (x.split("  ",1) for x in sums.decode().splitlines())}
    def read(rel):
        name=kind+"/"+rel
        raw=(old/name).read_bytes()
        need(digest(raw)==pins[name],"changed archived input "+rel)
        return raw
    spec=json.loads(read("plan.json"))
    identity=json.loads(read("attempt/identity.json"))
    sealed=json.loads(read("attempt/sealed.json"))
    receipt=json.loads(read("attempt/receipt.json"))
    need(spec["kind"]==kind and receipt["plan"]==pins[kind+"/plan.json"] and
         receipt["result_sha256"]==pins[kind+"/attempt/worker-result.json"] and
         spec["sources"]==D["sources"]() and not (old/kind/"attempt/failure.json").exists(),"incomplete input")
    width=spec["shape"][1]
    train=spec["train"]
    geometry,anchors={},{}
    for host in HOSTS:
        k="wave" if host=="both" else host
        pin=sealed["initial-"+k]
        raw=read("attempt/states/"+pin[7:]+".json")
        need(pin=="sha256:"+digest(raw),"changed initial artifact")
        D["OLD"]["check_geometry"](json.loads(raw),spec,identity["base"],k)
        geometry[host]={"pin":pin,"bytes":b64(raw)}
        anchors[host]={}
        for order,seed in zip(spec["order"],spec["seeds"]):
            stem="attempt/%s-%s_rival"%(seed,k)
            row=json.loads(read(stem+".jsonl").splitlines()[0])
            g=read(stem+".g32")[:width*2*4]
            case=train[order[0]]
            need(row["key"]==case["key"] and row["gold"]==case["gold"] and row["rival"]==case["rival"]
                 and row["step"]["origin_id"]==pin and digest(g)==row["grad_sha256"],"wrong first gradient")
            D["check_objective"]("rival",row["loss"],row["declared"])
            a=dict(gradient=b64(g),margin=row["declared"]["mean_target_minus_declared_rival"],
                   q=f32([-math.expm1(-row["declared"]["declared_ce"])])[0])
            need(a["q"]>0 and anchors[host].get(row["key"],a)==a,"invalid reference")
            anchors[host][row["key"]]=a
    if kind=="gpt2":
        need(model and spec["model_pins"]==P["PINS"] and all(P["file_hash"](model/k)==v for k,v in P["PINS"].items()),
             "wrong saved GPT-2")
    else:
        need(model is None,"tiny fixture must not use real model")
    with zipfile.ZipFile(wheel) as z:
        ext,=[n for n in z.namelist() if n.startswith("spiraltorch/") and n.endswith(".abi3.so")]
        ext_pin=digest(z.read(ext))
    plan=dict(schema="spiraltorch.wave_margin_study.v1",kind=kind,archive_manifest=ARCHIVE,source_plan=receipt["plan"],
              shape=spec["shape"],site="transformer.h.%d.attn"%(spec["shape"][0]-1),strength=spec["strength"],
              config=CFG,base=identity["base"],model=spec["model_pins"],train=train,geometry=geometry,anchors=anchors,
              schedule=[(c["key"],h) for i,c in enumerate(train) for h in (HOSTS if i%2==0 else HOSTS[::-1])],
              calls=2*len(train),seconds=300 if kind=="gpt2" else 60,anchor_tolerance=.001,
              extension=ext_pin,wheel=P["file_hash"](wheel),sources=sources())
    directory.mkdir(mode=0o700,exist_ok=False)
    raw=encoded(plan)
    write(directory/"plan.json",raw)
    print(digest(raw))


def vector(s,width):
    raw=un64(s)
    need(len(raw)==4*width and sys.byteorder=="little","wrong f32 row")
    v=memoryview(raw).cast("f")
    need(all(map(math.isfinite,v)),"nonfinite receipt values")
    return list(v)


def norm(v):
    return math.sqrt(math.fsum(x*x for x in v))


def dist(x,y):
    return norm(a-b for a,b in zip(x,y))


def pack(result):
    rec=dict(result)
    for key in ("site_input","upstream"):
        rec[key]=b64(result[key])
    rec["arms"]={h:{k:b64(v) for k,v in p.items()} for h,p in result["arms"].items()}
    return rec


def stats(plan, case, host, r):
    width=plan["shape"][1]
    con=r["contrast"]
    need(r["schema"]=="spiraltorch.wave_margin_parameter_pullbacks.v1" and r["objective"]=="target_logit_minus_alternative_logit"
         and r["encoding"]=="little_endian_f32" and r["factual"]==host and r["site"]==plan["site"] and
         r["geometry_artifact_id"]==plan["geometry"][host]["pin"] and r["base_model_identity"]==plan["base"] and
         r["lora_artifact_id"] is None and r["readout_accumulator"]=="f32_wide_backward" and r["strength"]==plan["strength"]
         and set(r["config"])==set(CFG) and all(f32([v])[0]==f32([r["config"][k]])[0] for k,v in CFG.items())
         and (con["target_id"],con["alternative_id"],con["prompt_tokens"])==(case["gold"],case["rival"],len(case["prefix"]))
         and con["target_minus_alternative"]==con["target_logit"]-con["alternative_logit"],"wrong local probe scope")
    vector(r["site_input"],width)
    upstream=norm(vector(r["upstream"],width))
    need(set(r["arms"])==set(ARMS),"incomplete parameter factorial")
    grads, ns, cos={}, {}, {}
    for k in ARMS:
        need(set(r["arms"][k])=={"gate","bias"},"wrong parameter blocks")
        g,b=(vector(r["arms"][k][p],width) for p in ("gate","bias"))
        grads[k]=g+b
        ns[k]={"gate":norm(g),"bias":norm(b),"all":norm(g+b)}
    A,S,Pv,B=(grads[k] for k in ARMS)
    scale=ns["affine"]["all"]
    ratio=lambda v:v/scale if scale else None
    for k in ARMS:
        denom=scale*ns[k]["all"]
        cos[k]=math.fsum(a*b for a,b in zip(A,grads[k]))/denom if denom else None
    dp,ds=dist(B,Pv),dist(B,S)
    anchor=plan["anchors"][host].get(case["key"])
    error=None
    if anchor:
        need(con["target_minus_alternative"]==anchor["margin"],"initial factual logits do not match archive")
        ref=[-v/anchor["q"] for v in vector(anchor["gradient"],2*width)]
        need(norm(ref)>0,"unusable archived reference")
        error=dist(grads[host],ref)/norm(ref)
        need(error<=plan["anchor_tolerance"],"factual margin VJP differs from archived binary gradient: "+str(error))
    return dict(upstream_l2=upstream,norms=ns,ratio={k:ratio(ns[k]["all"]) for k in ARMS},cosine=cos,
                distance_both_projection=ratio(dp),distance_both_saturation=ratio(ds),
                interaction=ratio(norm(b-s-p+a for a,s,p,b in zip(A,S,Pv,B))),
                projection=bool(scale and dp<ds and ns["projection"]["all"]<scale),
                saturation=bool(scale and ds<dp and ns["saturation"]["all"]<scale),anchor_error=error)


def case_map(plan):
    return {r["key"]:r for r in plan["train"]}


def replay(plan, directory):
    root=directory/"attempt"
    cases=case_map(plan)
    lines=(root/"rows.jsonl").read_bytes().splitlines()
    need(len(lines)==plan["calls"]==len(plan["schedule"]),"incomplete observation set")
    groups={h:{} for h in HOSTS}
    shared={}
    for (key,h),line in zip(plan["schedule"],lines):
        row=json.loads(line)
        need((row["key"],row["host"])==(key,h) and key not in groups[h],"changed order or repeated row")
        raw=row["receipt"]
        need(shared.setdefault(key,raw["site_input"])==raw["site_input"],"different incoming site inputs by host")
        groups[h][key]=stats(plan,cases[key],h,raw)
    def med(seq):
        values=[v for v in seq if v is not None]
        return median(values) if values else None
    summaries={}
    for host,rows in groups.items():
        need(set(cases)==set(rows),"missing comparison case")
        vals=list(rows.values())
        summaries[host]=dict(n=len(vals),projection=sum(v["projection"] for v in vals),
            saturation=sum(v["saturation"] for v in vals),zero_affine=sum(v["norms"]["affine"]["all"]==0 for v in vals),
            median_norm_ratio={k:med(v["ratio"][k] for v in vals) for k in ARMS},
            median_cosine={k:med(v["cosine"][k] for v in vals) for k in ARMS},
            median_distance_both_projection=med(v["distance_both_projection"] for v in vals),
            median_distance_both_saturation=med(v["distance_both_saturation"] for v in vals),
            median_interaction=med(v["interaction"] for v in vals),
            anchor_max_error=max(v["anchor_error"] for v in vals if v["anchor_error"] is not None))
    return dict(plan=digest(encoded(plan)),kind=plan["kind"],calls=len(lines),maps=4*len(lines),
                raw_sha256=P["file_hash"](root/"rows.jsonl"),by_item=groups,summary=summaries,
                gate={k:all(4*s[k]>=3*s["n"] for s in summaries.values()) for k in ("projection","saturation")})


def worker(plan,directory,model):
    need(all(os.environ.get(k)==v for k,v in D["OLD"]["ENV"].items()),"not the frozen CPU environment")
    for name in ("numpy","torch","transformers","safetensors"):
        need(name not in sys.modules,"unexpected foreign ML runtime")
        sys.modules[name]=None
    import spiraltorch as st
    need(P["file_hash"](Path(st._rs.__file__))==plan["extension"],"wrong installed extension")
    if plan["kind"]=="gpt2":
        cfg=(model/"config.json").read_bytes()
        weights=st.SafeTensorArchive(model/"model.safetensors")
        need(weights.content_id=="sha256:"+plan["model"]["model.safetensors"] and digest(cfg)==plan["model"]["config.json"],
             "changed owned base")
        base=st.NativeCausalLm.from_gpt2(cfg.decode(),weights)
        del weights
    else:
        need(not model,"unexpected model")
        base=N["tiny"](st)
    # This pinned revision owns one PackedB head; project_readout multiplies it directly.
    base=base.with_readout_accumulator("f32_wide_backward")
    need((base.base_model_identity,base.execution_backend,base.lora_artifact_id,base.geometry_artifact_id)==
         (plan["base"],"rust_f32_cpu_prepacked",None,None) and
         (base.readout_accumulator,base.readout_forward_precision,base.readout_backward_precision)==
         ("f32_wide_backward","f32","f64"),"wrong base or readout arithmetic")
    models={}
    for h,entry in plan["geometry"].items():
        art=st.GeometryAdapterArtifact.from_json(un64(entry["bytes"]).decode(),entry["pin"],plan["base"])
        models[h]=base.with_geometry(art)
    root=directory/"attempt"
    cases,shared=case_map(plan),{}
    with (root/"rows.jsonl").open("xb",buffering=0) as f:
        for key,host in plan["schedule"]:
            c=cases[key]
            print(key,host,flush=True)
            native=models[host].last_attention_wave_margin_pullbacks(c["prefix"],c["gold"],c["rival"],CFG)
            raw=pack(native)
            f.write(encoded(dict(key=key,host=host,receipt=raw)))
            stats(plan,c,host,raw)
            need(shared.setdefault(key,raw["site_input"])==raw["site_input"],"unmatched site input")
            need(models[host].geometry_artifact_id==plan["geometry"][host]["pin"],"model changed")
    write(root/"result.json",encoded(replay(plan,directory)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","worker","verify"))
    p.add_argument("directory",type=Path)
    p.add_argument("--archive",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--wheel",type=Path)
    p.add_argument("--kind",choices=("gpt2","rehearsal"),required=True)
    p.add_argument("--pin")
    a=p.parse_args()
    if a.action=="prepare":
        prepare(a.archive,a.kind,a.directory,a.model,a.wheel)
    else:
        raw=(a.directory/"plan.json").read_bytes()
        plan=json.loads(raw)
        need(digest(raw)==a.pin and encoded(plan)==raw and plan["kind"]==a.kind and sources()==plan["sources"],
             "changed plan/source")
        if a.action=="worker":
            worker(plan,a.directory,a.model)
        else:
            guard=json.loads((a.directory/"attempt/receipt.json").read_bytes())
            need(guard["success"] and not guard["retry"] and guard["exit_code"]==0 and guard["min_free"]>=4<<30 and
                 0<guard["elapsed"]<=guard["seconds_limit"]==plan["seconds"],"missing successful bounded attempt")
            result=replay(plan,a.directory)
            need(encoded(result)==(a.directory/"attempt/result.json").read_bytes(),"cannot reproduce saved result")
            print(json.dumps({k:result[k] for k in ("kind","calls","maps","gate")}))

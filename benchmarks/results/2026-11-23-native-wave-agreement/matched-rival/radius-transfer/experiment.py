"""One-shot matched radius transfer, using only native training/forward calls."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
FIT = runpy.run_path(str(HERE.parent / "run.py"))
D, P, N, OLD = (FIT[k] for k in ("D", "P", "N", "OLD"))
R = runpy.run_path(str(HERE.parent / "parameter-vjp/radius-sweep/sweep.py"))
bounded = runpy.run_path(str(HERE.parent / "parameter-vjp/bounded.py"))["run"]
need, enc, sha, f32 = (FIT[k] for k in ("need", "encoded", "digest", "f32"))
write, save, load = (OLD[k] for k in ("write", "save", "load"))
ROOTS = ("180c2c60b9dc5ee7c871fda336946af498324fec5199fa9235789025265a1201",
         "8edc3b8339e61dc1f0e8a1e30558af2c1b257f5d1f8e70fc3e3a1d3c76296e74",
         "2e60ffdf268a9a9e78be20f7816f2b94a2e32ddf3b41e49d7ee24696b3641be3")
ARMS = ("r1", "r4")
HELD = (("parent","parents","child","children"), ("brother","brothers","sister","sisters"),
        ("student","students","friend","friends"), ("chef","chefs","guest","guests"))
MIN_EFFECT = .01


def sources():
    names = ("README.md","experiment.py","../run.py","../design.py","../../probe.py",
             "../../protocol.py","../../native.py","../parameter-vjp/bounded.py",
             "../parameter-vjp/radius-sweep/sweep.py")
    return {n: P["file_hash"](HERE/n) for n in names}


def reader(root, expected):
    raw = (root/"SHA256SUMS").read_bytes()
    need(sha(raw) == expected, "unrecognized archive")
    pins = {p:h for h,p in (line.split("  ",1) for line in raw.decode().splitlines())}
    def read(name):
        pin = pins[name]
        raw = (root/name).read_bytes()
        need(sha(raw) == pin, "changed archive member: "+name)
        return raw
    return read


def fresh_cases(kind, train, model):
    if kind == "gpt2":
        previous = D["lexemes"](D["TRAIN"] + D["HELD"] + P["TRAIN"] + P["TEST"])
        need(not (D["lexemes"](HELD) & previous) and len(D["lexemes"](HELD)) == 4*len(HELD),
             "repeated lexical item")
        need(model is not None and all(P["file_hash"](model/k)==P["PINS"][k] for k in ("vocab.json","merges.txt")),
             "unexpected tokenizer")
        vocab = json.loads((model/"vocab.json").read_bytes())
        ranks = {tuple(s.split()):i for i,s in enumerate((model/"merges.txt").read_text().splitlines()[1:])}
        words = D["lexemes"](HELD+D["TRAIN"]) | {"The","that","the","met","is","are"}
        ids = {w:P["single_word_id"](w if w=="The" else chr(288)+w,vocab,ranks) for w in words}
        need(len(set(ids.values()))==len(ids) and D["cases"](D["TRAIN"],ids)==train, "training tokens differ")
        return D["cases"](HELD,ids)
    need(kind=="rehearsal" and model is None, "wrong rehearsal scope")
    first = train[0]
    the, _, that, the2, _, met = first["prefix"]
    ids = dict(The=the, that=that, the=the2, met=met, ee=19, ees=20, ff=21, ffs=22,
               **{"is":first["gold"], "are":first["rival"]})
    return D["cases"]((("ee","ees","ff","ffs"),),ids)


def plan(kind, roots, model):
    capture, sweep, matched = [reader(root,pin) for root,pin in zip(roots,ROOTS)]
    tag, sub = ("real","gpt2") if kind=="gpt2" else ("toy","rehearsal-pinned-head")
    p = json.loads(capture(sub+"/plan.json"))
    selection = json.loads(sweep(tag+"/summary.json"))
    need(p["kind"]==kind and (selection["selected"]==4 if kind=="gpt2" else selection["selected"] is None),
         "wrong selected control")
    states = [sweep("%s/radius-%d.json"%(tag,n)) for n in (1,4)]
    R["check_variant"](json.loads(states[0]),json.loads(states[1]),4)
    pins = dict(zip(ARMS,["sha256:"+sha(raw) for raw in states]))
    need(pins["r1"]==selection["parent_artifact"]==p["geometry"]["both"]["pin"] and
         (kind!="gpt2" or pins["r4"]==selection["selected_artifact"]), "wrong initialization")
    identity = json.loads(matched(kind+"/attempt/identity.json"))
    need(identity==dict(base=p["base"],extension=P["EXTENSION"],backend="rust_f32_cpu_prepacked",
                        head="original_packed",accumulator="f32_wide_backward"), "wrong source identity")
    sealed = json.loads(matched(kind+"/attempt/sealed.json"))
    train, seeds = p["train"], [17,31,73] if kind=="gpt2" else [11]
    order = [sorted(range(len(train)),key=lambda i:sha(("relative/%s/%s"%(seed,train[i]["key"])).encode()))
             for seed in seeds]
    anchors = {}
    for seed,idx in zip(seeds,order):
        stage = "%s-wave_rival"%seed
        rows = [json.loads(v) for v in matched(kind+"/attempt/"+stage+".jsonl").splitlines()]
        need([r["key"] for r in rows]==[train[i]["key"] for i in idx] and rows[0]["step"]["origin_id"]==
             sealed["initial-wave"]==pins["r1"] and rows[-1]["step"]["result_id"]==sealed[stage],"wrong legacy route")
        anchors[str(seed)] = [[r["grad_sha256"],r["step"]["result_id"]] for r in rows]
    ev = fresh_cases(kind,train,model)
    need(not ({tuple(r["prefix"]) for r in train} & {tuple(r["prefix"]) for r in ev})
         and len({tuple(r["prefix"]) for r in ev})==len(ev) and len({r["key"] for r in ev})==len(ev)
         and all(len(r["prefix"])==6 for r in train+ev), "split contamination")
    return dict(schema="spiraltorch.matched_wave_radius_learning.v1",kind=kind,shape=p["shape"],base=p["base"],
                site=p["site"],train=train,evaluation=ev,seeds=seeds,order=order,identity=identity,
                parent_manifests=ROOTS,initial=dict(zip(ARMS,[base64.b64encode(r).decode() for r in states])),
                pins=pins,anchors=anchors,target_l2=D["DISTANCE"],tolerance=1e-4,min_effect=MIN_EFFECT,
                updates=2*len(train)*len(seeds),scored_rows=(2+2*len(seeds))*len(ev),
                seconds=600 if kind=="gpt2" else 60, model_pins=p["model"],sources=sources())


def weights(raw,p,arm):
    rec = json.loads(raw)
    gate, bias = (R["tensor_at"](rec,k) for k in ("gate","bias"))
    vals = f32(gate.pop("values")+bias.pop("values"))
    need(len(vals)==2*p["shape"][1] and all(map(math.isfinite,vals)) and rec["base_model_identity"]==p["base"]
         and R["f32"](R["log_buffer"](rec)["values"][0])==R["f32"](math.log(int(arm[1:]))), "wrong state")
    return vals, sha(enc(rec))


def check_update(p,arm,seed,j,case,prev,old,new,raw,row):
    width = p["shape"][1]
    need((row["key"],row["gold"],row["rival"])==(case["key"],case["gold"],case["rival"]) and
         row["grad_sha256"]==sha(raw) and len(raw)==8*width, "wrong supervision or gradient tape")
    D["check_objective"]("rival",row["loss"],row["declared"])
    trace = row["step"]
    before,layout = weights(old,p,arm)
    after,new_layout = weights(new,p,arm)
    need("sha256:"+sha(old)==trace["origin_id"]==prev and trace["result_id"]=="sha256:"+sha(new)
         and layout==new_layout, "broken state lineage or changed frozen values")
    if arm=="r1":
        need(p["anchors"][str(seed)][j]==[sha(raw),trace["result_id"]], "R=1 differs from original at "+case["key"])
    return D["check_step"](before,after,memoryview(raw).cast("f"),trace,p)


def stages(p):
    for i,(seed,order) in enumerate(zip(p["seeds"],p["order"])):
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            yield seed, order, arm, "%s-%s"%(seed,arm)


def audit(p,root,sealed):
    need(set(sealed)=={"initial-"+a for a in ARMS}|{s for _,_,_,s in stages(p)}, "incomplete stages")
    routes = {}
    for seed,order,arm,name in stages(p):
        pin = sealed["initial-"+arm]
        init = base64.b64decode(p["initial"][arm],validate=True)
        need(pin==p["pins"][arm] and FIT["blob"](root,pin).read_bytes()==init, "wrong initial state bytes")
        journal = (root/(name+".jsonl")).read_bytes()
        rows = list(map(json.loads,journal.splitlines()))
        tape = (root/(name+".g32")).read_bytes()
        stride = p["shape"][1]*8
        need(len(rows)==len(order) and len(tape)==len(order)*stride, "partial training evidence")
        old = init
        steps,errors = [],[]
        for j,(i,row) in enumerate(zip(order,rows)):
            nxt = row["step"]["result_id"]
            new = FIT["blob"](root,nxt).read_bytes()
            step,err = check_update(p,arm,seed,j,p["train"][i],pin,old,new,tape[j*stride:(j+1)*stride],row)
            steps.append(step)
            errors.append(err)
            pin,old = nxt,new
        need(sealed[name]==pin, "wrong endpoint")
        endpoint = D["length"](D["delta"](weights(old,p,arm)[0],weights(init,p,arm)[0]))
        routes[name] = dict(n=len(steps),path=math.fsum(steps),endpoint=endpoint,max_error=max(errors),
                            journal_sha256=sha(journal),tape_sha256=sha(tape))
    need(sum(r["n"] for r in routes.values())==p["updates"], "wrong update count")
    return routes


def contrast(p,scores):
    def effect(stage,start,group):
        cases = [r for r in p["evaluation"] if group=="all" or r["conflict"]==(group=="conflict")]
        keys = [r["key"] for r in cases]
        a,b = scores[stage],scores[start]
        mean = lambda data,k: math.fsum(data[i][k] for i in keys)/len(keys)
        delta = lambda k: math.fsum(a[i][k]-b[i][k] for i in keys)/len(keys)
        return dict(n=len(keys),initial=mean(b,"margin"),final=mean(a,"margin"),change=delta("margin"),
                    nll_change=delta("nll"),initial_wins=sum(b[k]["margin"]>0 for k in keys),
                    final_wins=sum(a[k]["margin"]>0 for k in keys),initial_ties=sum(b[k]["margin"]==0 for k in keys),
                    final_ties=sum(a[k]["margin"]==0 for k in keys))
    orders = []
    for seed in p["seeds"]:
        arms = {k:{g:effect("%s-%s"%(seed,k),"initial-"+k,g) for g in ("all","conflict","congruent")} for k in ARMS}
        gain = {g:arms["r4"][g]["change"]-arms["r1"][g]["change"] for g in arms["r4"]}
        orders.append(dict(seed=seed,arms=arms,paired_gain=gain,
                           passed=arms["r4"]["all"]["change"]>0 and gain["all"]>=p["min_effect"]))
    return dict(orders=orders,advance=all(x["passed"] for x in orders))


def replay(p,root):
    sealed = load(root/"sealed.json")
    need(load(root/"identity.json")==p["identity"],"wrong execution identity")
    routes = audit(p,root,sealed)
    vocab, scored, hashes = p["shape"][-1], {}, {}
    for name in sealed:
        raw = (root/(name+".f32le")).read_bytes()
        need(sys.byteorder=="little" and len(raw)==len(p["evaluation"])*vocab*4,"incomplete logits")
        v=memoryview(raw).cast("f")
        scored[name]={r["key"]:P["metrics"](v[i*vocab:(i+1)*vocab],r["gold"],r["rival"])
                      for i,r in enumerate(p["evaluation"])}
        hashes[name]=sha(raw)
    need(sum(map(len,scored.values()))==p["scored_rows"],"prediction count differs")
    return dict(plan=sha(enc(p)),updates=p["updates"],scored=p["scored_rows"],paths=routes,
                endpoints=sealed,logit_sha256=hashes,by_item=scored,**contrast(p,scored))


def worker(p,root,model_dir):
    os.environ.update(OLD["ENV"], SPIRALTORCH_AUTOTUNE_STORE=str(root/"tuning.json"))
    for k in ("torch","numpy","safetensors","transformers"):
        need(k not in sys.modules,"unexpected ML runtime")
        sys.modules[k]=None
    import spiraltorch as st
    ext=P["file_hash"](Path(st._rs.__file__))
    need(ext==p["identity"]["extension"],"wrong training extension")
    if p["kind"]=="rehearsal":
        need(model_dir is None,"unexpected pretrained model")
        base=N["tiny"](st)
    else:
        need(model_dir is not None,"missing local model")
        config=(model_dir/"config.json").read_bytes()
        weights_archive=st.SafeTensorArchive(model_dir/"model.safetensors")
        need(sha(config)==p["model_pins"]["config.json"] and
             weights_archive.content_id=="sha256:"+p["model_pins"]["model.safetensors"],"wrong decoder")
        base=st.NativeCausalLm.from_gpt2(config.decode(),weights_archive)
        del weights_archive
    base=base.with_forward_readout_storage("original_packed").with_readout_accumulator("f32_wide_backward")
    identity=dict(base=base.base_model_identity,extension=ext,backend=base.execution_backend,
                  head=base.forward_readout_storage,accumulator=base.readout_accumulator)
    need(identity==p["identity"] and base.lora_artifact_id is None and base.geometry_artifact_id is None
         and (base.readout_forward_precision,base.readout_backward_precision)==("f32","f64"),"wrong backend or base")
    save(root/"identity.json",identity)
    (root/"states").mkdir()
    initial,sealed={},{}
    for arm in ARMS:
        raw=base64.b64decode(p["initial"][arm],validate=True)
        a=st.GeometryAdapterArtifact.from_json(raw.decode(),p["pins"][arm],p["base"])
        initial[arm]=base.with_geometry(a)
        sealed["initial-"+arm]=FIT["keep"](root,initial[arm],base)
    for seed,order,arm,name in stages(p):
        model=initial[arm]
        with (root/(name+".jsonl")).open("xb",buffering=0) as log, (root/(name+".g32")).open("xb",buffering=0) as tape:
            for j,i in enumerate(order):
                case=p["train"][i]
                prev=model.geometry_artifact_id
                g,report=FIT["gradient"](model,case,"rival")
                params=g.parameter_gradients()
                need(set(params)=={p["site"]} and set(params[p["site"]])=={"gate","bias"}
                     and all(len(v)==p["shape"][1] for v in params[p["site"]].values()),"unexpected trainables")
                raw=f32(params[p["site"]]["gate"]+params[p["site"]]["bias"]).tobytes()
                model,trace=model.step_geometry_gradient_at_distance(g,p["target_l2"])
                nxt=FIT["keep"](root,model,base)
                row=dict(key=case["key"],gold=case["gold"],rival=case["rival"],loss=g.loss,declared=report,
                         grad_sha256=sha(raw),step=trace)
                tape.write(raw)
                log.write(enc(row))
                check_update(p,arm,seed,j,case,prev,FIT["blob"](root,prev).read_bytes(),
                             FIT["blob"](root,nxt).read_bytes(),raw,row)
        need(initial[arm].geometry_artifact_id==p["pins"][arm],"initial state changed")
        sealed[name]=model.geometry_artifact_id
    save(root/"sealed.json",sealed)
    audit(p,root,sealed)
    FIT["measure"](st,base,p,root,sealed)
    save(root/"result.json",replay(p,root))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest="cmd",required=True)
    for cmd in ("prepare","run","verify","_worker"):
        s=subs.add_parser(cmd)
        s.add_argument("kind",choices=("rehearsal","gpt2"))
        s.add_argument("dir",type=Path)
        if cmd=="prepare":
            s.add_argument("roots",type=Path,nargs=3)
        else:
            s.add_argument("pin")
        if cmd!="verify":
            s.add_argument("--model-dir",type=Path)
    a=ap.parse_args()
    if a.cmd=="prepare":
        p=plan(a.kind,a.roots,a.model_dir)
        a.dir.mkdir(mode=0o700,exist_ok=False)
        save(a.dir/"plan.json",p)
        print(sha(enc(p)))
        return
    raw=(a.dir/"plan.json").read_bytes()
    p=json.loads(raw)
    need(sha(raw)==a.pin and raw==enc(p) and p["kind"]==a.kind and p["sources"]==sources(),"plan/source mismatch")
    root=a.dir/"attempt"
    if a.cmd=="verify":
        receipt=load(root/"receipt.json")
        need(receipt["success"] is True and receipt["retry"] is False and receipt["exit_code"]==0 and
             receipt["seconds_limit"]==p["seconds"] and 0<receipt["elapsed"]<=p["seconds"]
             and receipt["min_free"]>=4<<30,"unsuccessful attempt")
        result=replay(p,root)
        need((root/"result.json").read_bytes()==enc(result),"non-replaying result")
        print(json.dumps({k:result[k] for k in ("plan","updates","scored","advance")}))
    elif a.cmd=="run":
        need((a.kind=="gpt2")==bool(a.model_dir),"wrong model scope")
        args=[sys.executable,"-I","-B",str(HERE/"experiment.py"),"_worker",a.kind,str(a.dir.absolute()),a.pin]
        if a.model_dir is not None:
            args+=["--model-dir",str(a.model_dir.absolute())]
        bounded(root,p["seconds"],args)
    else:
        worker(p,root,a.model_dir)


if __name__=="__main__":
    main()

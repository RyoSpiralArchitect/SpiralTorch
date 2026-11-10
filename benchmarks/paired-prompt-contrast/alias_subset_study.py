"""Synthetic accepted-alias learning: fixed train, label-blind readout, then model-free scoring."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared_prefix_study import SITE, model, obj, pack, put, read, record, require, sha, write
from projection_language_fixture import weights

ARMS = ("rotating_gold", "absolute_set", "subset_rank")
MENU = [[23,k] for k in (2,3,4,14,15,16)]
STOP, RATE = 1, .035


def roster():
    train, queries, truth = [], [], []
    for entity in range(6):
        for c in range(3):
            meaning=(entity+c)%3
            train.append(dict(prompt=[49,32+entity,24+meaning,27+entity%2,29+meaning], meaning=meaning))
    for entity in range(4):
        for c in range(3):
            meaning=(entity+c)%3
            queries.append(dict(slot=len(queries),prompt=[52,29+meaning,43+entity,24+meaning,27+entity%2]))
            truth.append(dict(slot=len(truth),meaning=meaning))
    return train, queries, truth


def prepare(out):
    out.mkdir(mode=0o700)
    train, queries, truth=roster()
    config=dict(model_type="gpt2",vocab_size=64,n_positions=16,n_embd=32,n_head=4,n_layer=2,n_inner=48,tie_word_embeddings=False)
    pins={name:put(out/name,v) for name,v in (("config.json",config),("train.json",train),("queries.json",queries),("truth.json",truth))}
    pins["weights"]=write(out/"model.safetensors",weights())
    return put(out/"fixture.json",dict(files=pins, menu=MENU, tokens={"23":"status","2":"on","14":"enabled","3":"off",
        "15":"disabled","4":"idle","16":"standby"}, labels="declared equivalence in a symbolic fixture, not language understanding",
        model="deterministic, partly hand-structured tiny decoder; never pretrained",arms=ARMS,rate=RATE,stop=STOP,updates_per_arm=36,train=18,evaluation=12,
        controls="each training prompt twice; opposite single gold on its second appearance. Same frozen base and LoRA, anchored projection, order, and per-step realized distance. No endpoint rescaling.",
        gate="subset must beat both initial and rotating on mean conditional and absolute accepted NLL by >1e-6; worst-alias mean NLL no worse within 1e-6, accepted top1 and both outrank all negatives counts no lower.",
        limits="one CPU trial, 108 updates; 48 typed score reports plus 624 independent token scores; 120 seconds total model work; fail on first error, no retry"))


def open_model(st,folder,files):
    return model(st,folder,{"weights":files["weights"],"config":files["config.json"]})


def native():
    class Guard:
        def find_spec(self,name,path=None,target=None):
            if name.partition(".")[0] in {"torch","numpy","safetensors","transformers","accelerate","mlx"}:
                raise AssertionError("unexpected external ML import: "+name)
    sys.meta_path.insert(0,Guard())
    import spiraltorch as st
    return st


def train(folder,pin,out):
    st=native(); fx=obj(folder/"fixture.json",pin); pins=fx["files"]
    data=obj(folder/"train.json",pins["train.json"])
    require(len(data)==18 and fx["menu"]==MENU,"training plan mismatch")
    out.mkdir(mode=0o700)
    initial=open_model(st,folder,pins); arms={k:initial for k in ARMS}; frozen=initial.lora_to_json()
    with (out/"steps.jsonl").open("xb") as journal:
        for step in range(36):
            sample=data[step%18]; accepted=[sample["meaning"],sample["meaning"]+3]
            gold=accepted[((step%18)//3+step//18)%2]; p=sample["prompt"]
            radius=None
            for name in ARMS:
                m=arms[name]; pre=m.geometry_artifact_id
                record(journal,dict(event="start",step=step,arm=name,pre=pre))
                if name=="rotating_gold":
                    g=m.geometry_line_menu_gradients(p,MENU,gold_index=gold,stop_id=STOP,mix=1.)
                elif name=="absolute_set":
                    g=m.geometry_line_set_gradients(p,[MENU[i] for i in accepted],stop_id=STOP)
                else:
                    g=m.geometry_line_subset_gradients(p,MENU,accepted_indices=accepted,stop_id=STOP)
                params=g.parameter_gradients()
                require(set(params)=={SITE} and set(params[SITE])=={"gate","bias"} and all(len(xs)==32 for xs in params[SITE].values()) and g.geometry_artifact_id==pre
                    and g.base_model_identity==initial.base_model_identity and g.lora_artifact_id==initial.lora_artifact_id
                    and g.target_tokens==(6 if name=="absolute_set" else 18),"bad gradient scope")
                norm=math.sqrt(math.fsum(x*x for v in params[SITE].values() for x in v))
                require(math.isfinite(g.loss) and 0.<norm<math.inf,"invalid gradient")
                match=None
                if radius is None:
                    nxt=m.step_adaptation_sgd(g,"geometry",RATE)
                    radius=m.geometry_artifact().trainable_delta_l2(nxt.geometry_artifact())
                    require(0.<radius<math.inf,"empty reference step")
                else:
                    nxt,match=m.step_geometry_sgd_at_distance(g,RATE,radius)
                    require(match["relative_vector_error"]<=1e-4 and abs(match["realized_l2"]/radius-1.)<=1e-4,"step not matched")
                require(nxt.lora_to_json()==frozen and nxt.base_model_identity==initial.base_model_identity and
                    nxt.geometry_artifact_id!=pre,"frozen owner or no update")
                record(journal,dict(event="return",step=step,arm=name,gold=gold,accepted=accepted,pre=pre,
                    post=nxt.geometry_artifact_id,loss=g.loss,gradient_l2=norm,radius=radius,match=match))
                arms[name]=nxt
    ends={}
    for name,m in arms.items():
        h=write(out/(name+".json"),m.geometry_artifact().to_json().encode())
        require(m.geometry_artifact_id=="sha256:"+h,"endpoint digest")
        ends[name]=h
    return put(out/"training.json",dict(fixture=pin,base=initial.base_model_identity,lora=initial.lora_artifact_id,
        origin=initial.geometry_artifact_id,endpoints=ends,steps=sha((out/"steps.jsonl").read_bytes()),updates=108))


def score(folder,pin,trained,trpin,out):
    st=native(); fx=obj(folder/"fixture.json",pin); tr=obj(trained/"training.json",trpin)
    queries=obj(folder/"queries.json",fx["files"]["queries.json"]); out.mkdir(mode=0o700)
    origin=open_model(st,folder,fx["files"]); states={"initial":origin}
    require(tr["fixture"]==pin and (origin.base_model_identity,origin.lora_artifact_id,origin.geometry_artifact_id)==
        (tr["base"],tr["lora"],tr["origin"]),"different starting model")
    for name,h in tr["endpoints"].items():
        raw=read(trained/(name+".json"),h,100000)
        states[name]=origin.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),"sha256:"+h,tr["base"]))
    forwards=reports=0
    with (out/"scores.jsonl").open("xb") as journal:
        for name,m in states.items():
            for q in queries:
                require(set(q)=={"slot","prompt"},"query exposes labels")
                record(journal,dict(event="start",arm=name,slot=q["slot"]))
                # The same arbitrary partition for every query; actual labels enter only in audit().
                r=m.line_subset_loss(q["prompt"],MENU,accepted_indices=[0,3],stop_id=STOP);reports+=1
                cache={}; full=[]
                for line in MENU:
                    prefix=list(q["prompt"]); terms=[]
                    for k in [*line,STOP]:
                        key=(tuple(prefix),k)
                        if key not in cache: cache[key]=m.next_token_score(prefix,k)["nll"]; forwards+=1
                        terms.append(cache[key]); prefix.append(k)
                    full.append(math.fsum(terms))
                require(len(cache)==13 and all(abs(x-y)<2e-5 for x,y in zip(full,r["line_nlls"],strict=True)), "independent full-line costs disagree")
                record(journal,dict(event="return",arm=name,slot=q["slot"],owner=m.geometry_artifact_id,
                    costs=full,probe=r))
    require((reports,forwards)==(48,624),"score budget mismatch")
    return put(out/"scoring.json",dict(fixture=pin,training=trpin,reports=reports,forwards=forwards,rows=sha((out/"scores.jsonl").read_bytes())))


def metrics(costs,c):
    good=(c,c+3); bad=[i for i in range(6) if i not in good]
    def nll(ids):
        best=min(costs[i] for i in ids)
        return best-math.log(math.fsum(math.exp(best-costs[i]) for i in ids))
    a=nll(good); universe=nll(range(6))
    return dict(conditional=a-universe,absolute=a,worst_alias=max(costs[i] for i in good),
        negative=nll(bad),accepted_top1=min(range(6),key=costs.__getitem__) in good,
        both_beat_negatives=max(costs[i] for i in good)<min(costs[i] for i in bad))


def audit(folder,pin,trained,trpin,scored,scpin,out):
    fx=obj(folder/"fixture.json",pin); tr=obj(trained/"training.json",trpin); sc=obj(scored/"scoring.json",scpin)
    truth=obj(folder/"truth.json",fx["files"]["truth.json"])
    require(sc["training"]==trpin and sc["fixture"]==tr["fixture"]==pin and len(truth)==12,"lineage")
    data=obj(folder/"train.json",fx["files"]["train.json"]); expected={k:tr["origin"] for k in ARMS}
    steps=[json.loads(s) for s in read(trained/"steps.jsonl",tr["steps"],1000000).splitlines()]
    require(len(steps)==216,"step count")
    for k in range(108):
        a,b=steps[2*k:2*k+2]; name=ARMS[k%3]; step=k//3; meaning=data[step%18]["meaning"]
        require((a["event"],b["event"],a["step"],b["step"],a["arm"],b["arm"],a["pre"],b["pre"])==
            ("start","return",step,step,name,name,expected[name],expected[name]) and
            b["accepted"]==[meaning,meaning+3] and b["gold"]==b["accepted"][((step%18)//3+step//18)%2],"step chain")
        require(b["radius"]==steps[6*step+1]["radius"] and
            (b["match"] is None if name==ARMS[0] else abs(b["match"]["realized_l2"]/b["radius"]-1)<1e-4),"step distance")
        expected[name]=b["post"]
    require(expected=={k:"sha256:"+v for k,v in tr["endpoints"].items()},"endpoints")
    rows=[json.loads(s) for s in read(scored/"scores.jsonl",sc["rows"],1000000).splitlines()]
    require(len(rows)==96,"score count"); cells={}
    for a,b in zip(rows[::2],rows[1::2],strict=True):
        key=(b["arm"],b["slot"])
        require((a["event"],b["event"])==("start","return") and (a["arm"],a["slot"])==key and key not in cells
                and len(b["costs"])==6 and all(math.isfinite(x) for x in b["costs"]),"score ledger")
        require(b["owner"]==({"initial":tr["origin"],**expected})[key[0]],"score owner")
        require(abs(metrics(b["costs"],0)["conditional"]-b["probe"]["conditional_nll"])<2e-5,"fixed probe")
        cells[key]=metrics(b["costs"],truth[key[1]]["meaning"])
    summary={}
    for name in ("initial",*ARMS):
        vs=[cells[name,i] for i in range(12)]
        summary[name]={k:(sum(v[k] for v in vs) if isinstance(vs[0][k],bool) else math.fsum(v[k] for v in vs)/len(vs)) for k in vs[0]}
    s=summary["subset_rank"]
    gates={k:(all(s[m]<summary[k][m]-1e-6 for m in ("conditional","absolute")) and
              s["worst_alias"]<=summary[k]["worst_alias"]+1e-6 and
              all(s[m]>=summary[k][m] for m in ("accepted_top1","both_beat_negatives"))) for k in ("initial","rotating_gold")}
    out.mkdir(mode=0o700)
    return put(out/"results.json",dict(fixture=pin,training=trpin,scoring=scpin,results=summary,
        gates=gates,gate_pass=all(gates.values()),per_query=[dict(arm=k,slot=i,**v) for (k,i),v in cells.items()]))


if __name__=="__main__":
    command,*args=sys.argv[1:]
    if command not in {"prepare","train","score","audit"}: raise SystemExit("prepare|train|score|audit")
    print(globals()[command](*(Path(v) if i%2==0 else v for i,v in enumerate(args))))

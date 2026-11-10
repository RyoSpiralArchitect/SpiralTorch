"""Model-free readback of the matched synthetic language study. No scoring or update callback."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

if __name__ == "__main__":
    # A readback worker may open these pinned source files without permission
    # to enumerate their parent directory (which a sys.path importer needs).
    for name in ("field_candidates", "field_completions", "completion_evidence", "projection_language_fixture"):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from field_completions import fit_menus
from completion_evidence import rebuild
from projection_language_fixture import ARMS, V, STOP, pack, exclusive

PIN = "f0177d9c3c980737f35c9a1694bae9553fef9c319b72bedc0dc767979c2a753e"
LIMIT = 1e-4


def check(c, msg):
    if not c:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def raw(p, pin, cap):
    with p.open("rb") as f:
        b = f.read(cap+1)
    check(len(b)<=cap and sha(b)==pin, "wrong bounded source: "+p.name)
    return b


def get(p, pin, cap=65536):
    return json.loads(raw(p,pin,cap))


def f32(x):
    return struct.unpack("<f",struct.pack("<f",x))[0]


def decode(bits, size):
    check(len(bits)==size and all(type(x) is int and 0<=x<2**32 for x in bits), "bad f32 storage")
    values = struct.unpack(f"<{size}f", struct.pack(f"<{size}I", *bits))
    check(all(map(math.isfinite, values)), "nonfinite observation")
    return values


def close(a,b):
    return math.isclose(a,b,rel_tol=3e-12,abs_tol=1e-13)


def dot(xs,ys):
    return math.fsum(x*y for x,y in zip(xs,ys,strict=True))


def length(xs):
    return math.sqrt(dot(xs,xs))


def nll(row, token):
    a=max(row);i=row.index(a)
    return (a-row[token])+math.log1p(math.fsum(math.exp(x-a) for j,x in enumerate(row) if j!=i))


def objective(bits, answer, rival):
    labels=[*answer,STOP]
    check(len(bits)==len(labels), "wrong supervised row count")
    rows=[decode(b,V) for b in bits]
    full=[nll(r,t) for r,t in zip(rows,labels)]
    d=rows[0][rival]-rows[0][labels[0]]
    binary=max(d,0.)+math.log1p(math.exp(-abs(d)))
    return f32(math.fsum([.5*full[0]+.5*binary,*full[1:]])/len(labels)), full, binary


def displacement(before, gs, weights, goal):
    combined=[f32(weights[0]*a+weights[1]*b) for a,b in zip(*gs,strict=True)]
    proposal=[f32(x-f32(f32(.035)*g)) for x,g in zip(before,combined)]
    d=[y-x for x,y in zip(before,proposal)]; raw_len=length(d)
    if goal is not None and raw_len:
        after=[f32(x+v*(goal/raw_len)) if v else x for x,v in zip(before,d)]
    else:
        after=proposal
    delta=[y-x for y,x in zip(after,before)]
    magnitude=length(delta);dots=[dot(g,delta) for g in gs]
    both=all(v < -1e-12*length(g)*magnitude for v,g in zip(dots,gs))
    if goal is None:
        err=0.
    elif raw_len:
        err=length([a/goal-b/raw_len for a,b in zip(delta,d)])
    else:
        err=math.inf
    representable=goal is None or (err<=LIMIT and abs(magnitude/goal-1.)<=LIMIT)
    return after,dict(l2=magnitude,dots=dots,both=both,error=err,representable=representable)


def checked_move(report, actual):
    check(close(report["l2"],actual["l2"]) and report["both_descend"]==actual["both"] and
          all(close(x,y) for x,y in zip(report["loss_dots"],actual["dots"],strict=True)), "rounded movement mismatch")


def geometry(t, name, owner):
    check(owner.startswith("sha256:"), "unbound artifact")
    r=get(t/(name+".geometry.json"),owner[7:])
    p,=r["placements"]
    check(p["path"]=="transformer.h.1.attn","wrong site")
    params={}
    for tensor in p["tensors"]:
        if tensor["role"]=="parameter":
            check(tensor["requires_grad"] and tensor["shape"]==[32] and len(tensor["values"])==32,"wrong mutable coordinates")
            params[tensor["name"]]=list(map(f32,tensor["values"]))
            tensor["values"]=["trainable",32]
        else:
            check(tensor["role"]=="buffer" and not tensor["requires_grad"],"mutable anchor")
    check(set(params)=={"bias","gate"}, "wrong trainable names")
    return params["bias"]+params["gate"],sha(pack(r))


def typed_pair(it, kind, num, who):
    be=next(it);ret=next(it)
    ident=dict(kind=kind,n=num,who=who)
    check(be==dict(**ident,event="start") and ret["event"]=="return" and all(ret[k]==v for k,v in ident.items()),"misordered "+kind)
    return ret


def read_train(t,tr,pairs):
    events=iter(json.loads(x) for x in raw(t/"events.jsonl",tr["journal"],8<<20).splitlines())
    init=[geometry(t,k+"_start",pin) for k,pin in zip(("absolute","anchored"),tr["origins"],strict=True)]
    current={a:(init[j%2][0],tr["origins"][j%2]) for j,a in enumerate(ARMS)}
    history={a:[] for a in ARMS};stops={};n=u=probes=pre_exact=0;initial={}
    for step,p in enumerate(pairs,1):
        radius=None
        for j,a in enumerate(ARMS):
            if a in stops:
                continue
            v0,owner=current[a];who=dict(arm=a,step=step,pre=owner);gs=[];link=[]
            for side,(s,right) in enumerate(((p["short"],p["long"]),(p["long"],p["short"]))):
                sw=dict(**who,side=side);n+=1;r=typed_pair(events,"gradient",n,sw)
                g=decode(r["bits"],64);gs.append(g);rep=r["report"];probes+=1;pre=typed_pair(events,"pre_loss",probes,sw)
                loss,ces,bin_loss=objective(pre["row_bits"],s["answer"],right["answer"][0])
                check(pre["owner"]==owner and loss==f32(pre["loss"]) and abs(loss-rep["loss"])<=2e-6 and
                      rep["target_tokens"]==len(ces) and rep["contrasted_tokens"]==rep["rival_ids"]==1 and
                      rep["mix"]==.5 and abs(rep["full_ce"]-sum(ces)/len(ces))<=2e-6 and abs(bin_loss-rep["declared_ce"])<=2e-6,
                      "different pre objective or mask")
                pre_exact+=int(loss==f32(rep["loss"]));link.append((s,right,sw,loss))
                if step==1:
                    witness=(r["bits"],rep,pre["row_bits"])
                    check(initial.setdefault(side,witness)==witness,"different chart origin logits, report or gradient")
            base=next(events);r=next(events)
            check(base==dict(event="start",kind="update",who=who,requested=radius) and r["kind"]=="update" and r["who"]==who,"misbound step")
            eqv,eq=displacement(v0,gs,[.5,.5],radius)
            reweight=j>=2 and not eq["both"]
            norms=list(map(length,gs));weights=[.5,.5]
            if reweight and all(norms):
                weights=[norms[1]/sum(norms),norms[0]/sum(norms)]
                end,applied=displacement(v0,gs,weights,radius)
            else:
                end,applied=eqv,eq
            if r["event"]=="refused":
                check(j!=0 and (not eq["representable"] or not applied["representable"] or
                      (reweight and not applied["both"])), "refusal without numerical witness")
                stops[a]=dict(step=step,reason=r["reason"]);continue
            check(r["event"]=="return" and (eq["representable"] or (reweight and not eq["l2"])) and applied["representable"],"bad accepted step")
            receipt=r["receipt"];new_owner=receipt["post_geometry_id"];v,layout=geometry(t,f"{a}-{step:02}",new_owner)
            check(v==end and layout==init[j%2][1] and receipt["pre_geometry_id"]==owner and receipt["base_model_identity"]==tr["base"]
                  and receipt["lora_artifact_id"]==tr["lora"] and receipt["readout_accumulator"]=="f32_wide_backward"
                  and receipt["policy"]==("equal" if j<2 else "common_descent") and f32(receipt["learning_rate"])==f32(.035)
                  and receipt["trainable_values"]==64 and receipt["reweighted"]==reweight and
                  all(close(x,y) for x,y in zip(weights,receipt["weights"],strict=True)) and
                  all(close(x,y) for x,y in zip(norms,receipt["gradient_l2"],strict=True)),"not the paired proposal")
            checked_move(receipt["equal"],eq);checked_move(receipt["applied"],applied)
            if j==0:
                check(receipt["distance_match"] is None and applied["l2"]>0.,"no reference")
                radius=r["reference"]
                check(close(radius,applied["l2"]),"reference radius drift")
            else:
                m=receipt["distance_match"]
                check(m["requested_l2"]==r["reference"]==radius and m["origin_id"]==owner and m["result_id"]==new_owner and
                      close(m["relative_vector_error"],applied["error"]) and close(m["realized_l2"],applied["l2"]),"matching receipt drift")
            deltas=[]
            for s,right,sw,pre_loss in link:
                probes+=1;post=typed_pair(events,"post_loss",probes,sw);loss,_,_=objective(post["row_bits"],s["answer"],right["answer"][0])
                check(post["owner"]==new_owner and f32(post["loss"])==loss,"different post objective")
                deltas.append(loss-pre_loss)
            history[a].append(dict(step=step,field=p["field"],reweighted=reweight,delta=deltas,both=max(deltas)<-1e-6,
                any_harm=max(deltas)>1e-6,first_order_both=applied["both"],length=applied["l2"],direction_error=applied["error"]))
            current[a]=(v,new_owner);u+=1
    check(next(events,None) is None and (n,u,probes)==(tr["gradients"],tr["updates"],tr["measured_forwards"]) and
          (n,u,probes)==(2*(u+len(stops)),sum(map(len,history.values())),n+2*u) and u<=72,"training counts")
    check(len(tr["endpoints"])==4,"arm count")
    sums={}
    for j,(a,idx) in enumerate(zip(ARMS,tr["endpoints"],strict=True)):
        rr=history[a];v,owner=current[a]
        check(idx==dict(arm=a,steps=len(rr),owner=owner,refusal=stops[a]["reason"] if a in stops else None)
              and len(rr)==(stops[a]["step"]-1 if a in stops else 18),"endpoint mismatch")
        sums[a]=dict(steps=len(rr),reweighted=sum(x["reweighted"] for x in rr),both=sum(x["both"] for x in rr),
            any_harm=sum(x["any_harm"] for x in rr),first_order_both=sum(x["first_order_both"] for x in rr),
            path=sum(x["length"] for x in rr),end_l2=length([x-y for x,y in zip(v,init[j%2][0])]))
    return history,sums,stops,pre_exact


def read_scores(s,sc,tr,menus,qs):
    expected=[dict(state=k,owner=v) for k,v in zip(("absolute_start","anchored_start"),tr["origins"])]
    expected.extend(dict(state=x["arm"],owner=x["owner"]) for x in tr["endpoints"] if x["steps"]==18)
    check(sc["owners"]==expected and sc["documents"]==108,"wrong scoring roster")
    native=get(s/"by-answer.json",sc["native"],1<<20);spans=native["answers"];by=native["comparison"]["evaluations"]
    check(len(spans)==108 and len(by)==len(expected) and all(x["base_model_identity"]==tr["base"] and x["lora_artifact_id"]==tr["lora"]
        and x["geometry_artifact_id"]==m["owner"] and x["readout_accumulator"]=="f32_wide_backward" for x,m in zip(by,expected)),"native score owners")
    events=iter(json.loads(x) for x in raw(s/"calls.jsonl",sc["journal"],4<<20).splitlines())
    check(next(events)==dict(event="start",kind="by_answer",documents=108,owners=len(expected)) and
        next(events)==dict(event="return",kind="by_answer",sha=sc["native"]),"native scoring call")
    ledger=iter(json.loads(x) for x in raw(s/"lines.jsonl",sc["lines"],1<<20).splitlines())
    ranked=[];nr=nl=0;max_err=0.
    for slot,q in enumerate(qs):
        check(q["slot"]==slot and set(q)=={"slot","field","prompt"},"query contains a label")
        menu=menus[q["field"]];need=sorted({c[:k] for c in menu.choices for k in range(len(c)+1)})
        check(len(need)==10 and len(menu.choices)==6,"menu denominator");origin_rows={}
        for j,m in enumerate(expected):
            costs={}
            for part in need:
                nr+=1;who=dict(**m,slot=slot,prefix=list(part),n=nr);b,e=next(events),next(events)
                check(b==dict(event="start",kind="prefix",who=who) and e["who"]==who and e["event"]=="return" and e["kind"]=="prefix","scoring chronology")
                if j<2:
                    check(origin_rows.setdefault(part,e["row_bits"])==e["row_bits"],"zero-chart readout differs")
                row=decode(e["row_bits"],V)
                targets=menu.first_ids if not part else [t for pr,t in menu.edges if pr==part]
                costs[part]={t:nll(row,t) for t in targets}
            lines=rebuild(menu,costs[()],[(pr,t,costs[pr][t]) for pr,t in menu.edges])
            rec=next(ledger);check((rec["slot"],rec["state"])==(slot,m["state"]),"wrong line owner")
            check(tuple(tuple(l["content"]) for l in rec["lines"])==menu.choices,"line choice mapping")
            lmap={l.content:l for l in lines};native_order=[]
            for k,content in enumerate(menu.choices):
                e=rec["lines"][k];line=lmap[content];doc=slot*6+k;span=spans[doc]
                check((span["document"],span["start"],span["end"])==(doc,len(q["prompt"]),len(q["prompt"])+len(content)+1) and
                    len(span["scores"])==len(expected),"wrong answer or newline span")
                independent=line.total;nt=span["scores"][j]["nll"]*len(line.parts);nl+=1
                check(close(independent,e["line"]) and close(nt,e["native_line"]) and all(close(x,y) for x,y in zip(line.parts,e["parts"],strict=True)),
                    "native candidate totals or prefix decomposition differ")
                max_err=max(max_err,abs(independent-nt));native_order.append((nt,content))
            order=[x.content for x in lines]
            check([v for _,v in sorted(native_order)]==order,"rank differs between native answer API and prefix oracle")
            ranked.append(dict(state=m["state"],slot=slot,order=order,first=min(lines,key=lambda x:(x.parts[0],x.content)).content,
                costs={x.content:x.total for x in lines}))
    check(next(ledger,None) is None and next(events,None) is None and nr==sc["prefix_rows"]==180*len(expected) and nl==sc["line_cells"]==108*len(expected)
          and max_err<=2e-5 and abs(max_err-sc["max_abs_difference"])<1e-11,"incomplete score coverage")
    return ranked,max_err


def audit(fixture, trained, scored, out):
    manifest=get(fixture/"fixture.json",PIN);pins=manifest["files"]
    pairs=get(fixture/"train.json",pins["train.json"]);qs=get(fixture/"queries.json",pins["queries.json"])
    check(len(pairs)==18 and len(qs)==18 and len({tuple(p[s]["prompt"]) for p in pairs for s in ("short","long")}
          | {tuple(q["prompt"]) for q in qs})==54,"wrong or overlapping prompts")
    menus=fit_menus(((p["field"],tuple(p[s]["answer"])) for p in pairs for s in ("short","long")),V,STOP)
    sb=(scored/"scoring.json").read_bytes();sc=json.loads(sb);tr=get(trained/"training.json",sc["training"])
    check(tr["fixture"]==sc["fixture"]==PIN and get(trained/"menus.json",tr["menus"])=={f:list(map(list,m.choices)) for f,m in menus.items()},
          "menu isn't train-only")
    history,training,stops,exact=read_train(trained,tr,pairs);ranked,err=read_scores(scored,sc,tr,menus,qs)
    # All line decompositions and ranks are reconstructed before consulting the answers.
    truth=get(fixture/"truth.json",pins["truth.json"]);check(len(truth)==18,"wrong labels")
    joined=[]
    for r in ranked:
        i=r["slot"];g=truth[i];gold=tuple(g["answer"]);check(g["slot"]==i and gold in r["order"],"bad join")
        joined.append(dict(state=r["state"],slot=i,field=qs[i]["field"],gold=gold,winner=r["order"][0],rank=r["order"].index(gold)+1,
             first=r["first"]==gold,target_line_nll=r["costs"][gold],
             margin=min(v for k,v in r["costs"].items() if k!=gold)-r["costs"][gold]))
    quality={}
    for m in sc["owners"]:
        rr=[r for r in joined if r["state"]==m["state"]];check(len(rr)==18,"wrong endpoint denominator")
        quality[m["state"]]=dict(full=sum(r["rank"]==1 for r in rr),single=sum(r["rank"]==1 and len(r["gold"])==1 for r in rr),
            first=sum(r["first"] for r in rr),mean_gold_line_nll=math.fsum(r["target_line_nll"] for r in rr)/18)
    gates={};changes={}
    for chart in ("absolute","anchored"):
        a,b=chart+"_equal",chart+"_common"
        if a in stops or b in stops:
            gates[chart]=dict(mechanism=None,quality=None);continue
        ta,tb=training[a],training[b];qa,qb=quality[a],quality[b]
        gates[chart]=dict(mechanism=tb["both"]>ta["both"] and tb["any_harm"]<=ta["any_harm"],
                          quality=qb["full"]>qa["full"] and qb["single"]>=qa["single"])
        aa=[x for x in joined if x["state"]==a];bb=[x for x in joined if x["state"]==b]
        changes[chart]=dict(fixed=sum(x["rank"]!=1 and y["rank"]==1 for x,y in zip(aa,bb)),
            lost=sum(x["rank"]==1 and y["rank"]!=1 for x,y in zip(aa,bb)),
            changed_winner=sum(x["winner"]!=y["winner"] for x,y in zip(aa,bb)))
    out.mkdir(mode=0o700)
    r=dict(study="seeded synthetic templates, not pretrained language quality",fixture=PIN,training_index=sc["training"],
        scoring_index=sha(sb),gradient_calls=tr["gradients"],updates=tr["updates"],train_forwards=tr["measured_forwards"],pre_losses_bit_exact=exact,
        prefix_forwards=sc["prefix_rows"],line_cells=sc["line_cells"],line_max_abs_error=err,
        train=training,scores=quality,stops=stops,gates=gates,compare=changes,
        steps=exclusive(out/"steps.json",pack(history)),joined=exclusive(out/"joined.json",pack(joined)))
    return exclusive(out/"summary.json",pack(r))


if __name__=="__main__":
    if len(sys.argv)!=5:
        raise SystemExit("usage: projection_language_audit.py FIXTURE TRAINED SCORED NEW_DIR")
    print(audit(*(Path(s) for s in sys.argv[1:])))

"""Independent, model-free readback for shared_prefix_study. No gold is used to rank candidates."""
import importlib.util
import json
import math
from pathlib import Path
import sys

if __name__=="__main__":
    for name in ("field_candidates","field_completions","completion_evidence","projection_language_fixture",
                 "projection_language_audit","shared_prefix_study"):
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+".py"))
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)

from projection_language_audit import close, decode, f32, geometry, length, nll
from shared_prefix_study import ARMS, RATE, STOP, VOCAB, fit, obj, put, read, require


def choice(costs,gold):
    winner=min(range(len(costs)),key=costs.__getitem__)
    best=costs[winner]
    tail=math.fsum(math.exp(best-x) for i,x in enumerate(costs) if i!=winner)
    return winner,(costs[gold]-best)+math.log1p(tail),[math.exp(best-x)/(1+tail) for x in costs]


def report(r,menu,gold,mix):
    require((r["gold_index"],r["mix"],r["stop_id"],r["shared_content_tokens"])==(gold,mix,STOP,1) and
            r["target_counts"]==[len(c)+1 for c in menu],"different loss domain")
    full=r["line_nlls"];suffix=r["suffix_nlls"]
    require(len(full)==len(suffix)==len(menu) and all(math.isfinite(x) and x>=0 for x in full+suffix),
            "invalid complete-line costs")
    shared=full[0]-suffix[0]
    require(all(abs(x-y-shared)<2e-6 for x,y in zip(full,suffix)),"inconsistent shared prefix")
    winner,conditional,probs=choice(suffix,gold)
    ce=full[gold]/r["target_counts"][gold]
    require(r["winner_index"]==winner and close(conditional,r["menu_nll"]) and close(ce,r["gold_mean_nll"]) and
            all(close(x,y) for x,y in zip(probs,r["menu_probabilities"],strict=True)) and
            close(r["rival_mass"],math.fsum(p for i,p in enumerate(probs) if i!=gold)) and
            f32((1-mix)*ce+mix*conditional)==f32(r["loss"]),"incorrect menu normalization or objective")


def read_training(out,tr,samples,menu):
    it=iter(json.loads(x) for x in read(out/"steps.jsonl",tr["journal"]).splitlines())
    origin,layout=geometry(out,"start",tr["origin"])
    prior={a:(origin,tr["origin"]) for a in ARMS};obs={a:[] for a in ARMS}
    for i,s in enumerate(samples,1):
        ref=None;initial=None
        for a,mix in ARMS.items():
            b,e=next(it),next(it);v0,owner=prior[a];ident=dict(step=i,arm=a,pre=owner)
            require(b==dict(event="start",**ident) and e["event"]=="return" and
                    all(e[k]==v for k,v in ident.items()),"misbound update")
            gold=menu.index(s["answer"])
            before,after=e["before"],e["after"];report(before,menu,gold,mix);report(after,menu,gold,mix)
            if i==1:
                if initial is None:
                    initial=before["line_nlls"]
                require(initial==before["line_nlls"],"different starting scores")
            g=decode(e["gradient_bits"],64);direction=[f32(x-f32(f32(RATE)*y)) for x,y in zip(v0,g)]
            delta=[y-x for y,x in zip(direction,v0)];d=length(delta)
            v,layout2=geometry(out,f"{a}-{i}",e["post"])
            require(layout2==layout and d>0,"changed base, anchor, layout or zero direction")
            actual=length([y-x for x,y in zip(v0,v)])
            if ref is None:
                ref=e["reference"]
                require(e["match"] is None and close(actual,ref) and v==direction,"invalid CE reference")
            else:
                m=e["match"];expected=[f32(x+y*(ref/d)) if y else x for x,y in zip(v0,delta)]
                error=length([(y-x)/ref-z/d for x,y,z in zip(v0,v,delta)])
                require(e["reference"]==m["requested_l2"]==ref and v==expected and m["trainable_values"]==64
                        and (m["origin_id"],m["result_id"])==(owner,e["post"]) and close(actual,m["realized_l2"])
                        and close(error,m["relative_vector_error"]) and error<=1e-4 and abs(actual/ref-1.)<=1e-4,
                        "different matched endpoint")
            obs[a].append(dict(step=i,gold_index=gold,length=actual,objective_delta=after["loss"]-before["loss"],
                               conditional_delta=after["menu_nll"]-before["menu_nll"],gold_ce_delta=after["gold_mean_nll"]-before["gold_mean_nll"]))
            prior[a]=(v,e["post"])
    require(next(it,None) is None and len(samples)==12 and all(prior[a][1]==tr["endpoints"][a] for a in ARMS)
            and (tr["updates"],tr["gradients"],tr["objective_forwards"])==(24,24,48),"incomplete training")
    return obs


def read_scores(out,sc,tr,qs,menu):
    states={"start":tr["origin"],**tr["endpoints"]}
    require(states==sc["owners"],"incorrect scoring owner")
    it=iter(json.loads(x) for x in read(out/"rows.jsonl",sc["journal"],1<<20).splitlines())
    ranked=[];largest=0.;nr=0
    for i,q in enumerate(qs):
        require(set(q)=={"slot","prompt"} and q["slot"]==i,"not a blind query")
        for a,owner in states.items():
            b,e=next(it),next(it);ident=dict(state=a,owner=owner,slot=i)
            require(b==dict(event="start",**ident) and e["event"]=="return" and
                    all(e[k]==v for k,v in ident.items()),"misbound evaluation")
            parts=sorted({tuple(c[:j]) for c in menu for j in range(len(c)+1)})
            require([tuple(x["prefix"]) for x in e["rows"]]==parts,"missing prefix trace")
            logits={p:decode(x["bits"],VOCAB) for p,x in zip(parts,e["rows"])};nr+=len(parts)
            per_candidate=[[nll(logits[tuple(c[:j])],v) for j,v in enumerate(c+[STOP])] for c in menu]
            costs=[math.fsum(x) for x in per_candidate];suffix=[math.fsum(x[1:]) for x in per_candidate]
            r=e["report"];report(r,menu,0,1.)
            largest=max(largest,max(abs(x-y) for x,y in zip(costs,r["line_nlls"])),
                        max(abs(x-y) for x,y in zip(suffix,r["suffix_nlls"])))
            order=sorted(range(len(menu)),key=suffix.__getitem__)
            require(largest<=2e-5 and order==sorted(range(len(menu)),key=r["suffix_nlls"].__getitem__) and
                    all(x[0]==per_candidate[0][0] for x in per_candidate),"prefix oracle and native ranking differ")
            ranked.append(dict(slot=i,state=a,order=order,costs=costs,suffix=suffix,parts=per_candidate))
    require(next(it,None) is None and nr==sc["prefix_forwards"]==144 and len(ranked)==sc["reports"]==36,"incomplete endpoint scoring")
    return ranked,largest


def audit(f,pin,t,tp,s,sp,out):
    fx=obj(f/"fixture.json",pin);p=fx["files"];tr=obj(t/"training.json",tp);sc=obj(s/"scoring.json",sp)
    require(tr["fixture"]==sc["fixture"]==pin and sc["training"]==tp,"broken evidence chain")
    samples=obj(f/"train.json",p["train"]);qs=obj(f/"queries.json",p["queries"]);menu=fit(samples)
    require(menu==obj(t/"menu.json",tr["menu"]) and len({tuple(x["prompt"]) for x in samples+qs})==24,
            "different support or overlapping prompts")
    steps=read_training(t,tr,samples,menu);ranked,err=read_scores(s,sc,tr,qs,menu)
    truth=obj(f/"truth.json",p["truth"]);joined=[]
    require(len(truth)==len(qs)==12,"wrong evaluation count")
    for r in ranked:
        key=truth[r["slot"]];gold=menu.index(key["answer"]);require(key["slot"]==r["slot"],"wrong key")
        _,loss,_=choice(r["suffix"],gold)
        joined.append(dict(slot=r["slot"],state=r["state"],gold=gold,winner=r["order"][0],
            rank=r["order"].index(gold)+1,conditional_nll=loss,gold_line_nll=r["costs"][gold],
            gold_margin=min(v for j,v in enumerate(r["suffix"]) if j!=gold)-r["suffix"][gold]))
    summary={}
    for a in ("start",*ARMS):
        rr=[r for r in joined if r["state"]==a]
        require(len(rr)==12 and all(sum(r["gold"]==i for r in rr)==4 for i in range(3)),"wrong denominators")
        summary[a]=dict(correct=sum(x["rank"]==1 for x in rr),short=sum(x["rank"]==1 and x["gold"]==0 for x in rr),
            conditional_nll=math.fsum(x["conditional_nll"] for x in rr)/12,gold_line_nll=math.fsum(x["gold_line_nll"] for x in rr)/12)
    ce,rank=summary["gold_ce"],summary["line_mix"]
    out.mkdir(mode=0o700)
    return put(out/"summary.json",dict(fixture=pin,training=tp,scoring=sp,steps=put(out/"steps.json",steps),
        joined=put(out/"joined.json",joined),scores=summary,max_line_error=err,updates=24,gradient_calls=24,train_reports=48,
        eval_reports=36,eval_prefix_rows=144,complete_line_cells=108,
        lower_conditional_nll=rank["conditional_nll"]<ce["conditional_nll"],
        improved_decisions=rank["correct"]>ce["correct"] and rank["short"]>=ce["short"],
        limitation="One small synthetic model, one order, same label concepts in another prompt template. No free generation or pretrained quality test."))


if __name__=="__main__":
    args=sys.argv[1:]
    if len(args)!=7:
        raise SystemExit("FIXTURE FIXTURE_PIN TRAIN TRAIN_PIN SCORES SCORE_PIN NEW_DIR")
    print(audit(Path(args[0]),args[1],Path(args[2]),args[3],Path(args[4]),args[5],Path(args[6])))

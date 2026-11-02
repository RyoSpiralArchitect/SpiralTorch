"""Independent file-only coverage, checkpoint lineage, text scoring and bounded public receipt export."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as st  # noqa: E402
from endpoint_check import EndpointPlan  # noqa: E402
d,r=st.d,st.readout
check,read,sha=d.check,d.old.read,d.old.sha
COMMIT="ab63b0e3f64412b72814b8c32c89aebda9891964"
INPUT_PIN="72151805040cddbaa9d2149abc90ba5d424da8513eaa62fe2ca240363c7843c9"
POINTS=("initial",*d.ORDERS)
PREP_RECEIPTS=("manifest.json","prepared.json","prepare-attempt.json","prepare-launch.json","prepare-worker.json")
NAMES=("manifest.json","prepared.json","prepare-attempt.json","prepare-launch.json","prepare-worker.json",
       "attempt.json","fit-launch.json","fit-worker.json","grade-launch.json","grade-worker.json",
       "owners.json","fit.json","result.json","complete.json",
       *(f"{p}/{f}" for p in POINTS for f in ("scores.json","generated.jsonl","lines.jsonl")),
       *(f"{p}/{f}" for p in d.ORDERS for f in ("order.json","training.json","first-step.json","steps.jsonl","checkpoints.json")))


def verify_inputs(root, originals=None, tok=None):
    m=read(root/"manifest.json")
    rows=d.records()
    paths=[d.HERE/n for n in ("run.py","design.py","trial.py","readout.py")]+[d.BUILD,d.PREV/"protocol.py",d.p.PILOT,
           *[d.old.ROOT/"tools"/n for n in ("native_answer_generation.py","byte_release_artifacts.py","owned_process.py")]]
    prepared,attempt,launch,worker=(read(root/n) for n in PREP_RECEIPTS[1:])
    check(sha(root/"manifest.json")==INPUT_PIN==prepared["manifest_sha256"]
          and m["commit"]==COMMIT and d.pin(m["frozen"])==st.SPEC_SHA and m["frozen"]==d.freeze(rows)
          and m["source_sha256"]=={str(f.relative_to(d.old.ROOT)):sha(f) for f in paths} and m["tokenizer"]==d.old.FILES["tokenizer.json"]
          and m["prepare_attempt"]==sha(root/"prepare-attempt.json"),"input freeze or source")
    check(attempt["retries"]==0 and attempt["commit"]==COMMIT
          and 0<prepared["cpu_seconds"]<120 and 0<prepared["wall_seconds"]<120
          and attempt["started_epoch"]<=launch["started_epoch"]<=worker["started_epoch"]<launch["deadline"]
          and worker["cpu_limit"]==launch["cpu_limit"]<=120,"preparation budget")
    ident=m["installed"]
    check(ident["wheel_sha256"]==d.WHEEL and ident["native_sha256"]==d.NATIVE and ident["all_original_files_match"]
          and set(m["files"])==set(st.FILES),"installed package or corpus inventory")
    prompts=None
    if originals is not None:
        check(all(sha(originals/f)==h for f,h in m["files"].items()),"private token bytes")
        prompts=read(originals/"prompts.json")
    for split,total in (("train",d.TRAIN),("eval",d.EVAL)):
        stats=m["lengths"][split]
        check(len(stats)==total and [v[0] for v in stats]==[row["key"] for row in rows[split]]
              and all(0<a<b<=64 and b-a==n for _,b,a,n in stats),"bad target span")
        if originals is not None:
            data=read(originals/("train.json" if split=="train" else "eval-content.json"))
            check(data["keys"]==[v[0] for v in stats] and (split!="eval" or prompts["keys"]==data["keys"]),"token key join")
            for i,((_,b,a,n),doc,mask) in enumerate(zip(stats,data["token_documents"],data["labels"],strict=True)):
                check(len(doc)==b and mask==[-100]*a+doc[a:] and
                      (doc[-1]==d.NL and d.NL not in doc[a:-1] if split=="train" else d.NL not in doc[a:] and prompts["prompt_ids"][i]==doc[:a]),
                      "content mask, delimiter or prompt-only input")
                if tok is not None:
                    src=rows[split][i]
                    tail=src["answer"]+("\n" if split=="train" else "")
                    check(tok.encode(src["prompt"],add_special_tokens=False).ids==doc[:a] and
                          tok.decode(doc[a:],skip_special_tokens=False)==tail and
                          tok.encode(src["prompt"]+tail,add_special_tokens=False).ids==doc,"independent tokenization")
    check(sum(x[3] for x in m["lengths"]["train"])==305 and sum(x[3] for x in m["lengths"]["eval"])==111,
          "training or evaluation denominator")
    return m,rows


def checkpoint(path,entry):
    ck=read(path)
    check("sha256:"+sha(path)==entry["checkpoint_id"] and [ck["lora_id"],ck["geometry_id"]]==entry["owners"]
          and ck["data_id"]==entry["data_id"] and ck["updates"]==d.UPDATES
          and ck["config"]==dict(context_length=d.p.CONTEXT,epochs=d.EPOCHS,learning_rate=d.p.LR,mode="joint")
          and ck["data_policy"]==dict(window_policy="covering_targets",has_labels=True)
          and ck["base_model_identity"]==d.old.BASE_ID and ck["readout_accumulator"]=="f32","final checkpoint")
    lo,geo=ck["lora_json"],ck["geometry_json"]
    check(ck["lora_id"]=="sha256:"+hashlib.sha256(b"spiraltorch.native_lora.v1\0"+lo.encode()).hexdigest() and
          ck["geometry_id"]=="sha256:"+hashlib.sha256(geo.encode()).hexdigest(),"inner adapter identity")
    l,rec=json.loads(lo),json.loads(geo)
    check(l["rank"]==2 and l["alpha"]==4 and l["width"]==768 and l["layer"]==11 and len(l["a"])==len(l["b"])==1536
          and rec["base_model_identity"]==l["base_model_identity"]==d.old.BASE_ID and len(rec["placements"])==1,"LoRA shape or base")
    site=rec["placements"][0]
    arm=entry["arm"]
    recipe=site["recipe"]
    check(site["path"]=="transformer.h.11.attn" and recipe["strength"]==.25 and recipe["features"]==768,"site")
    expected={"affine":"spiraltorch.channel_affine_adapter.v1","both":"spiraltorch.wave_gate_adapter.v2"}.get(arm,"spiraltorch.wave_stage_adapter.v1")
    check(recipe["schema"]==expected,"wrong operator")
    if arm in ("saturation","projection"):
        stage=dict(kind="saturation",saturation=1.,porosity=.05) if arm=="saturation" else dict(kind="projection",curvature=-1.,log_radius=0.)
        check(recipe["stage"]==stage,"wrong factor")
    elif arm=="both":
        check(not recipe["learnable_radius"] and recipe["kernel"]==dict(curvature=-1.,porosity=.05,saturation=1.,max_values=65536),
              "wrong full-Wave recipe")
    tensors={t["name"]:t for t in site["tensors"]}
    check(set(tensors)==({"gate","bias","log_radius"} if arm=="both" else {"gate","bias"}),"geometry capacity")
    for k in ("gate","bias"):
        t=tensors[k]
        check(t["role"]=="parameter" and t["requires_grad"] and t["shape"]==[768] and len(t["values"])==768
              and all(map(math.isfinite,t["values"])),"untrained or invalid geometry")
    if arm=="both":
        t=tensors["log_radius"]
        check(t["role"]=="buffer" and not t["requires_grad"] and t["values"]==[0.] and t["shape"]==[],"radius drift")


def training(root,order,origin,train_rows,stats,originals,*,width=d.p.WIDTH):
    loc=root/order
    plan=read(loc/"order.json")
    perm=d.indices(train_rows,order)
    check(plan["owners"]==origin and plan["keys"]==[stats[i][0] for i in perm],"different start or training order")
    previous=origin.copy()
    steps=r.lines(loc/"steps.jsonl")
    check(len(steps)==d.UPDATES,"missing updates")
    epochs=read(loc/"training.json")
    check(len(epochs)==d.EPOCHS,"epoch count")
    for epoch in range(d.EPOCHS):
        losses=[[] for _ in d.ARMS]
        for i,idx in enumerate(perm):
            step=epoch*d.TRAIN+i+1
            group=steps[step-1]
            nt=stats[idx][3]
            check(group["depth"]==d.p.LAYER and len(group["reports"])==len(d.ARMS),"shared prefix or roster")
            for j,x in enumerate(group["reports"]):
                check((x["update"],x["epoch"],x["window_start"],x["window_end"],x["target_tokens"],x["mode"])==(step,epoch,i,i+1,nt,"joint")
                      and [x["pre_lora_artifact_id"],x["pre_geometry_artifact_id"]]==previous[j] and math.isfinite(x["loss"]),"update lineage")
                previous[j]=[x["post_lora_artifact_id"],x["post_geometry_artifact_id"]]
                losses[j].append(x["loss"]*nt)
            if step==1:
                check(len({z["loss"] for z in group["reports"]})==len({v[0] for v in previous})==1,"first-step tangent/LoRA")
        n=sum(v[3] for v in stats)
        check(epochs[epoch]==dict(epoch=epoch,targets=n,nll=[math.fsum(g)/n for g in losses]),"wrong training aggregate")
    first=read(loc/"first-step.json")
    check(set(first)=={"gate","bias"} and all(math.isfinite(x) and x>=0 for x in first.values()),"first-step report")
    for j,(a,e) in enumerate(zip(d.ARMS,read(loc/"checkpoints.json"),strict=True)):
        check(e["arm"]==a and e["data_id"]==plan["data_id"] and e["owners"]==previous[j] and
              e["update"]==d.UPDATES and e["parameters"]==2*width,"wrong boundary")
        if originals is not None:
            checkpoint(originals/order/(a+".ck"),e)
    return previous,plan["data_id"]


def verified_endpoint(loc, rows, plan, owners, *, base_id, layer, stops, vocabulary, cap, decode=None, initial=False):
    raw=read(loc/"scores.json")["scores"]
    plan.verify_scores(raw,len(d.ARMS))
    check(raw["comparison"]["shared_prefix_layers"]==layer,"wrong scoring prefix")
    ev=raw["comparison"]["evaluations"]
    for e,ids in zip(ev,owners,strict=True):
        check([e["lora_artifact_id"],e["geometry_artifact_id"]]==ids and e["base_model_identity"]==base_id
              and e["readout_accumulator"]=="f32","evaluation ownership or accumulator")
    gens=r.lines(loc/"generated.jsonl")
    scored=r.lines(loc/"lines.jsonl")
    n=len(rows)
    check(n==len(plan.stats) and len(gens)==len(d.ARMS)*n,"endpoint cardinality")
    for g,s in zip(gens,scored,strict=True):
        ids=g["generated_ids"]
        check(0<len(ids)<=cap and all(type(v) is int and 0<=v<vocabulary for v in ids)
              and not stops.intersection(ids[:-1]) and g["finish_reason"]==("eos" if ids[-1] in stops else "length")
              and (ids[-1] in stops or len(ids)==cap),"invalid prompt-only generation boundary")
        if decode is not None:
            check(decode(ids)==s["generated_text"],"wrong decoded generation text")
    if initial:
        check(all(a["scores"]==[a["scores"][0]]*len(d.ARMS) for a in raw["answers"]) and all(
              gens[i]["generated_ids"]==gens[j*n+i]["generated_ids"] for i in range(n) for j in range(1,len(d.ARMS))),
              "initial model behavior differs")
    return r.describe(loc,rows), (ev[0]["data_id"],plan.plan_id), len(gens)


def verify_fit_budget(root,prepared,attempt,end):
    start,deadline=attempt["started_epoch"],attempt["deadline"]
    limit=st.LIMIT
    cpu0,wall0=prepared["cpu_seconds"],prepared["wall_seconds"]
    check(attempt["retries"]==0 and attempt["limit_seconds"]==limit and
          deadline==start+limit-max(cpu0,wall0)-1 and end["exit_code"]==0 and
          start<=end["finished_epoch"]<=deadline and cpu0<=end["cpu_seconds"]<limit and
          wall0<=end["active_wall_seconds"]<limit and math.isclose(end["finished_epoch"]-start+wall0,
          end["active_wall_seconds"],rel_tol=0,abs_tol=.1),"invalid attempt budget")
    prior_cpu,prior_started=cpu0,start
    for phase in ("fit","grade"):
        launch,worker=(read(root/(phase+"-"+k+".json")) for k in ("launch","worker"))
        cap=launch["cpu_limit"]
        check(type(cap) is int and type(worker["cpu_limit"]) is int and 0<cap==worker["cpu_limit"] and
              launch["deadline"]==deadline and prior_started<=launch["started_epoch"]<=worker["started_epoch"]<deadline and
              worker["started_epoch"]<=end["finished_epoch"] and prior_cpu<=launch["prior_cpu"]<=end["cpu_seconds"] and
              launch["prior_cpu"]+cap<=limit, "invalid "+phase+" worker budget or chronology")
        prior_cpu,prior_started=launch["prior_cpu"],worker["started_epoch"]


def verify(root,originals=None,tok=None):
    m,rows=verify_inputs(root,originals,tok)
    content=None
    if originals is not None:
        raw_content=(originals/"eval-content.json").read_bytes()
        check(hashlib.sha256(raw_content).hexdigest()==m["files"]["eval-content.json"],"original evaluation content drift")
        content=json.loads(raw_content)
    endpoint_plan=EndpointPlan.build(m["lengths"]["eval"],d.p.CONTEXT,2,content=content)
    fit,result,end=(read(root/f) for f in ("fit.json","result.json","complete.json"))
    check(fit["manifest_sha256"]==result["manifest_sha256"]==INPUT_PIN and result["fit_sha256"]==sha(root/"fit.json")
          and end["result_sha256"]==sha(root/"result.json") and fit["receipt_sha256"]==st.fit_pins(root)
          and result["lines_sha256"]=={p:sha(root/p/"lines.jsonl") for p in POINTS},"receipt/line hashes")
    attempt=read(root/"attempt.json")
    prepared=read(root/"prepared.json")
    check(attempt["manifest_sha256"]==INPUT_PIN and attempt["prepared_sha256"]==sha(root/"prepared.json"),"wrong attempt inputs")
    verify_fit_budget(root,prepared,attempt,end)
    origins=read(root/"owners.json")
    origin=origins["owners"]
    check(origins["arms"]==list(d.ARMS) and origins["trainable_counts"]==[4608]*4 and len(origin)==4
          and len({v[0] for v in origin})==1 and len({v[1] for v in origin})==4,"unmatched initial capacity")
    data_ids,plans,descriptions=set(),set(),{}
    stream_count=0
    for point in POINTS:
        if point=="initial":
            owner=origin
        else:
            owner,data_id=training(root,point,origin,rows["train"],m["lengths"]["train"],originals)
            check(data_id not in data_ids,"repeated permutation")
            data_ids.add(data_id)
        descriptions[point],identity,n=verified_endpoint(root/point,rows["eval"],endpoint_plan,owner,
                base_id=d.old.BASE_ID,layer=d.p.LAYER,stops={d.NL,d.EOS},vocabulary=d.EOS+1,cap=d.MAX_NEW,
                decode=None if tok is None else lambda ids:tok.decode(ids,skip_special_tokens=False),initial=point=="initial")
        plans.add(identity)
        stream_count+=n
    check(len(plans)==1 and next(iter(plans))[0]==fit["eval_data_id"] and descriptions==result["endpoints"] and
          d.decision([descriptions[o] for o in d.ORDERS])==result["decision"],"changed heldout set, contrasts or decision")
    check((fit["arm_updates"],fit["restored_checkpoints"],fit["scoring_points"],fit["greedy_streams"])==
          (5760,12,4,stream_count) and stream_count==1152,"experiment count")
    return dict(arm_updates=5760,checkpoint_boundaries=12,original_checkpoints_verified=12 if originals else 0,
                distinct_eval_records=72,endpoint_sets=4,greedy_streams=stream_count,generated_text_redecoded=tok is not None,
                aggregate_scores_reconciled=len(POINTS)*len(d.ARMS),eval_plan_verified=True,
                eval_content_id_recomputed=endpoint_plan.data_id is not None,model_executions_by_verifier=0,decision=result["decision"])


def export(root,zip_path):
    with zipfile.ZipFile(zip_path,"x") as out:
        for name in sorted(NAMES):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100600<<16
            data=(root/name).read_bytes()
            check(len(data)<=4*1024**2,"oversized receipt")
            out.writestr(info,data)


def open_archive(path,dest):
    check(path.is_file() and not path.is_symlink() and path.stat().st_size<=40*1024**2,"archive size/type")
    with zipfile.ZipFile(path) as z:
        items=z.infolist()
        check(len(items)==len(NAMES) and {i.filename for i in items}==set(NAMES)
              and sum(i.file_size for i in items)<=40*1024**2,"receipt allowlist")
        for name in NAMES:
            i=z.getinfo(name)
            check(i.file_size<=4*1024**2 and i.compress_type==zipfile.ZIP_DEFLATED,"oversized or unsupported entry")
            with z.open(name) as f:
                data=f.read(4*1024**2+1)
                check(len(data)==i.file_size,"partial or oversized receipt")
            target=dest/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as out:
                out.write(data)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path",type=Path)
    ap.add_argument("--prepared-only",action="store_true")
    ap.add_argument("--originals",type=Path)
    ap.add_argument("--model",type=Path,help="tokenizer directory only; model weights are never opened")
    ap.add_argument("--codec-site",type=Path)
    ap.add_argument("--export",type=Path)
    a=ap.parse_args()
    check((a.model is None)==(a.codec_site is None),"supply tokenizer directory and package together")
    tok=st.codec(a) if a.model is not None else None
    if a.prepared_only:
        verify_inputs(a.path,a.originals,tok)
        if a.export is not None:
            check(a.originals is not None and tok is not None,"re-tokenize and verify original inputs before export")
            a.export.mkdir(exist_ok=False)
            for name in PREP_RECEIPTS:
                shutil.copyfile(a.path/name,a.export/name)
            verify_inputs(a.export)
        result=dict(prepared=True,train=120,eval=72,train_targets=305,eval_content_targets=111,
                    original_data_verified=a.originals is not None,model_executions_by_verifier=0)
    elif a.path.is_file():
        check(a.export is None,"cannot re-export an archive")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            open_archive(a.path,root)
            result=verify(root,a.originals,tok)
    else:
        result=verify(a.path,a.originals,tok)
        if a.export:
            check(a.originals is not None and tok is not None,"verify original bytes and generated text first")
            export(a.path,a.export)
            with tempfile.TemporaryDirectory() as tmp:
                open_archive(a.export,Path(tmp))
                verify(Path(tmp))
    print(json.dumps(result,sort_keys=True))

"""Shared native SGD, answer-only teacher forcing and prompt-only free-running endpoints."""
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as d  # noqa: E402
from native_answer_generation import greedy_completions  # noqa: E402

check,save,read,sha=d.check,d.old.save,d.old.read,d.old.sha


def owners(model):
    return [model.lora_artifact_id,model.geometry_artifact_id]


def endpoint(st,models,data,prompts,directory,deadline,*,context,layer,stop_ids,cap,cache):
    keys=prompts["keys"]
    n=len(keys)
    check(len(models)==4 and keys==data["keys"] and len(set(keys))==len(prompts["prompt_ids"])==n,"endpoint key join")
    target_count=sum(v!=-100 for mask in data["labels"] for v in mask)
    r=st.NativeCausalLm.evaluate_variants_by_answer(models,data["token_documents"],context,
                labels=data["labels"],microbatch_windows=2)
    raw=json.loads(r.to_json())
    ev=raw["comparison"]["evaluations"]
    check(r.shared_prefix_layers==layer and r.scored_documents==r.document_count==len(r)==n and len(ev)==4
          and len({(e["data_id"],e["plan_id"]) for e in ev})==1,"masked endpoint coverage")
    for e,model in zip(ev,models,strict=True):
        check([e["lora_artifact_id"],e["geometry_artifact_id"]]==owners(model) and e["target_tokens"]==target_count
              and e["window_count"]==n and e["skipped_windows"]==0,"wrong endpoint owner or denominator")
    for i,(row,mask,prompt) in enumerate(zip(raw["answers"],data["labels"],prompts["prompt_ids"],strict=True)):
        a,b=row["start"],row["end"]
        check(row["document"]==i and 0<a==len(prompt)<b==len(mask) and mask[:a]==[-100]*a and -100 not in mask[a:]
              and data["token_documents"][i][:a]==prompt,"wrong content span or prompt")
    save(directory/"scores.json",dict(scores=raw,contrasts={k:r.contrast(*ab) for k,ab in d.PAIRS.items()}))
    with (directory/"generated.jsonl").open("x") as stream:
        for arm,model in zip(d.ARMS,models,strict=True):
            completions=greedy_completions(st,model,prompts["prompt_ids"],max_new_tokens=cap,stop_ids=stop_ids,max_cache_values=cache)
            for i,(key,out) in enumerate(zip(keys,completions,strict=True)):
                check(time.time()<deadline,"expired during generation")
                stream.write(json.dumps(dict(arm=arm,index=i,key=key,owners=owners(model),**out),separators=(",",":"))+"\n")
                stream.flush()
    return ev[0]["data_id"]


def fit_order(st,base,origin,data,indices,directory,deadline,*,context,layer,epochs):
    directory.mkdir(mode=0o700,exist_ok=False)
    n=len(data["keys"])
    check(sorted(indices)==list(range(n)) and len(origin)==4,"invalid order or arm roster")
    docs,masks=([data[k][i] for i in indices] for k in ("token_documents","labels"))
    Session=st.NativeCausalLmTrainingSession
    first=Session(origin[0],docs,context,mode="joint",labels=masks,window_policy="covering_targets",
                  batch_windows=1,epochs=epochs,learning_rate=d.p.LR)
    sessions=[first]+[first.fork_variant(m,"joint") for m in origin[1:]]
    previous=[owners(m) for m in origin]
    check(all(s.completed_updates==0 and s.total_updates==n*epochs and s.updates_per_epoch==n and
              s.data_id==first.data_id and owners(s.model)==o for s,o in zip(sessions,previous,strict=True)),"unmatched origin")
    save(directory/"order.json",dict(keys=[data["keys"][i] for i in indices],data_id=first.data_id,owners=previous))
    epoch_losses=[]
    with (directory/"steps.jsonl").open("x") as stream:
        for epoch in range(epochs):
            losses=[[] for _ in sessions]
            tokens=0
            for i,mask in enumerate(masks):
                check(time.time()<deadline,"expired before joint update")
                depth,reports=Session.step_matched(sessions)
                raw=[json.loads(r.to_json()) for r in reports]
                stream.write(json.dumps(dict(depth=depth,reports=raw),separators=(",",":"),allow_nan=False)+"\n")
                stream.flush()
                step=epoch*n+i+1
                t=sum(v!=-100 for v in mask)
                check(depth==layer and len(raw)==4 and t>0,"invalid matched prefix or target count")
                for j,r in enumerate(raw):
                    check((r["epoch"],r["update"],r["window_start"],r["window_end"],r["target_tokens"],r["mode"])==
                          (epoch,step,i,i+1,t,"joint") and [r["pre_lora_artifact_id"],r["pre_geometry_artifact_id"]]==previous[j]
                          and math.isfinite(r["loss"]),"wrong update, row or owner")
                    previous[j]=[r["post_lora_artifact_id"],r["post_geometry_artifact_id"]]
                    losses[j].append(r["loss"]*t)
                tokens+=t
                if step==1:
                    check(len({r["loss"] for r in raw})==len({o[0] for o in previous})==1,"unmatched first loss or LoRA")
                    save(directory/"first-step.json",d.first_step([s.model for s in sessions]))
            epoch_losses.append(dict(epoch=epoch,targets=tokens,nll=[math.fsum(x)/tokens for x in losses]))
            print(f"{directory.name}: epoch {epoch+1}/{epochs}, {n*(epoch+1)} updates/arm; no held-out probe",flush=True)
    save(directory/"training.json",epoch_losses)
    restored,entries=[],[]
    for arm,s,ids in zip(d.ARMS,sessions,previous,strict=True):
        check(s.exhausted and owners(s.model)==ids,"incomplete session")
        path=directory/(arm+".ck")
        ck=s.save_checkpoint_new(path)
        check(ck=="sha256:"+sha(path),"checkpoint pin")
        model=st.NativeCausalLm.from_session_checkpoint_at(base,path.read_bytes(),ck,update=n*epochs,recorded_data_id=s.data_id,
                post_lora_id=ids[0],post_geometry_id=ids[1])
        check(owners(model)==ids,"wrong restored endpoint")
        state,count=d.p.parameter_state(model.geometry_artifact())
        check(count==2*len(state["gate"]) and (arm!="both" or state["log_radius"]==[0.]),"fixed radius or capacity changed")
        entries.append(dict(arm=arm,checkpoint_id=ck,data_id=s.data_id,owners=ids,update=n*epochs,parameters=count))
        restored.append(model)
    save(directory/"checkpoints.json",entries)
    return restored,first.data_id

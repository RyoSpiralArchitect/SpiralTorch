"""Artificial complete-line SGD: direct prior wheel oracle and cross-runtime continuation."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
shared = runpy.run_path(str(HERE/"native_cached_lines_fixture.py"))
st = shared["st"]
PLAN = (HERE/"native_line_session_plan.json").read_bytes()
RECIPE = json.loads(PLAN)
CUT, RETURN = 2, 4
N = len(RECIPE["examples"])*RECIPE["epochs"]
pin = lambda raw: "sha256:"+hashlib.sha256(raw).hexdigest()


def save(path, data):
    with path.open("x") as f:
        json.dump(data,f,allow_nan=False,sort_keys=True,indent=2)
        f.write("\n")


def hosts():
    return ((name,m) for name,m,_ in shared["models"]() if m.geometry_artifact_id is not None)


def direct(m, i):
    e=RECIPE["examples"][i%len(RECIPE["examples"])]
    spec=dict(e["objective"])
    kind=spec.pop("kind")
    g=getattr(m,"geometry_line_"+kind+"_gradients")(e["prompt_ids"],e["candidates"],stop_id=e["stop_id"],**spec)
    out=m.step_adaptation_sgd(g,mode="geometry",learning_rate=RECIPE["learning_rate"])
    return out,dict(kind=kind,scores=getattr(g,"line_"+kind+"_report")(),loss=g.loss,targets=g.target_tokens,
                    before=m.geometry_artifact_id,after=out.geometry_artifact_id)


def endpoint(m):
    rec=json.loads(m.geometry_artifact().state_json())
    params={p["path"]:{t["name"]:t["values"] for t in p["tensors"]} for p in rec["placements"]}
    return dict(parameters=params,logits=m.forward([1,2],[1,2],last_only=True).values())


def reference(out):
    assert not hasattr(st,"NativeCausalLmLineTrainingSession"), "reference must use the previous wheel"
    result={}
    for name,origin in hosts():
        m=origin
        steps=[]
        for i in range(N):
            m,r=direct(m,i)
            steps.append(r)
        assert origin.geometry_artifact_id!=m.geometry_artifact_id
        result[name]=dict(steps=steps,endpoint=endpoint(m),origin=origin.geometry_artifact_id,base=origin.base_model_identity)
    assert len(result)==4
    save(out,result)


def check_step(r,ref,i):
    assert (r["plan_id"],r["update"],r["epoch"],r["example_index"]) == (pin(PLAN),i+1,i//3,i%3)
    assert (r["pre_geometry_id"],r["post_geometry_id"])==(ref["before"],ref["after"])
    assert (r["target_tokens"],r["loss"])==(ref["targets"],ref["loss"])
    assert r["objective"]==dict(kind=ref["kind"],scores=ref["scores"])
    assert r["work"]["peak_logit_values"]==11 and r["work"]["backward_head_calls"]>1


def issue(ref_path,out):
    ref=json.loads(ref_path.read_bytes())
    cases={}
    for name,m in hosts():
        s=st.NativeCausalLmLineTrainingSession(m,PLAN)
        assert s.plan_id==pin(PLAN) and s.total_updates==N and not s.exhausted
        for i,r in enumerate(ref[name]["steps"]):
            check_step(s.step(),r,i)
            if i+1==CUT:
                payload,ck=s.checkpoint_with_id()
                assert pin(payload.encode())==ck
        assert s.exhausted and s.step() is None and endpoint(s.model)==ref[name]["endpoint"]
        cases[name]=dict(payload=payload,id=ck,final_id=s.model.geometry_artifact_id,final_checkpoint=s.checkpoint_with_id()[1],
                         endpoint=ref[name]["endpoint"],losses=[r["loss"] for r in ref[name]["steps"]])
    save(out,dict(plan_id=pin(PLAN),cut=CUT,return_at=RETURN,total=N,cases=cases))


def resume(ref_path,bridge):
    ref=json.loads(ref_path.read_bytes())
    b=json.loads(bridge.read_bytes())
    for name,origin in hosts():
        row=b["cases"][name]
        s=st.NativeCausalLmLineTrainingSession.from_json(origin,PLAN,row["payload"],row["id"])
        assert s.completed_updates==CUT and s.checkpoint_with_id()==(row["payload"],row["id"])
        for i in range(CUT,N):
            check_step(s.step(),ref[name]["steps"][i],i)
        assert s.step() is None and s.checkpoint_with_id()[1]==row["final_checkpoint"]
        assert s.model.geometry_artifact_id==row["final_id"] and endpoint(s.model)==row["endpoint"]
    print(json.dumps({"cold_native_resumes":len(b["cases"]),"exact_suffix_steps":len(b["cases"])*(N-CUT)}))


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max(close(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max(close(x,y) for x,y in zip(a,b))
    err=abs(a-b)
    assert err<=3e-6,(a,b)
    return err


def take_back(bridge,transfer,out):
    b=json.loads(bridge.read_bytes())
    receipt=json.loads(transfer.read_bytes())
    assert receipt["plan_id"]==pin(PLAN)==b["plan_id"]
    rows={}
    for name,origin in hosts():
        r=receipt["cases"][name]
        assert pin(r["payload"].encode())==r["id"]
        s=st.NativeCausalLmLineTrainingSession.from_json(origin,PLAN,r["payload"],r["id"])
        assert s.completed_updates==RETURN and s.checkpoint_with_id()==(r["payload"],r["id"])
        m=s.model
        loss_errors=[]
        for i in range(RETURN,N):
            m,d=direct(m,i)
            step=s.step()
            check_step(step,d,i)
            loss_errors.append(close(step["loss"],r["suffix_losses"][i-RETURN]))
        final=endpoint(s.model)
        assert s.step() is None and s.model.geometry_artifact_id==m.geometry_artifact_id
        rows[name]=dict(prefix_pin=r["id"],suffix_loss_error=max(loss_errors),wasm_final_error=close(final,r["endpoint"]),
                        native_final_error=close(final,b["cases"][name]["endpoint"]),result_id=s.model.geometry_artifact_id)
    assert set(rows)==set(receipt["cases"])==set(b["cases"]) and len(rows)==4
    save(out,rows)


if __name__=="__main__":
    functions={"reference":reference,"issue":issue,"resume":resume,"take-back":take_back}
    functions[sys.argv[1]](*map(Path,sys.argv[2:]))

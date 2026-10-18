"""Qualify the full bounded run, then decode fixed prompts from each checkpoint."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct

GATE_SHA256="668b98136318fc91325ad6af58c35bbb82d4aa99be284233ca36ab5773621298"
SELECTED=("seed11_ordinary","mixture-11-0-train-query-train","mixture-11-0-train-query-frozen")


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path,value):
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),"w",encoding="utf-8") as out:
        json.dump(value,out,indent=2,allow_nan=False)
        out.write("\n")


def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline(request,cursor):
    t=request["config"]["steps"]
    unigram,bigram=Counter(),Counter()
    for batch in request["train_batches"][:cursor]:
        for d,start in batch:
            window=request["train_documents"][d][start:start+t+1]
            require(len(window)==t+1,"incomplete train window")
            unigram.update(window[1:])
            bigram.update(zip(window[:-1],window[1:]))
    counts=Counter()
    for (x,y),count in bigram.items():
        counts[x]+=count
    labels=[]
    for batch in request["validation_batches"]:
        for d,start in batch:
            window=request["validation_documents"][d][start:start+t+1]
            require(len(window)==t+1,"incomplete validation window")
            labels.extend(zip(window[:-1],window[1:]))
    total=sum(unigram.values())
    return dict(revision=cursor,training_target_presentations=total,validation_targets=len(labels),
                unigram_ce=-math.fsum(math.log((unigram[y]+1)/(total+256)) for x,y in labels)/len(labels),
                bigram_ce=-math.fsum(math.log((bigram[x,y]+1)/(counts[x]+256)) for x,y in labels)/len(labels))


def repetition(rows):
    longest=[]
    for row in rows:
        best=current=1
        for a,b in zip(row,row[1:]):
            current=current+1 if a==b else 1
            best=max(best,current)
        longest.append(best)
    return dict(bytes=sum(len(r) for r in rows),space_bytes=sum(v==32 for r in rows for v in r),
                distinct_bytes_per_row=[len(set(r)) for r in rows],longest_identical_run_per_row=longest)


def bits(values):
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values),"invalid saved scalar")
    return [struct.pack("<f",v) for v in values]


def bind_tensors(descriptors,stored,reported):
    require(len(descriptors)==len(stored)==len(reported),"checkpoint tensor coverage")
    for descriptor,saved,values in zip(descriptors,stored,reported):
        require(saved["shape"]==descriptor["shape"] and
                len(saved["values"])==len(values)==math.prod(descriptor["shape"]),"checkpoint tensor shape")
        require(bits(saved["values"])==bits(values),"checkpoint weights differ from qualified report")


def bind_history(saved,reported):
    require(len(saved["training"])==len(reported["training"]),"checkpoint training coverage")
    for a,b in zip(saved["training"],reported["training"]):
        require(type(a["revision"]) is int and a["revision"]==b["revision"] and bits([a["ce"]])==bits([b["ce"]]),"checkpoint training history differs")
    require(len(saved["validation"])==len(reported["validation"]),"checkpoint validation coverage")
    for a,b in zip(saved["validation"],reported["validation"]):
        require(type(a["revision"]) is int and a["revision"]==b["revision"] and bits(a["batch_losses"])==bits(b["batch_losses"]),"checkpoint validation history differs")


def bind_checkpoint(request,request_hash,checkpoint,report,cursor,helpers):
    require(checkpoint["request_sha256"]==request_hash and type(checkpoint["completed_updates"]) is int
            and checkpoint["completed_updates"]==cursor,"checkpoint source/cursor")
    require(len(checkpoint["cases"])==len(report["cases"])==len(request["cases"]),"checkpoint case coverage")
    for desc,saved,reported in zip(request["cases"],checkpoint["cases"],report["cases"]):
        require(desc["name"]==saved["name"]==reported["name"],"checkpoint case identity")
        model=helpers.STUDY.decode_json(saved["model_json"])
        stored=helpers.stored_parameters(model,cursor,desc.get("pair_metric"),desc.get("head_pair_metrics"),desc.get("mixture"))
        bind_tensors(desc["parameters"],stored,reported["final_parameters"])
        bind_history(saved,reported)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","pilot","gate"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    raw=(args.pilot/"request.json").read_bytes()
    request=json.loads(raw)
    prepared=json.loads((args.pilot/"prepared.json").read_bytes())
    claim=json.loads((args.pilot/"native-run-claim.json").read_bytes())
    require(sha(raw)==prepared["request_sha256"]==claim["request_sha256"],"run identity changed")
    require(sha((args.pilot/"PLAN.md").read_bytes())==prepared["plan_sha256"],"plan changed")
    require(sha(args.gate.read_bytes())==GATE_SHA256,"fixed generation prompts changed")
    gate=json.loads(args.gate.read_bytes())
    require(gate["cohorts"][0]["name"]=="autograd-prose","prompt cohort")
    t,b=request["config"]["steps"],request["config"]["batch"]
    require(t==32 and b==2 and gate["steps"]==32,"fixed model/window shape")
    selection=gate["cohorts"][0]["batches"][0]
    prompts=[bytes(gate["documents"][d][s:s+t]) for d,s in selection]
    require(len(prompts)==b and all(len(p)==t for p in prompts),"prompt coverage")
    s=load(args.repo/"tools/byte_corpus_study.py")
    native=s.decode_json((args.pilot/"native-512.json").read_bytes())
    reference=s.decode_json((args.pilot/"torch.json").read_bytes())
    comparison=s.compare(raw,reference,native)
    write(args.pilot/"comparison.json",comparison)
    reports={stop:s.decode_json((args.pilot/f"native-{stop}.json").read_bytes()) for stop in (128,256,512)}
    checkpoints={stop:(args.pilot/f"native-checkpoint-{stop}.json").read_bytes() for stop in reports}
    resume_helpers=load(args.repo/"tools/verify_byte_corpus_resume.py")
    for stop,report in reports.items():
        require(report["request_sha256"]==sha(raw) and report["engine"]=="spiraltorch","segment report identity")
        require([c["name"] for c in report["cases"]]==[c["name"] for c in native["cases"]],"segment case coverage")
        for partial,final in zip(report["cases"],native["cases"]):
            require(partial["training"]==final["training"][:stop],"training prefix changed across resume")
            require(partial["validation"]==[p for p in final["validation"] if p["revision"]<=stop],"validation prefix changed across resume")
        bind_checkpoint(request,sha(raw),s.decode_json(checkpoints[stop]),report,stop,resume_helpers)
    import spiraltorch as st
    study=st.nn.ByteCorpusStudy.from_json(raw)
    oracle=load(args.repo/"benchmarks/results/2026-10-15-language-greedy-serving/verify_generation.py").oracle
    generated=[]
    for stop in (128,256,512):
        checkpoint=study.checkpoint_from_json(checkpoints[stop])
        require(checkpoint.completed_updates==stop,"checkpoint cursor")
        for name in SELECTED:
            artifact=study.case_artifact(checkpoint,name,policy="fixed_window_frozen.v1")
            study.validate_artifact(checkpoint,artifact)
            pristine=artifact.restore_wgpu()
            require(pristine.tensor_device().adapter_info()["device_type"]!="Cpu","not actual GPU")
            expected,hashes=oracle(st,pristine,prompts,32)
            require(pristine.attempted_updates==stop,"oracle changed revision")
            del pristine
            owner=artifact.restore_wgpu()
            result=json.loads(owner.generate_greedy(prompts,max_new_bytes=32))
            require(result["artifact_id"]==owner.artifact_id==artifact.artifact_id,"generation identity")
            require(owner.attempted_updates==stop and result["serving"] is None,"generation owner policy")
            require(result["prompt_sha256"]==[sha(p) for p in prompts],"generation prompt identity")
            require(result["generated_bytes"]==expected and result["next_logits_f32le_sha256"]==hashes,"pristine full-forward oracle mismatch")
            row=dict(revision=stop,case=name,receipt=result,pristine_full_forward_oracle_exact=True,
                     repetition=repetition(result["generated_bytes"]))
            generated.append(row)
            write(args.pilot/f"generation-{stop}-{name}.json",row)
            print(json.dumps(dict(event="generation_qualified",revision=stop,case=name,**row["repetition"])),flush=True)
            del owner
    trajectories=[]
    for case,expected in zip(native["cases"],reference["cases"]):
        trajectories.append(dict(name=case["name"],seed=case["seed"],geometry=case["geometry"],
            geometry_update=case.get("geometry_update"),pair_metric=case.get("pair_metric"),
            mixture=case.get("mixture"),parameter_scalars=case["parameter_scalars"],
            native_validation=case["validation"],torch_validation=expected["validation"]))
    inputs=["PLAN.md","prepared.json","historical-differences.json","request.json","torch.json","native-run-claim.json"]
    inputs += [f"native-{stop}.json" for stop in (128,256,512)]
    inputs += [f"native-checkpoint-{stop}.json" for stop in (128,256,512)]
    summary=dict(schema="spiraltorch.byte_corpus.learning_budget_qualification.v1",
        request_sha256=sha(raw),inputs={p:sha((args.pilot/p).read_bytes()) for p in inputs},
        numeric_comparison_passed=True,comparison_sha256=sha((args.pilot/"comparison.json").read_bytes()),
        adapter=native["adapter"],torch_version=reference["torch_version"],
        cases=trajectories,baselines=[baseline(request,stop) for stop in (128,256,512)],generation=generated,
        all_resume_history_prefixes_equal=True,all_checkpoint_weights_and_histories_bound_to_reports=True,
        native_accepted_model_updates=25*512,
        gate_sha256=GATE_SHA256,qualifier_sha256=sha(Path(__file__).read_bytes()),
        scope="One fresh seed, all25 controls, unchanged numeric gates. Historical initial bits differ; compare budgets within this run only. No speed, broad quality, seed stability, promotion or PyPI claim.")
    write(args.pilot/"qualification.json",summary)
    print(json.dumps(dict(qualified=True,arms=25,max_updates=512,generations=len(generated))),flush=True)


if __name__=="__main__":
    main()

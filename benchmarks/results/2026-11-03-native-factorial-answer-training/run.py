"""Model-free prepare; one separately pinned, bounded fit + offline text adjudication."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as d  # noqa: E402
import trial  # noqa: E402
import readout  # noqa: E402
from owned_process import run as owned  # noqa: E402

check,save,read,sha=d.check,d.old.save,d.old.read,d.old.sha
SPEC_SHA="52c9c589de05f0c6a1f7d2b8b95493ac7a9df717730443c4b981f6ddf0f7b537"
LIMIT=3600
FILES=("train.json","eval-content.json","prompts.json")


def sources(commit):
    paths=[d.HERE/n for n in ("run.py","design.py","trial.py","readout.py")]+[d.BUILD,d.PREV/"protocol.py",d.p.PILOT,
           *[d.old.ROOT/"tools"/n for n in ("native_answer_generation.py","byte_release_artifacts.py","owned_process.py")]]
    pins={str(f.relative_to(d.old.ROOT)):sha(f) for f in paths}
    for n,pin in pins.items():
        raw=subprocess.check_output(["git","-C",str(d.old.ROOT),"show",f"{commit}:{n}"])
        check(hashlib.sha256(raw).hexdigest()==pin,"uncommitted source: "+n)
    return pins


def codec(a):
    path=a.model/"tokenizer.json"
    check(sha(path)==d.old.FILES["tokenizer.json"],"wrong tokenizer bytes")
    spec=importlib.machinery.PathFinder.find_spec("tokenizers",[str(a.codec_site)])
    mod=importlib.util.module_from_spec(spec)
    sys.modules["tokenizers"]=mod
    spec.loader.exec_module(mod)
    check(mod.__version__=="0.22.2","wrong tokenizer package")
    tok=mod.Tokenizer.from_file(str(path))
    check(tok.encode("\n",add_special_tokens=False).ids==[d.NL] and tok.token_to_id("<|endoftext|>")==d.EOS,"wrong stops")
    return tok


def package(a,phase):
    a.action=phase
    a.wheel_pin=d.WHEEL
    st,identity=d.old.package(a)
    check(identity["native_sha256"]==d.NATIVE,"wrong loaded wheel/extension")
    return st,identity


def prepare(a):
    rows=d.records()
    frozen=d.freeze(rows)
    check(d.pin(frozen)==SPEC_SHA,"prospective split or protocol changed")
    st,identity=package(a,"prepare")
    tok=codec(a)
    lengths,seen={},set()
    for split in ("train","eval"):
        keys,docs,masks,prompts=[],[],[],[]
        for row in rows[split]:
            prompt,answer=row["prompt"],row["answer"]
            text=prompt+answer+"\n"
            ids,mask=st.encode_supervised_document(tok,text,[(len(prompt),len(text))])
            prefix=tok.encode(prompt,add_special_tokens=False).ids
            start=len(prefix)
            check(0<start<len(ids)-1<len(ids)<=d.p.CONTEXT and ids[:start]==prefix and ids[-1]==d.NL
                  and mask==[-100]*start+ids[start:] and d.NL not in ids[start:-1] and d.EOS not in ids
                  and tok.decode(prefix,skip_special_tokens=False)==prompt and tok.decode(ids[start:-1],skip_special_tokens=False)==answer
                  and tok.decode(ids,skip_special_tokens=False)==text and tuple(prefix) not in seen,"supervision boundary or prompt leakage")
            seen.add(tuple(prefix))
            keys.append(row["key"])
            docs.append(ids if split=="train" else ids[:-1])
            masks.append(mask if split=="train" else mask[:-1])
            prompts.append(prefix)
        filename="train.json" if split=="train" else "eval-content.json"
        save(a.output/filename,dict(keys=keys,token_documents=docs,labels=masks))
        lengths[split]=[[k,len(v),len(pr),len(v)-len(pr)] for k,v,pr in zip(keys,docs,prompts,strict=True)]
        if split=="eval":
            save(a.output/"prompts.json",dict(keys=keys,prompt_ids=prompts))
    check(not d.old.FORBIDDEN & {n.partition(".")[0] for n in sys.modules},"foreign ML during preparation")
    save(a.output/"manifest.json",dict(schema="spiraltorch.factorial_sft_inputs.v1",commit=a.commit,
         source_sha256=sources(a.commit),frozen=frozen,lengths=lengths,installed=identity,tokenizer=d.old.FILES["tokenizer.json"],
         files={n:sha(a.output/n) for n in FILES},prepare_attempt=sha(a.output/"prepare-attempt.json")))


def inputs(a):
    root=a.output
    check(sha(root/"manifest.json")==a.manifest_pin,"input manifest pin")
    m=read(root/"manifest.json")
    rows=d.records()
    check(d.pin(m["frozen"])==SPEC_SHA and d.freeze(rows)==m["frozen"] and m["commit"]==a.commit
          and sources(a.commit)==m["source_sha256"] and m["prepare_attempt"]==sha(root/"prepare-attempt.json")
          and m["files"]=={n:sha(root/n) for n in FILES},"input bytes, source or plan drift")
    return rows,m


def fit(a):
    root=a.output
    rows,m=inputs(a)
    st,identity=package(a,"fit")
    check(identity==m["installed"] and all(sha(a.model/n)==h for n,h in d.old.FILES.items()),"wrong frozen decoder or wheel")
    train,ev,prompts=(read(root/n) for n in FILES)
    base=st.NativeCausalLm.from_gpt2((a.model/"config.json").read_text(),st.SafeTensorArchive(a.model/"model.safetensors"))
    check(base.base_model_identity==d.old.BASE_ID and base.execution_backend=="rust_f32_cpu_prepacked"
          and base.readout_accumulator=="f32","wrong native backend")
    models=d.origin(st,base,d.p.WIDTH,d.p.LAYER)
    save(root/"owners.json",dict(arms=d.ARMS,owners=[trial.owners(x) for x in models],trainable_counts=[4608]*4))
    kw=dict(context=d.p.CONTEXT,layer=d.p.LAYER,stop_ids=[d.NL,d.EOS],cap=d.MAX_NEW,cache=d.CACHE)
    initial=root/"initial"
    initial.mkdir(mode=0o700,exist_ok=False)
    ev_id=trial.endpoint(st,models,ev,prompts,initial,a.deadline,**kw)
    start=read(initial/"scores.json")["scores"]["answers"]
    check(all(r["scores"]==[r["scores"][0]]*4 for r in start),"initial predictions differ")
    gens=readout.lines(initial/"generated.jsonl")
    check(all(gens[j*d.EVAL+i]["generated_ids"]==gens[i]["generated_ids"] for i in range(d.EVAL) for j in range(1,4)),
          "initial greedy streams differ")
    data_ids=set()
    for order in d.ORDERS:
        dest=root/order
        restored,data_id=trial.fit_order(st,base,models,train,d.indices(rows["train"],order),dest,a.deadline,
                                         context=d.p.CONTEXT,layer=d.p.LAYER,epochs=d.EPOCHS)
        check(data_id not in data_ids,"duplicate training order")
        data_ids.add(data_id)
        check(trial.endpoint(st,restored,ev,prompts,dest,a.deadline,**kw)==ev_id,"changed evaluation plan")
        print(order+": stored four restored endpoints and greedy streams",flush=True)
    check(not (d.old.FORBIDDEN|{"tokenizers"}) & {n.partition(".")[0] for n in sys.modules},"foreign model runtime")
    save(root/"fit.json",dict(arm_updates=d.UPDATES*4*3,restored_checkpoints=12,scoring_points=4,
         greedy_streams=(1+len(d.ORDERS))*4*d.EVAL,eval_data_id=ev_id,manifest_sha256=a.manifest_pin,
         receipt_sha256=fit_pins(root)))


def fit_pins(root):
    names=["owners.json"]+[f"{e}/{file}" for e in ("initial",*d.ORDERS) for file in ("scores.json","generated.jsonl")]+[
           f"{o}/{file}" for o in d.ORDERS for file in ("checkpoints.json","steps.jsonl","training.json","first-step.json","order.json")]
    return {n:sha(root/n) for n in names}


def grade(a):
    root=a.output
    rows,_=inputs(a)
    check(sha(root/"fit.json")==a.fit_pin and read(root/"fit.json")["receipt_sha256"]==fit_pins(root),"fit receipt pin")
    check("spiraltorch" not in sys.modules,"text adjudication must not load the model runtime")
    tok=codec(a)
    eval_rows=rows["eval"]
    result={}
    for e in ("initial",*d.ORDERS):
        readout.decode_endpoint(root/e,eval_rows,lambda ids:tok.decode(ids,skip_special_tokens=False))
        result[e]=readout.describe(root/e,eval_rows)
    check(not (d.old.FORBIDDEN|{"spiraltorch"}) & {n.partition(".")[0] for n in sys.modules},"model used for adjudication")
    save(root/"result.json",dict(schema="spiraltorch.factorial_sft_outcome.v1",manifest_sha256=a.manifest_pin,
         fit_sha256=a.fit_pin,lines_sha256={e:sha(root/e/"lines.jsonl") for e in result},
         endpoints=result,decision=d.decision([result[o] for o in d.ORDERS])))


def cpu():
    c=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+c.ru_utime+c.ru_stime


def launch(a,phase,deadline,prior,extra=()):
    remaining=min(deadline-time.time(),LIMIT-prior-cpu())-2
    check(remaining>2,"cumulative CPU or wall budget exhausted")
    cap=math.floor(remaining)
    save(a.output/(phase+"-launch.json"),dict(deadline=deadline,cpu_limit=cap,started_epoch=time.time(),prior_cpu=prior+cpu()))
    command=[sys.executable,"-I","-B",__file__,*sys.argv[1:],"--worker",phase,"--deadline",str(deadline),"--cap",str(cap),*extra]
    from owned_process import run as supervise
    with (a.output/(phase+".log")).open("x") as log:
        status=supervise(command,timeout=remaining,stdout=log).returncode
    check(status==0,f"{phase} exited {status}; retain first failure, no retry")


if __name__=="__main__":
    os.umask(0o077)
    check(sys.flags.isolated and sys.dont_write_bytecode,"use python -I -B")
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("prepare","run"))
    for name in ("model","wheel","output","codec-site"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--commit",required=True)
    ap.add_argument("--manifest-pin")
    ap.add_argument("--fit-pin")
    ap.add_argument("--worker",choices=("prepare","fit","grade"),help=argparse.SUPPRESS)
    ap.add_argument("--deadline",type=float,help=argparse.SUPPRESS)
    ap.add_argument("--cap",type=int,help=argparse.SUPPRESS)
    a=ap.parse_args()
    check(subprocess.check_output(["git","-C",str(d.old.ROOT),"rev-parse","HEAD"],text=True).strip()==a.commit,"wrong checkout")
    if a.worker:
        check(0<a.cap<=LIMIT and 0<a.deadline-time.time()<=LIMIT,"invalid worker admission")
        resource.setrlimit(resource.RLIMIT_CPU,(a.cap,a.cap))
        save(a.output/(a.worker+"-worker.json"),dict(cpu_limit=a.cap,started_epoch=time.time()))
        try:
            {"prepare":prepare,"fit":fit,"grade":grade}[a.worker](a)
        except Exception as ex:
            save(a.output/(a.worker+"-failure.json"),dict(type=type(ex).__name__,message=str(ex),epoch=time.time()))
            raise
    elif a.action=="prepare":
        a.output.mkdir(mode=0o700,exist_ok=False)
        start=time.time()
        save(a.output/"prepare-attempt.json",dict(commit=a.commit,started_epoch=start,retries=0))
        launch(a,"prepare",start+120,0.)
        pin=sha(a.output/"manifest.json")
        save(a.output/"prepared.json",dict(manifest_sha256=pin,cpu_seconds=cpu(),wall_seconds=time.time()-start))
        print(json.dumps(read(a.output/"prepared.json")),flush=True)
    else:
        prepared=read(a.output/"prepared.json")
        check(a.manifest_pin==prepared["manifest_sha256"]==sha(a.output/"manifest.json"),"externally pin the prepared data")
        start=time.time()
        deadline=start+LIMIT-max(prepared["wall_seconds"],prepared["cpu_seconds"])-1
        save(a.output/"attempt.json",dict(started_epoch=start,deadline=deadline,limit_seconds=LIMIT,retries=0,
                                        manifest_sha256=a.manifest_pin,prepared_sha256=sha(a.output/"prepared.json")))
        launch(a,"fit",deadline,prepared["cpu_seconds"])
        launch(a,"grade",deadline,prepared["cpu_seconds"],["--fit-pin",sha(a.output/"fit.json")])
        total=cpu()+prepared["cpu_seconds"]
        wall=time.time()-start+prepared["wall_seconds"]
        check(total<LIMIT and wall<LIMIT,"exceeded cumulative budget")
        save(a.output/"complete.json",dict(cpu_seconds=total,active_wall_seconds=wall,finished_epoch=time.time(),
                                          result_sha256=sha(a.output/"result.json"),exit_code=0))
        print(read(a.output/"result.json")["decision"],flush=True)

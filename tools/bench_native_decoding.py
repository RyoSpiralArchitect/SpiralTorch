"""Two installed wheels, matched synthetic logits, balanced CPU generation timings. No pretrained data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
from statistics import median
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"bindings/st-py/tests/test_native_causal_lm.py"
METHODS=("old_report","new_report","new_id","new_chunk")
DESIGNS=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))*2
SHAPE=dict(width=16,heads=2,layers=2,positions=32,vocabulary=50_257)
PROMPTS=([12,4,26,81,17,2,63,482],[7,900,4,63,18,33,401,2,9,6,111])
STEPS=8
CASES={"plain":{"selection":"greedy"},
       "sample32":{"selection":"sample","top_k":32,"top_p":.9,"distribution":{"kind":"softmax","temperature":.85}},
       "zspace32":{"selection":"greedy","top_k":32,"top_p":.9,"distribution":{"kind":"zspace","config":{
           "curvature":-1.,"repression_strength":2.,"last_token_repression":.5}}}}
THREADS=("RAYON_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")
LIMIT=240


def check(ok,why):
    if not ok: raise ValueError(why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(p): return json.loads(p.read_bytes())


def save(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False,separators=(",",":"))
        f.write("\n")


def worker(a):
    resource.setrlimit(resource.RLIMIT_CPU,(35,35))
    for name in THREADS: os.environ[name]="1"
    os.environ["SPIRALTORCH_AUTOTUNE_STORE"]=str(a.output.resolve()/f"autotune-{a.index}-{a.worker}.json")
    foreign={"torch","numpy","transformers","accelerate","peft","safetensors","tokenizers"}
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.partition(".")[0] in foreign: raise ModuleNotFoundError("native-only benchmark",name=fullname)
    sys.meta_path.insert(0,Guard())
    sys.path.insert(0,str(ROOT/"tools"))
    if a.worker=="admit":
        import byte_release_artifacts as wheels
        inv=wheels.wheel_archive_inventory(a.wheel)
        check(inv["sha256"]==a.pin,"wrong wheel")
        identity=wheels.installed_identity(inv)
        import spiraltorch as st
        check(sha(st._rs.__file__)==identity["native_sha256"],"wrong loaded extension")
        save(a.output/f"identity-{a.index}.json",identity)
        return
    import spiraltorch as st
    proto=read(a.output/"protocol.json")
    flavor="old" if a.worker=="old_report" else "new"
    check(sha(st._rs.__file__)==proto["installed"][flavor]["native_sha256"],"wrong running wheel")
    base=st.NativeCausalLm.from_gpt2(*runpy.run_path(str(FIXTURE))["fixture"](**SHAPE))
    check(base.execution_backend=="rust_f32_cpu_prepacked","not the CPU decoder")
    case_names=list(CASES)
    case_names=case_names[a.index%3:]+case_names[:a.index%3]

    def measure(case):
        histories,stops,pins,reports=[],[],[],[]
        ns=0
        for j,prompt in enumerate(PROMPTS):
            cfg=json.dumps(dict(max_new_tokens=STEPS,seed=123+j,decoding=CASES[case],execution={"kind":"kv_cache","max_values":32768}))
            steps=[]
            start=time.perf_counter_ns()
            gen=st.NativeCausalLmGenerator(base,prompt,cfg)
            if a.worker=="new_chunk":
                ids=gen.advance_ids(STEPS)
            elif a.worker=="new_id":
                ids=[gen.step_token_id() for _ in range(STEPS)]
            else:
                steps=[gen.step() for _ in range(STEPS)]
                ids=[s.token_id for s in steps]
            ns+=time.perf_counter_ns()-start
            check(len(ids)==STEPS and ids==gen.generated_ids and gen.finished and gen.finish_reason=="length","incomplete generation")
            histories.append(ids);stops.append(gen.finish_reason);pins.append(gen.checkpoint_id)
            reports.extend(s.to_json() for s in steps)
        digest=hashlib.sha256(("\n".join(reports)+"\n").encode()).hexdigest() if reports else None
        return dict(ns=ns,histories=histories,finish=stops,checkpoints=pins,reports_sha256=digest)
    warm={k:measure(k) for k in case_names}
    measured={k:measure(k) for k in case_names}
    for k,r in measured.items():
        check({x:y for x,y in r.items() if x!="ns"}=={x:y for x,y in warm[k].items() if x!="ns"},"nondeterministic warm/cold signature")
    check(not foreign.intersection(n.partition(".")[0] for n in sys.modules),"foreign runtime imported")
    save(a.output/f"cell-{a.index}-{a.worker}.json",dict(index=a.index,method=a.worker,base=base.base_model_identity,cases=measured))


def reduce(runs):
    ratios={}
    for case in CASES:
        reference=runs[0]["cases"][case]
        for r in runs:
            v=r["cases"][case]
            check(r["base"]==runs[0]["base"] and all(v[k]==reference[k] for k in ("histories","finish","checkpoints"))
                  and v["ns"]>0,"different generation, checkpoint, or model")
            if r["method"].endswith("report"):
                check(v["reports_sha256"]==reference["reports_sha256"] and v["reports_sha256"] is not None,"report bytes differ")
            else:
                check(v["reports_sha256"] is None,"unexpected ID-only report")
        times=[[next(r["cases"][case]["ns"] for r in runs if r["index"]==i and r["method"]==m) for m in METHODS] for i in range(8)]
        ratios[case]={"median_ms":{m:median(t[j] for t in times)/1e6 for j,m in enumerate(METHODS)}}
        for name,x,y in (("report_old_over_new",0,1),("id_old_over_new",0,2),("chunk_old_over_new",0,3),("chunk_new_report_over_ids",1,3)):
            values=[t[x]/t[y] for t in times]
            ratios[case][name]=dict(pairs=values,median=median(values))
    return ratios


def main(a):
    a.output.mkdir(mode=0o700,exist_ok=False)
    start=time.monotonic()
    deadline=start+LIMIT
    save(a.output/"attempt.json",dict(wall_seconds=LIMIT,worker_cpu_seconds=35,worker_wall_seconds=40,
                                      retries=0,started_epoch=time.time(),old_wheel=a.old_pin,new_wheel=a.new_pin))
    sys.path.insert(0,str(ROOT/"tools"))
    from owned_process import run
    wheels={"old":(a.old_python,a.old_wheel,a.old_pin),"new":(a.new_python,a.new_wheel,a.new_pin)}
    def invoke(which,method,index):
        python,path,pin=wheels[which]
        with (a.output/f"{index}-{method}.log").open("x") as log:
            proc=run([str(python),"-I","-B",__file__,"--output",str(a.output),"--worker",method,
                      "--index",str(index),"--wheel",str(path),"--pin",pin],timeout=min(40.,deadline-time.monotonic()),stdout=log)
        check(proc.returncode==0,f"{method}/{index} failed; no retry")
    installed={}
    for i,which in enumerate(wheels):
        invoke(which,"admit",i)
        installed[which]=read(a.output/f"identity-{i}.json")
    check(installed["old"]["native_sha256"]!=installed["new"]["native_sha256"],"identical extensions")
    order=[[METHODS[k] for k in row] for row in DESIGNS]
    proto=dict(schema="spiraltorch.synthetic_exact_decoding_bench.v1",installed=installed,shape=SHAPE,prompts=PROMPTS,
               max_new_tokens=STEPS,policies=CASES,order=order,thread_env={n:"1" for n in THREADS},warmup="each worker and policy once",
               script_sha256=sha(__file__),fixture_sha256=sha(FIXTURE),host=platform.platform(),
               scope="fresh synthetic CPU model; timed generator construction, KV prefill, eight tokens for two prompts; report JSON, setup and disk excluded")
    save(a.output/"protocol.json",proto)
    runs=[]
    with (a.output/"runs.jsonl").open("x") as f:
        for i,row in enumerate(order):
            for method in row:
                invoke("old" if method=="old_report" else "new",method,i)
                cell=read(a.output/f"cell-{i}-{method}.json")
                f.write(json.dumps(cell,separators=(",",":"),allow_nan=False)+"\n");f.flush()
                runs.append(cell)
                for case,v in cell["cases"].items():
                    ref=runs[0]["cases"][case]
                    check(all(v[k]==ref[k] for k in ("histories","finish","checkpoints")) and cell["base"]==runs[0]["base"]
                          and (not method.endswith("report") or v["reports_sha256"]==ref["reports_sha256"]),"first parity mismatch")
    ratios=reduce(runs)
    elapsed=time.monotonic()-start
    check(elapsed<=LIMIT,"total wall budget exceeded")
    result=dict(ratios=ratios,attempt_sha256=sha(a.output/"attempt.json"),protocol_sha256=sha(a.output/"protocol.json"),
                runs_sha256=sha(a.output/"runs.jsonl"),cells=len(runs),elapsed_seconds=elapsed)
    save(a.output/"result.json",result)
    print(json.dumps({k:{n:c[n]["median"] for n in ("report_old_over_new","id_old_over_new","chunk_old_over_new")} for k,c in result["ratios"].items()}))


if __name__=="__main__":
    os.umask(0o077)
    check(sys.flags.isolated and sys.dont_write_bytecode,"use python -I -B")
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    for side in ("old","new"):
        for param in ("python","wheel","pin"):
            ap.add_argument(f"--{side}-{param}",type=str if param=="pin" else Path)
    ap.add_argument("--worker",choices=("admit",*METHODS),help=argparse.SUPPRESS)
    ap.add_argument("--index",type=int,help=argparse.SUPPRESS)
    ap.add_argument("--wheel",type=Path,help=argparse.SUPPRESS)
    ap.add_argument("--pin",help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.worker: worker(a)
    else:
        check(all(getattr(a,s+"_"+p) for s in ("old","new") for p in ("python","wheel","pin")),"specify both pinned wheels/interpreters")
        main(a)

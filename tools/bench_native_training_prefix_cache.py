"""Synthetic paired CPU benchmark of cold-start, four-epoch native training. No pretrained data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"bindings/st-py/tests/test_native_causal_lm.py"
ARMS=ROOT/"benchmarks/results/2026-11-03-native-factorial-answer-training/design.py"
CFG=dict(width=64,heads=4,layers=6,positions=32,vocabulary=97)
DOCUMENTS,CONTEXT,BATCH,EPOCHS=12,25,2,4
PAIRS=6


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def save(path,value):
    with path.open("x") as f:
        json.dump(value,f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")


def check(ok,msg):
    if not ok: raise ValueError(msg)


def main(out,wheel):
    out.mkdir(mode=0o700,exist_ok=False)
    threads=("RAYON_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")
    for k in threads: os.environ[k]="1"
    os.environ["SPIRALTORCH_AUTOTUNE_STORE"]=str(out.resolve()/"autotune.json")
    forbidden={"numpy","torch","transformers","accelerate","peft","safetensors","tokenizers"}
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.partition(".")[0] in forbidden:
                raise ModuleNotFoundError("synthetic native benchmark only",name=fullname)
    sys.meta_path.insert(0,Guard())
    sys.path.insert(0,str(ROOT/"tools"))
    import byte_release_artifacts as wheels
    identity=wheels.installed_identity(wheels.wheel_archive_inventory(wheel))
    import spiraltorch as st
    check(identity["native_sha256"]==sha(st._rs.__file__),"different loaded extension")
    batches=DOCUMENTS//BATCH
    order=[["plain","cached"] if i%2==0 else ["cached","plain"] for i in range(PAIRS)]
    rn=CFG["width"]*DOCUMENTS*(CONTEXT-1)
    protocol=dict(schema="spiraltorch.synthetic_prefix_cache_bench.v1",shape=CFG,
                  arms=["affine","saturation","projection","both"],layer=CFG["layers"]-1,
                  context=CONTEXT,documents=DOCUMENTS,batch_windows=BATCH,epochs=EPOCHS,
                  max_values=rn,max_entries=batches,rank=2,alpha=4.,lr=.01,geometry_strength=.25,
                  order=order,warmup=["plain","cached"],target_tokens_per_batch=BATCH*4,
                  thread_env={k:os.environ[k] for k in threads},installed=identity,
                  script_sha256=sha(__file__),fixture_sha256=sha(FIXTURE),arm_constructor_sha256=sha(ARMS),
                  python=platform.python_version(),platform=platform.platform(),
                  scope="empty cache for each timed model cohort; only update loop timed; no real weights or data")
    save(out/"protocol.json",protocol)
    base=st.NativeCausalLm.from_gpt2(*runpy.run_path(str(FIXTURE))["fixture"](**CFG))
    models=runpy.run_path(str(ARMS))["origin"](st,base,CFG["width"],CFG["layers"]-1)
    check(base.execution_backend=="rust_f32_cpu_prepacked","unexpected execution")
    docs=[[(7*d+13*t+t*t)%CFG["vocabulary"] for t in range(CONTEXT)] for d in range(DOCUMENTS)]
    masks=[[-100]*(len(row)-4)+row[-4:] for row in docs]
    sess=st.NativeCausalLmTrainingSession
    template=sess(models[0],docs,CONTEXT,mode="joint",labels=masks,learning_rate=.01,
                  epochs=EPOCHS,batch_windows=BATCH,window_policy="covering_targets")
    check(template.updates_per_epoch==batches,"wrong training schedule")

    def measure(use):
        cohort=[template.fork_variant(m,"joint") for m in models]
        cache=st.NativeCausalLmTrainingPrefixCache(base,max_values=rn,max_entries=batches) if use else None
        reports,durations=[],[]
        for _ in range(EPOCHS):
            start=time.perf_counter_ns()
            for _ in range(batches):
                reports.append(sess.step_matched(cohort,prefix_cache=cache))
            durations.append(time.perf_counter_ns()-start)
        digest=hashlib.sha256()
        for step,(depth,items) in enumerate(reports,1):
            check(depth==CFG["layers"]-1 and len(items)==4,"incorrect prefix/roster")
            for r in items:
                check(r.update==step and r.target_tokens==BATCH*4,"step or labels differ")
                digest.update(r.to_json().encode()+b"\n")
        check(all(s.exhausted for s in cohort),"incomplete cohort")
        stats=None if cache is None else cache.stats()
        if use:
            check(stats["hits"]==batches*(EPOCHS-1) and stats["entries"]==stats["insertions"]==batches
                  and stats["retained_values"]==rn and stats["capacity_bypasses"]==stats["no_frozen_block"]==0
                  and stats["skipped_block_calls"]==batches*(EPOCHS-1)*(CFG["layers"]-1),"unexpected cache work")
        return dict(epoch_ns=durations,total_ns=sum(durations),cache=stats,trace_sha256=digest.hexdigest(),
                    checkpoint_ids=[s.checkpoint_id for s in cohort])

    signature=None
    ratios,cold,repeat=[],[],[]
    with (out/"runs.jsonl").open("x") as f:
        for i,sequence in enumerate([protocol["warmup"],*order]):
            pair={}
            for method in sequence:
                row=measure(method=="cached")
                f.write(json.dumps(dict(pair=i,method=method,**row),allow_nan=False,separators=(",",":"))+"\n")
                f.flush()
                sig=[row["trace_sha256"],row["checkpoint_ids"]]
                if signature is None: signature=sig
                check(sig==signature,"different step or checkpoint; stop without retry")
                pair[method]=row
            if i:
                a,b=pair["plain"],pair["cached"]
                ratios.append(a["total_ns"]/b["total_ns"])
                cold.append(a["epoch_ns"][0]/b["epoch_ns"][0])
                repeat.append(sum(a["epoch_ns"][1:])/sum(b["epoch_ns"][1:]))
    check(not forbidden & {s.partition(".")[0] for s in sys.modules},"foreign ML loaded")
    result=dict(base=base.base_model_identity,data_id=template.data_id,protocol_sha256=sha(out/"protocol.json"),
                runs_sha256=sha(out/"runs.jsonl"),all_trace_sha256=signature[0],all_checkpoint_ids=signature[1],
                paired_ratios=ratios,median_total_ratio=statistics.median(ratios),
                fill_ratios=cold,median_fill_ratio=statistics.median(cold),
                repeated_ratios=repeat,median_repeated_ratio=statistics.median(repeat),
                limits="synthetic native CPU, one host/shape/roster, no quality or GPU claim")
    save(out/"result.json",result)
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    os.umask(0o077)
    check(sys.flags.isolated and sys.dont_write_bytecode,"use python -I -B")
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--wheel",type=Path,required=True)
    a=ap.parse_args()
    main(a.output,a.wheel)

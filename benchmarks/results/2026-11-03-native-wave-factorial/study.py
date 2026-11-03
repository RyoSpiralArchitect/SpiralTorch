"""Run one fixed, train-input-only, local 2x2 operator comparison. No optimizer."""
import argparse
import hashlib
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
m, old = d.m, d.old
LIMIT = 1800


def source_pins(commit):
    files = [d.HERE / n for n in ("study.py","design.py","build.json")] + [
             d.PREV/"observe.py",d.PREV/"metrics.py",m.ROOT/"tools/owned_process.py",m.ROOT/"tools/byte_release_artifacts.py"]
    pins = {str(f.relative_to(m.ROOT)):m.sha(f) for f in files}
    for n,pin in pins.items():
        m.check(hashlib.sha256(subprocess.check_output(["git","-C",str(m.ROOT),"show",f"{commit}:{n}"])).hexdigest()==pin,
                "uncommitted study source")
    return pins


def run_worker(a):
    root = a.output
    start = m.read(root/"attempt.json")["started_epoch"]
    remaining = min(LIMIT,start+LIMIT-time.time())-3
    m.check(remaining > 1,"CPU budget expired")
    cap = math.floor(remaining)
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap))
    m.write(root/"worker.json",dict(started_epoch=time.time(),cpu_limit=cap))
    train = a.originals/"train.json"
    entries = {k:v for k,v in m.boundaries().items() if k[1] in d.SOURCES}
    paths = {k:a.originals/k[0]/(k[1]+".ck") for k in entries}
    old.allow_train_only(a.originals, [train,*paths.values()])
    m.check(len(paths)==6 and m.sha(train)==m.TRAIN and all("sha256:"+m.sha(f)==entries[k]["checkpoint_id"] for k,f in paths.items()),
            "wrong original input bytes")
    data = m.read(train)
    plan = old.train_plan(data)
    expected = d.anchors()
    for (o,arm,i),row in expected.items():
        m.validate_row(row,o,arm,i,entries[o,arm],plan[i])
    for k in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
        os.environ[k]="1"
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",SPIRALTORCH_AUTOTUNE_STORE=str(root/"autotune.json"))
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.partition(".")[0] in old.FOREIGN:
                raise ModuleNotFoundError("external ML disabled", name=fullname)
    sys.meta_path.insert(0,Guard())
    m.check(not old.FOREIGN & {n.partition(".")[0] for n in sys.modules},"foreign runtime preloaded")
    build = m.read(d.HERE/"build.json")
    inv = old.wheels.wheel_archive_inventory(a.wheel)
    m.check(inv["sha256"]==build["wheel_sha256"] and inv["native"]["sha256"]==build["native_sha256"]
            and build["core_commit"]==d.CORE,"instrument identity")
    identity = old.wheels.installed_identity(inv)
    import spiraltorch as st
    m.check(m.sha(Path(st._rs.__file__))==build["native_sha256"] and all(m.sha(a.model/n)==h for n,h in m.MODEL.items()),
            "wrong loaded extension or base")
    m.write(root/"manifest.json",dict(schema="spiraltorch.wave_factorial_inputs.v1",study_commit=a.commit,build=build,
            source_sha256=source_pins(a.commit),installed=identity,base=m.BASE,base_sha256=m.MODEL,train_sha256=m.TRAIN,
            original_checkpoint_ids={o:{arm:entries[o,arm]["checkpoint_id"] for arm in d.SOURCES} for o in m.ORDERS},
            factual_archive_sha256=d.ANCHOR,config=d.CFG,plan=plan))
    base = st.NativeCausalLm.from_gpt2((a.model/"config.json").read_text(),st.SafeTensorArchive(a.model/"model.safetensors"))
    m.check(base.base_model_identity==m.BASE and base.readout_accumulator=="f32" and base.execution_backend=="rust_f32_cpu_prepacked",
            "wrong decoder")
    calls = 0
    with (root/"observations.jsonl").open("x") as f:
        for order in m.ORDERS:
            for arm in d.SOURCES:
                e = entries[order,arm]
                model = st.NativeCausalLm.from_session_checkpoint_at(base,paths[order,arm].read_bytes(),e["checkpoint_id"],
                        update=96,recorded_data_id=e["data_id"],post_lora_id=e["owners"][0],post_geometry_id=e["owners"][1])
                for i,(ids,mask) in enumerate(zip(data["token_documents"],data["labels"],strict=True)):
                    m.check(time.time()<start+LIMIT,"wall deadline")
                    obs = json.loads(model.wave_factorial_observation_json(ids,[1,len(ids)],d.CFG,labels=mask))
                    row = dict(order=order,arm=arm,index=i,key=plan[i][0],observation=obs)
                    f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                    f.flush()
                    calls+=1
                    d.validate(row,expected[order,arm,i])
                m.check([model.lora_artifact_id,model.geometry_artifact_id]==e["owners"],"snapshot mutated")
                print(f"{order}/{arm}: {calls}/576, all factual rows reproduced",flush=True)
    m.check(calls==576 and not old.FOREIGN & {n.partition(".")[0] for n in sys.modules},"execution count or runtime")
    with (root/"observations.jsonl").open("rb") as f:
        orders = d.summarize(iter(f),expected)
    m.write(root/"result.json",dict(schema="spiraltorch.local_wave_factorial_study.v1",model_calls=calls,local_maps=4*calls,updates=0,
            eval_windows=0,orders=orders,manifest_sha256=m.sha(root/"manifest.json"),journal_sha256=m.sha(root/"observations.jsonl"),
            cpu_seconds=time.process_time(),finished_epoch=time.time()))


if __name__=="__main__":
    os.umask(0o077)
    m.check(sys.flags.isolated and sys.dont_write_bytecode,"use python -I -B")
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("originals","wheel","model","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--commit",required=True)
    ap.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    a=ap.parse_args()
    m.check(subprocess.check_output(["git","-C",str(m.ROOT),"rev-parse","HEAD"],text=True).strip()==a.commit,"wrong checkout")
    if a.worker:
        try:
            run_worker(a)
        except Exception as exc:
            m.write(a.output/"first-error.json",dict(type=type(exc).__name__,message=str(exc),at=time.time()))
            raise
    else:
        m.check(not a.output.resolve().is_relative_to(a.originals.resolve()),"refuse to write into source")
        a.output.mkdir(mode=0o700,exist_ok=False)
        start=time.time()
        m.write(a.output/"attempt.json",dict(started_epoch=start,limit_seconds=LIMIT,retries=0,commit=a.commit))
        with (a.output/"process.log").open("x") as log:
            outcome=old.run([sys.executable,"-I","-B",__file__,*sys.argv[1:],"--worker"],timeout=LIMIT-(time.time()-start),stdout=log)
        m.write(a.output/"exit.json",dict(status=outcome.returncode,finished_epoch=time.time()))
        sys.exit(outcome.returncode)

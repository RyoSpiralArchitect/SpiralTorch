"""One preregistered CPU GPT-2 field-extraction pilot. Prepare and fit are separate processes."""
import argparse
from collections import Counter
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"tools"))
import byte_release_artifacts as wheels  # noqa: E402

FIELDS = ("color", "shape", "material")
VALUES = (
    ("red","blue","green","yellow","dark red","pale blue","light green","bright yellow"),
    ("circle","square","oval","star","small circle","large square","thin oval","five point star"),
    ("wood","glass","steel","clay","pine wood","clear glass","brushed steel","red clay"),
)
ARMS = ("lora", "frozen_wave_lora", "joint_wave_lora")
FILES = {"config.json":"0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
         "model.safetensors":"248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
         "tokenizer.json":"8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"}
BASE_ID = "sha256:0c552e7a3d7fae610edd7a22831f91e0fa718b212d38d15a07ff875e3979fb3c"
SPEC = {"schema":"spiraltorch.native_field_extraction_pilot.v1","arms":ARMS,"train":96,"eval":48,
        "context":64,"train_batch":1,"eval_batch":2,"epochs":1,"lr":.01,"layer":11,"width":768,"rank":2,"alpha":4.,
        "wave":{"curvature":-1.,"saturation":1.,"porosity":.05,"max_values":65536},
        "strength":.25,"gate_bias_log_radius_initial":"all zeros, all trainable in joint only",
        "order_seed":"field-extraction-v1-order-20261103","checkpoints":[32,64,96],
        "score_updates":[0,96],"arithmetic":"f32","base_id":BASE_ID,"model_sha256":FILES,
        "primary":"final joint minus frozen WaveGate, equal-answer unsmoothed NLL",
        "pilot_positive_if":"primary < 0 and joint all-targets-top1 answer count >= frozen",
        "secondary":"token-weighted NLL, per-answer W/T/L, supplied-prefix top1 counts, comparisons to bare LoRA",
        "limits":"one ordering, one closed-vocabulary synthetic task, no decoding, no tuned threshold, no replication or parameter-matched affine control"}
FORBIDDEN = {"torch","numpy","transformers","peft","accelerate","safetensors"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path, value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,allow_nan=False,sort_keys=True,separators=(",",":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(path.read_bytes())


def records():
    used, splits = set(), {}
    for split, repeats in (("train",4),("eval",2)):
        rows = []
        for f,field in enumerate(FIELDS):
            for k in range(8):
                for rep in range(repeats):
                    def rank(combo):
                        return digest(f"field-v1:{split}:{f}:{k}:{rep}:{combo}".encode())
                    triple = min((c for c in itertools.product(range(8),repeat=3) if c[f]==k and c not in used),key=rank)
                    used.add(triple)
                    values = {name:VALUES[j][triple[j]] for j,name in enumerate(FIELDS)}
                    pos = (f+k+rep)%3
                    order = [n for n in FIELDS if n!=field]
                    if (f+k+rep)%2:
                        order.reverse()
                    order.insert(pos,field)
                    template=(f+k+rep)%2
                    if template:
                        prompt="Record: "+"; ".join(n+" = "+values[n] for n in order)+f".\nReturn the {field}.\nAnswer:"
                    else:
                        prompt=" | ".join(n+": "+values[n] for n in order)+f"\nWhat is the {field}?\nAnswer:"
                    key="-".join(map(str,triple))
                    rows.append(dict(key=key,field=field,values=values,position=pos,template=template,prompt=prompt,answer=" "+values[field]))
        rows.sort(key=lambda r:digest((SPEC["order_seed"]+":"+split+":"+r["key"]).encode()))
        splits[split]=rows
    return splits


def admit_time(deadline, previous_cpu=0.):
    remaining=min(deadline-time.time(),3600-previous_cpu)-2
    require(remaining>1 and deadline-time.time()<=3601,"invalid or exhausted one-hour budget")
    hard=math.floor(remaining)
    resource.setrlimit(resource.RLIMIT_CPU,(hard,hard))
    require(os.statvfs(".").f_bavail * os.statvfs(".").f_frsize >= 1024**3,"less than 1 GiB free")
    return hard


def package(args):
    for env in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
        os.environ[env]="1"
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",
                      SPIRALTORCH_AUTOTUNE_STORE=str(args.output/"autotune.json"))
    banned=FORBIDDEN|({"tokenizers"} if args.action=="fit" else set())
    require(not (banned & {n.partition(".")[0] for n in sys.modules}),"foreign module already loaded")
    class Guard:
        def find_spec(self, fullname,path=None,target=None):
            if fullname.partition(".")[0] in banned:
                raise ModuleNotFoundError("foreign ML disabled for pilot",name=fullname)
    sys.meta_path.insert(0,Guard())
    inventory=wheels.wheel_archive_inventory(args.wheel)
    require(inventory["sha256"]==args.wheel_pin,"wrong wheel")
    identity=wheels.installed_identity(inventory)
    import spiraltorch as st
    require(sha(st._rs.__file__)==identity["native_sha256"],"wrong loaded native extension")
    return st,identity


def prepare(args):
    admit_time(args.deadline)
    out=args.output
    out.mkdir(mode=0o700,exist_ok=False)
    st,installed=package(args)
    save(out/"protocol.json",dict(spec=SPEC,script_sha256=sha(__file__),installed=installed,
                                  deadline_epoch=args.deadline,source_commit=args.commit))
    require(sha(args.model_dir/"tokenizer.json")==FILES["tokenizer.json"],"unexpected tokenizer bytes")
    spec=importlib.machinery.PathFinder.find_spec("tokenizers",[str(args.tokenizers_site)])
    codec=importlib.util.module_from_spec(spec)
    sys.modules["tokenizers"]=codec
    spec.loader.exec_module(codec)
    require(codec.__version__=="0.22.2","unexpected tokenizer version")
    tok=codec.Tokenizer.from_file(str(args.model_dir/"tokenizer.json"))
    seen_ids,seen_keys=set(),set()
    stats,files={},{}
    for split,rows in records().items():
        ids,labels=[],[]
        for row in rows:
            text=row["prompt"]+row["answer"]
            sequence,mask=st.encode_supervised_document(tok,text,[(len(row["prompt"]),len(text))])
            boundary=next(i for i,v in enumerate(mask) if v!=-100)
            require(mask==[-100]*boundary+sequence[boundary:] and boundary>=1 and len(sequence)<=SPEC["context"]
                    and tok.decode(sequence[:boundary],skip_special_tokens=False)==row["prompt"]
                    and tok.decode(sequence[boundary:],skip_special_tokens=False)==row["answer"],"split or alignment differs")
            require(tuple(sequence) not in seen_ids and row["key"] not in seen_keys,"record leakage/duplicate")
            require(row["answer"]==" "+row["values"][row["field"]],"incorrect answer key")
            seen_ids.add(tuple(sequence))
            seen_keys.add(row["key"])
            ids.append(sequence)
            labels.append(mask)
        require(len(rows)==SPEC[split],"wrong split count")
        repeats=4 if split=="train" else 2
        require(set(Counter((r["field"],r["answer"]) for r in rows).values())=={repeats}
                and set(Counter(r["position"] for r in rows).values())=={len(rows)//3},"imbalanced split")
        for name,value in ((split+"-text.json",rows),(split+".json",dict(token_documents=ids,labels=labels))):
            save(out/name,value)
            files[name]=sha(out/name)
        stats[split]=dict(documents=len(rows),answers=len(rows),target_tokens=sum(x!=-100 for row in labels for x in row),
                          tokens=sum(map(len,ids)),longest_document=max(map(len,ids)))
    manifest=dict(protocol_sha256=sha(out/"protocol.json"),files=files,splits=stats,distinct_records=len(seen_keys),
                  tokenizer_version=codec.__version__,prep_cpu_seconds=time.process_time())
    save(out/"manifest.json",manifest)
    print(json.dumps(dict(manifest_sha256=sha(out/"manifest.json"),splits=stats)),flush=True)


def make_wave(st,base):
    width=SPEC["width"]
    def param(name,shape,values):
        return dict(name=name,shape=shape,values=values,role="parameter",requires_grad=True)
    return st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema":"spiraltorch.geometry_adapter_artifact.v1","base_model_identity":base.base_model_identity,
        "policy":"full_prefix_f32_no_cache.v1","training":True,"placements":[{
            "path":f"transformer.h.{SPEC['layer']}.attn","training":True,
            "recipe":{"schema":"spiraltorch.wave_gate_adapter.v2","features":width,"strength":SPEC["strength"],
                      "learnable_radius":True,"kernel":SPEC["wave"]},
            "tensors":[param("gate",[width],[0.]*width),param("bias",[width],[0.]*width),param("log_radius",[],[0.])]}]}))


def score(st,models,data,stats):
    r=st.NativeCausalLm.evaluate_variants_by_answer(models,data["token_documents"],SPEC["context"],
                                                   labels=data["labels"],microbatch_windows=SPEC["eval_batch"])
    require(r.shared_prefix_layers==SPEC["layer"] and (len(r),r.scored_documents,r.document_count)==(48,48,48)
            and all(v.target_tokens==stats["target_tokens"] and v.window_count==48 for v in r.evaluations),"evaluation coverage differs")
    raw=json.loads(r.to_json())
    require([a["document"] for a in raw["answers"]]==list(range(48)),"evaluation mapping differs")
    return dict(scores=raw,contrasts={"joint_minus_frozen":r.contrast(1,2),
                                     "joint_minus_lora":r.contrast(0,2),"frozen_minus_lora":r.contrast(0,1)})


def fit(args):
    out=args.output
    require(sha(out/"manifest.json")==args.manifest_pin,"manifest pin mismatch")
    m=read(out/"manifest.json")
    require(sha(out/"protocol.json")==m["protocol_sha256"],"protocol pin mismatch")
    p=read(out/"protocol.json")
    require(p["deadline_epoch"]==args.deadline and p["script_sha256"]==sha(__file__)
            and p["spec"]==json.loads(json.dumps(SPEC)) and p["source_commit"]==args.commit,"source or conditions changed")
    limit=admit_time(args.deadline,m["prep_cpu_seconds"])
    save(out/"attempt.json",dict(manifest_sha256=args.manifest_pin,started_epoch=time.time(),cpu_limit_seconds=limit,
                                 no_retry=True,deadline_epoch=args.deadline))
    st,installed=package(args)
    require(installed==p["installed"] and set(m["files"])=={"train.json","eval.json","train-text.json","eval-text.json"}
            and all(sha(out/n)==h for n,h in m["files"].items()),"frozen input or installation changed")
    require(all(sha(args.model_dir/n)==h for n,h in FILES.items()),"pretrained model/tokenizer bytes changed")
    base=st.NativeCausalLm.from_gpt2((args.model_dir/"config.json").read_text(),st.SafeTensorArchive(args.model_dir/"model.safetensors"))
    require(base.base_model_identity==BASE_ID and base.readout_accumulator=="f32"
            and base.execution_backend=="rust_f32_cpu_prepacked","wrong model/arithmetic/backend")
    lora=base.with_lora(SPEC["layer"],SPEC["rank"],SPEC["alpha"])
    wave=make_wave(st,base)
    both=lora.with_geometry(wave)
    train,evaluation=read(out/"train.json"),read(out/"eval.json")
    Session=st.NativeCausalLmTrainingSession
    control=Session(lora,train["token_documents"],SPEC["context"],labels=train["labels"],mode="lora",window_policy="covering_targets",
                    batch_windows=SPEC["train_batch"],epochs=SPEC["epochs"],learning_rate=SPEC["lr"])
    arms=[control,control.fork_variant(both,"lora"),control.fork_variant(both,"joint")]
    require(all(s.completed_updates==0 and s.total_updates==96 and s.data_id==control.data_id for s in arms),"wrong training plan")
    initial=score(st,[s.model for s in arms],evaluation,m["splits"]["eval"])
    save(out/"scores-000.json",initial)
    require(initial["contrasts"]["joint_minus_frozen"]["tied"]==48 and initial["scores"]["comparison"]["evaluations"][1:]==
            [initial["scores"]["comparison"]["evaluations"][1]]*2,"frozen and joint did not start identically")
    save(out/"initial-owners.json",dict(data_id=control.data_id,lora=lora.lora_artifact_id,wave=wave.artifact_id))
    print("initial evaluation saved; beginning 96 lockstep CPU updates",flush=True)
    def checkpoint(update):
        entries=[]
        for name,s in zip(ARMS,arms,strict=True):
            file=f"{name}-{update:03}.ck"
            pin=s.save_checkpoint_new(out/file)
            require(pin=="sha256:"+sha(out/file),"saved checkpoint differs")
            entry=dict(checkpoint_id=pin,recorded_data_id=s.data_id,completed_updates=s.completed_updates,
                       post_geometry_id=s.model.geometry_artifact_id,post_lora_id=s.model.lora_artifact_id)
            boundary=f"{name}-{update:03}-boundary.json"
            save(out/boundary,entry)
            entries.append(dict(file=file,boundary=boundary,**entry))
        save(out/f"cohort-{update:03}.json",entries)
        return entries
    targets=0
    with (out/"steps.jsonl").open("x") as journal:
        for step in range(1,97):
            require(time.time()<args.deadline,"wall budget expired")
            depth,reports=Session.step_matched(arms)
            expected=sum(v!=-100 for v in train["labels"][step-1])
            targets+=expected
            record=[json.loads(r.to_json()) for r in reports]
            journal.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
            journal.flush()
            require(depth==11 and all(r.update==step and r.window_start==step-1 and r.window_end==step
                    and r.target_tokens==expected and math.isfinite(r.loss) for r in reports),"incorrect matched update")
            require(reports[0].post_geometry_artifact_id is None and reports[1].post_geometry_artifact_id==wave.artifact_id,
                    "a frozen owner changed")
            if step==1:
                require(reports[1].loss==reports[2].loss and reports[1].post_lora_artifact_id==reports[2].post_lora_artifact_id,
                        "first update did not start from matched owners")
            if step in SPEC["checkpoints"]:
                entries=checkpoint(step)
            if step%8==0:
                print(f"completed {step}/96, {targets} supervised targets per arm",flush=True)
    require(targets==m["splits"]["train"]["target_tokens"] and all(s.exhausted for s in arms),"incomplete training")
    restored=[st.NativeCausalLm.from_session_checkpoint_at(base,(out/e["file"]).read_bytes(),e["checkpoint_id"],
              update=96,recorded_data_id=e["recorded_data_id"],post_lora_id=e["post_lora_id"],
              post_geometry_id=e["post_geometry_id"]) for e in entries]
    final=score(st,restored,evaluation,m["splits"]["eval"])
    save(out/"scores-096.json",final)
    require(final["scores"]["comparison"]["evaluations"][0]["data_id"]==initial["scores"]["comparison"]["evaluations"][0]["data_id"],
            "evaluation data changed")
    c=final["contrasts"]["joint_minus_frozen"]
    require(not (FORBIDDEN|{"tokenizers"}) & {n.partition(".")[0] for n in sys.modules},"foreign runtime loaded")
    result=dict(manifest_sha256=args.manifest_pin,source_commit=args.commit,installed=installed,base_id=BASE_ID,
                updates_per_arm=96,training_targets_per_arm=targets,evaluation=m["splits"]["eval"],arms=ARMS,
                receipt_sha256={n:sha(out/n) for n in ["attempt.json","initial-owners.json","steps.jsonl",
                               "scores-000.json","scores-096.json","cohort-032.json","cohort-064.json","cohort-096.json"]},
                primary=c,pilot_positive=c["answer_weighted_delta_nll"]<0 and
                    c["candidate_all_correct_answers"]>=c["reference_all_correct_answers"],
                final_probe_nll=[v["nll"] for v in final["scores"]["comparison"]["evaluations"]],
                start_probe_nll=[v["nll"] for v in initial["scores"]["comparison"]["evaluations"]],
                train_cpu_seconds=time.process_time(),wall_seconds=time.time()-read(out/"attempt.json")["started_epoch"],
                final_checkpoints_verified=True,free_running_generation=False,gpu=False)
    save(out/"result.json",result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    require(sys.flags.isolated and sys.dont_write_bytecode,"require python -I -B")
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","fit"])
    for key in ("output","model-dir","wheel"):
        ap.add_argument("--"+key,type=Path,required=True)
    for key in ("wheel-pin","commit"):
        ap.add_argument("--"+key,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    ap.add_argument("--tokenizers-site",type=Path)
    ap.add_argument("--manifest-pin")
    ap.add_argument("--owned-worker",action="store_true",help=argparse.SUPPRESS)
    a=ap.parse_args()
    require(subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()==a.commit,
            "checkout does not match the selected source commit")
    if not a.owned_worker:
        from owned_process import run
        wall=a.deadline-time.time()
        require(0<wall<=3601,"invalid or expired absolute deadline")
        result=run([sys.executable,"-I","-B",__file__,*sys.argv[1:],"--owned-worker"],
                   timeout=min(wall,3600),stdout=sys.stdout)
        raise SystemExit(result.returncode)
    if a.action=="prepare":
        require(a.tokenizers_site is not None,"supply the local tokenizer package location")
        prepare(a)
    else:
        require(a.manifest_pin is not None and a.tokenizers_site is None,"fit requires a manifest pin and no external tokenizer")
        fit(a)

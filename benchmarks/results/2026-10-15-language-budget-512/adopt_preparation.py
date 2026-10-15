"""Revalidate a retained fresh preparation; never relabel old training state."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path,value):
    raw = value if isinstance(value,bytes) else (json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),"wb") as out:
        out.write(raw)


def load(repo,name):
    spec=importlib.util.spec_from_file_location(name,repo/"tools"/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def differences(old,new,path=""):
    if isinstance(old,dict) and isinstance(new,dict):
        require(old.keys()==new.keys(),"recipe field set changed: "+path)
        return [item for key in old for item in differences(old[key],new[key],path+"/"+key)]
    if isinstance(old,list) and isinstance(new,list):
        require(len(old)==len(new),"recipe shape changed: "+path)
        return [item for i,(a,b) in enumerate(zip(old,new)) for item in differences(a,b,path+"/"+str(i))]
    if type(old) in (int,float) and type(new) in (int,float):
        if struct.pack("<f",old)!=struct.pack("<f",new):
            return [dict(path=path,absolute_difference=abs(old-new))]
    else:
        require(type(old) is type(new) and old==new,"non-numeric policy changed: "+path)
    return []


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","source","output","historical"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    original=(args.source/"v3/request.json").read_bytes()
    raw=original
    artifacts={"v3/request.json":sha(original)}
    for version,verifier in (("v5","verify_byte_corpus_preparation"),("v7","verify_byte_corpus_mixture_preparation"),("v8","byte_corpus_query")):
        packet_raw=(args.source/(version+"-packet.json")).read_bytes()
        packet=json.loads(packet_raw)
        if version=="v8":
            candidate,report=load(args.repo,verifier).verify_preparation(raw,packet)
        else:
            report=load(args.repo,verifier).verify(raw,packet)
            candidate=packet["request_json"].encode()
        require(candidate==(args.source/(version+"-request.json")).read_bytes(),"retained stage bytes changed")
        write(args.output/(version+"-verification.json"),report)
        artifacts[version+"-packet.json"]=sha(packet_raw)
        raw=candidate
    new=json.loads(raw)
    old_raw=args.historical.read_bytes()
    old=json.loads(old_raw)
    require(len(new["cases"])==25 and len(new["train_batches"])==512,"bounded study size")
    for key in ("config","train_documents","validation_documents","validation_batches","rate","checkpoint_every"):
        require(new[key]==old[key],"recipe changed beyond budget/preparation: "+key)
    require(new["train_batches"][:128]==old["train_batches"],"historical batch prefix differs")
    old_cases={c["name"]:c for c in old["cases"] if c["seed"]==11}
    require(set(old_cases)=={c["name"] for c in new["cases"]},"case coverage")
    deltas=[dict(case=c["name"],changes=differences(old_cases[c["name"]],c)) for c in new["cases"]]
    write(args.output/"historical-differences.json",dict(historical_request_sha256=sha(old_raw),
          request_sha256=sha(raw),equal=False,scope="Initial preparation differences only; no model trained in rejected v1 pilot",cases=deltas))
    import spiraltorch as st
    study=st.nn.ByteCorpusStudy.from_json(raw)
    study.validate_segment()
    require(study.total_updates==512,"wrong runtime budget")
    write(args.output/"request.json",raw)
    write(args.output/"prepared.json",dict(schema="spiraltorch.byte_corpus.fresh_budget_preparation.v1",
          request_sha256=sha(raw),plan_sha256=sha((args.output/"PLAN.md").read_bytes()),
          runner_sha256=sha((args.source/"run_budget.py").read_bytes()),
          adopter_sha256=sha(Path(__file__).read_bytes()),historical_request_sha256=sha(old_raw),
          retained_preparation_hashes=artifacts,primary_contrast="within_new_run_128_256_512",
          historical_initial_bits_equal=False,arms=25,updates=512,seed=11,
          checkpoint_size_bound=study.checkpoint_size_bound()))
    print(json.dumps(dict(prepared=True,request_sha256=sha(raw),historical_initial_bits_equal=False)),flush=True)


if __name__=="__main__":
    main()

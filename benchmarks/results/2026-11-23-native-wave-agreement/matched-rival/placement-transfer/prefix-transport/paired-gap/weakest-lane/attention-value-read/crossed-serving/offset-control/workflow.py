"""Exclusive, fail-first calibration and immutable score replay on the previous synthetic study."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze
import contract as k
import measure


def build(out):
    dest=out/"bin";dest.mkdir(mode=0o700)
    common=["rustc","--edition=2021","-C","debuginfo=0","-C","strip=symbols","-C","opt-level=1"]
    core=k.REPO/k.RUST;lib=dest/"libpaired_offset.rlib"
    commands=[common+["--test",str(core),"-o",str(dest/"rust-tests")],
              [str(dest/"rust-tests")],
              common+["--crate-name","paired_offset","--crate-type","rlib",str(core),"-o",str(lib)],
              common+[str(k.HERE/"offset_stream.rs"),"--extern","paired_offset="+str(lib),"-o",str(dest/"offset_stream")],
              [sys.executable,"-ISB",str(k.HERE/"test_offset.py")]]
    for args in commands:
        subprocess.run(args,check=True)
    version=subprocess.check_output(["rustc","-Vv"],text=True)
    print(k.d.save(out/"built.json",dict(compiler=version,code=k.sources(),engine=k.d.digest((dest/"offset_stream").read_bytes()))))


def runner():
    return runpy.run_path(str(k.GOV))["run"]


def doctor(old,cross,out):
    runner()(out/"build-cpu",120,[sys.executable,"-ISB",str(k.HERE/"workflow.py"),"build",str(out)])
    build=k.c.s.k.read(out/"built.json")
    k.need(k.c.s.k.read(out/"build-cpu/receipt.json")["success"] and build["code"]==k.sources(),"bad build")
    cf=k.c.x.read(cross/"frozen.json",k.CROSS)
    k.need(cf["wheel"]==k.c.WHEEL and cf["extension"]==k.c.EXTENSION,"wheel identity")
    for kind in k.RESULT:
        # Bind parent journal/plan identities here. No transfer logits needed for this or fitting.
        old_receipt=k.c.x.read(old/kind/"result.json",k.c.PINS[kind])
        k.need(tuple(old_receipt[n] for n in ("inputs","trained","checked"))==k.PARENT[kind],"parent hashes")
        k.c.x.read(cross/(kind+"-plan.json"),cf["plans"][kind])
    print(k.d.save(out/"frozen.json",dict(schema="spiraltorch.paired_token_offset_control.v1",code=build["code"],
        engine=build["engine"],compiler=build["compiler"],source_result=k.c.PINS,parents=k.RESULT,cross=k.CROSS,
        plans=cf["plans"],native_training_forwards={"rehearsal":64,"formal":4608},native_transfer_forwards=0,updates=0)))


def fit_once(old,cross,out,kind,pin):
    k.c.x.read(out/"frozen.json",pin)
    if kind=="formal":
        k.need(k.c.s.k.read(out/"rehearsal-cpu/receipt.json")["success"],"no successful calibration rehearsal")
    os.environ.update(SPIRALTORCH_BACKEND="cpu",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",RAYON_NUM_THREADS="1")
    runner()(out/(kind+"-cpu"),60 if kind=="rehearsal" else 120,[sys.executable,"-ISB",str(k.HERE/"measure.py"),
             str(old),str(cross),str(out),kind,pin])
    print(k.d.digest((out/kind/"fitted.json").read_bytes()))


def records(path):
    return list(k.c.s.s.legacy.records(path))


def seal(out,pin):
    freeze=k.c.x.read(out/"frozen.json",pin);prog=out/"bin/offset_stream"
    k.need(k.d.digest(prog.read_bytes())==freeze["engine"] and k.sources()==freeze["code"],"binary/source drift")
    seals={}
    for kind in ("rehearsal","formal"):
        dest=out/kind;raw=(dest/"fitted.json").read_bytes();fit=json.loads(raw)
        cpub=(out/(kind+"-cpu")/"receipt.json").read_bytes();cpu=json.loads(cpub)
        k.need(cpu["success"] and cpu["exit_code"]==0 and not cpu["retry"] and cpu["min_free"]>=4<<30
               and cpu["elapsed"]<=cpu["seconds_limit"]==(60 if kind=="rehearsal" else 120),"failed native budget")
        k.need(fit["frozen"]==pin and fit["forwards"]==freeze["native_training_forwards"][kind] and fit["updates"]==0
            and fit["diagonal"]*4==fit["forwards"] and k.d.digest((dest/"training.jsonl").read_bytes())==fit["source"]
            and k.fit(records(dest/"training.jsonl"),prog)==fit["fits"],"failed training-only replay")
        seals[kind]=dict(fitted=k.d.digest(raw),training=fit["source"],receipt=k.d.digest(cpub))
    print(k.d.save(out/"sealed.json",dict(frozen=pin,seals=seals)))


def apply(old,cross,out,pin,locked):
    freeze=k.c.x.read(out/"frozen.json",pin);seal=k.c.x.read(out/"sealed.json",locked)
    prog=out/"bin/offset_stream"
    k.need(seal["frozen"]==pin and set(seal["seals"])==set(k.RESULT) and freeze["code"]==k.sources()
           and k.d.digest(prog.read_bytes())==freeze["engine"],"unsealed fit or wrong binary")
    def no_model(event,args):
        if event=="open" and isinstance(args[0],str) and old/"site" in Path(args[0]).absolute().parents:
            raise ValueError("scoring attempted to load a model")
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                ("spiraltorch","numpy","torch","transformers","safetensors")):
            raise ValueError("unexpected model/network in analysis")
    sys.addaudithook(no_model)
    results={}
    for kind in ("rehearsal","formal"):
        dest=out/kind;fit=k.c.x.read(dest/"fitted.json",seal["seals"][kind]["fitted"])
        groups=measure.transfer(old,cross,out,kind,fit)
        with (dest/"transfer.jsonl").open("xb") as f:
            for g in groups: k.c.s.old.archive_step(f,g)
        report=analyze.analysis(groups,fit["fits"],prog)
        prior=k.c.x.read(cross/kind/"result.json",k.RESULT[kind])
        previous={(z["base"],z["teacher"],z["order"],z["template"],z["scale"]):z["outcomes"] for z in prior["strata"]}
        for z in report["cells"]:
            p=previous[tuple(z[n] for n in ("base","teacher","order","template","scale"))]
            k.need(z["target_both"]==p["UU"]["both"],"original UU pair count")
            if z["method"]=="unshifted" and z["source"]=="UA":
                k.need((z["both"],z["individual"],z["ties"],z["mean_weaker"])==
                       (p["UA"]["both"],p["UA"]["correct"],p["UA"]["ties"],p["UA"]["mean_weaker"]),"original UA outcomes")
        results[kind]=k.d.save(dest/"result.json",dict(frozen=pin,seal=locked,fit=seal["seals"][kind]["fitted"],
                transfer=k.d.digest((dest/"transfer.jsonl").read_bytes()),native_transfer_forwards=0,updates=0,**report))
    print(k.d.save(out/"results.json",dict(frozen=pin,seal=locked,results=results)))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="build": build(Path(sys.argv[2]))
    elif action=="freeze": doctor(*map(Path,sys.argv[2:5]))
    elif action=="fit": fit_once(*map(Path,sys.argv[2:5]),*sys.argv[5:])
    elif action=="seal": seal(Path(sys.argv[2]),sys.argv[3])
    elif action=="apply":
        old,cross,out=map(Path,sys.argv[2:5])
        runner()(out/"readback-cpu",60,[sys.executable,"-ISB",str(k.HERE/"workflow.py"),"read",*sys.argv[2:]])
    elif action=="read": apply(*map(Path,sys.argv[2:5]),*sys.argv[5:])
    else: raise ValueError("unknown command")

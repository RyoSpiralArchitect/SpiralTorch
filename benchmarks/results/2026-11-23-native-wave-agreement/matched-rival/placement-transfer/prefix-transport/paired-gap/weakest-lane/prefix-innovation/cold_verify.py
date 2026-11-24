"""Verify the source bundle and replay the archived evidence without loading any model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
import tarfile
import tempfile

def verify(capture,objects):
    live=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    sub=Path(__file__).resolve().parent.relative_to(live)
    env=dict(os.environ,GIT_ALTERNATE_OBJECT_DIRECTORIES=str(objects))
    def git(*args,cold=True,**kw):
        flags=["--git-dir="+str(capture/"restore.git")] if cold else []
        return subprocess.check_output(["git","--no-lazy-fetch",*flags,*args],cwd=live,env=env,**kw)
    def require(ok,why):
        if not ok:raise ValueError(why)
    sha=lambda data:hashlib.sha256(data).hexdigest()
    base="5962fbb81412ead6f8014c0de4b826c6844e1e2c"
    head=git("rev-parse","refs/heads/innovation").decode().strip()
    require(head==git("rev-parse","HEAD",cold=False).decode().strip(),"source head")
    ids=git("rev-list","--objects","--no-object-names","--missing=print",head,"^"+base,cold=False).splitlines()
    checks=git("cat-file","--batch-check=%(objectname) %(objecttype)",input=b"\n".join(ids)+b"\n").splitlines()
    require(len(ids)==len(checks)==len(set(ids)) and all(o==s.split()[0] and s.split()[1] in
            (b"commit",b"blob",b"tree",b"tag") for o,s in zip(ids,checks)),"missing restore object")
    changes=git("diff","--name-only","-z","677e7757ec9cdeddeccfc0ede120792c72c90c06",head,cold=False).split(b"\0")
    changed={}
    for name in filter(None,changes):
        n=name.decode();buf=git("show",head+":"+n);require(buf==(live/n).read_bytes(),n);changed[n]=sha(buf)
    reads=set()
    def opened(event,args):
        if event=="open" and isinstance(args[0],str):
            p=Path(args[0])
            if p.is_absolute() and live in p.parents and p.is_file():reads.add(p.relative_to(live).as_posix())
    sys.addaudithook(opened)
    audit=runpy.run_path(str(live/sub/"audit.py"))
    needed=set(audit["s"].sources())|reads|{(sub/"readback.py").as_posix()}
    results={}
    with tempfile.TemporaryDirectory(dir=capture,prefix="cold-") as staging:
        staging=Path(staging);source=staging/"source"
        for n in sorted(needed):
            rel=PurePosixPath(n);require(not rel.is_absolute() and ".." not in rel.parts,"source path")
            blob=git("show",head+":"+n);require(blob==(live/n).read_bytes(),n)
            dst=source/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(blob)
        ev=staging/"evidence";hosts=staging/"parents/base"
        with tarfile.open(capture/"journals-portable.tar.gz","r:gz") as tar:
            pins=json.load(tar.extractfile("SHA256.json"));seen=set()
            for item in tar:
                rel=PurePosixPath(item.name)
                require(item.isreg() and not rel.is_absolute() and ".." not in rel.parts and item.name not in seen,"archive member")
                seen.add(item.name)
                if item.name=="SHA256.json":continue
                raw=tar.extractfile(item).read();require(sha(raw)==pins[item.name],item.name)
                if item.name.startswith(("evidence/rehearsal/","evidence/formal/","parents/base/")) or item.name=="evidence/installed.json":
                    dst=staging/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
            require(seen==set(pins)|{"SHA256.json"},"incomplete archive")
        def run(script,*args):
            return subprocess.check_output([sys.executable,"-ISB",str(source/sub/script),*map(str,args)],cwd=staging)
        for kind in ("rehearsal","formal"):
            pin=results[kind]=run("audit.py",ev/kind,hosts/kind).decode().strip()
            require(pin==pins["evidence/"+kind+"/result.json"],"cold audit")
        table=run("readback.py",ev/"formal",results["formal"])
        require(sha(table)==pins["evidence/orders.tsv"] and table==(live/sub/"orders.tsv").read_bytes(),"decision table")
    for name,tests in (("rust-cpu",12),("causal-lm-cpu",376),("binding-cpu",4)):
        log=(capture/name/"process.log").read_text()
        require((("%d passed; 0 failed"%tests in log) if name!="binding-cpu" else ("Ran 4 tests" in log and "OK" in log)),name)
    for kind,limit in (("formal",300),("rehearsal",120)):
        receipt=json.loads((capture/(kind+"-cpu")/"receipt.json").read_bytes())
        require(receipt["success"] and receipt["exit_code"]==0 and not receipt["retry"] and receipt["elapsed"]<limit==receipt["seconds_limit"],kind)
    return dict(head=head,prerequisite=base,objects=len(ids),changed=changed,source_files=len(needed),
                archived_files=len(pins),cold_results=results,synthetic_formal_updates=2304,new_pretrained_calls=0,
                captures={n:sha((capture/n).read_bytes()) for n in ("source-portable.bundle","journals-portable.tar.gz","orders.tsv")})

if __name__=="__main__":
    root,objects=map(Path,sys.argv[1:])
    report=verify(root,objects)
    with (root/"closed.json").open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(report["head"],report["objects"],report["archived_files"],report["cold_results"])

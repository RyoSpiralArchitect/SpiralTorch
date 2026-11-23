"""Cold, no-model replay. Combine the small delta with the two sealed parent archives."""
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

HERE=Path(__file__).resolve().parent
VALUE="8a0d747f394342a5af7a525aca1076a5b29f6cf377fb535f636ec1e5076a9e19"
CROSS="7708b8d246434556115e6e4ce11fce2fb43294effd33a8629fabc32a2220eb49"
FROZEN="1027a252b47f765628f94b9ac5f6f05b8aca2c14952b85a8e0f9a6adc33ec277"
SEAL="dc8d25cad74316d711b4d829008cbe603a27632cf181c497280a17bf82232178"
RESULT={"formal":"4bba04cc6802282aa625cd53d4519e35bf767a9f0533cb06d900905def902658",
        "rehearsal":"990dad3af3e7539a34fa22aaaad8a559534caaa6c913559895a9919370348acd"}
WITNESS="7411634d97fd3635212c7cf93590788a32146c3a2a0f86a746b5efc61bab9ebc"
TABLE="9856782b9c08c946b5a68897f873aa440a0f43f8a0094dcc4a1aaf44ebf0e0d1"


def sha(buf): return hashlib.sha256(buf).hexdigest()
def need(test,msg):
    if not test: raise ValueError(msg)
def blob(raw): return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def pack(out):
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    sub=HERE.relative_to(repo).as_posix()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    entries=subprocess.check_output(["git","--no-lazy-fetch","ls-tree","-rz",head,"--",sub,
        "crates/st-nn/src/language/paired_offset.rs","crates/st-nn/src/language/mod.rs","crates/st-nn/examples/paired_token_offset.rs"])
    files={};tree={}
    for rec in filter(None,entries.split(b"\0")):
        fields,name=rec.split(b"\t"); mode,kind,oid=fields.decode().split()
        name=name.decode();v=(repo/name).read_bytes()
        need(kind=="blob" and mode=="100644" and blob(v)==oid,"source not at HEAD")
        files["source/"+name]=v;tree[name]=oid
    names=["built.json","frozen.json","sealed.json","results.json","decision-witnesses.json","strata.tsv",
           "metadata/typecheck.json","metadata/process.log"]
    for stage in ("build-cpu","rehearsal-cpu","formal-cpu","readback-cpu"):
        names.extend(stage+"/"+n for n in ("process.log","receipt.json"))
    for stage in RESULT:
        names.extend(stage+"/"+n for n in ("training.jsonl","fitted.json","transfer.jsonl","result.json"))
    files.update({"capture/"+n:(out/n).read_bytes() for n in names})
    meta=dict(source=sub,head=head,tree=tree,parents=[VALUE,CROSS],files={n:sha(v) for n,v in files.items()})
    files["manifest.json"]=(json.dumps(meta,sort_keys=True)+"\n").encode()
    dest=out/"offset-portable.tar.gz"
    with dest.open("xb") as f,gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0) as gz,tarfile.open(fileobj=gz,mode="w") as tar:
        for n,v in sorted(files.items()):
            info=tarfile.TarInfo(n);info.size=len(v);info.mode=0o600
            tar.addfile(info,io.BytesIO(v))
    return sha(dest.read_bytes())


def extract(path,pin,tag,root):
    need(sha(path.read_bytes())==pin,"wrong "+tag+" archive")
    with tarfile.open(path,"r:gz") as tar:
        idx="SHA256.json" if tag=="old" else "manifest.json"
        obj=json.load(tar.extractfile(idx))
        pins=obj if tag=="old" else obj["files"]; seen=set()
        for r in tar:
            rel=PurePosixPath(r.name)
            need(r.isreg() and not rel.is_absolute() and ".." not in rel.parts and str(rel)==r.name and r.name not in seen,"bad member")
            seen.add(r.name)
            if r.name==idx:continue
            first,*rest=rel.parts;dst=None
            if tag=="old":
                if first=="evidence" and len(rest)>=2 and rest[0] in RESULT and (len(rest)==2 or rest[1:]==["training","scores.jsonl"] or rest[1:]==["evaluation","scores.jsonl"]):
                    dst=root/"old"/Path(*rest)
            elif first=="source":
                dst=root/"repo"/Path(*rest)
            elif first=="capture":
                dst=root/tag/Path(*rest)
            if dst is not None:
                raw=tar.extractfile(r).read();need(sha(raw)==pins[r.name],"corrupt member "+r.name)
                if first=="source":need(blob(raw)==obj["tree"][str(Path(*rest))],"source object differs")
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    need(dst.read_bytes()==raw,"unexpected sourcetree overlay: "+str(dst))
                else:
                    with dst.open("xb") as f:f.write(raw)
        need(seen==set(pins)|{idx},"missing member")
    return obj


def replay(root):
    sys.path.insert(0,str(HERE))
    import analyze
    import contract as k
    import measure
    import readback
    old,cross,out=root/"old",root/"cross",root/"capture"
    freeze=k.c.x.read(out/"frozen.json",FROZEN)
    seal=k.c.x.read(out/"sealed.json",SEAL)
    need(freeze["code"]==k.sources() and seal["frozen"]==FROZEN and
         k.c.s.k.read(out/"results.json")==dict(frozen=FROZEN,seal=SEAL,results=RESULT),"bad source/seal")
    core=root/"repo"/k.RUST;rlib=root/"libpaired_offset.rlib";engine=root/"offset_stream"
    subprocess.run(["rustc","--edition=2021","--crate-name","paired_offset","--crate-type=rlib",str(core),"-o",str(rlib)],check=True)
    subprocess.run(["rustc","--edition=2021",str(HERE/"offset_stream.rs"),"--extern","paired_offset="+str(rlib),"-o",str(engine)],check=True)
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo") or (event=="import" and args[0].split(".")[0] in
                ("spiraltorch","torch","numpy","transformers","safetensors")): raise ValueError("unexpected model/network")
    sys.addaudithook(guard)
    for kind in ("rehearsal","formal"):
        dest=out/kind; s=seal["seals"][kind];fit=k.c.x.read(dest/"fitted.json",s["fitted"])
        need(fit["frozen"]==FROZEN and sha((dest/"training.jsonl").read_bytes())==fit["source"]==s["training"],"training lineage")
        rc=k.c.x.read(out/(kind+"-cpu")/"receipt.json",s["receipt"])
        need(rc["success"] and not rc["retry"] and rc["exit_code"]==0 and rc["min_free"]>=4<<30 and
             0<rc["elapsed"]<=rc["seconds_limit"]==(60 if kind=="rehearsal" else 120),"budget evidence")
        sp,owners,bank,_=k.training_context(old,cross,kind)
        train=list(k.c.s.s.legacy.records(dest/"training.jsonl"))
        expected=[(b,t,o,sc) for b,t,o in k.c.groups(sp) for sc in k.c.SCALES]
        diag=calls=0
        for z,(b,t,o,sc) in zip(train,expected,strict=True):
            need(z["group"]==k.group(b,t,o,sc),"changed fit group")
            factual=k.c.s.arm(b,"uniform",t,o)
            if sc=="raw":
                need(fit["owners"][k.c.name(b,t,o,sc,"uniform","uniform")]==owners[factual],"factual model ownership")
            for p,(l,r) in zip(z["pairs"],k.c.x.pairs(sp["train"],t),strict=True):
                need(p["keys"]==[l["key"],r["key"]] and p["template"]==l["template"]==r["template"],"fit pair provenance")
                for i,key in enumerate(p["keys"]):
                    bits=k.c.s.v.u.bits
                    need(bits(p["scores"]["origin"][i])==bits(bank[k.c.s.arm(b,"attention")][key]),"not saved origin")
                    bits(p["scores"]["UA"][i]);u=bits(p["scores"]["UU"][i]);calls+=2
                    if sc=="raw":
                        need(u==bits(bank[factual][key]),"not saved factual 11-logit row");diag+=1
        need(calls==fit["forwards"]==freeze["native_training_forwards"][kind] and diag==fit["diagonal"] and
             k.fit(train,engine)==fit["fits"],"fitted offsets did not replay")
        groups=measure.transfer(old,cross,out,kind,fit)
        log=b"".join(k.d.encoded(g) for g in groups);need(log==(dest/"transfer.jsonl").read_bytes(),"wrong transfer rows")
        report=dict(frozen=FROZEN,seal=SEAL,fit=s["fitted"],transfer=sha(log),native_transfer_forwards=0,updates=0,
                    **analyze.analysis(groups,fit["fits"],engine))
        need(k.d.encoded(report)==(dest/"result.json").read_bytes() and sha(k.d.encoded(report))==RESULT[kind],"cold results")
    witness=k.d.encoded(readback.diagnose(out/"formal",RESULT["formal"]))
    table=readback.table(k.c.x.read(out/"formal/result.json",RESULT["formal"]))
    need(witness==(out/"decision-witnesses.json").read_bytes() and sha(witness)==WITNESS,"wrong post hoc witnesses")
    need(table==(out/"strata.tsv").read_bytes()==(HERE/"strata.tsv").read_bytes() and sha(table)==TABLE,"wrong outcomes table")
    need("7 passed; 0 failed" in (out/"build-cpu/process.log").read_text() and "Ran 4 tests" in (out/"build-cpu/process.log").read_text()
         and k.c.s.k.read(out/"build-cpu/receipt.json")["success"] and k.c.s.k.read(out/"metadata/typecheck.json")["success"]
         and k.c.s.k.read(out/"readback-cpu/receipt.json")["success"],"incomplete verification evidence")


if __name__=="__main__":
    action=sys.argv[1]
    if action=="pack":print(pack(Path(sys.argv[2])))
    elif action=="replay":replay(Path(sys.argv[2]))
    elif action=="verify":
        delta,old,cross,pin,scratch=sys.argv[2:]
        with tempfile.TemporaryDirectory(dir=scratch) as stage:
            root=Path(stage);extract(Path(old),VALUE,"old",root);extract(Path(cross),CROSS,"cross",root)
            note=extract(Path(delta),pin,"capture",root)
            need(note["parents"]==[VALUE,CROSS] and not Path(note["source"]).is_absolute() and ".." not in Path(note["source"]).parts,"unbound source path")
            subprocess.run([sys.executable,"-ISB",str(root/"repo"/note["source"]/"recover.py"),"replay",str(root)],cwd=root/"repo",check=True)
            print(json.dumps(dict(delta=pin,head=note["head"],sources=len(note["tree"]),
                        evidence_files=len(note["files"])-len(note["tree"]),frozen=FROZEN,seal=SEAL,results=RESULT,
                        witness=WITNESS,table=TABLE,model_calls_in_replay=0),sort_keys=True))
    else:raise ValueError("expected pack, replay or verify")

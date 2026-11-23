"""Build a source/evidence delta; verify in empty storage with just it and the pinned parent archive."""
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
FROZEN = "fdeda563994b622f94a54338369d237bbd68a425131a5fb7d4143ec7c1bf64b0"
PARENT = "8a0d747f394342a5af7a525aca1076a5b29f6cf377fb535f636ec1e5076a9e19"
RESULT = {"formal":"26dc374dced142e1c69644347c1e43d0d010b4d69033d4f40805657198c32eb0",
          "rehearsal":"ca4cad8b11b605c9a40217146b0458c6ebef9de1bcdfe457c35ebd64fc5ccb48"}
GAP = "425b98de5ae6c76978fa25a90df78d754d3d0269dbc4f463f418e720aa50604f"
TABLE = "d91cdb3071919726b35ae9a1f98bce6382ad5b42688f8fd2044faf4fc59525b9"


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()


def pack(out):
    repo = Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    sub = HERE.relative_to(repo)
    seen = set()
    def opened(event, args):
        if event == "open" and isinstance(args[0], str):
            p = Path(args[0])
            if p.is_absolute() and repo in p.parents and p.is_file():
                seen.add(p.relative_to(repo).as_posix())
    sys.addaudithook(opened)
    c = runpy.run_path(str(HERE/"cross_design.py"))
    seen.update(c["s"].sources())
    c["s"].old.s.fixtures_cached()  # configuration/test fixture bytes, not model execution
    seen.update(f.relative_to(repo).as_posix() for f in HERE.iterdir() if f.is_file())
    seen.add((HERE.parents[5]/"parameter-vjp/bounded.py").relative_to(repo).as_posix())
    paths = sorted(seen)
    head = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    tree = subprocess.check_output(["git","--no-lazy-fetch","ls-tree","-z",head,"--",*paths]).split(b"\0")
    ids = {}
    for entry in filter(None,tree):
        meta, path = entry.split(b"\t")
        mode, obj, oid = meta.split()
        need(mode in (b"100644",b"100755") and obj==b"blob", "not a source blob")
        ids[path.decode()] = oid.decode()
    need(set(paths)==set(ids), "untracked source in capsule")
    files = {}
    for name in paths:
        raw = (repo/name).read_bytes()
        oid = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()
        need(oid == ids[name], "worktree differs from Git: "+name)
        files["source/"+name] = raw
    extras = ["frozen.json","rehearsal-plan.json","formal-plan.json","admitted.json","completed.json",
              "gap-readback.json","strata.tsv"]
    for kind in RESULT:
        extras += [kind+"/"+n for n in ("started.json","scored.json","scores.jsonl","witnesses.jsonl","result.json")]
        extras += [kind+"-cpu/"+n for n in ("receipt.json","process.log")]
    files.update({"capture/"+n:(out/n).read_bytes() for n in extras})
    manifest = dict(head=head, parent=PARENT, source=sub.as_posix(), tree=ids,
                    files={k:sha(v) for k,v in files.items()})
    files["manifest.json"] = encoded(manifest)
    dest = out/"cross-portable.tar.gz"
    with dest.open("xb") as fp, gzip.GzipFile(filename="",mode="wb",fileobj=fp,mtime=0) as stream:
        with tarfile.open(fileobj=stream,mode="w") as tar:
            for nm,raw in sorted(files.items()):
                info = tarfile.TarInfo(nm); info.size=len(raw); info.mode=0o600
                tar.addfile(info,io.BytesIO(raw))
    return sha(dest.read_bytes())


def unpack(path, root, main):
    with tarfile.open(path,"r:gz") as tar:
        where = "manifest.json" if main else "SHA256.json"
        note = json.load(tar.extractfile(where))
        pins = note["files"] if main else note
        seen = set()
        for item in tar:
            name = item.name
            rel = PurePosixPath(name)
            need(item.isreg() and not rel.is_absolute() and ".." not in rel.parts and str(rel)==name
                 and name not in seen, "unsafe or duplicate archive member")
            seen.add(name)
            if name == where:
                continue
            dst = None
            if main:
                need(rel.parts[0] in ("source","capture"), "unexpected delta path")
                dst = root/rel
            elif name.startswith("parents/base/"):
                dst = root/"capture/base"/rel.relative_to("parents/base")
            elif len(rel.parts)>=3 and rel.parts[:2] in (("evidence","formal"),("evidence","rehearsal")):
                # The parent also contains the historical training journal and wheel; do not materialize either.
                keep = (len(rel.parts)==3 or rel.parts[2:] == ("evaluation","scores.jsonl")
                        or (len(rel.parts)==4 and rel.parts[2]=="training" and name.endswith("-unit.json"))
                        or rel.parts[2:4] == ("training","weights"))
                if keep:
                    dst = root/"parent"/rel.relative_to("evidence")
            if dst is not None:
                raw = tar.extractfile(item).read()
                need(sha(raw)==pins[name], name)
                if main and rel.parts[0]=="source":
                    oid = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()
                    need(oid==note["tree"][str(rel.relative_to("source"))], "source object drift")
                dst.parent.mkdir(parents=True,exist_ok=True)
                with dst.open("xb") as f:
                    f.write(raw)
        need(seen==set(pins)|{where}, "incomplete archive")
    return note


def verify(delta, parent, pin, scratch):
    need(sha(delta.read_bytes())==pin and sha(parent.read_bytes())==PARENT, "archive digest")
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        root = Path(temp)
        m = unpack(delta,root,True)
        need(m["parent"]==PARENT and not PurePosixPath(m["source"]).is_absolute()
             and ".." not in PurePosixPath(m["source"]).parts, "foreign parent/source")
        unpack(parent,root,False)
        repo, ev = root/"source", root/"capture"
        src = repo/m["source"]
        need(sha((ev/"frozen.json").read_bytes())==FROZEN, "frozen input digest")
        def execute(script,*args):
            cmd = [sys.executable,"-ISB",str(src/script),*map(str,args)]
            return subprocess.check_output(cmd,cwd=repo,stderr=subprocess.STDOUT).decode()
        need("Ran 4 tests" in execute("test_cross.py"), "unit checks did not execute")
        need("Ran 3 tests" in execute("test_gap_readback.py"), "gap readback checks did not execute")
        for kind,limit in (("rehearsal",60),("formal",120)):
            pinfile = "admitted.json" if kind=="rehearsal" else "completed.json"
            rcraw = (ev/(kind+"-cpu")/"receipt.json").read_bytes(); rc=json.loads(rcraw)
            need(json.loads((ev/pinfile).read_bytes())==dict(frozen=FROZEN,result=RESULT[kind],receipt=sha(rcraw))
                 and rc["success"] and not rc["retry"] and rc["exit_code"]==0 and rc["seconds_limit"]==limit
                 and 0<rc["elapsed"]<=limit and rc["min_free"]>=4<<30, "failed one-shot budget")
            got = execute("audit.py",root/"parent",ev,kind,FROZEN,"--verify").strip()
            need(got==RESULT[kind],"cold prediction/site audit")
        execute("gap_readback.py",ev/"formal",RESULT["formal"],root)
        for name,expected in (("gap-readback.json",GAP),("strata.tsv",TABLE)):
            need(sha((root/name).read_bytes())==expected==sha((ev/name).read_bytes()),"post hoc readback drift")
        need((root/"strata.tsv").read_bytes()==(src/"strata.tsv").read_bytes(),"published table drift")
        return dict(head=m["head"],source_files=len(m["tree"]),evidence_files=len(m["files"])-len(m["tree"]),
                    delta=pin,parent=PARENT,frozen=FROZEN,results=RESULT,decomposition=GAP,table=TABLE,
                    new_model_calls=0)


if __name__ == "__main__":
    if sys.argv[1]=="pack":
        print(pack(Path(sys.argv[2])))
    elif sys.argv[1]=="verify":
        delta,parent,pin,scratch=sys.argv[2:]
        print(json.dumps(verify(Path(delta),Path(parent),pin,Path(scratch)),sort_keys=True))
    else:
        raise ValueError("expected pack or verify")

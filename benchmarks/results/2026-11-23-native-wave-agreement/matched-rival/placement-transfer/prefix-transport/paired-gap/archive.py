"""Pack and byte-check a native synthetic paired-gap campaign and its minimal parent inputs."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(root, parent, bank, selector, residual, output, *, additional_parents=(),
         evidence_files=("installed.json","rehearsal-succeeded.json","installed.py","paired-decisions.json","orders.tsv"),
         evidence_dirs=("formal-cpu","rehearsal-cpu","binding-cpu","rust-cpu","rust-layout-cpu","wheel-cpu","wheel")):
    files={}
    def take(name,path):
        if name in files or not path.is_file() or path.is_symlink():
            raise ValueError("missing, duplicate or non-regular input: "+name)
        files[name]=path
    def subtree(name,folder):
        for p in folder.rglob("*"):
            if p.is_file():
                take(name+"/"+p.relative_to(folder).as_posix(),p)
    for kind in ("formal","rehearsal"):
        subtree("evidence/"+kind,root/kind)
        sp=json.loads((root/kind/"training.json").read_bytes())
        for rel in ["complete.json","attempt/hosts.json"]+["base-%s.safetensors"%b for b in sp["bases"]]:
            take("parents/base/"+kind+"/"+rel,parent/kind/rel)
        ancestors=(("bank",bank,("inputs.json","training.json","evaluation.json")),
                   ("selector",selector,("inputs.json","transfer.json")),("residual",residual,("spec.json",)))
        for label,folder,names in ancestors+tuple(additional_parents):
            for n in names:take("parents/"+label+"/"+kind+"/"+n,folder/kind/n)
    for n in evidence_files:
        take("evidence/"+n,root/n)
    for n in evidence_dirs:
        subtree("evidence/"+n,root/n)
    hashes={}
    def add(t,name,raw):
        entry=tarfile.TarInfo(name)
        entry.mode=0o600
        entry.size=len(raw)
        t.addfile(entry,io.BytesIO(raw))
    with output.open("xb") as dst, gzip.GzipFile(fileobj=dst,mode="wb",mtime=0,filename="") as gz:
        with tarfile.open(fileobj=gz,mode="w") as tar:
            for name,path in sorted(files.items()):
                raw=path.read_bytes()
                hashes[name]=digest(raw)
                add(tar,name,raw)
            manifest=(json.dumps(hashes,sort_keys=True,separators=(",",":"))+"\n").encode()
            add(tar,"SHA256.json",manifest)
    verified=set()
    with tarfile.open(output,"r:gz") as tar:
        for m in tar:
            if not m.isreg() or m.name in verified or m.name not in set(files)|{"SHA256.json"}:
                raise ValueError("unexpected archive item")
            raw=tar.extractfile(m).read()
            if m.name=="SHA256.json":
                ok=raw==manifest
            else:
                ok=digest(raw)==hashes[m.name] and raw==files[m.name].read_bytes()
            if not ok:raise ValueError("read-back mismatch: "+m.name)
            verified.add(m.name)
    if verified!=set(files)|{"SHA256.json"}:raise ValueError("incomplete archive")
    return dict(archive_sha256=digest(output.read_bytes()),bytes=output.stat().st_size,
                verified_files=len(files),manifest_sha256=digest(manifest))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent","bank","selector","residual","output"):p.add_argument(name,type=Path)
    print(json.dumps(pack(**vars(p.parse_args())),sort_keys=True))

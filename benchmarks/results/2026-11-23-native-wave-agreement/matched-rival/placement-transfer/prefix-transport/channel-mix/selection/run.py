"""One synthetic CPU execution: collect training logits, freeze route choices, then open transfer."""
import argparse
import json
from pathlib import Path
import runpy
import struct
import subprocess
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
u,s,d,need=a.u,a.s,a.d,a.need

def native_selector(binary):
    def choose(groups,t):
        buf=bytearray(struct.pack("<I",len(groups)))
        for f in groups:
            for q in f:
                for pair in q:
                    buf.extend(struct.pack("<2f",*pair))
        run=subprocess.run([str(binary),"first" if t=="head" else "second"],input=buf,stdout=subprocess.PIPE,
                           timeout=5,check=True)
        n,i,threshold,*g=struct.unpack("<II7d",run.stdout)
        need(n==len(groups) and i<3,"invalid native selection")
        return dict(frames=n,selected=i,threshold=threshold,worst=g[:3],mean=g[3:])
    return choose

def collect(path,rows,bank,parent,sp,manifest,st):
    hosts=u.old.hosts(parent,sp)
    count=0;previous=None
    with path.open("xb") as f:
        for b,m,name in u.cells(sp):
            if b!=previous:
                host=hosts[str(b)]
                base=u.old.decoder(st,parent,sp,b,host)
                previous=b
            raw=u.parent_artifact(bank,manifest,name)
            ident,vec=u.c.state(raw,host,m)
            art=st.GeometryAdapterArtifact.from_json(raw.decode(),ident,host["base"])
            restored=base.with_geometry(art)
            need(u.old.current(restored,host,m)==(ident,vec,raw),"checkpoint changed on restore")
            for r in rows:
                prompt=r["prompt"]
                vs=restored.forward(prompt,[1,len(prompt)],last_only=True).values()
                u.c.v.u.bits(vs)
                u.old.archive_step(f,dict(name=name,owner=ident,key=r["key"],logits=vs))
                count+=1
    return count

def run(root,bank,parent,signal,binary,kind,inputs):
    root,bank,parent,signal=(p/kind for p in (root,bank,parent,signal))
    cfg,sp,manifest=u.load_inputs(root,bank,inputs)
    need(kind==cfg["kind"] and u.sha(binary)==cfg["selector"],"split or selector identity")
    d.save(root/"started.json",dict(inputs=inputs,selector=cfg["selector"],extension=cfg["extension"],retry=False))
    st=u.c.v.sig.native(cfg["extension"])
    path=root/"training-logits.jsonl"
    count=collect(path,sp["train"],bank,parent,sp,manifest,st)
    z=a.journal(path,bank,manifest,sp,sp["train"],u.c.raw_signals(signal,sp))
    selection=a.choices(sp,z,native_selector(binary))
    calibration=d.save(root/"calibration.json",dict(inputs=inputs,forwards=count,scores=u.sha(path),choices=selection))
    checked=d.save(root/"checked.json",a.review_training(root,bank,signal,inputs))
    print("sealed training-only choices",calibration,checked,flush=True)
    # No path to transfer logits, labels or inputs is read before the independent replay.
    rows=u.read(root/"transfer.json",cfg["transfer"])
    need(rows==u.fresh(kind),"unexpected transfer split")
    consumed=d.save(root/"consumed.json",dict(inputs=inputs,checked=checked,transfer=cfg["transfer"],retry=False))
    path=root/"transfer-logits.jsonl"
    count=collect(path,rows,bank,parent,sp,manifest,st)
    print("sealed transfer",d.save(root/"complete.json",dict(checked=checked,consumed=consumed,scores=u.sha(path),
                                                            forwards=count)),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("root","bank","parent","signal","binary"): cli.add_argument(n,type=Path)
    cli.add_argument("kind",choices=u.PINS)
    cli.add_argument("inputs")
    run(**vars(cli.parse_args()))

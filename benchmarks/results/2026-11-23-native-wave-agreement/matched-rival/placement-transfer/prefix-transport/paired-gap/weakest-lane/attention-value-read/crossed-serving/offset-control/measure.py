"""Get only missing training-side cohorts; never open a transfer score or target while fitting."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as k


def fence(old,cross,kind):
    allow={old/kind/n for n in ("inputs.json","training.json","trained.json","training-checked.json","training/scores.jsonl")}
    allow.update((cross/"frozen.json",cross/(kind+"-plan.json")))
    def guard(event,args):
        if event!="open" or not isinstance(args[0],str):
            return
        path=Path(args[0]).absolute()
        if (old==path or old in path.parents or cross==path or cross in path.parents):
            k.need(path in allow or old/"site" in path.parents or cross/"base"/kind in path.parents,
                   "fitting attempted to open a non-training input: "+str(path))
    sys.addaudithook(guard)


def train(old,cross,out,kind,pin):
    freeze=k.c.x.read(out/"frozen.json",pin)
    k.need(freeze["code"]==k.sources() and freeze["parents"]==k.RESULT,"changed source or cohorts")
    program=out/"bin/offset_stream"
    k.need(k.d.digest(program.read_bytes())==freeze["engine"],"changed Rust engine")
    fence(old,cross,kind)
    sp,owners,original,cells=k.training_context(old,cross,kind)
    rows=sp["train"]
    hosts=k.c.s.old.hosts(cross/"base"/kind,sp)
    sys.path.insert(0,str(old/"site"))
    st=k.c.s.v.sig.native(k.c.EXTENSION)
    models={b:k.c.s.old.decoder(st,cross/"base"/kind,sp,b,hosts[str(b)]) for b in sp["bases"]}
    dest=out/kind
    dest.mkdir(mode=0o700)
    groups=[]; ids={}; calls=diagonal=0
    with (dest/"training.jsonl").open("xb") as log:
        for b,t,o in k.c.groups(sp):
            for scale in k.c.SCALES:
                bank={"origin":original[k.c.s.arm(b,"attention")]}
                for code,route in (("UA","attention"),("UU","uniform")):
                    n=k.c.name(b,t,o,scale,"uniform",route)
                    cell=cells[n];h=hosts[str(b)]
                    art=st.GeometryAdapterArtifact.create_json(json.dumps(cell["artifact"]))
                    m=models[b].with_geometry(art)
                    ident,vec,raw=k.c.s.owner(m,h,route)
                    k.need(vec==cell["params"],"wrong serving weights")
                    ids[n]=ident
                    factual=code=="UU" and scale=="raw"
                    target=k.c.s.arm(b,"uniform",t,o)
                    if factual:
                        k.need(owners[target]==ident==cell["donor_id"] and k.d.digest(raw)==cell["donor_file"],"wrong raw UU")
                    bucket=bank[code]={}
                    for r in rows:
                        z=m.forward(r["prompt"],[1,len(r["prompt"])],last_only=True).values()
                        key=r["key"]; packed=k.c.s.v.u.bits(z);calls+=1
                        if factual:
                            if packed!=k.c.s.v.u.bits(original[target][key]):
                                k.d.save(dest/"first-mismatch.json",dict(cell=n,key=key,actual=z,expected=original[target][key]))
                                raise ValueError("changed factual training logits")
                            diagonal+=1
                        bucket[key]=z
                    k.need(k.c.s.owner(m,h,route)==(ident,vec,raw),"serving mutated weights")
                record=k.pairs_record(b,t,o,scale,rows,bank)
                k.c.s.old.archive_step(log,record);groups.append(record)
    expected=(4608,1152,36) if kind=="formal" else (64,16,4)
    k.need((calls,diagonal,len(groups))==expected and len(ids)==2*len(groups),"wrong training budget")
    fit=k.fit(groups,program)
    print(k.d.save(dest/"fitted.json",dict(frozen=pin,source=k.d.digest((dest/"training.jsonl").read_bytes()),
          owners=ids,fits=fit,forwards=calls,diagonal=diagonal,updates=0)),flush=True)


def transfer(old,cross,out,kind,fit):
    """Read the two already sealed evaluations. Not in the scope of the training worker."""
    prior=k.c.saved(old,kind)
    rows,z0=k.c.transfer(prior)
    result=k.c.x.read(cross/kind/"result.json",k.RESULT[kind])
    rec=k.c.x.read(cross/kind/"scored.json",result["scored"])
    path=cross/kind/"scores.jsonl"
    k.need(rec["scores"]==k.d.digest(path.read_bytes()) and result["frozen"]==k.CROSS,"changed old serving scores")
    by={name:{} for name in fit["owners"]}
    count=0
    for r in k.c.s.s.legacy.records(path):
        count+=1;n=r["name"]
        if n in by:
            k.need(r["owner"]==fit["owners"][n] and r["key"] not in by[n],"transfer owner/duplicate")
            k.c.s.v.u.bits(r["logits"]);by[n][r["key"]]=r["logits"]
    k.need(count==result["forwards"] and all(set(z)=={r["key"] for r in rows} for z in by.values()),"transfer coverage")
    groups=[]
    for b,t,o in k.c.groups(prior.sp):
        for scale in k.c.SCALES:
            bank={"origin":z0[k.c.s.arm(b,"attention")]}
            for label,route in (("UA","attention"),("UU","uniform")):
                bank[label]=by[k.c.name(b,t,o,scale,"uniform",route)]
            groups.append(k.pairs_record(b,t,o,scale,rows,bank))
    return groups


if __name__=="__main__":
    train(*map(Path,sys.argv[1:4]),*sys.argv[4:])

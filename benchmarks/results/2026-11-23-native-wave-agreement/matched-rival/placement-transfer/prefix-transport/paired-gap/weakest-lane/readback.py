"""Describe paired decisions and separate ordering from center movement on the same stored logits."""
import argparse
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
x,d,need=a.x,a.d,a.need


def describe(folder,pin):
    res=x.read(folder/"result.json",pin)
    inp,sp=a.s.data(folder,res["inputs"])
    checked=x.read(folder/"training-checked.json",res["checked"])
    out={}
    for split in ("train","transfer"):
        local=split=="train"
        rows=sp["train"] if local else x.read(folder/"transfer.json",inp["transfer"])
        part=folder/("training" if local else "evaluation")
        manifest=x.read(folder/("trained.json" if local else "scored.json"),res["trained" if local else "scored"])
        a.c.pinned(part/"scores.jsonl",manifest["files"]["scores.jsonl"])
        z=a.bank(sp,rows,part/"scores.jsonl",checked["owners"])
        need(a.m.analyze(sp,rows,z)==res[split],"cell or gate drift")
        split_report=out[split]={}
        for t in x.TEACHERS:
            table=[r for r in res[split]["cells"] if r["teacher"]==t]
            totals={}
            for arm in a.s.ARMS:
                cs=[r for r in table if r["arm"]==arm]
                total=lambda k:sum(r[k] for r in cs)
                block=lambda k,f:sum(r[k][f] for r in cs)
                totals[arm]=dict(n=total("n"),pairs=total("pairs"),both=total("both"),origin_both=total("both_before"),
                      raw=block("raw","correct"),origin_raw=block("raw","before"),raw_up=block("raw","up"),raw_down=block("raw","down"),
                      rank=block("rank","correct"),origin_rank=block("rank","before"),ties=total("weak_ties"),
                      mean_min=math.fsum(r["mean_min"]*r["pairs"] for r in cs)/total("pairs"),
                      mean_min_before=math.fsum(r["mean_min_before"]*r["pairs"] for r in cs)/total("pairs"))
            details={}
            for b in sp["bases"]:
                deltas=[];changes=[];worse=[];raw_counts=[]
                for o in sp["orders"]:
                    ce,w=(z[x.name(b,g,t,o)] for g in ("binary","weakest_lane"))
                    good=lambda bank,r:(1-2*r[t])*(bank[r["key"]][7]-bank[r["key"]][8])>0
                    for r in rows:
                        lhs,rhs=good(ce,r),good(w,r)
                        if lhs!=rhs:changes.append(dict(key=r["key"],order=o,gain=rhs))
                    for pair in x.pairs(rows,t):
                        wc,rc,cc=a.m.lanes(ce,pair)
                        ww,rw,cw=a.m.lanes(w,pair)
                        delta,rank,cent=ww-wc,rw-rc,abs(cc)-abs(cw)
                        need(math.isclose(delta,rank+cent,abs_tol=2e-13,rel_tol=2e-13),"margin decomposition")
                        deltas.append((delta,rank,cent))
                        sc,sw=(sum(good(bank,r) for r in pair) for bank in (ce,w))
                        raw_counts.append((sc,sw))
                        if sc and not sw:
                            worse.append(dict(key="/".join(r["key"] for r in pair),order=o,
                                              delta_min=delta,delta_rank=rank,center_reduction=cent))
                mean=lambda vs:math.fsum(vs)/len(vs)
                cnt=lambda rows:dict(gains=sum(p["gain"] for p in rows),losses=sum(not p["gain"] for p in rows),
                                     unique_changed_inputs=len({p["key"] for p in rows}))
                details[str(b)]=dict(pairs=len(deltas),weak_minus_ce={k:mean(v) for k,v in zip(
                    ("min","rank","center_reduction"),zip(*deltas))},raw_changes=cnt(changes),changed_inputs=changes,
                    ce_both_wrong=sum(p==0 for p,_ in raw_counts),weak_both_wrong=sum(q==0 for _,q in raw_counts),
                    new_both_wrong=worse)
            split_report[t]=dict(totals=totals,by_base=details)
    return dict(result=pin,descriptive=out,selection=res["selection"],transfer_gates=res["transfer"]["base_gates"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path);p.add_argument("pin");p.add_argument("output",type=Path)
    args=p.parse_args()
    print(d.save(args.output,describe(args.folder,args.pin)))

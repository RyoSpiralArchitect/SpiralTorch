"""No-model replay: compare two causal contexts at fixed lexical frame and opposite number."""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
from statistics import median
import struct
from types import SimpleNamespace

u=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))
d,k,s,need=u.d,u.k,u.s,u.need


def packed(values):
    return struct.pack("<%df"%len(values),*values)


def summary(seq):
    seq=[v for v in seq if v is not None]
    return dict(n=len(seq),median=median(seq) if seq else None,
                min=min(seq) if seq else None,max=max(seq) if seq else None)


def routes(singular,plural,role):
    sa,st=singular["trace"],plural["trace"]
    n,w=sa["shape"]
    pos={"head":1,"attractor":4}[role]
    need(st["shape"]==[n,w] and sa["rotation"]==st["rotation"] and pos<n,"context shape")
    need(packed(sa["site_input"][:pos*w])==packed(st["site_input"][:pos*w]),"noncausal site input")
    a,b=sa["source_features"],st["source_features"]
    U,V=sa["bias_gradient"],st["bias_gradient"]
    feature,readout=[],[]
    for i in range(n):
        f=[(a[i*w+j]-b[i*w+j])*(U[j]+V[j])/2 for j in range(w)]
        r=[(a[i*w+j]+b[i*w+j])*(U[j]-V[j])/2 for j in range(w)]
        feature.append(f);readout.append(r)
    total=lambda vs:[math.fsum(col) for col in zip(*vs)]
    f,r=total(feature),total(readout)
    expected=k.sub(sa["gate_gradient"],st["gate_gradient"])
    discrepancy=k.norm([x+y-z for x,y,z in zip(f,r,expected)])
    need(discrepancy<2e-6*max(1.,k.norm(expected)),"gate contrast failed exact symmetric decomposition")
    fn,rn=k.norm(f),k.norm(r)
    fp,rp=map(lambda vs:sum(map(k.norm,vs)),(feature,readout))
    dz=[math.fsum(a[j::w])-math.fsum(b[j::w]) for j in range(w)]
    avg=[(x+y)/2 for x,y in zip(U,V)]
    denom=k.norm(dz)*k.norm(avg)
    return dict(feature_norm=fn,readout_norm=rn,gradient_delta=k.norm(expected),residual=discrepancy,
                input_contrast=k.norm(k.sub(sa["site_input"][pos*w:(pos+1)*w],st["site_input"][pos*w:(pos+1)*w])),
                prefix_contrast=k.norm(dz),readout_contrast=k.norm(k.sub(U,V)),
                feature_coupling=fn/denom if denom else None,
                feature_kept=fn/fp if fp else None,readout_kept=rn/rp if rp else None,
                direct_feature_norm=k.norm(feature[pos]),source_feature_norms=list(map(k.norm,feature)),
                source_readout_norms=list(map(k.norm,readout)),feature_vector=f,readout_vector=r)


def verify(root,parent,masked):
    raw=(root/"spec.json").read_bytes();sp=json.loads(raw)
    need(d.encoded(u.preparation(sp["kind"],parent,masked,sp["extension"]))==raw,"plan/source change")
    out=root/"attempt";ids=k.identities(out); manifest=k.read(root/"complete.json")
    need(set(ids)=={"rows.jsonl"} and manifest==dict(spec=d.digest(raw),files=ids),"partial or altered journal")
    cells=[];pairs=[];oracle=[];worst=dict(add=0.,ce=0.,fd=0.)
    nrows=0
    with (out/"rows.jsonl").open("rb") as stream:
        for b in sp["bases"]:
            identities={}
            for turn in sp["turns"]:
                scenarios={};opens=[]
                for row in sp["train"]:
                    e=json.loads(next(stream));nrows+=1
                    checks=u.validate(e,sp,row,b,turn)
                    for key in worst: worst[key]=max(worst[key],checks[key])
                    t=e["trace"];key=row["key"]
                    sig=packed(t["site_input"]+t["upstream"]+t["bias_gradient"])
                    need(identities.setdefault(key,sig)==sig,"rotation changed the factual zero-weight site or adjoint")
                    frame=scenarios.setdefault(row["frame"],{})
                    key2=row["head"],row["attractor"]
                    need(key2 not in frame,"duplicate number cell");frame[key2]=e
                    g=t["contrast"]["target_minus_alternative"];sgn=1-2*row["head"]
                    if sgn*g<0:
                        dn=k.norm(t["gate_gradient"])
                        linear=u.RADIUS*dn/abs(g)
                        actual=(e["oracle"]["gap"]-g)*sgn/abs(g) if e["oracle"] is not None else None
                        record=dict(base=b,rotation=turn,key=key,origin_gap=g,gradient_norm=dn,linear_fraction=linear,
                                    actual_fraction=actual,linear_reaches=linear>=1.,actual_reaches=actual is not None and actual>=1.,
                                    actual_wins=e["oracle"] is not None and sgn*e["oracle"]["gap"]>0.)
                        opens.append(record);oracle.append(record)
                for role in ("head","attractor"):
                    matched=[]
                    for frame,grid in scenarios.items():
                        need(set(grid)=={(0,0),(0,1),(1,0),(1,1)},"incomplete factorial frame")
                        for fixed in (0,1):
                            before,after=((0,fixed),(1,fixed)) if role=="head" else ((fixed,0),(fixed,1))
                            data=routes(grid[before],grid[after],role)
                            pair=dict(base=b,rotation=turn,role=role,frame=frame,fixed_number=fixed,**data)
                            pairs.append(pair);matched.append(pair)
                    rates={field:summary(x[field] for x in matched) for field in
                           ("feature_norm","readout_norm","input_contrast","prefix_contrast","readout_contrast",
                            "feature_coupling","feature_kept","readout_kept","direct_feature_norm","gradient_delta")}
                    cells.append(dict(base=b,rotation=turn,role=role,pairs=len(matched),
                            feature_bigger=sum(x["feature_norm"]>x["readout_norm"] for x in matched),
                            exact_ties=sum(x["feature_norm"]==x["readout_norm"] for x in matched),
                            nonzero_local_input=sum(x["input_contrast"]>0 for x in matched),
                            wrong=len(opens),oracle_calls=sum(x["actual_fraction"] is not None for x in opens),
                            linear_reaches=sum(x["linear_reaches"] for x in opens),actual_reaches=sum(x["actual_reaches"] for x in opens),
                            actual_wins=sum(x["actual_wins"] for x in opens),
                            linear_fraction=summary(x["linear_fraction"] for x in opens),
                            actual_fraction=summary(x["actual_fraction"] for x in opens),**rates))
        need(not stream.read(1),"extra measurements")
    return dict(schema="spiraltorch.prefix_margin_signal_result.v1",kind=sp["kind"],spec=d.digest(raw),
                manifest=d.digest((root/"complete.json").read_bytes()),captures=nrows,ce_vjps=2*nrows,
                fd_forwards=6*nrows,oracle_forwards=sum(x["actual_fraction"] is not None for x in oracle),
                worst_abs_error=worst,cells=cells,pairs=pairs,oracle=oracle)


def table(path,records):
    def format_row(x):
        return {key:(json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(dict,list)) else v)
                for key,v in x.items()}
    with path.open("x",encoding="ascii",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader();writer.writerows(format_row(x) for x in records)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent","masked","json","cells","pairs"):
        cli.add_argument(name,type=Path)
    q=cli.parse_args();res=verify(q.root,q.parent,q.masked)
    print(d.save(q.json,res))
    table(q.cells,res["cells"]);table(q.pairs,res["pairs"])

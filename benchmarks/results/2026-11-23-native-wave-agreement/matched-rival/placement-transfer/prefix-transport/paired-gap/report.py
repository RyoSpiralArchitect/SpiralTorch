"""Descriptive read-back of raw decisions and all template/order cells. No decoder imports."""
import argparse
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("audit.py"))))
x,d,need=a.x,a.d,a.need


def describe(root,result_pin):
    res=x.read(root/"result.json",result_pin)
    sp=x.read(root/"training.json",x.read(root/"inputs.json",res["inputs"])["training"])
    chk=x.read(root/"training-checked.json",res["checked"])
    rows=[]
    report={}
    for split in ("train","transfer"):
        local=split=="train"
        work=root/("training" if local else "evaluation")
        manifest=x.read(root/("trained.json" if local else "scored.json"),res["trained" if local else "scored"])
        x.c.pinned(work/"scores.jsonl",manifest["files"]["scores.jsonl"])
        cases=sp["train"] if local else x.read(root/"transfer.json",x.read(root/"inputs.json",res["inputs"])["transfer"])
        z=a.load_scores(sp,cases,work/"scores.jsonl",chk["owners"])
        need(a.analyze(sp,cases,z)==res[split],"table does not match saved logits")
        table=res[split]["cells"]
        census={}
        for t in x.TEACHERS:
            totals={}
            for arm in x.ARMS:
                cs=[r for r in table if r["teacher"]==t and r["arm"]==arm]
                sums=lambda k,sub:sum(r[k][sub] for r in cs)
                totals[arm]=dict(decisions=sum(r["n"] for r in cs), pairs=sum(r["pairs"] for r in cs),
                    raw_before=sums("raw","before"),raw=sums("raw","correct"),raw_up=sums("raw","up"),raw_down=sums("raw","down"),
                    rank_before=sums("rank","before"),rank=sums("rank","correct"),rank_up=sums("rank","up"),
                    rank_down=sums("rank","down"),signed_margin=math.fsum(r["n"]*r["signed_margin"] for r in cs)/sum(r["n"] for r in cs),
                    mean_half_gap=math.fsum(r["pairs"]*r["mean_half_gap"] for r in cs)/sum(r["pairs"] for r in cs),
                    mean_abs_center_drift=math.fsum(r["pairs"]*r["abs_center_drift"] for r in cs)/sum(r["pairs"] for r in cs))
            def sign(n):return (n>0)-(n<0)
            key = lambda r:r["key"]
            gap = lambda cs,r:cs[key(r)][7]-cs[key(r)][8]
            by_base={}
            for b in sp["bases"]:
                disagree=0
                rep={}
                paired={arm:dict(both_correct=0,ordered=0,same_side=0,both_wrong=0,ties=0)
                        for arm in ("origin",)+x.ARMS}
                for order in sp["orders"]:
                    logits={arm:z[x.name(b,arm,t,order)] for arm in x.ARMS}
                    for row in cases:
                        disagree+=len({sign(gap(bank,row)) for bank in logits.values()})>1
                    for pos,neg in x.pairs(cases,t):
                        obs={}
                        for arm,bank in dict(logits,origin=z[x.name(b)]).items():
                            p,n=gap(bank,pos),gap(bank,neg)
                            half,center=(p-n)/2,(p+n)/2
                            weaker=min(p,-n)
                            need(math.isclose(weaker,half-abs(center),abs_tol=2e-13,rel_tol=2e-13),
                                 "weaker lane is not half-gap minus absolute center")
                            obs[arm]=(sign(half),half,center,weaker)
                            tally=paired[arm]
                            tally["ordered"]+=half>0
                            tally["both_correct"]+=p>0>n
                            tally["both_wrong"]+=p<0<n
                            tally["same_side"]+=sign(p)==sign(n)!=0
                            tally["ties"]+=p==0 or n==0
                        if obs["anchored_rank"][0]!=obs["binary"][0]:
                            tag=key(pos)+"/"+key(neg)
                            sg,r,c,w=obs["anchored_rank"]
                            rep.setdefault(tag,[]).append(dict(order=order,binary=obs["binary"][0],anchored=sg,
                                                               anchored_half_gap=r,anchored_center=c,weakest_lane=w))
                need(all(sum(s[k] for k in ("same_side","both_correct","both_wrong","ties"))==
                         len(sp["orders"])*len(cases)//2 for s in paired.values()),"pair outcome partition")
                by_base[str(b)]=dict(raw_disagreements_across_any_arm=disagree,rank_differences=rep,paired=paired)
            cs=[r for r in table if r["teacher"]==t and r["arm"]=="anchored_rank"]
            controls={(r["base"],r["order"],r["template"]):r for r in table if r["teacher"]==t and r["arm"]=="rank"}
            matches=[(r,controls[r["base"],r["order"],r["template"]]) for r in cs]
            census[t]=dict(totals=totals,base_detail=by_base,center_cells=len(cs),
                           reduced_center=sum(r["abs_center_drift"]<s["abs_center_drift"] for r,s in matches),
                           ten_percent=sum(0< s["abs_center_drift"] and r["abs_center_drift"]<=.9*s["abs_center_drift"] for r,s in matches))
        report[split]=census
        for r in table:
            record={k:r[k] for k in ("base","order","template","teacher","arm","n","pairs","signed_margin","origin_half_gap","mean_half_gap","abs_center_drift")}
            record["split"]=split
            for k in ("rank","raw"):
                record.update({k+"_"+field:value for field,value in r[k].items()})
            rows.append(record)
    return dict(result_sha256=result_pin,execution=res["execution"],transfer_forwards=res["transfer_forwards"],
                transfer_gates=res["transfer"]["base_gates"],descriptive=report), rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("pin")
    p.add_argument("report",type=Path);p.add_argument("table",type=Path)
    args=p.parse_args()
    summary,rows=describe(args.root,args.pin)
    sha=d.save(args.report,summary)
    fields=["split"]+[f for f in rows[0] if f!="split"]
    # Preserve binary64 round trips for independent analysis.
    with args.table.open("x",encoding="ascii",newline="") as out:
        print(*fields,sep="\t",file=out)
        for r in rows:print(*(r[f] for f in fields),sep="\t",file=out)
    print(sha,d.digest(args.table.read_bytes()),len(rows))

"""Identify contradictory signed rows and export *all* template cells. No model or fitting."""
import itertools
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze as a
import contract as k


def diagnose(folder, pin):
    r=k.c.x.read(folder/"result.json",pin)
    path=folder/"transfer.jsonl";k.need(k.d.digest(path.read_bytes())==r["transfer"],"wrong source rows")
    certs=[]
    for g in k.c.s.s.legacy.records(path):
        for src in k.SOURCES:
            rows=[]
            for p in g["pairs"]:
                for key,sg,tg in zip(p["keys"],k.contrasts(p["scores"][src]),k.contrasts(p["scores"]["UU"])):
                    rows.append(dict(key=key,template=p["template"],source_gap=sg,target_gap=tg))
            k.need(len(rows)==len({v["key"] for v in rows}),"duplicate prompt")
            for tp in [None]+sorted({v["template"] for v in rows}):
                part=[v for v in rows if tp is None or v["template"]==tp]
                need_ties=any(v["target_gap"]==0 for v in part)
                contradictions=[(p,n) for p,n in itertools.product(part,part)
                                if p["target_gap"]>0>n["target_gap"] and p["source_gap"]<=n["source_gap"]]
                example=max(contradictions,key=lambda e:(e[1]["source_gap"]-e[0]["source_gap"],e[0]["key"],e[1]["key"]),default=None)
                feasible=a.exact_decisions([[v["source_gap"]] for v in part],[[v["target_gap"]] for v in part])
                k.need(not (contradictions and feasible),"invalid threshold witness")
                certs.append(dict(group=g["group"],base=g["base"],teacher=g["teacher"],order=g["order"],scale=g["scale"],
                                  source=src,template=tp,rows=len(part),reference_has_ties=need_ties,
                                  exact_signs_possible=feasible,contradictions=len(contradictions),
                                  witness=None if example is None else {"wants_positive":example[0],"wants_negative":example[1]}))
    for v in r["group_bounds"]:
        g=k.group(v["base"],v["teacher"],v["order"],v["scale"])
        q,=[x for x in certs if x["group"]==g and x["source"]==v["source"] and x["template"] is None]
        k.need(v["exact_signs_possible"]==q["exact_signs_possible"],"certificate disagreement")
    return dict(result=pin,post_hoc=True,new_model_calls=0,certificates=certs)


def table(r):
    columns="base teacher order template scale source method offset pairs both individual ties target_both shared missed other identical_sign mean_weaker contrast_rmse".split()
    lines=["\t".join(columns)]
    for cell in r["cells"]:
        lines.append("\t".join(format(cell[n],".17g") if isinstance(cell[n],float) else str(cell[n]) for n in columns))
    return ("\n".join(lines)+"\n").encode()


if __name__=="__main__":
    folder,pin,out=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
    print(k.d.save(out/"decision-witnesses.json",diagnose(folder,pin)))
    raw=table(k.c.x.read(folder/"result.json",pin))
    with (out/"strata.tsv").open("xb") as f: f.write(raw)
    print(k.d.digest(raw))

"""Post-hoc scale control from a sealed journal. No perturbation is executed here."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

v=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
u,k=v.u,v.k


def read(root,parent,masked):
    # Require the original validations, not just plausible-looking gradient fields.
    result=v.verify(root,parent,masked)
    sp=k.read(root/"spec.json")
    head={s["key"]:1-2*s["head"] for s in sp["train"]}
    groups={}
    with (root/"attempt/rows.jsonl").open("rb") as f:
        for line in f:
            r=json.loads(line);t=r["trace"]
            g,b=k.norm(t["gate_gradient"]),k.norm(t["bias_gradient"])
            n,w=t["shape"]
            means=[v.math.fsum(t["source_features"][j::w]) for j in range(w)]
            base=v.math.sqrt(sum(x*x for x in means))
            z=t["contrast"]["target_minus_alternative"]
            groups.setdefault((r["base"],r["turn"]),[]).append(dict(readout_norm=k.norm(t["upstream"]),
                feature_norm=base,gate_norm=g,bias_norm=b,gate_over_bias=g/b if b else None,
                coupling=g/(b*base) if b and base else None,
                bias_fraction=u.RADIUS*b/abs(z) if head[r["key"]]*z<0 else None))
    rows=[]
    for (b,t),group in groups.items():
        fractions=[r["bias_fraction"] for r in group if r["bias_fraction"] is not None]
        row=dict(base=b,rotation=t,wrong=len(fractions),linear_bias_reaches=sum(x>=1 for x in fractions))
        for key in group[0]:
            vals=v.summary(r[key] for r in group)
            row.update({key+"_"+stat:vals[stat] for stat in ("n","median","max")})
        rows.append(row)
    return dict(source_manifest=result["manifest"],new_calls=0,rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent","masked","output"): p.add_argument(name,type=Path)
    q=p.parse_args();v.table(q.output,read(q.root,q.parent,q.masked)["rows"])

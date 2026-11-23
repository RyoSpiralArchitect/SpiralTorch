"""Four-map factorial read-back at fixed factual inputs, without model execution."""
import argparse
import json
import math
from pathlib import Path
import runpy
from statistics import median

reader = runpy.run_path(str(Path(__file__).with_name("block_readback.py")))
study = reader["study"]


def interaction(arms):
    keys = ("affine","saturation","projection","both")
    if set(arms) != set(keys):
        raise ValueError("all four local maps required")
    a,s,p,b = [arms[k] for k in keys]
    if not a or any(len(v)!=len(a) or not all(math.isfinite(x) for x in v) for v in (a,s,p,b)):
        raise ValueError("nonfinite or unmatched coordinates")
    additive = [y+z-x for x,y,z in zip(a,s,p)]
    n = study["norm"]
    residual = n([x-y for x,y in zip(b,additive)])
    an,bn,fn = n(a),n(b),n(additive)
    return dict(residual_over_affine=residual/an if an else None,
                residual_over_full=residual/bn if bn else None,
                additive_cosine_to_full=math.fsum(x*y for x,y in zip(additive,b))/(fn*bn) if fn and bn else None)


def summarize(values):
    present = [v for v in values if v is not None]
    return dict(n=len(present),undefined=len(values)-len(present),min=min(present) if present else None,
                median=median(present) if present else None,max=max(present) if present else None)


def read(root):
    original = reader["summarize"](root)
    def checked(name):
        raw = (root/name).read_bytes()
        if study["digest"](raw)!=original["pins"][name]:
            raise ValueError("source changed after original replay: "+name)
        return raw
    width = json.loads(checked("plan.json"))["shape"][1]
    raw = [json.loads(line) for line in checked("attempt/rows.jsonl").splitlines()]
    summary, per_row = {}, []
    for host in study["HOSTS"]:
        selected = [row for row in raw if row["host"]==host]
        if len(selected)!=24 or len({row["key"] for row in selected})!=24:
            raise ValueError("not the fixed 24 training prefixes per host")
        summary[host] = {}
        for blocks in (("gate",),("bias",),("gate","bias")):
            label = "+".join(blocks)
            vals = []
            for row in selected:
                coords = {arm:[x for key in blocks for x in study["vector"](params[key],width)]
                          for arm,params in row["receipt"]["arms"].items()}
                val = interaction(coords)
                vals.append(val)
                per_row.append(dict(host=host,key=row["key"],block=label,**val))
            summary[host][label] = {k:summarize([v[k] for v in vals]) for k in vals[0]}
    return dict(schema="spiraltorch.wavegate_local_interaction.v1",pins=original["pins"],
                original_calls=original["original_calls"],original_maps=original["original_local_maps"],
                new_model_calls=0,new_weight_updates=0,summary=summary,rows=per_row)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("report",type=Path)
    q=p.parse_args()
    encoded=study["encoded"](read(q.root))
    with q.report.open("xb") as f:
        f.write(encoded)
    print(study["digest"](encoded))

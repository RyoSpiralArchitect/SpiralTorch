"""Secondary, post-hoc check: local gold-minus-rival margin sensitivity at visited states."""
import argparse
import json
import math
from pathlib import Path
import runpy
from statistics import median

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE/"analyze.py"))
D, R, need, digest = (A[k] for k in ("D","R","need","digest"))


def local(loss, report, step):
    D["check_objective"]("rival", loss, report)
    margin = report["mean_target_minus_declared_rival"]
    t = math.exp(-abs(margin))
    # RestrictedRow::blend writes +/- wrong_mass / partition as f32.
    q = D["f32"]([(t if margin>=0 else 1.)/(1+t)])[0]
    need(q>0 and q==D["f32"]([-math.expm1(-report["declared_ce"])])[0], "undefined binary VJP scale")
    norm = step["gradient_l2"]/q
    rise = -step["gradient_dot_step"]/q
    dist = step["realized_l2"]
    need(all(math.isfinite(x) and x>0 for x in (norm,rise,dist)) and
         rise<=norm*dist*(1+1e-10), "invalid directional derivative")
    return dict(rival_probability_f32=q, margin_gradient_l2=norm,
                first_order_own_margin_delta=rise, realized_step_l2=dist)


def derive(archive, paths_file, paths_pin):
    raw = paths_file.read_bytes()
    need(digest(raw)==paths_pin, "wrong path report")
    path = json.loads(raw)
    need(path["schema"]=="spiraltorch.matched_update_cancellation.v1" and
         path["source_sha256"]==digest((HERE/"analyze.py").read_bytes()), "wrong path analysis")
    kind = path["kind"]
    need(kind in ("gpt2","rehearsal"), "unknown input scope")
    A["observe_inputs"](archive, kind)
    sums = (archive/"SHA256SUMS").read_bytes()
    need(digest(sums)==path["manifest"], "changed archive manifest")
    files = {file:pin for pin,file in (s.split("  ",1) for s in sums.decode().splitlines())}
    order_reports = {}
    for seed in path["comparisons"]:
        stage = {}
        for geom in ("affine","wave"):
            name = seed+"-"+geom+"_rival"
            rel = kind+"/attempt/"+name+".jsonl"
            journal = (archive/rel).read_bytes()
            need(digest(journal)==files[rel], "changed validated step journal")
            rows = [json.loads(s) for s in journal.splitlines()]
            need(len(rows)==path["routes"][name]["n"] and all(r["objective"]=="rival" for r in rows),
                 "wrong row scope")
            stage[geom] = [(r["key"], local(r["loss"], r["declared"], r["step"])) for r in rows]
        paired = []
        for (affine_key,a),(wave_key,w) in zip(stage["affine"],stage["wave"]):
            need(affine_key==wave_key, "unmatched training rows")
            paired.append(dict(key=affine_key,affine=a,wave=w,
                               norm_ratio_affine_over_wave=a["margin_gradient_l2"]/w["margin_gradient_l2"]))
        ratios = [p["norm_ratio_affine_over_wave"] for p in paired]
        order_reports[seed] = dict(rows=paired, first=paired[0]["key"],
                                   min_ratio=min(ratios), median_ratio=median(ratios), max_ratio=max(ratios),
                                   affine_larger=sum(r>1 for r in ratios))
    need(sum(len(s["rows"]) for s in order_reports.values())*4==path["archived_updates"], "partial comparison")
    return dict(schema="spiraltorch.archived_binary_margin_sensitivity.v1",kind=kind,
                path_report=paths_pin,input_manifest=path["manifest"],source=digest(Path(__file__).read_bytes()),
                orders=order_reports)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("--paths",type=Path,required=True)
    p.add_argument("--paths-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root,out=a.archive.resolve(),a.output.resolve()
    need(out!=root and root not in out.parents and not out.exists(), "output must be new and outside archive")
    result=derive(root,a.paths,a.paths_sha256)
    R["write"](out,D["encoded"](result))
    print("paired archived steps:", sum(len(s["rows"]) for s in result["orders"].values()))

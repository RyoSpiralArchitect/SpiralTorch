"""Verify archived model outputs and project-specific comparisons without loading any model."""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

p = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
d, k, s, u, a, need = p.d, p.k, p.s, p.u, p.a, p.need


def position_stats(rows, z):
    frames = {}
    for r in rows:
        key = (1-2*r["head"], 1-2*r["attractor"])
        frame = frames.setdefault(r["frame"], {})
        need(key not in frame, "duplicate number cell")
        logits = z[r["key"]]
        frame[key] = logits[7]-logits[8]
    components, counts = [], {"conflict_wins":0, "congruent_wins":0}
    for f in frames.values():
        need(set(f) == {(-1,-1),(-1,1),(1,-1),(1,1)}, "incomplete lexical frame")
        factors = [[1,h,t,h*t] for h,t in f]
        coeff = [math.fsum(v*x for v,x in zip(f.values(),axis))/4 for axis in zip(*factors)]
        for ((h,t),v),x in zip(f.items(),factors):
            need(math.isclose(k.dot(x,coeff),v,rel_tol=1e-12,abs_tol=1e-12), "factorial reconstruction")
            counts["congruent_wins" if h == t else "conflict_wins"] += h*v > 0
        components.append(coeff)
    return dict(zip(("offset","head","attractor","interaction"),k.avg(components)),**counts)


def archive(root, previous):
    raw = (root/"spec.json").read_bytes()
    sp = json.loads(raw)
    need(raw == d.encoded(p.plan(previous,sp["kind"])), "unsealed plan")
    complete = (root/"complete.json").read_bytes()
    out = root/"attempt"
    starts = {s.key(*x) for x in p.starts(sp)}
    names = starts | {s.key(*x) for x in p.cells(sp)}
    expected = {"steps.jsonl"} | {f+"/"+n+".json" for n in names for f in ("logits","weights")}
    ids = k.identities(out)
    need(ids.keys() == expected and json.loads(complete) == dict(spec=d.digest(raw),files=ids),
         "missing, modified or surplus evidence")
    anchors = p.anchor_rows(previous)
    hosts = k.read(previous/"attempt/hosts.json")
    states, zs, checks = {}, {}, {}
    keys = {r["key"] for r in sp["train"]+sp["test"]}
    for b in sp["bases"]:
        host = hosts[str(b)]
        ref = k.read(previous/"attempt"/("frozen-%d.json"%b))
        need(ref.keys() == keys, "reference coverage")
        for name in sorted(names):
            prefix,turn,owner = name.split("-")[:3]
            if prefix != str(b):
                continue
            pin, v = p.state((out/"weights"/(name+".json")).read_bytes(),host,turn,owner)
            r = k.read(out/"logits"/(name+".json"))
            z = r["rows"]
            need(r["owner"] == pin and z.keys() == keys, "unbound or partial logits")
            for key, zz in z.items():
                bits = u.bits(zz)
                if name in starts:
                    need(v == [0.]*6 and bits == u.bits(ref[key]), "origin mismatch")
            if owner == "bias" and name not in starts:
                key = (b,)+tuple(name.split("-")[3:])
                blob = b"".join(u.bits(z[key]) for key in sorted(z))
                need(checks.setdefault(key,blob) == blob, "bias-only endpoint depends on rotation")
            states[name], zs[name] = (pin,v), z
    control, n, max_error, checked = {}, 0, 0., 0
    with (out/"steps.jsonl").open("rb") as f:
        for b,t,o,teacher,order in p.cells(sp):
            name = s.key(b,t,o,teacher,order)
            start = s.key(b,t,o)
            before, v = states[start]
            for i,pair in enumerate(k.schedule(sp["train"],order,"pair")*sp["epochs"]):
                rec = json.loads(next(f))
                need((rec["cell"],rec["i"]) == (name,i), "wrong cell or step order")
                p.check_members(rec,pair,teacher,zs[start],anchors[s.key(b,t,teacher,order)])
                max_error = max(max_error,p.check_motion(rec,v,before,hosts[str(b)],o))
                if o == "bias":
                    sig = p.bias_signature(rec)
                    key = (b,teacher,order,i)
                    need(control.setdefault(key,sig) == sig, "bias-only active step depends on rotation")
                    checked += t != "identity"
                before, v = rec["id"], rec["values"]
                n += 1
            need((before,v) == states[name], "trajectory did not reach archived endpoint")
        need(not f.read(1), "extra updates")
    need(n == (len(names)-len(starts))*len(sp["train"])*sp["epochs"]//2, "step count")
    counts = dict(ends=len(names)-len(starts),origins=len(starts),forwards=len(names)*len(keys),updates=n,
                  training_vjps=2*n,origin_gradient_anchors=2*(len(names)-len(starts)),
                  bias_cross_rotation_steps=checked,worst_motion_error=max_error)
    return sp, zs, counts, d.digest(complete)


def analyze(sp, zs):
    allrows, quality = [], {}
    for b in sp["bases"]:
        for split in ("train","test"):
            rows = sp[split]
            for order in sp["orders"]:
                effects = {}
                for turn in sp["turns"]:
                    for owner in sp["owners"]:
                        origin_z = zs[s.key(b,turn,owner)]
                        origin = a.metrics(rows,origin_z)
                        z = {h:zs[s.key(b,turn,owner,h,order)] for h in d.TEACHERS}
                        grammatical = {h:a.metrics(rows,z[h]) for h in d.TEACHERS}
                        e = a.effect(origin,grammatical["head"],grammatical["attractor"])
                        r = dict(base=b,split=split,order=order,rotation=turn,owner=owner,**e)
                        pos = position_stats(rows,origin_z)
                        for h in d.TEACHERS:
                            actual = [dict(q,head=q[h]) for q in rows]
                            g = a.metrics(actual,z[h])
                            pe = position_stats(rows,z[h])
                            need(pe["conflict_wins"]+pe["congruent_wins"] == grammatical[h]["wins"] and
                                 abs(pe["head"]-grammatical[h]["H"]) < 1e-12 and
                                 abs(pe["attractor"]-grammatical[h]["A"]) < 1e-12, "role statistics disagree")
                            for key in ("wins","ties","gold_nll","menu_nll","menu_mass","global_wins"):
                                r[h+"_"+key] = g[key]
                            for key in pos:
                                r[h+"_delta_"+key] = pe[key]-pos[key]
                            movement = [abs(m-origin["margins"][i]) for i,m in grammatical[h]["margins"].items()]
                            r[h+"_max_abs_change"] = max(movement)
                            changes = a.effect(origin,grammatical[h],grammatical[h])
                            r[h+"_switches"] = changes["switches"]
                        effects[turn,owner] = r
                for (t,o),r in effects.items():
                    for label,ctrl in (("position",effects["identity",o]),("owner",effects[t,"bias"])):
                        r[label+"_gap_h"] = r["h"]-ctrl["h"]
                        r[label+"_gap_S"] = r["S"]-ctrl["S"]
                        r[label+"_wins"] = ctrl["wins"]
                    allrows.append(r)
        for t in u.TURNS[1:]:
            subset = [r for r in allrows if (r["base"],r["split"],r["rotation"],r["owner"]) == (b,"test",t,"gate")]
            need(len(subset) == len(sp["orders"]), "missing comparison order")
            passes = {ctrl:p.qualify(subset,ctrl) for ctrl in ("owner","position")}
            passes["joint"] = {g:all(x[g] for x in passes.values()) for g in ("decision","probability")}
            quality[s.key(b,t)] = passes
    return allrows,quality


def verify(root, previous):
    sp, zs, counts, archive_hash = archive(root,previous)
    rows, quality = analyze(sp,zs)
    return dict(schema="spiraltorch.prefix_owner_results.v1",kind=sp["kind"],counts=counts,manifest=archive_hash,
                spec=d.digest((root/"spec.json").read_bytes()),by_base=quality,rows=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root","previous","result","table"):
        cli.add_argument(name,type=Path)
    args = cli.parse_args()
    result = verify(args.root,args.previous)
    print(d.save(args.result,result))
    with args.table.open("x",encoding="ascii",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(result["rows"][0]),delimiter="\t",lineterminator="\n")
        w.writeheader()
        w.writerows(result["rows"])

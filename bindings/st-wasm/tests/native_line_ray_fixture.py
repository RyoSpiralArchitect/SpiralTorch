"""Compare three adapter owner modes at equal rate and equal raw parameter distance.
Only the preceding tiny synthetic fixture is imported. Python observes and verifies;
the Rust owners always perform the actual pullback, update and checkpoint.
"""
import json
import math
from pathlib import Path
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
common = runpy.run_path(str(HERE / "native_line_owners_fixture.py"))
Session, MODES, CASES = (common[k] for k in ("Session", "MODES", "CASES"))
D = 1 / 256
CUT, BACK, TOTAL = 2, 4, 6


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def plan(mode, policy):
    p = json.loads(common["plan"](mode))
    rate = p.pop("learning_rate")
    p.update(schema="spiraltorch.native_causal_lm_line_plan.v3",
             update={"kind":"parameter_gradient_l2", "target_l2":D} if policy == "ray" else
                    {"kind":"sgd", "learning_rate":rate})
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def blocks(model):
    a = json.loads(model.lora_to_json())
    path = "transformer.h.%d.attn.c_proj" % a["layer"]
    values = {(path, k): [f32(x) for x in a[k]] for k in ("a", "b")}
    for p in json.loads(model.geometry_artifact().to_json())["placements"]:
        for t in p["tensors"]:
            assert t["role"] == "parameter" and t["requires_grad"], "unexpected artificial layout"
            values[p["path"], t["name"]] = list(map(f32, t["values"]))
    return values


def distance(left, right):
    assert left.keys() == right.keys()
    contributions = [[], []]
    for k in left:
        assert len(left[k]) == len(right[k])
        bucket = contributions[0 if k[0].endswith("c_proj") else 1]
        bucket.extend((a-b)**2 for a,b in zip(left[k],right[k]))
    lg, gg = (math.fsum(row) for row in contributions)
    return {"lora":math.sqrt(lg), "geometry":math.sqrt(gg), "total":math.sqrt(lg+gg)}


def logits(m):
    return [m.forward(e["prompt_ids"], [1, len(e["prompt_ids"])], last_only=True).values() for e in CASES]


def centered_rms(a, b):
    changes = []
    assert len(a) == len(b) == len(CASES)
    for before, after in zip(a,b):
        assert len(before) == len(after) == 11
        row = [y-x for x,y in zip(before,after)]
        mean = math.fsum(row)/len(row)
        changes.extend((x-mean)**2 for x in row)
    return math.sqrt(math.fsum(changes)/len(changes))


def check_ray(start, end, gradient, report, observed, mode):
    groups = gradient.parameter_gradients()
    norm = math.sqrt(math.fsum(g*g for p in groups.values() for xs in p.values() for g in xs))
    assert norm > 0 and math.isclose(norm, report["gradient_l2"], rel_tol=1e-12)
    relative = []
    dot = []
    for (path,name), xs in start.items():
        grad = groups.get(path, {}).get(name)
        if grad is None:
            assert end[path,name] == xs
            continue
        for x,y,g in zip(xs,end[path,name],grad):
            assert y == f32(x + D * (-g/norm))
            relative.append(((y-x)/D+g/norm)**2)
            dot.append(g*(y-x))
    near = lambda a,b: math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14)
    assert report["mode"] == mode and report["requested_l2"] == D
    assert near(math.sqrt(math.fsum(relative)),report["relative_vector_error"])
    assert near(math.fsum(dot),report["gradient_dot_step"])
    assert near(report["realized_l2"],observed["total"]) and abs(observed["total"]/D-1) <= 1e-4
    for k in ("lora","geometry"):
        if report[k] is None:
            assert observed[k] == 0
        else:
            assert near(observed[k],report[k]["realized_l2"])


def run(out, bridge_out):
    origin, seed = common["host"](common["HOST"])
    before = common["losses"](origin)
    x0, z0 = blocks(origin), logits(origin)
    old = json.loads((HERE/"native_line_owners_bridge.json").read_bytes())
    assert old["seed_id"] == seed and old["lora_id"] == origin.lora_artifact_id and old["before"] == before
    results, bridges = {}, {}
    for policy in ("sgd", "ray"):
        for mode in MODES:
            p = plan(mode,policy)
            session = Session(origin,p)
            m = origin
            steps = []
            for i in range(TOTAL):
                e = CASES[i%len(CASES)]
                x = blocks(m)
                g = common["grad"](m,e,mode)
                if policy == "sgd":
                    m = m.step_adaptation_sgd(g,mode=mode,learning_rate=0.125)
                    report = None
                else:
                    m, report = m.step_adaptation_gradient_at_distance(g,mode=mode,target_l2=D)
                y = blocks(m)
                observed = distance(x,y)
                r = session.step()
                assert common["owners"](m) == common["owners"](session.model)
                assert r["update"] == i+1 and r["example_index"] == i%len(CASES) and r["mode"] == mode
                assert r["loss"] == g.loss and r.get("parameter_ray") == report
                if report is not None:
                    check_ray(x,y,g,report,observed,mode)
                else:
                    assert r["loss"] == old["cases"][mode]["steps"][i]
                steps.append(dict(update=i+1, loss=r["loss"], displacement=observed, ray=report))
                if i+1 == CUT:
                    payload, ck = session.checkpoint_with_id()
                    assert common["pin"](payload) == ck
            assert session.exhausted and session.step() is None
            after, end = common["losses"](m), common["state"](m)
            if policy == "sgd":
                assert after == old["cases"][mode]["after"] and end == old["cases"][mode]["endpoint"]
            else:
                bridges[mode] = dict(plan=p,payload=payload,id=ck,steps=[s["loss"] for s in steps],
                                     endpoint=end,final_pin=session.checkpoint_with_id()[1])
            results[policy+"/"+mode] = dict(steps=steps, after=after,
                endpoint_distance=distance(x0,blocks(m)), path_l2=math.fsum(s["displacement"]["total"] for s in steps),
                centered_logit_rms=centered_rms(z0,logits(m)))
    common["save"](out,dict(before=before,target_l2=D,host=common["HOST"],cases=results))
    measures = {}
    for policy in ("sgd","ray"):
        measures[policy] = {}
        for mode in MODES:
            r = results[policy+"/"+mode]
            measures[policy][mode] = dict(path_l2=r["path_l2"], endpoint_l2=r["endpoint_distance"]["total"],
                                           centered_logit_rms=r["centered_logit_rms"])
    common["save"](bridge_out,dict(cut=CUT,back=BACK,total=TOTAL,cases=bridges,measurements=measures))
    print(json.dumps(dict(matched_updates=6*TOTAL, direct_reference_updates=6*TOTAL, pretrained_calls=0,
                          matched_distances_checked=3*TOTAL, legacy_baseline_steps_checked=3*TOTAL)))


def resume(ref_path, back_path=None):
    ref = json.loads(ref_path.read_bytes())
    back = json.loads(back_path.read_bytes()) if back_path else None
    source, _ = common["host"](common["HOST"])
    assert set(ref["cases"])==set(MODES) and (back is None or set(back)==set(MODES))
    worst=0.
    for mode in MODES:
        r=ref["cases"][mode]
        c=r if back is None else back[mode]
        s=Session.from_json(source,r["plan"],c["payload"],c["id"])
        start=CUT if back is None else BACK
        assert s.completed_updates==start and s.checkpoint_with_id()==(c["payload"],c["id"])
        for i in range(start,TOTAL):
            rep=s.step()
            assert rep["mode"]==mode and abs(rep["parameter_ray"]["realized_l2"]/D-1)<=1e-4
            worst=max(worst,common["max_error"](rep["loss"],r["steps"][i]))
        worst=max(worst,common["max_error"](common["state"](s.model),r["endpoint"]))
        if back is None:
            assert s.checkpoint_with_id()[1]==r["final_pin"]
        else:
            worst=max(worst,common["max_error"](common["state"](s.model),c["endpoint"]))
    print(json.dumps(dict(suffix_updates=(TOTAL-start)*len(MODES),max_error=worst)))


if __name__=="__main__":
    if len(sys.argv)==4 and sys.argv[1]=="run":
        run(*map(Path,sys.argv[2:]))
    elif len(sys.argv) in (3,4) and sys.argv[1]=="resume":
        resume(*map(Path,sys.argv[2:]))
    else:
        sys.exit("run NEW_RESULT NEW_BRIDGE | resume BRIDGE [RETURNED]")

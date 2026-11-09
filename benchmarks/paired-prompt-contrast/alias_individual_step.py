"""Apply previously predicted nine-objective repairs at frozen synthetic training points."""
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_interference import STATES, fit_inputs, forbid_nontraining_files, signature
from alias_interference_audit import coordinates, dot, mean_vec
from alias_protection import CROSS_TOL, MARGIN, MENU, RATE, STEPS, STOP, checkpoint, native, open_model, scores, sealed, sources
from shared_prefix_study import obj, put, read, record, require

FLOOR = 1e-5
BUDGET = dict(backwards=1536, four_group_steps=64, guards=64, forwards=3072, token_scores=36)


def pins():
    return {**sources(), "alias_individual_step": signature(Path(__file__)),
            "alias_individual_step_audit": signature(Path(__file__).with_name("alias_individual_step_audit.py"))}


def archive(folder, pp, mp):
    p = obj(folder / "protocol/plan.json", pp)
    m = obj(folder / "measured/measured.json", mp)
    require(m["plan"] == pp and p["kind"] == "spiraltorch.alias_absolute_protection.v1" and
            p["steps"] == STEPS and p["classes"] == ["on", "off", "idle"] and p["protected_lines"] == {"2":[23,4], "5":[23,16]},
            "not the nominated protection experiment")
    it = (r for raw in io.BytesIO(read(folder / "measured/journal.jsonl", m["journal"], 24 << 20))
          if (r := json.loads(raw))["arm"] == "guard_standby")
    paths = {}
    for state in STATES:
        prefix = p["origins"][state]
        path = []
        for t in range(STEPS + 1):
            tags = dict(state=state, t=t, owner=prefix)
            def take(event, **extra):
                r = next(it)
                require(r["event"] == event and all(r[k] == v for k, v in (tags | extra).items()), "bad archive order")
                return r
            take("start")
            rows = [take("example", slot=i) for i in range(18)]
            step = take("step") if t < STEPS else None
            path.append((prefix, rows, step))
            if step:
                prefix = "sha256:" + step["post"]
        paths[state] = path
    require(next(it, None) is None, "extra prior points")
    return p, paths


def weight_path(folder, state, t):
    return folder / "measured" / (f"{state}-guard_standby-{t:02}.json" if t else f"{state}.json")


def means(rows, groups):
    return [mean_vec([rows[i]["v"] for i in ids]) for ids in groups] + [rows[i]["singletons"][1]["v"] for i in groups[2]]


def witness(gs, displacement):
    def unit(x):
        n = math.hypot(*x)
        require(n > 0 and math.isfinite(n), "undefined unit vector")
        return [v/n for v in x]
    units = list(map(unit, gs))
    first = unit([-d for d in displacement])
    slopes = [dot(g, first) for g in units]
    k = slopes.index(min(slopes))
    mix, interval, u = 0., None, first
    if slopes[k] <= FLOOR:
        lo, hi = 0., 1.
        for a, g in zip(slopes, units, strict=True):
            b = dot(g, units[k]) - a
            if b > 0:
                lo = max(lo, (FLOOR-a)/b)
            elif b < 0:
                hi = min(hi, (FLOOR-a)/b)
            else:
                require(a >= FLOOR, "unavailable segment")
        require(lo < hi, "no open interval")
        interval, mix = [lo, hi], (lo+hi)/2
        u = unit([(1-mix)*a + mix*b for a,b in zip(first,units[k],strict=True)])
    return dict(worst=k, interval=interval, mix=mix, margin=min(dot(g,u) for g in units),
                cosine=dot(first,u), direction=[-x for x in u])


def freeze(previous, pp, mp, rp, extension, out):
    p, paths = archive(previous, pp, mp)
    rb = obj(previous / "readback.json", rp, 2 << 20)
    require(rb["pins"]["plan"] == pp and rb["pins"]["measured"] == mp and
            p["code"] == sources() and len(extension) == 64 and all(k in "0123456789abcdef" for k in extension),
            "incompatible reference")
    refs = {}
    for state, path in paths.items():
        refs[state] = []
        for t, ((owner, rows, step), (nxt, _, _)) in enumerate(zip(path[:-1], path[1:], strict=True)):
            a, layout = coordinates(weight_path(previous, state, t), owner[7:])
            b, later = coordinates(weight_path(previous, state, t+1), nxt[7:])
            require(layout == later and nxt == "sha256:" + step["post"], "wrong archived weights")
            ref = witness(means(rows, p["groups"]), [y-x for x,y in zip(a,b,strict=True)])
            old = rb["exploratory_only_nine_slope_witness"][state][t]
            require(old["already"] == (ref["interval"] is None) and abs(ref["mix"]-old.get("mix",0.)) < 1e-12
                    and abs(ref["margin"]-old["margin"]) < 1e-12 and abs(ref["cosine"]-old.get("cosine",1.)) < 1e-12,
                    "different retrospective witness")
            refs[state].append(ref)
    out.mkdir(mode=0o700)
    return put(out / "plan.json", dict(kind="spiraltorch.alias_individual_finite_step.v1", code=pins(),
        old_plan=pp, old_measured=mp, old_readback=rp, native=extension, refs=refs, budget=BUDGET, seconds=120,
        retries=0, min_cosine=FLOOR, margin=MARGIN, relative_prediction_tolerance=.15,
        primary="All 64 independent steps: the three mean conditional NLLs and six separate absolute standby NLLs "
                "must each decrease by >5e-6 in measurement and prediction; agreement within max(5e-6,15% of abs(prediction)). "
                "Both singleton and full menu readouts must pass. For pass-through, weights and scores equal the control exactly.",
        controls="Every input gradient, candidate receipt, candidate endpoint and its scores must replay the archived originals. "
                 "Refused/failed runs retained. This is not a continued repaired trajectory, held-out score or global solver."))


def measure(fixture, training, previous, protocol, pp, out):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (120,120))
    forbid_nontraining_files(fixture, training)
    pl = obj(protocol / "plan.json", pp)
    require(pl["code"] == pins(), "code changed after freeze")
    p, paths = archive(previous, pl["old_plan"], pl["old_measured"])
    fx, _, data = fit_inputs(fixture, p["fixture"], training, p["training"])
    groups = [[i for i,e in enumerate(data) if e["meaning"] == k] for k in range(3)]
    require(groups == p["groups"] and 6 in groups[2], "wrong training roster")
    st = native()
    require(signature(st._rs.__file__) == pl["native"], "wrong local wheel")
    base = open_model(st,fixture,fx["files"])
    frozen = base.lora_to_json()
    counts = dict.fromkeys(BUDGET, 0)
    with (out / "observations.jsonl").open("xb") as f:
        for state, path in paths.items():
            for t, ((owner, old, step), (_, old_next, _)) in enumerate(zip(path[:-1], path[1:], strict=True)):
                tags = dict(state=state, t=t, owner=owner)
                record(f, dict(event="start", **tags))
                raw = read(weight_path(previous,state,t),owner[7:],100000)
                model = base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),owner,p["base"]))
                hs, tail = [], []
                for i,e in enumerate(data):
                    c, prompt = e["meaning"], e["prompt"]
                    accepted = [c,c+3]
                    g = model.geometry_line_subset_gradients(prompt, MENU, accepted_indices=accepted, stop_id=STOP)
                    vals = sealed(g, owner, p, 18) | scores(g.line_subset_report(), accepted)
                    require(all(old[i][k] == v for k,v in vals.items()), "class gradient or loss changed")
                    hs.append(g)
                    counts["backwards"] += 1
                    if c == 2:
                        g = model.geometry_line_set_gradients(prompt, [MENU[5]], stop_id=STOP)
                        vals = sealed(g, owner, p, 3) | dict(cost=g.line_set_report()["set_nll"], index=5)
                        require(old[i]["singletons"][1] == vals, "individual gradient or loss changed")
                        tail.append(g)
                        counts["backwards"] += 1
                class_handles = [[hs[i] for i in ids] for ids in groups]
                candidate, receipt = model.step_common_geometry_sgd([*class_handles,tail], RATE, target_l2=p["radius"])
                counts["four_group_steps"] += 1
                require(receipt == step["receipt"] and candidate.geometry_artifact_id == "sha256:" + step["post"], "control update changed")
                guarded, guard = model.guard_geometry_step(candidate, [*class_handles,*([g] for g in tail)], min_cosine=FLOOR)
                counts["guards"] += 1
                require(guarded.lora_to_json() == frozen and guarded.base_model_identity == p["base"] and model.geometry_artifact_id == owner,
                        "changed frozen model")
                pin = checkpoint(out/f"{state}-{t:02}.json", guarded)
                record(f, dict(event="guard", **tags, post=pin, receipt=guard))
                for name, endpoint in (("candidate",candidate), ("guarded",guarded)):
                    for i,e in enumerate(data):
                        c = e["meaning"]
                        ids = [c,c+3]
                        v = scores(endpoint.line_subset_loss(e["prompt"], MENU, accepted_indices=ids, stop_id=STOP), ids)
                        counts["forwards"] += 1
                        if name == "candidate" or guard["feasible_mix"] is None:
                            require(all(x == old_next[i][k] for k,x in v.items()), "control full-menu scores changed")
                        if c == 2:
                            v["single"] = endpoint.line_set_loss(e["prompt"], [MENU[5]], stop_id=STOP)["set_nll"]
                            require(abs(v["single"] - v["lines"][5]) <= CROSS_TOL, "different singleton/full-menu line")
                            if name == "candidate" or guard["feasible_mix"] is None:
                                require(v["single"] == old_next[i]["singletons"][1]["cost"], "control singleton changed")
                            counts["forwards"] += 1
                        record(f,dict(event="score", **tags, arm=name, slot=i, **v))
                if t in (0,STEPS-1):
                    for name, m in (("before",model),("candidate",candidate),("guarded",guarded)):
                        prefix = list(data[6]["prompt"])
                        terms = []
                        for k in [*MENU[5],STOP]:
                            terms.append(m.next_token_score(prefix,k)["nll"])
                            prefix.append(k)
                            counts["token_scores"] += 1
                        record(f,dict(event="tokens", **tags, arm=name, slot=6, terms=terms))
    require(counts == BUDGET, "wrong call count")
    return put(out/"measured.json",dict(plan=pp,native=pl["native"],counts=counts,journal=signature(out/"observations.jsonl")))


def once(fixture, training, previous, protocol, pp, out):
    pl = obj(protocol / "plan.json", pp)
    require(pl["code"] == pins() and pl["budget"] == BUDGET and pl["seconds"] == 120 and pl["retries"] == 0, "not a frozen run")
    out.mkdir(mode=0o700)
    put(out/"attempt.json",dict(attempt=1,plan=pp,native=pl["native"]))
    env = dict(os.environ, SPIRALTORCH_BACKEND="cpu", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
    cmd = [sys.executable,"-I","-B",str(Path(__file__).resolve()),"_worker",
           *map(str,(fixture,training,previous,protocol,pp,out))]
    start = time.monotonic()
    with (out/"process.log").open("xb") as log:
        try:
            status = subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    put(out/"process.json",dict(exit=status,seconds=time.monotonic()-start,log=signature(out/"process.log")))
    require(status == 0,"first failure retained; see process.log")
    return signature(out/"measured.json")


if __name__ == "__main__":
    from alias_individual_step_audit import audit
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 6:
        print(freeze(Path(a[0]), *a[1:5], Path(a[5])))
    elif cmd in ("measure","_worker") and len(a) == 6:
        print((once if cmd == "measure" else measure)(*(Path(s).resolve() for s in a[:4]), a[4], Path(a[5]).resolve()))
    elif cmd == "audit" and len(a) == 6:
        print(audit(Path(a[0]), Path(a[1]), a[2], Path(a[3]), a[4], Path(a[5])))
    else:
        raise SystemExit("freeze OLD PP MP RB EXTENSION OUT | measure FIXTURE TRAIN OLD PLAN PP OUT | audit OLD PLAN PP MEASURED MP OUT")

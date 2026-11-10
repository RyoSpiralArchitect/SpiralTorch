"""Chain native guarded geometry updates, with a prospective cold-process continuation check."""
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_individual_step import FLOOR, archive, pins as one_step_sources, weight_path
from alias_interference import STATES, fit_inputs, forbid_nontraining_files, signature
from alias_interference_audit import f32
from alias_protection import CROSS_TOL, MARGIN, MENU, RATE, RADIUS, STOP, native, open_model, scores, sealed, term_scores
from shared_prefix_study import obj, pack, put, read, record, require, sha, write

ARMS = ("four_group", "individual_guard")
HORIZON, SPLIT, WALL = 128, 64, 120
MILESTONES = (0, 32, 64, 96, 128)
VECTOR = struct.Struct("<64f")
BLOCK = 24 * VECTOR.size
BUDGET = dict(backwards=12384, proposals=516, guards=258, forwards=18432,
              token_scores=600, adopted=512, uncommitted_probes=4)


def sources():
    return {**one_step_sources(), "alias_guarded_training": signature(Path(__file__)),
            "alias_guarded_training_audit": signature(Path(__file__).with_name("alias_guarded_training_audit.py"))}


def fingerprint(x):
    return sha(pack(x))


def row_view(r):
    v = {k: r[k] for k in ("slot", "cost", "absolute", "lines")}
    if "single" in r:
        v["single"] = r["single"]
    elif "singletons" in r:
        require(r["singletons"][1]["index"] == 5, "different prior alias")
        v["single"] = r["singletons"][1]["cost"]
    return v


def block(vs):
    require(len(vs) == 24, "expected 18 class and six individual gradients")
    return b"".join(VECTOR.pack(*v) for v in vs)


def freeze(four, one, old_pin, out):
    rb = obj(one / "readback.json", old_pin)
    p = obj(one / "protocol/plan.json", rb["pins"]["plan"])
    m = obj(one / "measured/measured.json", rb["pins"]["measurement"])
    result = obj(one / "report/results.json", rb["pins"]["report"], 2 << 20)
    require(p["code"] == one_step_sources() and p["native"] == m["native"] and
            m["plan"] == rb["pins"]["plan"] == result["plan"] and result["measurement"] == rb["pins"]["measurement"] and
            result["primary_pass"] and rb["chained_training_updates"] == 0, "not the pinned independent-step trial")
    previous, paths = archive(four, p["old_plan"], p["old_measured"])
    entries = [json.loads(raw) for raw in io.BytesIO(read(one / "measured/observations.jsonl", m["journal"], 8 << 20))]
    refs = {}
    out.mkdir(mode=0o700)
    for state, path in paths.items():
        owner = previous["origins"][state]
        write(out / (state+".json"), read(weight_path(four,state,0), owner[7:], 100000))
        refs[state] = []
        for t,(pre,old,proposal) in enumerate(path[:-1]):
            rs = [r for r in entries if (r["state"], r["t"]) == (state,t)]
            guard, = [r for r in rs if r["event"] == "guard"]
            require(guard["owner"] == pre and guard["receipt"]["candidate_geometry_id"] == "sha256:"+proposal["post"], "foreign prior step")
            v = dict(pre=pre, before=fingerprint([row_view(r) for r in old]),
                     gradient=sha(block([r["v"] for r in old] + [old[i]["singletons"][1]["v"] for i in previous["groups"][2]])),
                     candidate="sha256:"+proposal["post"], proposal=fingerprint(proposal["receipt"]),
                     guarded="sha256:"+guard["post"], guard=fingerprint(guard["receipt"]))
            for arm in ("candidate","guarded"):
                rows = [r for r in rs if r["event"] == "score" and r["arm"] == arm]
                require([r["slot"] for r in rows] == list(range(18)), "incomplete reference scores")
                v[arm+"_scores"] = fingerprint([row_view(r) for r in rows])
            refs[state].append(v)
    return put(out/"plan.json", dict(kind="spiraltorch.alias_guarded_chain.v1", code=sources(), prior_readback=old_pin,
        native=p["native"], wheel=rb["pins"]["wheel"], fixture=previous["fixture"], training=previous["training"],
        base=previous["base"], lora=previous["lora"], groups=previous["groups"], origins=previous["origins"],
        refs=refs, states=STATES, arms=ARMS, horizon=HORIZON, split=SPLIT, milestones=MILESTONES,
        radius=RADIUS, rate=RATE, floor=FLOOR, margin=MARGIN, prediction_rel=.15, budget=BUDGET, wall=WALL, retries=0,
        primary="At BOTH starts: all 128 adopted guarded steps lower every one of nine training NLLs by >5e-6; "
                "predictions must also clear 5e-6 and agree within max(5e-6,15% of |prediction|) in both line layouts.",
        secondary="Report 32-step gate separately, physical harm versus a threshold miss, same-owner rescues, "
                  "repair/pass streaks, endpoint tradeoffs, and whether the last 16 steps pass without repairs. "
                  "This last observation says nothing about future steps.",
        continuation="At 64: persist geometry and boundary. Compute one unapplied next proposal per path in memory. "
                     "Fresh process restores each path from validated bytes; its first committed step must reproduce the "
                     "before scores, all gradient bits, proposal and guard receipts, and both endpoint IDs.",
        failure="First API refusal, timeout, ownership or integrity error stops the trial. No fallback, retry or resizing. "
                "Scientific threshold misses are recorded, not used to tune, backtrack or discard a successful native step.",
        scope="Same 18 synthetic training prompts; deterministic full cohort, frozen base and LoRA, no evaluation data."))


def geometry_path(root, owner):
    require(isinstance(owner,str) and owner.startswith("sha256:") and len(owner)==71 and
            all(c in "0123456789abcdef" for c in owner[7:]), "invalid geometry content ID")
    return root/"geometry"/(owner[7:]+".json")


def retain(root, model):
    owner = model.geometry_artifact_id
    data = model.geometry_artifact().to_json().encode()
    require("sha256:"+sha(data) == owner, "weights do not match model identity")
    dst = geometry_path(root, owner)
    try:
        write(dst, data)
    except FileExistsError:
        require(read(dst, owner[7:], 100000) == data, "colliding saved geometry")
    return owner


def forward(model, e):
    c, prompt = e["meaning"], e["prompt"]
    ids = [c,c+3]
    v = scores(model.line_subset_loss(prompt, MENU, accepted_indices=ids, stop_id=STOP), ids)
    if c == 2:
        v["single"] = model.line_set_loss(prompt,[MENU[5]],stop_id=STOP)["set_nll"]
        require(abs(v["single"]-v["lines"][5]) <= CROSS_TOL, "line-scoring layouts disagree")
    return v


def prepare(model, arm, examples, p, counts, on_refusal=None):
    owner = model.geometry_artifact_id
    hs, ts, vs, tail, rows = [], [], [], [], []
    for slot,e in enumerate(examples):
        c, prompt = e["meaning"], e["prompt"]
        ids = [c,c+3]
        g = model.geometry_line_subset_gradients(prompt,MENU,accepted_indices=ids,stop_id=STOP)
        counts["backwards"] += 1
        v = sealed(g,owner,p,18)
        r = scores(g.line_subset_report(),ids)
        require(v["loss"] == f32(r["cost"]), "wrong conditional gradient")
        hs.append(g)
        vs.append(v["v"])
        if c == 2:
            g = model.geometry_line_set_gradients(prompt,[MENU[5]],stop_id=STOP)
            counts["backwards"] += 1
            v = sealed(g,owner,p,3)
            r["single"] = g.line_set_report()["set_nll"]
            require(v["loss"] == f32(r["single"]) and abs(r["single"]-r["lines"][5]) <= CROSS_TOL,
                    "wrong absolute gradient")
            ts.append(g)
            tail.append(v["v"])
        rows.append(dict(slot=slot, **r))
    classes = [[hs[i] for i in group] for group in p["groups"]]
    raw = block(vs+tail)
    try:
        candidate, proposal = model.step_common_geometry_sgd([*classes,ts],RATE,target_l2=RADIUS)
    except ValueError as ex:
        if on_refusal is not None:
            on_refusal("proposal", rows, raw, None, None, ex)
        raise
    counts["proposals"] += 1
    require(proposal["pre_geometry_id"] == owner and proposal["post_geometry_id"] == candidate.geometry_artifact_id,
            "candidate from wrong source")
    selected, guard = candidate, None
    if arm == "individual_guard":
        try:
            selected, guard = model.guard_geometry_step(candidate,[*classes,*([g] for g in ts)],min_cosine=FLOOR)
        except ValueError as ex:
            if on_refusal is not None:
                on_refusal("guard", rows, raw, candidate, proposal, ex)
            raise
        counts["guards"] += 1
        require((guard["pre_geometry_id"],guard["candidate_geometry_id"],guard["post_geometry_id"]) ==
                (owner,candidate.geometry_artifact_id,selected.geometry_artifact_id), "guard lineage mismatch")
    require(model.geometry_artifact_id == owner, "proposal mutated the source model")
    proof = dict(pre=owner,before=fingerprint(rows),gradient=sha(raw),candidate=candidate.geometry_artifact_id,
                 chosen=selected.geometry_artifact_id,proposal=fingerprint(proposal),guard=fingerprint(guard) if guard else None)
    return rows,raw,candidate,selected,proposal,guard,proof


def check_replay(p, state, arm, t, proof, cs, chosen):
    if t >= len(p["refs"][state]):
        return False
    ref = p["refs"][state][t]
    if arm == "four_group":
        require(proof["pre"] == ref["pre"], "four-group baseline diverged from archive")
    if proof["pre"] != ref["pre"]:
        return False
    require(all(proof[k] == ref[k] for k in ("gradient","before","candidate","proposal")) and cs==ref["candidate_scores"],
            "same-owner candidate did not replay")
    if arm == "individual_guard":
        require(proof["chosen"]==ref["guarded"] and proof["guard"]==ref["guard"] and chosen==ref["guarded_scores"],
                "same-owner guard did not replay")
    return True


def boundary(marker, pp, stage, state, arm):
    require(marker["plan"]==pp and marker["stage"]==stage and marker["cursor"]==SPLIT and
            marker["state"]==state and marker["arm"]==arm and marker["shadow"]["pre"]==marker["owner"] and
            marker["shadow"]["before"]==marker["scores"], "foreign resume boundary")


def worker(fixture, training, protocol, pp, root, stage, previous):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (WALL//2,WALL//2))
    forbid_nontraining_files(fixture,training)
    p = obj(protocol/"plan.json",pp)
    require(p["code"] == sources() and stage in (0,1), "code or stage differs from plan")
    part = root/f"part{stage}"
    part.mkdir(mode=0o700)
    prior = None
    if stage:
        prior = obj(root/"part0/complete.json",previous)
        require(prior["plan"]==pp and prior["stage"]==0 and prior["pid"]!=os.getpid() and
                set(prior["files"])=={"steps.jsonl","gradients.f32le"} and
                set(prior["endpoints"])=={s+"/"+a for s in STATES for a in ARMS} and
                all(signature(root/"part0"/name)==pin for name,pin in prior["files"].items()), "invalid predecessor")
    else:
        require(previous == "-", "unexpected predecessor")
    fx, _, examples = fit_inputs(fixture,p["fixture"],training,p["training"])
    groups = [[i for i,e in enumerate(examples) if e["meaning"]==c] for c in range(3)]
    require(groups == p["groups"], "different training roster")
    st = native()
    require(signature(st._rs.__file__)==p["native"], "different native extension")
    base = open_model(st,fixture,fx["files"])
    lora = base.lora_to_json()
    require((base.base_model_identity,base.lora_artifact_id)==(p["base"],p["lora"]), "wrong base/LoRA")
    counts, ends = dict.fromkeys(BUDGET,0), {}
    cursor = {}
    def preserve_refusal(phase, rows, raw, candidate, proposal, error):
        raw_pin = write(part/"refused.f32le", raw)
        candidate_id = retain(root,candidate) if candidate is not None else None
        put(part/"refusal.json",dict(plan=pp, cursor=cursor, phase=phase, before=rows,
            gradient=raw_pin, candidate=candidate_id, proposal=proposal, reason=str(error)))
    with (part/"steps.jsonl").open("xb") as journal, (part/"gradients.f32le").open("xb") as grad:
        try:
            for state in STATES:
                for arm in ARMS:
                    name = state + "/" + arm
                    mark = prior["endpoints"][name] if prior else None
                    if mark:
                        boundary(mark,pp,0,state,arm)
                    owner = mark["owner"] if mark else p["origins"][state]
                    path = geometry_path(root,owner) if mark else protocol/(state+".json")
                    art = st.GeometryAdapterArtifact.from_json(read(path,owner[7:],100000).decode(),owner,p["base"])
                    model = base.with_geometry(art)
                    require(retain(root,model)==owner, "restore changed weights")
                    prev_scores = mark["scores"] if mark else None
                    for t in range(stage*SPLIT,(stage+1)*SPLIT):
                        cursor = dict(state=state,arm=arm,t=t,owner=owner)
                        record(journal,dict(event="start",**cursor))
                        before,raw,candidate,selected,proposal,guard,proof = prepare(model,arm,examples,p,counts,preserve_refusal)
                        require(proof["pre"]==owner and (prev_scores is None or prev_scores==proof["before"]),
                                "previous adopted endpoint was not used for the next gradient")
                        if mark and t == SPLIT:
                            require(proof==mark["shadow"], "first resumed update differs from the live witness")
                        require(all(m.base_model_identity==p["base"] and m.lora_to_json()==lora for m in (candidate,selected))
                                and model.geometry_artifact_id==owner, "frozen owner changed")
                        retain(root,candidate)
                        retain(root,selected)
                        candidate_scores = [dict(slot=i,**forward(candidate,e)) for i,e in enumerate(examples)]
                        counts["forwards"] += 24
                        guard_scores = None
                        if guard is not None:
                            guard_scores = [dict(slot=i,**forward(selected,e)) for i,e in enumerate(examples)]
                            counts["forwards"] += 24
                            if guard["feasible_mix"] is None:
                                require(proof["candidate"]==proof["chosen"] and candidate_scores==guard_scores, "changed pass-through")
                        chosen_scores = candidate_scores if guard_scores is None else guard_scores
                        prev_scores = fingerprint(chosen_scores)
                        known = check_replay(p,state,arm,t,proof,fingerprint(candidate_scores),prev_scores)
                        components = {}
                        for timepoint,endpoint in ((t,model),(t+1,selected)):
                            if timepoint in MILESTONES and (timepoint==0 or timepoint==t+1):
                                components[str(timepoint)] = [term_scores(endpoint,examples[i]["prompt"]) for i in groups[2]]
                                counts["token_scores"] += len(groups[2])*5
                        grad.write(raw)
                        record(journal,dict(event="adopt",**cursor,proof=proof,proposal=proposal,guard=guard,
                               before=before,candidate_scores=candidate_scores,guard_scores=guard_scores,
                               archived_replay=known,terms=components))
                        owner, model = proof["chosen"], selected
                        counts["adopted"] += 1
                    end = dict(plan=pp,stage=stage,cursor=(stage+1)*SPLIT,state=state,arm=arm,owner=owner,scores=prev_scores)
                    if stage==0:
                        cursor = dict(state=state,arm=arm,t=SPLIT,owner=owner)
                        proof = prepare(model,arm,examples,p,counts,preserve_refusal)[-1]
                        require(proof["pre"]==owner and proof["before"]==prev_scores, "shadow changed boundary")
                        end["shadow"] = proof
                        counts["uncommitted_probes"] += 1
                    ends[name] = end
        except Exception as ex:
            record(journal,dict(event="stop",cursor=cursor,error_type=type(ex).__name__,error=str(ex),counts=counts))
            raise
    files = {name:signature(part/name) for name in ("steps.jsonl","gradients.f32le")}
    return put(part/"complete.json",dict(plan=pp,stage=stage,pid=os.getpid(),previous=previous,counts=counts,endpoints=ends,files=files))


def run(fixture, training, protocol, pp, out):
    p = obj(protocol/"plan.json",pp)
    require(p["code"]==sources() and p["budget"]==BUDGET and (p["wall"],p["retries"])==(WALL,0), "not the frozen trial")
    out.mkdir(mode=0o700)
    (out/"geometry").mkdir(mode=0o700)
    put(out/"attempt.json",dict(plan=pp,native=p["native"],attempt=1))
    env = dict(os.environ, SPIRALTORCH_BACKEND="cpu", HF_HUB_OFFLINE="1", RAYON_NUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    deadline = time.monotonic()+WALL
    previous, parts = "-", []
    for stage in (0,1):
        log = out/f"worker{stage}.log"
        cmd = [sys.executable,"-I","-B",str(Path(__file__).resolve()),"_worker",*map(str,(fixture,training,protocol,pp,out,stage,previous))]
        env["SPIRALTORCH_AUTOTUNE_STORE"] = str(out/f"tuning{stage}.json")
        with log.open("xb") as f:
            try:
                rc = subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=max(0.,deadline-time.monotonic())).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        seconds = WALL-(deadline-time.monotonic())
        put(out/f"exit{stage}.json",dict(exit=rc,elapsed=seconds,log=signature(log),previous=previous))
        require(rc==0 and seconds<=WALL, f"stage {stage} stopped; original evidence retained, no retry")
        previous = signature(out/f"part{stage}/complete.json")
        require(log.read_text().splitlines()[-1:]==[previous],"worker returned different evidence")
        parts.append(previous)
    return put(out/"complete.json",dict(plan=pp,parts=parts,native=p["native"]))


if __name__ == "__main__":
    from alias_guarded_training_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==4:
        print(freeze(Path(a[0]),Path(a[1]),a[2],Path(a[3])))
    elif cmd=="run" and len(a)==5:
        print(run(*(Path(s).resolve() for s in a[:3]),a[3],Path(a[4]).resolve()))
    elif cmd=="_worker" and len(a)==7:
        print(worker(*map(Path,a[:3]),a[3],Path(a[4]),int(a[5]),a[6]))
    elif cmd=="audit" and len(a)==5:
        print(audit(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    else:
        raise SystemExit("freeze FOUR ONE ONE_READBACK_SHA OUT | run FIXTURE TRAIN PLAN PLAN_SHA OUT | audit PLAN PLAN_SHA MEASURED PIN OUT")

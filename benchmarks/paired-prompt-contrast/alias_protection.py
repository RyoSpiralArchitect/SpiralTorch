"""Absolute verbalization protection: freeze, measure once with a native wheel, audit without it."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_common_trajectory import CODE, MARGIN, RADIUS, SECONDS, STEPS, checkpoint, scores
from alias_interference import CLASSES, STATES, fit_inputs, forbid_nontraining_files, signature
from alias_subset_study import MENU, RATE, STOP, native, open_model
from shared_prefix_study import SITE, obj, pack, put, read, record, require, sha

# Both singletons are correct on exactly these idle prompts; their objectives are not interchangeable.
ARMS = {"three": None, "guard_standby": 5, "guard_idle": 2}
ALIASES = (2, 5)
BUDGET = dict(subset_backwards=3456, singleton_backwards=2304, subset_forwards=108,
              singleton_forwards=72, separate_token_scores=240, updates=192)
CROSS_TOL = 2e-5


def sources():
    names = (*CODE, "alias_protection", "alias_protection_audit")
    return {k: signature(Path(__file__).with_name(k + ".py")) for k in names}


def control_point(rows):
    return sha(pack([{k: r[k] for k in ("slot", "cost", "absolute", "lines", "loss", "v") if k in r} for r in rows]))


def freeze(previous, pp, mp, rp, out):
    p = obj(previous / "protocol/plan.json", pp)
    m = obj(previous / "measured/measured.json", mp)
    r = obj(previous / "report/results.json", rp, 2 << 20)
    require(p["kind"] == "spiraltorch.alias_common_trajectory.v1" and r["primary_pass"] and
            p["radius"] == RADIUS and p["steps"] == STEPS and r["plan"] == m["plan"] == pp and
            p["native"] == m["native"] and r["measurement"] == mp, "wrong three-class reference")
    current = sources()
    require(p["code"] == {k: current[k] for k in CODE} and
            r["states"]["initial"]["arms"]["common"]["harmed_examples"][2]["baseline_weak"] == 6,
            "different source or discovery cohort")
    rows = [json.loads(x) for x in read(previous / "measured/journal.jsonl", m["journal"], 16 << 20).splitlines()]
    replay = {}
    for s in STATES:
        path = [d for d in rows if d["state"] == s and d["arm"] == "common"]
        replay[s] = []
        for t in range(STEPS + 1):
            point = [d for d in path if d["time"] == t]
            samples = [d for d in point if d["event"] == "example"]
            require([v["slot"] for v in samples] == list(range(18)), "missing control prompt")
            item = dict(scores=control_point(samples), owner=samples[0]["owner"])
            if t < STEPS:
                step, = [d for d in point if d["event"] == "step"]
                item.update(post=step["post"], receipt=sha(pack(step["receipt"])))
            replay[s].append(item)
    out.mkdir(mode=0o700)
    return put(out / "plan.json", dict(kind="spiraltorch.alias_absolute_protection.v1", code=current,
        prior=[pp, mp, rp], replay=replay, states=STATES, arms=ARMS, arm_order=list(ARMS), classes=CLASSES, protected_class=2,
        protected_lines={str(i): MENU[i] for i in ALIASES}, groups=p["groups"], origins=p["origins"], base=p["base"],
        lora=p["lora"], fixture=p["fixture"], training=p["training"], native=m["native"], rate=RATE,
        steps=STEPS, radius=RADIUS, margin=MARGIN, cross_layout_tolerance=CROSS_TOL, budget=BUDGET, seconds=SECONDS, retries=0,
        primary="At both starts: each of 32 guard_standby steps reduces all three conditional class means "
                "AND the mean complete standby NLL on the six idle prompts by >5e-6 under both scoring layouts.",
        specificity="At both starts the guarded standby endpoint must beat both the replayed three-group arm "
                    "and the other-alias (idle) guard by >5e-6 on mean absolute standby NLL. Report separately.",
        secondary="Test separately: every idle prompt benefits at endpoint; the standby word gets more likely after status; "
                  "standby's share among both accepted complete lines grows. Track costs paid by the other objectives. "
                  "No conditional-on-two-alias metric is an absolute word or line probability.",
        scope="Post-discovery experiment on 18 reused artificial training prompts only. No held-out or pretrained access."
    ))


def sealed(g, owner, pl, tokens):
    require((g.geometry_artifact_id, g.base_model_identity, g.lora_artifact_id, g.target_tokens, g.readout_accumulator) ==
            (owner, pl["base"], pl["lora"], tokens, "f32_wide_backward"), "foreign gradient")
    p = g.parameter_gradients()
    require(set(p) == {SITE} and set(p[SITE]) == {"bias", "gate"}, "foreign coordinates")
    v = p[SITE]["bias"] + p[SITE]["gate"]
    require(len(v) == 64 and all(map(math.isfinite, v)) and math.isfinite(g.loss), "bad numeric gradient")
    return dict(loss=g.loss, v=v)


def term_scores(model, prompt):
    status, idle = MENU[2]
    _, standby = MENU[5]
    choices = [(prompt, status), (prompt + [status], idle), (prompt + [status, idle], STOP),
               (prompt + [status], standby), (prompt + [status, standby], STOP)]
    values = []
    for prefix, token in choices:
        r = model.next_token_score(prefix, token)
        require(r["target_id"] == token and r["vocabulary"] == 64 and math.isfinite(r["nll"]), "wrong token readout")
        values.append(r["nll"])
    return values


def train(fixture, training, protocol, pp, out):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (SECONDS, SECONDS))
    forbid_nontraining_files(fixture, training)
    p = obj(protocol / "plan.json", pp)
    require(p["code"] == sources() and p["arms"] == ARMS, "changed trial")
    fx, tr, examples = fit_inputs(fixture, p["fixture"], training, p["training"])
    groups = [[i for i, e in enumerate(examples) if e["meaning"] == c] for c in range(3)]
    require(groups == p["groups"] and [MENU[i] for i in ALIASES] == [[23, 4], [23, 16]] and
            [fx["tokens"][str(i)] for i in (23, 4, 16)] == ["status", "idle", "standby"], "different prompts or labels")
    st = native()
    require(signature(st._rs.__file__) == p["native"], "different extension")
    base = open_model(st, fixture, fx["files"])
    h = tr["endpoints"]["subset_rank"]
    art = st.GeometryAdapterArtifact.from_json(read(training / "subset_rank.json", h, 100000).decode(), "sha256:" + h, p["base"])
    counts = dict.fromkeys(BUDGET, 0)
    frozen = base.lora_to_json()
    with (out / "journal.jsonl").open("xb") as f:
        for state, origin in (("initial", base), ("subset_rank", base.with_geometry(art))):
            require((origin.geometry_artifact_id, origin.base_model_identity, origin.lora_artifact_id) ==
                    (p["origins"][state], p["base"], p["lora"]), "wrong origin")
            checkpoint(out / f"{state}.json", origin)
            for arm, guard in ARMS.items():
                model = origin
                for t in range(STEPS + 1):
                    owner = model.geometry_artifact_id
                    tags = dict(state=state, arm=arm, t=t, owner=owner)
                    record(f, dict(event="start", **tags))
                    handles, extra, rows = [], {k: [] for k in ALIASES}, []
                    for slot, e in enumerate(examples):
                        c, prompt = e["meaning"], e["prompt"]
                        ids = [c, c + 3]
                        kw = dict(accepted_indices=ids, stop_id=STOP)
                        if t < STEPS:
                            g = model.geometry_line_subset_gradients(prompt, MENU, **kw)
                            r = dict(**scores(g.line_subset_report(), ids), **sealed(g, owner, p, 18))
                            handles.append(g)
                            counts["subset_backwards"] += 1
                        else:
                            r = scores(model.line_subset_loss(prompt, MENU, **kw), ids)
                            counts["subset_forwards"] += 1
                        r["slot"] = slot
                        if c == 2:
                            r["singletons"] = []
                            for k in ALIASES:
                                if t < STEPS:
                                    g = model.geometry_line_set_gradients(prompt, [MENU[k]], stop_id=STOP)
                                    one = sealed(g, owner, p, 3)
                                    sr = g.line_set_report()
                                    extra[k].append(g)
                                    counts["singleton_backwards"] += 1
                                else:
                                    sr = model.line_set_loss(prompt, [MENU[k]], stop_id=STOP)
                                    one = {}
                                    counts["singleton_forwards"] += 1
                                require(sr["stop_id"] == STOP and sr["target_counts"] == [3] and sr["decoder_lanes"] == 1 and
                                        abs(sr["set_nll"] - r["lines"][k]) <= CROSS_TOL, "absolute loss differs across layouts")
                                r["singletons"].append(dict(index=k, cost=sr["set_nll"], **one))
                            if (arm == "three" and t == 0) or t == STEPS:
                                r["terms"] = term_scores(model, prompt)
                                counts["separate_token_scores"] += 5
                        rows.append(r)
                        record(f, dict(event="example", **tags, **r))
                    prior = p["replay"][state][t]
                    if arm == "three" or t == 0:
                        require(owner == prior["owner"] and control_point(rows) == prior["scores"], "three-class scores/gradients changed")
                    if t == STEPS:
                        break
                    objectives = [[handles[i] for i in ids] for ids in groups]
                    if guard is not None:
                        objectives.append(extra[guard])
                    nxt, receipt = model.step_common_geometry_sgd(objectives, RATE, target_l2=RADIUS)
                    counts["updates"] += 1
                    post = checkpoint(out / f"{state}-{arm}-{t+1:02}.json", nxt)
                    require(owner == model.geometry_artifact_id == receipt["pre_geometry_id"] and
                            "sha256:" + post == receipt["post_geometry_id"] and nxt.base_model_identity == p["base"] and
                            nxt.lora_to_json() == frozen and abs(receipt["applied_l2"] / RADIUS - 1) <= 1e-4, "bad update")
                    if arm == "three":
                        require(post == prior["post"] and sha(pack(receipt)) == prior["receipt"], "control update changed")
                    record(f, dict(event="step", **tags, post=post, receipt=receipt))
                    model = nxt
    require(counts == BUDGET, "wrong number of model calls")
    return put(out / "measured.json", dict(plan=pp, native=p["native"], counts=counts, journal=signature(out / "journal.jsonl")))


def run(fixture, training, protocol, pp, out):
    p = obj(protocol / "plan.json", pp)
    require(p["code"] == sources() and (p["budget"], p["seconds"], p["retries"]) == (BUDGET, SECONDS, 0), "protocol mismatch")
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(attempt=1, plan=pp, native=p["native"]))
    env = dict(os.environ, SPIRALTORCH_BACKEND="cpu", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"))
    cmd = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_worker", *map(str, (fixture, training, protocol, pp, out))]
    start = time.monotonic()
    with (out / "process.log").open("xb") as f:
        try:
            status = subprocess.run(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=SECONDS).returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    put(out / "process.json", dict(exit=status, seconds=time.monotonic() - start, log=signature(out / "process.log")))
    require(status == 0, "stopped on first failure: see process.log")
    return signature(out / "measured.json")


if __name__ == "__main__":
    from alias_protection_audit import audit
    cmd, *a = sys.argv[1:]
    if cmd == "freeze" and len(a) == 5:
        print(freeze(Path(a[0]), *a[1:4], Path(a[4])))
    elif cmd in ("measure", "_worker") and len(a) == 5:
        print((run if cmd == "measure" else train)(*(Path(s).resolve() for s in a[:3]), a[3], Path(a[4]).resolve()))
    elif cmd == "audit" and len(a) == 5:
        print(audit(Path(a[0]), a[1], Path(a[2]), a[3], Path(a[4])))
    else:
        raise SystemExit("freeze OLD PP MP RP OUT | measure FIXTURE TRAIN PLAN PP OUT | audit PLAN PP MEASURED MP OUT")

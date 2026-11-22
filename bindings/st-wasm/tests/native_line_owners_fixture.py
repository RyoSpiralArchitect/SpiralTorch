"""Matched, six-step complete-answer training on generated tiny weights only."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
shared = runpy.run_path(str(HERE / "native_cached_lines_fixture.py"))
st = shared["st"]
Session = st.NativeCausalLmLineTrainingSession
RECIPE = json.loads((HERE / "native_line_session_plan.json").read_bytes())
CASES = RECIPE["examples"]
MODES = ("lora", "geometry", "joint")
HOST = "ordinary/wave"
CUT, RETURN, TOTAL = 2, 4, 6


def pin(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def host(name):
    base = next(m for n, m, _ in shared["models"]() if n == name)
    seed = base.with_lora(0, 2, 2.)
    r = json.loads(seed.lora_to_json())
    # Nonzero A and B exercise both parameter derivatives from the first step.
    for key, phase in [("a", 2), ("b", 5)]:
        r[key] = [((i * 3 + phase) % 9 - 4) / 16 for i in range(len(r[key]))]
    m = base.with_lora_json(json.dumps(r))
    assert base.with_lora_json(m.lora_to_json().encode(), expected_id=m.lora_artifact_id).lora_artifact_id == m.lora_artifact_id
    return m, seed.lora_artifact_id


def plan(mode):
    return json.dumps(dict(RECIPE, schema="spiraltorch.native_causal_lm_line_plan.v2", mode=mode),
                      sort_keys=True, separators=(",", ":"))


def owners(m):
    return (m.lora_artifact_id, m.geometry_artifact_id)


def state(m):
    r = json.loads(m.lora_to_json())
    artifact = m.geometry_artifact()
    tensors = [] if artifact is None else json.loads(artifact.to_json())["placements"][0]["tensors"]
    return dict(a=r["a"], b=r["b"], geometry=[t["values"] for t in tensors],
                logits=m.forward([1, 2], [1, 2], last_only=True).values())


def losses(m):
    values = []
    for e in CASES:
        spec = dict(e["objective"])
        kind = spec.pop("kind")
        r = getattr(m, "line_" + kind + "_loss")(e["prompt_ids"], e["candidates"], stop_id=e["stop_id"], **spec)
        values.append(r["loss"])
    return values


def grad(m, e, mode):
    return m.line_adaptation_gradients(e["prompt_ids"], e["candidates"], stop_id=e["stop_id"],
                                       objective=json.dumps(e["objective"]), mode=mode, max_tile_values=11)


def issue(out):
    origin, seed_id = host(HOST)
    cases = {}
    for mode in MODES:
        p = plan(mode)
        s = Session(origin, p)
        m = origin
        steps = []
        for i in range(TOTAL):
            e = CASES[i % len(CASES)]
            g = grad(m, e, mode)
            r = s.step()
            start = owners(m)
            m = m.step_adaptation_sgd(g, mode=mode, learning_rate=RECIPE["learning_rate"])
            end = owners(m)
            assert end == owners(s.model)
            assert [x != y for x,y in zip(start,end)] == [mode != "geometry", mode != "lora"]
            assert (r["pre_lora_id"], r["pre_geometry_id"]) == start
            assert (r["post_lora_id"], r["post_geometry_id"]) == end
            assert r["mode"] == mode and r["update"] == i+1 and r["loss"] == g.loss
            assert r["objective"]["scores"] == getattr(g,"line_"+e["objective"]["kind"]+"_report")()
            assert r["work"] == g.line_training_work() and r["work"]["peak_logit_values"] == 11
            steps.append(r["loss"])
            if i+1 == CUT:
                raw, ck = s.checkpoint_with_id()
                assert pin(raw) == ck
        assert s.exhausted and s.step() is None
        cases[mode] = dict(plan=p, plan_id=s.plan_id, payload=raw, id=ck, steps=steps,
                           after=losses(m), endpoint=state(m), final_checkpoint=s.checkpoint_with_id()[1])
    # No attached geometry is a distinct control from an attached but frozen one.
    plain, _ = host("ordinary/plain")
    only = Session(plain, plan("lora"))
    assert only.step()["pre_geometry_id"] is None
    raw, ck = only.checkpoint_with_id()
    assert Session.from_json(plain, plan("lora"), raw, ck).model.lora_artifact_id == only.model.lora_artifact_id
    assert json.loads(raw)["geometry"] == dict(origin_id=None,id=None,weights=None)
    save(out,dict(host=HOST, seed_id=seed_id, lora=origin.lora_to_json(), lora_id=origin.lora_artifact_id,
                  before=losses(origin), cut=CUT, back=RETURN, total=TOTAL, cases=cases))
    print(json.dumps(dict(host=HOST, matched_updates=len(MODES)*TOTAL, manual_reference_updates=len(MODES)*TOTAL,
                          pure_lora_updates=1, before=losses(origin), after={k:r["after"] for k,r in cases.items()})))


def max_error(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((max_error(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((max_error(x,y) for x,y in zip(a,b)), default=0.)
    delta = abs(a-b)
    assert delta <= 3e-6, (a,b)
    return delta


def resume(ref_path, exported=None):
    bridge = json.loads(ref_path.read_bytes())
    back = json.loads(exported.read_bytes()) if exported else None
    origin, _ = host(bridge["host"])
    assert set(bridge["cases"]) == set(MODES)
    if back is not None:
        assert set(back) == set(MODES)
    worst = 0.
    for mode in MODES:
        ref = bridge["cases"][mode]
        c = ref if back is None else back[mode]
        s = Session.from_json(origin,ref["plan"],c["payload"],c["id"])
        cut = CUT if back is None else RETURN
        assert s.completed_updates == cut and s.checkpoint_with_id() == (c["payload"],c["id"])
        for i in range(cut,TOTAL):
            r=s.step()
            assert r["mode"] == mode and r["update"] == i+1
            worst=max(worst,max_error(r["loss"], ref["steps"][i]))
        worst=max(worst,max_error(state(s.model),ref["endpoint"]))
        if back is None:
            assert s.checkpoint_with_id()[1] == ref["final_checkpoint"]
        else:
            worst=max(worst,max_error(state(s.model),c["endpoint"]))
    print(json.dumps(dict(resumed_modes=len(MODES),suffix_updates=(TOTAL-cut)*len(MODES),max_abs_error=worst)))


if __name__ == "__main__":
    if sys.argv[1] == "issue" and len(sys.argv)==3:
        issue(Path(sys.argv[2]))
    elif sys.argv[1] == "resume" and len(sys.argv) in (3,4):
        resume(*map(Path,sys.argv[2:]))
    else:
        sys.exit("issue OUT | resume BRIDGE [RETURNED]")

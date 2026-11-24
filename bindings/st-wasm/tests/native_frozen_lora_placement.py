"""Issue and resume a six-site synthetic geometry handoff. No pretrained inputs."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
fixture = runpy.run_path(str(HERE/"native_line_owners_fixture.py"))
Session = fixture["Session"]
SITES = ["transformer.h.%d%s" % (layer, tail) for layer in range(2) for tail in (".attn", ".mlp", "")]
PROMPT = [2, 5, 1]
PLAN = json.dumps(dict(schema="spiraltorch.native_causal_lm_line_plan.v2", mode="geometry", epochs=1,
    learning_rate=.0625, max_tile_values=11, examples=[
        dict(prompt_ids=PROMPT, candidates=[[3], [3, 4]], stop_id=0, objective=dict(kind="set")),
        dict(prompt_ids=[5, 2], candidates=[[1, 5], [2]], stop_id=0,
             objective=dict(kind="menu", gold_index=0, mix=.5)),
        dict(prompt_ids=[3, 1], candidates=[[2], [2, 3], [5]], stop_id=0,
             objective=dict(kind="subset", accepted_indices=[0, 2])),
    ]), sort_keys=True, separators=(",", ":"))


def origin():
    active, _ = fixture["host"]("ordinary/wave")
    frozen = active.freeze_lora()
    assert frozen.frozen_lora_artifact_id == active.lora_artifact_id and frozen.lora_artifact_id is None
    assert frozen.frozen_lora_to_json() == active.lora_to_json()
    assert active.forward(PROMPT, [1, 3]).values() == frozen.forward(PROMPT, [1, 3]).values()
    return active, frozen


def state(model, path):
    placement, = json.loads(model.geometry_artifact().to_json())["placements"]
    assert placement["path"] == path and model.lora_artifact_id is None
    return dict(parameters={t["name"]: t["values"] for t in placement["tensors"]},
                logits=model.forward(PROMPT, [1, 3], last_only=True).values())


def check_step(session, path, fixed):
    m = session.model
    previous = m.geometry_artifact_id
    raw = m.frozen_lora_to_json()
    r = session.step()
    m = session.model
    assert (r["pre_geometry_id"], r["post_geometry_id"]) == (previous, m.geometry_artifact_id)
    assert "pre_lora_id" not in r and "post_lora_id" not in r and previous != m.geometry_artifact_id
    assert m.frozen_lora_artifact_id == fixed and m.frozen_lora_to_json() == raw
    return {"loss": r["loss"], **state(m, path)}


def issue(out):
    active, base = origin()
    cases = {}
    for path in SITES:
        model = base.with_single_geometry_at(path)
        s = Session(model, PLAN)
        start = state(model, path)
        step1 = check_step(s, path, active.lora_artifact_id)
        cut = s.checkpoint_with_id()
        step2 = check_step(s, path, active.lora_artifact_id)
        step3 = check_step(s, path, active.lora_artifact_id)
        assert s.exhausted and s.step() is None
        cases[path] = dict(start=start, origin=model.geometry_artifact_id, cut=cut, steps=[step1,step2,step3])
    with out.open("x") as f:
        json.dump(dict(schema="spiraltorch.synthetic_frozen_lora_placement.v1", plan=PLAN, cases=cases,
                       base=base.base_model_identity, frozen=active.lora_artifact_id),
                  f, sort_keys=True, allow_nan=False, separators=(",", ":"))
    print(json.dumps(dict(sites=len(cases), native_updates=len(cases)*3, pretrained_calls=0)))


def finish(reference, returned):
    native, foreign = [json.loads(p.read_bytes()) for p in (reference, returned)]
    assert foreign["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    _, base = origin()
    assert set(native["cases"]) == set(foreign["cases"]) == set(SITES)
    worst = 0.
    for path in SITES:
        payload, pin = foreign["cases"][path]
        assert fixture["pin"](payload) == pin
        s = Session.from_json(base.with_single_geometry_at(path), PLAN, payload, pin)
        assert s.checkpoint_with_id() == (payload, pin) and s.completed_updates == 2
        steps = native["cases"][path]["steps"]
        old = {k:v for k,v in steps[1].items() if k != "loss"}
        measure = fixture["max_error"]
        worst = max(worst, measure(state(s.model, path), old),
                    measure(check_step(s, path, base.frozen_lora_artifact_id), steps[2]))
        assert s.exhausted
    print(json.dumps(dict(returned_sites=len(SITES), native_suffix_updates=len(SITES),
                          maximum_absolute_difference=worst, pretrained_calls=0)))


if __name__ == "__main__":
    if not __debug__:
        sys.exit("assertions required")
    if len(sys.argv) == 3 and sys.argv[1] == "issue":
        issue(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "finish":
        finish(*map(Path, sys.argv[2:]))
    else:
        sys.exit("issue NEW_BRIDGE | finish BRIDGE WASM_RETURN")

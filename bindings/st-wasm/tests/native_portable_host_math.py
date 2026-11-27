"""Issue and accept one synthetic joint LoRA/WaveGate cross-client handoff."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import struct
import sys

HERE = Path(__file__).resolve().parent
PLAN_ID = "sha256:d8deba486de69e07582505001e2a67e2356d2b88039f9278b3a335d40eb5f263"
CUT_ID = "sha256:4261432f6dfc12d8ff646761dece12a2837bc7da51349d423bd66663032b0c8b"
NEXT_ID = "sha256:4d187e365f7f98eac0e2cf8a5c92bf4602cf5b4ba1f6caa5c97479eb043360e7"
POLICIES = ["portable_libm_tanh_f32.v1", "portable_libm_exp_f64.v1"]


def pin(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def word(x):
    assert math.isfinite(x)
    return struct.unpack(">I", struct.pack(">f", x))[0]


def fixture():
    bridge = json.loads((HERE / "native_line_owners_bridge.json").read_bytes())
    plan = json.loads(bridge["cases"]["joint"]["plan"])
    del plan["learning_rate"]
    plan.update(
        schema="spiraltorch.native_causal_lm_line_plan.v3",
        update=dict(
            kind="anchor_forward_kl", target_kl=1 / 65536, max_parameter_l2=1 / 8,
            anchor_examples=[0, 1, 2],
            criterion=dict(kind="mean_with_cap", max_anchor_kl=2 / 65536),
            direction=dict(kind="raw_then_sensitivity_ray", basis="parameter_block",
                           select="gradient_norm", probe_l2=1 / 256),
        ),
    )
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    assert pin(text) == PLAN_ID
    helper = runpy.run_path(str(HERE / "native_line_owners_fixture.py"))
    origin, _ = helper["host"](bridge["host"])
    assert origin.lora_artifact_id == bridge["lora_id"]
    assert [origin.host_gelu_tanh, origin.host_attention_exp] == POLICIES
    anchors = [plan["examples"][i]["prompt_ids"] for i in plan["update"]["anchor_examples"]]
    return helper["Session"], origin, text, plan, anchors


def snapshot(session, anchors):
    payload, identity = session.checkpoint_with_id()
    assert pin(payload) == identity
    model = session.model
    rows = [[word(x) for x in model.forward(p, [1, len(p)], last_only=True).values()]
            for p in anchors]
    assert [len(row) for row in rows] == [11, 11, 11]
    return dict(payload=payload, id=identity, cursor=session.completed_updates, rows=rows)


def issue(path):
    session_cls, origin, text, plan, anchors = fixture()
    session = session_cls(origin, text)
    for cursor in range(1, 5):
        attempt = session.attempt_step()
        assert attempt["status"] == "applied" and attempt["step"]["update"] == cursor
    before = snapshot(session, anchors)
    assert (before["cursor"], before["id"]) == (4, CUT_ID)
    e = plan["examples"][4 % len(plan["examples"])]
    gradients = session.model.line_adaptation_gradients(
        e["prompt_ids"], e["candidates"], stop_id=e["stop_id"],
        objective=json.dumps(e["objective"]), mode="joint", max_tile_values=plan["max_tile_values"],
    )
    blocks = {site: {name: list(map(word, data)) for name, data in block.items()}
              for site, block in gradients.parameter_gradients().items()}
    assert sorted(len(data) for block in blocks.values() for data in block.values()) == [3, 3, 6, 6]
    assert before == snapshot(session, anchors)
    attempt = session.attempt_step()
    assert attempt["status"] == "applied"
    step = attempt["step"]
    assert (step["update"], step["mode"]) == (5, "joint")
    assert step["pre_lora_id"] != step["post_lora_id"]
    assert step["pre_geometry_id"] != step["post_geometry_id"]
    after = snapshot(session, anchors)
    assert (after["cursor"], after["id"]) == (5, NEXT_ID)
    receipt = dict(plan=text, policies=POLICIES, before=before, after=after,
                   blocks=blocks, loss=word(step["loss"]))
    with path.open("x", encoding="utf-8") as fd:
        json.dump(receipt, fd, allow_nan=False, sort_keys=True)
        fd.write("\n")


def receive(native_path, wasm_path):
    session_cls, origin, text, _, anchors = fixture()
    native, wasm = (json.loads(p.read_bytes()) for p in (native_path, wasm_path))
    assert native == wasm and native["plan"] == text and native["policies"] == POLICIES
    for key, identity, cursor in (("before", CUT_ID, 4), ("after", NEXT_ID, 5)):
        state = wasm[key]
        assert (state["id"], state["cursor"]) == (identity, cursor)
        restored = session_cls.from_json(origin, text, state["payload"], identity)
        assert snapshot(restored, anchors) == state
    print(json.dumps(dict(gradient_words=18, anchor_logit_words_per_cut=33,
                          continued_update=5, cut=CUT_ID, endpoint=NEXT_ID)))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["issue", dest]:
            issue(Path(dest))
        case ["receive", native, wasm]:
            receive(Path(native), Path(wasm))
        case _:
            sys.exit("usage: issue NEW_NATIVE.json | receive NATIVE.json WASM.json")

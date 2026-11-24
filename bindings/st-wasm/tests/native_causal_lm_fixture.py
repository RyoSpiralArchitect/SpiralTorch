"""Generate tiny portable decoder checkpoints and native reference values. No pretrained input."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import sys


def build():
    helpers = runpy.run_path(str(Path(__file__).resolve().parents[2] / "st-py/tests/test_native_causal_lm.py"))
    st = helpers["st"]
    cases = []
    for outside, distances in [
        ([(0., 2.)], [("repair", 1/65536)]),
        ([(0., 0.)], [("infeasible", 1/65536)]),
        ([(-4., 0.), (6., 0.)], [("accept", 1/65536), ("finite_reject", 1/128)]),
    ]:
        config, raw = helpers["uniform_set_fixture"](*outside)
        base = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(raw))
        art = helpers["artifact"](base, path="transformer.h.0", gate=0., trainable=True, trainable_names=("bias",))
        origin = base.with_geometry(art)
        steps = []
        for name, distance in distances:
            kw = {"contrasts": [(1, 0, [1])]}
            try:
                selected, receipt = origin.step_declared_set_geometry_at_distance(
                    [2, 0], [1, 2], target_l2=distance, **kw)
            except ValueError as e:
                if name != "infeasible" or "feasible interval" not in str(e):
                    raise
                selected, proposed = origin, None
            else:
                assert name != "infeasible" and receipt["accepted"] == (name != "finite_reject")
                proposed = receipt["proposed"]
            steps.append(dict(name=name, distance=distance, proposed=proposed,
                              logits=selected.forward([2], [1, 1], last_only=True).values(),
                              scores=selected.declared_set_nll([2, 0], [1, 2], **kw)))
        cases.append(dict(config=config, weights=base64.b64encode(raw).decode("ascii"),
                          base_id=base.base_model_identity, origin_id=art.artifact_id,
                          geometry=art.to_json(), steps=steps))
    return {"schema": "spiraltorch.wasm.artificial_declared_step.v1", "cases": cases}


def build_cache_reference():
    """The oracle only uses full-prefix and the original one-token API, never with_suffix."""
    directory = Path(__file__).resolve().parents[2] / "st-py/tests"
    helpers = runpy.run_path(str(directory / "test_native_causal_lm.py"))
    recipes = runpy.run_path(str(directory / "test_native_geometry_runtime.py"))
    st = helpers["st"]
    config, raw = helpers["fixture"](width=3, heads=3, raw_weights=True)
    weights = st.SafeTensorArchive.from_bytes(raw)
    prompt, chunk, tails = [1, 2], [3, 4, 5], [[6, 7], [2, 8]]
    cases = []
    for lag in (False, True):
        cfg = json.loads(config)
        if lag:
            cfg.update(model_type="spiraltorch_gpt2_key_lag_v1", spiraltorch_key_lag={"layer": 1, "lag": 2})
        cfg = json.dumps(cfg)
        base = st.NativeCausalLm.from_gpt2(cfg, weights)
        for kind in ("plain", "wave", "history_gain"):
            m = base
            if kind != "plain":
                state = json.loads(recipes["artifact"](kind).state_json())
                state["base_model_identity"] = base.base_model_identity
                state["placements"][0]["path"] = "transformer.h.0.attn"
                m = base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
                grad = m.geometry_loss_gradients([2, 1, 3, 4], [1, 4])
                learned = m.step_geometry_sgd(grad, .125)
                if learned.geometry_artifact_id == m.geometry_artifact_id:
                    raise ValueError("artificial geometry did not move")
                m = learned
            refs = []
            for suffix in ([], chunk, *[chunk + tail for tail in tails]):
                ids = prompt + suffix
                full = m.forward(ids, [1, len(ids)], last_only=True).values()
                ctx = st.NativeCausalLmContext(m, prompt, max_cache_values=1024)
                for t in suffix:
                    ctx.append(t)
                if not all(abs(a-b) <= 3e-6 for a, b in zip(ctx.logits().values(), full)):
                    raise ValueError("legacy scalar cache differs from full prefix")
                refs.append(full)
            art = m.geometry_artifact()
            cases.append(dict(name=("lag/" if lag else "ordinary/") + kind, config=cfg, base_id=m.base_model_identity,
                              geometry=art.to_json() if art else None, geometry_id=m.geometry_artifact_id, refs=refs))
    return dict(weights=base64.b64encode(raw).decode("ascii"), prompt=prompt, chunk=chunk, tails=tails, cases=cases)


def verify_wasm(fixture_file, selected_file):
    import spiraltorch as st
    fixture = json.loads(Path(fixture_file).read_bytes())
    selected = json.loads(Path(selected_file).read_bytes())
    if [x["name"] for x in selected] != ["repair", "accept", "finite_reject"]:
        raise ValueError("missing or unordered WASM results")
    errors = []
    for item in selected:
        f, = [f for f in fixture["cases"] if f["base_id"] == item["base_id"]]
        reference, = [s for s in f["steps"] if s["name"] == item["name"]]
        base = st.NativeCausalLm.from_gpt2(f["config"], st.SafeTensorArchive.from_bytes(base64.b64decode(f["weights"])))
        art = st.GeometryAdapterArtifact.from_json(item["geometry"], expected_id=item["geometry_id"],
                                                    expected_base_model_identity=base.base_model_identity)
        if (art.artifact_id == f["origin_id"]) != (item["name"] == "finite_reject"):
            raise ValueError("wrong selected weights")
        m = base.with_geometry(art)
        logits = m.forward([2], [1, 1], last_only=True).values()
        scores = m.declared_set_nll([2, 0], [1, 2], contrasts=[(1, 0, [1])])
        for ref in (item, reference):
            if len(logits) != len(ref["logits"]) or scores.keys() != ref["scores"].keys():
                raise ValueError("returned model shape or score keys changed")
            errors.extend(abs(a-b) for a, b in zip(logits, ref["logits"]))
            errors.extend(abs(a-ref["scores"][k]) for k, a in scores.items())
    if not all(error <= 2e-6 for error in errors):
        raise ValueError("Python does not reproduce returned WASM weights")
    print(json.dumps(dict(wasm_to_python=len(selected), max_absolute_error=max(errors))))


if __name__ == "__main__":
    def native_only(event, args):
        if event == "import" and args[0].split(".")[0] in {"torch", "numpy", "transformers", "safetensors", "accelerate"}:
            raise AssertionError("unexpected foreign ML import: " + args[0])
    sys.addaudithook(native_only)
    if len(sys.argv) == 4 and sys.argv[1] == "--verify-wasm":
        verify_wasm(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[1] == "--cache"):
        data = build() if len(sys.argv) == 2 else build_cache_reference()
        payload = (json.dumps(data, allow_nan=False, indent=2) + "\n").encode()
        with Path(sys.argv[-1]).open("xb") as output:
            output.write(payload)
        print(hashlib.sha256(payload).hexdigest())
    else:
        sys.exit("usage: [--cache] OUTPUT | --verify-wasm FIXTURE_JSON SELECTED_JSON")

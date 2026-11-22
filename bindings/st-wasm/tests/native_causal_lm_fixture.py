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
    elif len(sys.argv) == 2:
        payload = (json.dumps(build(), allow_nan=False, indent=2) + "\n").encode()
        with Path(sys.argv[1]).open("xb") as output:
            output.write(payload)
        print(hashlib.sha256(payload).hexdigest())
    else:
        sys.exit("usage: OUTPUT | --verify-wasm FIXTURE_JSON SELECTED_JSON")

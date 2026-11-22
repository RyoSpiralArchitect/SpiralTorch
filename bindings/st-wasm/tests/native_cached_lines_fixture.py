"""Generate the padded-path oracle, or check the new installed binding. Artificial weights only."""
import base64
import json
from pathlib import Path
import sys


def no_foreign(event, args):
    if event == "import" and args[0].partition(".")[0] in {"torch", "numpy", "transformers", "safetensors", "accelerate"}:
        raise RuntimeError("unexpected external ML import: " + args[0])


sys.addaudithook(no_foreign)
import spiraltorch as st

HERE = Path(__file__).resolve().parent
CONTENTS = [[1], [1, 2, 3], [1, 2, 4], [1, 5], [7, 8]]
STOP = 0


def models():
    source = json.loads((HERE / "native_causal_lm_cache_fixture.json").read_bytes())
    arch = st.SafeTensorArchive.from_bytes(base64.b64decode(source["weights"], validate=True))
    for case in source["cases"]:
        model = st.NativeCausalLm.from_gpt2(case["config"], arch)
        if case["geometry"] is not None:
            model = model.with_geometry(st.GeometryAdapterArtifact.from_json(
                case["geometry"], expected_id=case["geometry_id"], expected_base_model_identity=case["base_id"]))
        assert (model.base_model_identity, model.geometry_artifact_id) == (case["base_id"], case["geometry_id"])
        yield case["name"], model, source["prompt"]


def generate(dest):
    cases = {name: model.score_lines(prompt, CONTENTS, stop_id=STOP) for name, model, prompt in models()}
    assert len(cases) == 6
    with dest.open("x") as f:
        json.dump(dict(contents=CONTENTS, stop=STOP, cases=cases), f, sort_keys=True, allow_nan=False, indent=2)
        f.write("\n")


def verify(src):
    ref = json.loads(src.read_bytes())
    rows = 0
    max_error = 0.
    for name, model, prompt in models():
        root = st.NativeCausalLmContext(model, prompt, max_cache_values=1024)
        before = root.logits().values()
        receipt = root.score_lines(CONTENTS, stop_id=STOP)
        actual = receipt["scores"]
        expected = ref["cases"][name]
        assert actual.keys() == expected.keys()
        for k in ("target_counts", "stop_id", "winner_index", "prefix_nodes", "decoder_lanes", "shared_content_tokens"):
            assert actual[k] == expected[k], (name, k)
        for key in ("menu_set_nll", "line_nlls", "conditional_nlls", "token_nlls"):
            def errors(a, b):
                if isinstance(a, list):
                    assert len(a) == len(b)
                    return [e for x, y in zip(a, b) for e in errors(x, y)]
                d = abs(a-b)
                assert d <= 3e-6, (name, key, d)
                return [d]
            max_error = max(max_error, max(errors(actual[key], expected[key])))
        assert [receipt[k] for k in ("cached_prompt_tokens", "decoded_content_tokens", "decoder_chunks")] == [2, 7, 6]
        assert root.token_ids == prompt and before == root.logits().values()
        rows += sum(actual["target_counts"])
    assert ref["contents"] == CONTENTS and ref["stop"] == STOP and rows == 96
    print(json.dumps(dict(checked_token_costs=rows, max_abs_old_padded=max_error)))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"generate", "verify"}:
        sys.exit("usage: generate|verify PATH")
    {"verify": verify, "generate": generate}[sys.argv[1]](Path(sys.argv[2]))

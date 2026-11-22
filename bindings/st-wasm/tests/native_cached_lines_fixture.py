"""Generate/check padded scores or snapshot dense training gradients. Artificial weights only."""
import base64
import json
import math
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


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for k in ("target_counts", "stop_id", "winner_index", "prefix_nodes", "decoder_lanes", "shared_content_tokens"):
        assert actual[k] == expected[k], k
    def error(a, b):
        if isinstance(a,list):
            assert len(a) == len(b)
            return max(error(x,y) for x,y in zip(a,b))
        d = abs(a-b)
        assert d <= 3e-6, d
        return d
    return max(error(actual[k], expected[k]) for k in ("menu_set_nll","line_nlls","conditional_nlls","token_nlls"))


def verify(src):
    ref = json.loads(src.read_bytes())
    rows, maxima = [0,0], [0.,0.]
    for name, model, prompt in models():
        root = st.NativeCausalLmContext(model, prompt, max_cache_values=1024)
        before = root.logits().values()
        for i, (cap, batches, peak) in enumerate(((None,6,22),(11,7,11))):
            r = root.score_lines(CONTENTS, stop_id=STOP, max_tile_values=cap)
            maxima[i] = max(maxima[i],compare(r["scores"],ref["cases"][name]))
            assert [r[k] for k in ("cached_prompt_tokens","decoded_content_tokens","decoder_chunks")] == [2,7,6]
            assert [r[k] for k in ("scanned_logit_values","retained_edge_logits","readout_batches",
                                   "peak_projected_logit_values")] == [88,12,batches,peak]
            rows[i] += sum(r["scores"]["target_counts"])
        assert root.token_ids == prompt and before == root.logits().values()
    assert ref["contents"] == CONTENTS and ref["stop"] == STOP and rows == [96,96]
    print(json.dumps(dict(checked_token_costs=rows[0], tiled_token_costs=rows[1],
                          max_abs_old_padded=maxima[0], max_abs_tiled_padded=maxima[1])))


def gradients(dest):
    """Check dense training stayed unchanged; same frozen artificial models, no update."""
    lines = [[1], [1,2], [3,4]]
    records = []
    for name, model, prompt in models():
        if model.geometry_artifact_id is None:
            continue
        for kind, extra in (("menu",dict(gold_index=1,mix=0.5)),("set",{}),("subset",dict(accepted_indices=[0,2]))):
            kw = dict(extra, stop_id=STOP)
            report = getattr(model,"line_"+kind+"_loss")(prompt,lines,**kw)
            g = getattr(model,"geometry_line_"+kind+"_gradients")(prompt,lines,**kw)
            assert getattr(g,"line_"+kind+"_report")() == report and g.target_tokens == 8
            values = g.parameter_gradients()
            flat = [v for tensors in values.values() for block in tensors.values() for v in block]
            assert flat and all(math.isfinite(v) for v in flat)
            records.append(dict(host=name, kind=kind, geometry_id=g.geometry_artifact_id, loss=g.loss,
                                targets=g.target_tokens, parameters=values, nonzero=sum(v!=0 for v in flat), report=report))
    assert len(records)==12 and all(r["nonzero"] for r in records)
    with dest.open("x") as f:
        json.dump(records,f,sort_keys=True,allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    modes = {"verify": verify, "generate": generate, "gradients": gradients}
    if len(sys.argv) != 3 or sys.argv[1] not in modes:
        sys.exit("usage: generate|verify|gradients PATH")
    modes[sys.argv[1]](Path(sys.argv[2]))

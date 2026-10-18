"""Optional independent Torch CE reference; never used by native evaluation."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))
data = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_evaluation.py")))


@pytest.mark.parametrize("kind", (None, *helpers["KINDS"]))
@pytest.mark.parametrize("masked", [False, True])
def test_native_corpus_matches_independent_target_sum_with_geometry(kind, masked):
    reference, native = helpers["pair"](width=3, heads=1)
    handles = []
    if kind is not None:
        record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
        record["base_model_identity"] = native.base_model_identity
        record["placements"][0]["path"] = "transformer.h.0.attn"
        if kind == "affine":
            record["placements"][0]["recipe"]["max_values"] = 96
        artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
        native = native.with_geometry(artifact)
        _, handles = helpers["attach"](reference, artifact)
    docs = data["DOCS"]
    labels = [list(d) for d in docs]
    if masked:
        labels = [[-100]*len(d) for d in docs]
        labels[3][4:7] = [1, 9, 0]
        labels[4][2] = 7
    total, count = 0., 0
    try:
        with torch.no_grad():
            for doc, start, end in data["SPANS"]:
                ids = torch.tensor([docs[doc][start:end-1]])
                targets = torch.tensor(labels[doc][start+1:end])
                active = int((targets != -100).sum())
                if active:
                    logits = reference(ids, use_cache=False).logits.flatten(0, 1)
                    total += torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100, reduction="sum").item()
                    count += active
        for batch in (1, 2, 8):
            result = native.evaluate_documents(docs, 4, labels=labels if masked else None, microbatch_windows=batch)
            assert result.target_tokens == count
            assert result.nll == pytest.approx(total/count, abs=3e-6)
    finally:
        for handle in handles:
            handle.remove()

"""Optional Torch decoder/probability reference, not a runtime or RNG dependency."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))


@pytest.mark.parametrize("kind", (None, *helpers["KINDS"]))
@pytest.mark.parametrize("sampling,k,p,temperature", [(False, None, 1., 1.), (False, 8, .8, .7), (True, 8, .8, .7)])
def test_native_sequence_matches_reference_probabilities(kind, sampling, k, p, temperature):
    reference, base = helpers["pair"](width=3, heads=1)
    reference.requires_grad_(False)
    model, handles = base, []
    if kind:
        record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
        record["base_model_identity"] = base.base_model_identity
        record["placements"][0]["path"] = "transformer.h.0.attn"
        if kind == "affine":
            record["placements"][0]["recipe"]["max_values"] = 96
        artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
        model = base.with_geometry(artifact)
        stack, handles = helpers["attach"](reference, artifact)
    generator = st.NativeCausalLmGenerator(model, [1, 5, 7], json.dumps({"max_new_tokens": 5, "seed": 581,
        "decoding": {"selection": "sample" if sampling else "greedy", "top_k": k, "top_p": p,
                     "distribution": {"kind": "softmax", "temperature": temperature}}}))
    try:
        for _ in range(5):
            with torch.no_grad():
                logits = reference(torch.tensor([generator.token_ids]), use_cache=False).logits[0, -1].double()
            # Independent stable rank and first-crossing nucleus, not Torch's RNG.
            ids = torch.argsort(logits, descending=True, stable=True)[:k]
            probabilities = torch.softmax(logits[ids] / temperature, dim=-1)
            if p < 1.:
                keep = int(torch.searchsorted(probabilities.cumsum(0), torch.tensor(p, dtype=torch.float64)).item()) + 1
                ids, probabilities = ids[:keep], probabilities[:keep]
            probabilities = probabilities / probabilities.sum()
            report = generator.step()
            values = json.loads(report.to_json())
            assert values["candidates"] == len(ids)
            assert values["raw_top_token"] == int(logits.argmax())
            assert values["zspace_contract"] is None
            if not sampling:
                assert report.token_id == int(ids[0])
            position = ids.tolist().index(report.token_id)
            assert report.probability == pytest.approx(float(probabilities[position]), abs=2e-6, rel=2e-5)
            entropy = -(probabilities * probabilities.log()).sum().item()
            assert report.entropy == pytest.approx(entropy, abs=2e-6, rel=2e-5)
    finally:
        for handle in handles:
            handle.remove()

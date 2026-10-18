"""Torch is only the optional numerical oracle, never the native execution owner."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))


@pytest.mark.parametrize("kind", helpers["KINDS"])
@pytest.mark.parametrize("source", ["prior_continuation", "model_topk_history", "model_topk_periodic"])
@pytest.mark.parametrize("arithmetic", ["f32", "f64"])
def test_repetition_loss_geometry_vjp_and_update_match_optional_torch(kind, source, arithmetic):
    reference, base = helpers["pair"](width=3, heads=1)
    base = base.with_readout_accumulator(arithmetic)
    reference.requires_grad_(False)
    record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
    record["base_model_identity"] = base.base_model_identity
    site = record["placements"][0]
    site["path"] = "transformer.h.0.attn"
    if kind == "affine":
        site["recipe"]["max_values"] = 96
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(artifact)
    stack, handles = helpers["attach"](reference, artifact)
    ids = [[1, 2, 1, 2, 1, 3], [4, 5, 6, 7, 8, 9]]
    labels = [[-100, -100, -100, -100, -100, 3], [-100, -100, -100, -100, -100, 9]]
    # The reference spells out these simple negative sets independently of the planner.
    # All vocabulary proposals avoid cross-backend top-k cut/tie differences.
    vocabulary = reference.config.vocab_size
    candidates = [[1, 2], [4, 5, 6, 7, 8]] if source == "model_topk_history" else [[2], []]
    config = {"kind": "native_v1", "config": {
        "strength": .7, "context_window": 16, "max_candidates_per_position": 8,
        "candidate_source": {"kind": source, **({"ngram_order": 2} if source == "prior_continuation"
                                               else {"proposal_top_k": vocabulary})}}}
    try:
        logits = reference(torch.tensor(ids)[:, :-1], use_cache=False).logits[:, -1, :]
        ce = torch.nn.functional.cross_entropy(logits, torch.tensor([3, 9]), label_smoothing=.1)
        probabilities = logits.double().softmax(-1)
        terms = [-torch.log1p(-probabilities[i, row].clamp(max=1. - 1e-6)).mean()
                 for i, row in enumerate(candidates) if row]
        auxiliary = torch.stack(terms).mean()
        loss = ce + .7 * auxiliary
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    gradient = native.geometry_loss_gradients([v for row in ids for v in row], [2, 6],
        labels=[v for row in labels for v in row], label_smoothing=.1, repetition=config)
    assert gradient.repetition_positions == len(terms)
    assert gradient.repetition_loss == pytest.approx(auxiliary.item(), abs=2e-7)
    assert gradient.loss == pytest.approx(loss.item(), abs=3e-6)
    assert gradient.causal_loss == pytest.approx(ce.item(), abs=3e-6)
    for name, parameter in stack.adapters[0].named_parameters():
        actual = gradient.parameter_gradients()[site["path"]][name]
        torch.testing.assert_close(torch.tensor(actual).reshape_as(parameter), parameter.grad, atol=3e-6, rtol=4e-4)
    updated = native.step_geometry_sgd(gradient, .03)
    torch.optim.SGD(stack.parameters(), lr=.03).step()
    restored = st.GeometryAdapterStack.from_artifact(updated.geometry_artifact(),
                                                     expected_base_model_identity=base.base_model_identity)
    for (name, actual), (other, expected) in zip(restored.named_parameters(), stack.named_parameters()):
        assert name == other
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert all(p.grad is None for p in reference.parameters())

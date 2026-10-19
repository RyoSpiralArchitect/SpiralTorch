"""Optional ragged-window reference; no Torch execution in the native accumulator."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))


@pytest.mark.parametrize("kind", helpers["KINDS"])
@pytest.mark.parametrize("method", ["sgd", "adamw"])
def test_ragged_loss_gradient_mean_and_single_update_match_reference(kind, method):
    reference, base = helpers["pair"](width=3, heads=1)
    reference.requires_grad_(False)
    record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"][0]["path"] = "transformer.h.0.attn"
    if kind == "affine":
        record["placements"][0]["recipe"]["max_values"] = 96
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(artifact)
    stack, handles = helpers["attach"](reference, artifact)
    named = dict(stack.adapters[0].named_parameters())
    rate, decay, clip = .02, .01, .001
    optimizer = (torch.optim.SGD(stack.parameters(), lr=rate) if method == "sgd" else
                 torch.optim.AdamW(stack.parameters(), lr=rate, betas=(.7, .9), eps=.05, weight_decay=decay, foreach=False))
    native_opt = st.NativeGeometryAdamW(artifact, beta1=.7, beta2=.9, epsilon=.05, max_gradient_norm=clip)
    batches = [torch.tensor([[1, 5, 7]]), torch.tensor([[2, 8, 4, 6], [3, 1, 9, 2]]), torch.tensor([[4, 3, 2, 1, 5]])]
    try:
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            accumulation = native.geometry_accumulator()
            expected_loss = 0.
            for tokens in batches:
                logits = reference(tokens[:, :-1], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum") / 12
                loss.backward()
                expected_loss += loss.item()
                accumulation.accumulate_loss(native, tokens.flatten().tolist(), list(tokens.shape))
            assert (accumulation.microbatches, accumulation.target_tokens) == (3, 12)
            assert accumulation.loss == pytest.approx(expected_loss, abs=3e-6)
            for name, parameter in named.items():
                actual = accumulation.parameter_gradients()["transformer.h.0.attn"][name]
                torch.testing.assert_close(torch.tensor(actual).reshape_as(parameter), parameter.grad, atol=3e-6, rtol=3e-4)
            if method == "adamw":
                torch.nn.utils.clip_grad_norm_(stack.parameters(), clip)
                native, native_opt = accumulation.step_adamw(native, native_opt, rate, weight_decay=decay)
                assert native_opt.revision == update + 1
                assert {s["steps"] for s in json.loads(native_opt.to_json())["slots"]} == {update + 1}
            else:
                native = accumulation.step_sgd(native, rate)
            optimizer.step()
            current = json.loads(native.geometry_artifact().state_json())["placements"][0]
            for parameter in current["tensors"]:
                if parameter["role"] == "parameter":
                    expected = named[parameter["name"]]
                    torch.testing.assert_close(torch.tensor(parameter["values"]).reshape_as(expected), expected, atol=3e-6, rtol=3e-4)
        assert all(p.grad is None for p in reference.parameters())
    finally:
        for handle in handles:
            handle.remove()

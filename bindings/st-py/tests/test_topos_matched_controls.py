"""Identify Topos limiting cases, not language quality or execution speed.

Torch controls exist only in tests. Production recurrence/VJP remain in Rust.
Randomly initialized HF models use synthetic IDs, no downloads or weight steps.
"""

import copy
import json

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")

COUPLING, ITERATIONS = .25, 4
GAIN = sum(COUPLING ** i for i in range(ITERATIONS))
STRENGTH = .5


class MatchedGate(torch.nn.Module):
    """Same gate count, zero initialization and local gain, with optional clip."""

    def __init__(self, features, saturation=None):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32, device="cpu"))
        self.saturation = saturation

    def forward(self, value):
        output = (value * self.gate) * GAIN
        if self.saturation is not None:
            output = output.clamp(-self.saturation, self.saturation)
        return value + STRENGTH * output


def topos(features, porosity, saturation=1.):
    return st.ToposResonatorAdapter(features, strength=STRENGTH, coupling=COUPLING,
                                   iterations=ITERATIONS, porosity=porosity,
                                   saturation=saturation)


def pointwise(adapter, values):
    value = values.detach().clone().requires_grad_()
    output = adapter(value)
    upstream = torch.linspace(.1, 1., value.numel(), dtype=value.dtype).reshape(value.shape)
    dx, dg = torch.autograd.grad(output, (value, adapter.gate), upstream)
    return output.detach(), dx, dg


@pytest.mark.parametrize("porosity", [0., .3, 1.])
@pytest.mark.parametrize("gate", [0., .125])
def test_unsaturated_topos_matches_ordinary_gate_with_matched_gain(porosity, gate):
    values = torch.tensor([[-.8, .3, 1.], [.2, -.7, -.1]], dtype=torch.float32)
    ordinary, geometry = MatchedGate(3), topos(3, porosity)
    with torch.no_grad():
        ordinary.gate.fill_(gate)
        geometry.gate.copy_(ordinary.gate)
    assert float((values * ordinary.gate.detach() * GAIN).abs().max()) < 1.
    actual, expected = pointwise(geometry, values), pointwise(ordinary, values)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-6, atol=2e-7)
    assert torch.count_nonzero(actual[2]) == 3


def test_hard_topos_matches_clipped_control_but_porous_tail_is_distinct():
    values = torch.tensor([[-2., -.1, 2.], [-3., .2, 3.]], dtype=torch.float32)
    ordinary, hard, porous = MatchedGate(3, 1.), topos(3, 0.), topos(3, .3)
    for adapter in (ordinary, hard, porous):
        with torch.no_grad():
            adapter.gate.fill_(1.)
    expected, hard_values, porous_values = [pointwise(a, values) for a in (ordinary, hard, porous)]
    for actual, reference in zip(hard_values, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-7)
    upstream = torch.linspace(.1, 1., values.numel(), dtype=values.dtype).reshape(values.shape)
    assert torch.equal(hard_values[1][:, [0, 2]], upstream[:, [0, 2]])
    assert torch.all(porous_values[1][:, [0, 2]] < upstream[:, [0, 2]])
    assert torch.all(hard_values[2][[0, 2]] == 0.)
    assert torch.all(porous_values[2][[0, 2]] != 0.)


@pytest.fixture(params=["gpt2", "llama"])
def frozen_model(request):
    transformers = pytest.importorskip("transformers")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(281)
            if request.param == "gpt2":
                model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
                    vocab_size=32, n_positions=8, n_embd=8, n_layer=2, n_head=2,
                    resid_pdrop=0., attn_pdrop=0., embd_pdrop=0., use_cache=False))
                path = "transformer.h.0.mlp"
            else:
                model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
                    vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
                    attention_dropout=0., use_cache=False))
                path = "model.layers.0.mlp"
            model = model.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
            before = copy.deepcopy(model.state_dict())
            yield request.param, model, path
            assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            assert not model.get_submodule(path)._forward_hooks and not model._forward_pre_hooks
    finally:
        torch.set_num_threads(threads)


def model_pullback(model, path, adapter):
    ids = torch.arange(12, device="cpu").reshape(2, 6)
    embedded = model.get_input_embeddings()(ids).detach().requires_grad_()
    stack = st.GeometryAdapterStack({path: adapter})
    assert sum(p.numel() for p in stack.parameters()) == 8
    hidden = []
    handle = model.get_submodule(path).register_forward_hook(
        lambda _module, _args, output: hidden.append(output.detach().clone()))
    try:
        with stack.attach(model):
            prediction = model(inputs_embeds=embedded, labels=ids, use_cache=False)
            dx, dg = torch.autograd.grad(prediction.loss, (embedded, adapter.gate))
    finally:
        handle.remove()
    assert len(hidden) == 1
    result = {"logits": prediction.logits.detach(), "loss": prediction.loss.detach(),
              "input_gradient": dx, "gate_gradient": dg, "hidden": hidden[0]}
    assert all(torch.isfinite(value).all() for value in result.values())
    return result


def test_zero_gate_cannot_identify_porosity_in_language_model_loss(frozen_model):
    _, model, path = frozen_model
    reference = model_pullback(model, path, MatchedGate(8))
    for porosity in (0., .3, 1.):
        actual = model_pullback(model, path, topos(8, porosity))
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=3e-5, atol=1e-8)
        assert torch.count_nonzero(actual["gate_gradient"]) > 0


def test_porous_difference_reaches_language_model_loss_pullback(frozen_model):
    architecture, model, path = frozen_model
    saturation = .001
    ordinary, hard, porous = MatchedGate(8, saturation), topos(8, 0., saturation), topos(8, 1., saturation)
    for adapter in (ordinary, hard, porous):
        with torch.no_grad():
            adapter.gate.fill_(64.)
    reference, hard_values, porous_values = [model_pullback(model, path, a) for a in (ordinary, hard, porous)]
    for name in reference:
        torch.testing.assert_close(hard_values[name], reference[name], rtol=3e-5, atol=1e-8)
    assert torch.count_nonzero((reference["hidden"] * 64.).abs() > saturation) > 0
    assert float((porous_values["logits"] - hard_values["logits"]).abs().max()) > 1e-7
    assert not torch.allclose(porous_values["gate_gradient"], hard_values["gate_gradient"], rtol=1e-4, atol=1e-12)
    print(json.dumps({"architecture": architecture, "scope": "synthetic frozen model; no updates or quality claim",
                      "max_logit_difference": float((porous_values["logits"] - hard_values["logits"]).abs().max()),
                      "max_gate_gradient_difference": float((porous_values["gate_gradient"] - hard_values["gate_gradient"]).abs().max())}))

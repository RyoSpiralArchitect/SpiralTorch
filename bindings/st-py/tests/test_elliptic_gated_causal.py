import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticGatedCausalLearningBatch"), reason="native gated batch required"
)


def inputs(sequence=4):
    generator = torch.Generator().manual_seed(101)
    x = torch.randn(2, sequence, 3, generator=generator)
    x[..., 0] = 1
    return x.requires_grad_()


@pytest.mark.parametrize("raw", [-0.7, 0.0, 0.8])
@pytest.mark.parametrize("sequence", [4, 128])
def test_equivalent_torch_formula_including_shared_gate_and_training_shape(raw, sequence):
    warp = st.EllipticWarp(1.3, 3, 2)
    x = inputs(sequence)
    other = x.detach().clone().requires_grad_()
    gate = torch.tensor(raw, requires_grad=True)
    ref_gate = gate.detach().clone().requires_grad_()
    actual = st.elliptic_gated_causal_autograd(warp, x, gate)
    local = st.elliptic_warp_autograd(warp, other).double()
    scores = local @ local.transpose(-1, -2) / 3
    mask = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    context = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ local).float().double()
    g = ref_gate.double().tanh()
    expected = ((1 - g) * local + g * context).float()
    seed = torch.cos(torch.arange(actual.numel()) * 0.3).reshape_as(actual)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(x.grad, other.grad, atol=3e-5, rtol=1e-4)
    torch.testing.assert_close(gate.grad, ref_gate.grad, atol=3e-5, rtol=1e-4)
    assert gate.grad.abs() > 0


def test_zero_exactness_native_sum_and_immutable_snapshot():
    warp = st.EllipticWarp(1.0, 3, 2)
    x = inputs(3)
    other = x.detach().clone().requires_grad_()
    gate = torch.tensor(0.0, requires_grad=True)
    actual = st.elliptic_gated_causal_autograd(warp, x, gate)
    local = st.elliptic_warp_autograd(warp, other)
    assert torch.equal(actual, local)
    seed = torch.cos(torch.arange(actual.numel()) * 0.3).reshape_as(actual)
    (actual * seed).sum().backward()
    (local * seed).sum().backward()
    assert torch.equal(x.grad, other.grad)
    snapshot = warp.map_gated_causal_batch(
        x.detach().flatten().tolist(), batch_size=2, sequence_length=3, raw_mix=0.0
    )
    assert snapshot.mix == 0
    dx, dg = snapshot.vjp(seed.flatten().tolist())
    assert torch.equal(torch.tensor(dx).reshape_as(x), x.grad)
    assert dg == gate.grad.item()
    duplicate = warp.map_gated_causal_batch(
        x.detach().flatten().tolist() * 2, batch_size=4, sequence_length=3, raw_mix=0.0
    )
    assert duplicate.vjp(seed.flatten().tolist() * 2)[1] == pytest.approx(2 * dg)
    features = snapshot.features
    features[0] = float("nan")
    warp.configure(sheet_count=5, spin_harmonics=4)
    assert snapshot.vjp(seed.flatten().tolist()) == (dx, dg)
    assert all(torch.isfinite(torch.tensor(snapshot.features)))


@pytest.mark.parametrize("raw", [-0.7, 0.8])
def test_causal_boundaries(raw):
    warp = st.EllipticWarp(1.0)
    x = inputs()
    gate = torch.tensor(raw, requires_grad=True)
    full = st.elliptic_gated_causal_autograd(warp, x, gate)
    changed = x.detach().clone()
    changed[0, 2:, 1:] += 3
    changed[1, :, 1:] -= 2
    assert torch.equal(full[0, :2], st.elliptic_gated_causal_autograd(warp, changed, gate)[0, :2])
    assert torch.equal(full[:1, :2], st.elliptic_gated_causal_autograd(warp, x[:1, :2], gate))
    full[0, 1].sum().backward()
    assert x.grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[0, 2:]) == 0
    assert torch.count_nonzero(x.grad[1]) == 0


def test_validation_and_saved_gate_version():
    warp = st.EllipticWarp(1.0)
    x = inputs()
    for gate in (0.0, torch.tensor(0.0, dtype=torch.float64)):
        with pytest.raises(TypeError, match="float32"):
            st.elliptic_gated_causal_autograd(warp, x, gate)
    with pytest.raises(ValueError, match="scalar"):
        st.elliptic_gated_causal_autograd(warp, x, torch.zeros(1))
    for raw in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            st.elliptic_gated_causal_autograd(warp, x, torch.tensor(raw))
        with pytest.raises(ValueError):
            st.EllipticGatedCausalResidualAdapter(3, raw_mix=raw)
    with pytest.raises(ValueError, match="budget"):
        st.elliptic_gated_causal_autograd(warp, x, torch.tensor(0.0), max_pairs=1)
    with pytest.raises(ValueError, match="nonempty"):
        st.elliptic_gated_causal_autograd(warp, x[0], torch.tensor(0.0))
    gate = torch.tensor(0.0, requires_grad=True)
    y = st.elliptic_gated_causal_autograd(warp, x, gate)
    with torch.no_grad():
        gate.add_(0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        y.sum().backward()


def test_adapter_initialization_pairing_and_checkpoint_contract():
    torch.manual_seed(103)
    local = st.EllipticResidualAdapter(4)
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(103)
    gated = st.EllipticGatedCausalResidualAdapter(4)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert torch.equal(local.orientation.weight, gated.orientation.weight)
    assert torch.equal(local.orientation.bias, gated.orientation.bias)
    assert gated.raw_mix.item() == 0
    state = copy.deepcopy(gated.state_dict())
    for other in (local, st.EllipticCausalResidualAdapter(4)):
        with pytest.raises(ValueError, match="incompatible"):
            other.load_state_dict(state)
        with pytest.raises(ValueError, match="incompatible"):
            gated.load_state_dict(other.state_dict())

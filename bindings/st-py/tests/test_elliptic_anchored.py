import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticAnchoredLearningBatch"), reason="native anchored batch required"
)


@pytest.mark.parametrize("shape", [(3,), (2, 4, 3), (2, 128, 3), (2, 0, 3)])
@pytest.mark.parametrize("raw", [-0.7, 0.0, 0.8])
def test_matched_torch_formula_and_both_vjps(shape, raw):
    generator = torch.Generator().manual_seed(101)
    warp = st.EllipticWarp(1.3, 3, 2)
    x = torch.randn(shape, generator=generator)
    x[..., 0] = 1
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    gate = torch.tensor(raw, requires_grad=True)
    ref_gate = gate.detach().clone().requires_grad_()
    actual = st.elliptic_anchored_autograd(warp, x, gate)
    local = st.elliptic_warp_autograd(warp, other).double()
    anchor = torch.tensor(warp.map_orientations_batch([1., 0., 0.]).features).double()
    mix = ref_gate.double().tanh()
    expected = ((1 - mix) * local + mix * anchor).float()
    seed = torch.randn(actual.shape, generator=generator)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x.grad, other.grad, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(gate.grad, ref_gate.grad, atol=2e-5, rtol=1e-4)
    if x.numel():
        assert gate.grad.abs() > 0
    else:
        assert gate.grad == 0


def test_zero_exactness_independent_rows_and_native_snapshot():
    warp = st.EllipticWarp(1.0, 3, 2)
    x = torch.tensor([[1., .2, .3], [1., -.4, .2]], requires_grad=True)
    other = x.detach().clone().requires_grad_()
    gate = torch.tensor(0.0, requires_grad=True)
    actual = st.elliptic_anchored_autograd(warp, x, gate)
    local = st.elliptic_warp_autograd(warp, other)
    assert torch.equal(actual, local)
    actual.sum().backward()
    local.sum().backward()
    assert torch.equal(x.grad, other.grad)
    snapshot = warp.map_anchored_batch(x.detach().flatten().tolist(), raw_mix=-0.5)
    seed = [1.] * 18
    gradient = snapshot.vjp(seed)
    duplicate = warp.map_anchored_batch(x.detach().flatten().tolist() * 2, raw_mix=-0.5)
    assert duplicate.vjp(seed * 2)[1] == pytest.approx(2 * gradient[1])
    first = snapshot.vjp([1.] * 9 + [0.] * 9)
    assert first[0][3:] == [0.] * 3 and first[1] != 0
    for row in range(2):
        one = warp.map_anchored_batch(x[row].detach().tolist(), raw_mix=-0.5)
        assert one.features == snapshot.features[row*9:row*9+9]
    snapshot.features[0] = float("nan")
    warp.configure(sheet_count=5, spin_harmonics=4)
    assert snapshot.vjp(seed) == gradient
    assert torch.isfinite(torch.tensor(snapshot.features)).all()


def test_validation_version_checks_and_no_causal_pair_budget():
    warp = st.EllipticWarp(1.0)
    x = torch.tensor([1., .2, .3], requires_grad=True)
    for gate in (0.0, torch.tensor(0.0, dtype=torch.float64)):
        with pytest.raises(TypeError, match="float32"):
            st.elliptic_anchored_autograd(warp, x, gate)
    with pytest.raises(ValueError, match="scalar"):
        st.elliptic_anchored_autograd(warp, x, torch.zeros(1))
    with pytest.raises(ValueError, match="budget"):
        st.elliptic_anchored_autograd(warp, x.expand(65537, 3), torch.tensor(0.0))
    with pytest.raises(ValueError, match="budget"):
        st.elliptic_anchored_autograd(warp, torch.zeros(2, 4), torch.tensor(0.0))
    for raw in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            st.elliptic_anchored_autograd(warp, x, torch.tensor(raw))
        with pytest.raises(ValueError):
            st.EllipticAnchoredResidualAdapter(3, raw_mix=raw)
    for parameter in ("gate", "orientation"):
        value = x.detach().clone().requires_grad_()
        gate = torch.tensor(0.0, requires_grad=True)
        output = st.elliptic_anchored_autograd(warp, value, gate)
        with torch.no_grad():
            (gate if parameter == "gate" else value).add_(0.1)
        with pytest.raises(RuntimeError, match="inplace"):
            output.sum().backward()
    # A pointwise path must not inherit the old quadratic attention budget.
    long = st.elliptic_anchored_autograd(warp, x.detach().expand(1, 1025, 3), torch.tensor(-0.5))
    assert long.shape == (1, 1025, 9)


def test_identity_initialization_and_distinct_checkpoint_schema():
    torch.manual_seed(103)
    local = st.EllipticResidualAdapter(4)
    rng = torch.get_rng_state()
    torch.manual_seed(103)
    anchored = st.EllipticAnchoredResidualAdapter(4)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(local.orientation.weight, anchored.orientation.weight)
    assert torch.equal(local.orientation.bias, anchored.orientation.bias)
    assert torch.equal(local.readout.weight, anchored.readout.weight)
    assert sum(p.numel() for p in anchored.parameters()) == 47
    assert anchored.raw_mix.item() == 0
    x = torch.randn(2, 3, 4)
    assert torch.equal(anchored(x), x)
    state = copy.deepcopy(anchored.state_dict())
    for other in (local, st.EllipticCausalResidualAdapter(4), st.EllipticGatedCausalResidualAdapter(4)):
        with pytest.raises((ValueError, RuntimeError)):
            other.load_state_dict(state)
        with pytest.raises((ValueError, RuntimeError)):
            anchored.load_state_dict(other.state_dict())


def test_real_hf_loss_updates_all_parameters_exact_adam_resume_and_cached_inference():
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    torch.manual_seed(109)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=16, n_embd=8, n_head=2, n_layer=1,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0,
    )).eval().requires_grad_(False)
    original = model.transformer.h[0].mlp
    base = {k: p.detach().clone() for k, p in model.named_parameters()}
    adapter = st.EllipticAnchoredResidualAdapter(8)
    model.transformer.h[0].mlp = torch.nn.Sequential(original, adapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    tokens = torch.arange(12).reshape(2, 6) % 32

    def update(module, opt):
        opt.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        loss.backward()
        assert torch.isfinite(loss)
        gradients = {k: p.grad.detach().clone() for k, p in module.named_parameters()}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        opt.step()
        return gradients

    first = update(adapter, optimizer)
    assert first["raw_mix"] == 0 and first["orientation.weight"].count_nonzero() == 0
    gradients = update(adapter, optimizer)
    assert all(g.abs().sum() > 0 for g in gradients.values())
    saved, adam = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
    update(adapter, optimizer)
    expected = copy.deepcopy(adapter.state_dict())
    expected_adam = copy.deepcopy(optimizer.state_dict())
    restored = st.EllipticAnchoredResidualAdapter(8)
    restored.load_state_dict(saved)
    resumed = torch.optim.Adam(restored.parameters(), lr=0.01)
    resumed.load_state_dict(adam)
    model.transformer.h[0].mlp = torch.nn.Sequential(original, restored)
    update(restored, resumed)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    assert equal(expected, restored.state_dict())
    assert equal(expected_adam, resumed.state_dict())
    model.eval()
    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        parts, past = [], None
        for i in range(tokens.shape[1]):
            output = model(tokens[:, i:i+1], past_key_values=past, use_cache=True)
            past = output.past_key_values
            parts.append(output.logits)
        torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-6, rtol=2e-5)
    model.transformer.h[0].mlp = original
    assert all(p.grad is None and torch.equal(base[k], p) for k, p in model.named_parameters())


def test_mps_is_explicit_host_transport_not_resident_execution():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    warp = st.EllipticWarp(1.0)
    cpu = torch.tensor([[1., .2, .3]], requires_grad=True)
    gpu = cpu.detach().to("mps").requires_grad_()
    raw_cpu = torch.tensor(-0.5, requires_grad=True)
    raw_gpu = raw_cpu.detach().to("mps").requires_grad_()
    a = st.elliptic_anchored_autograd(warp, cpu, raw_cpu)
    b = st.elliptic_anchored_autograd(warp, gpu, raw_gpu)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(a, b.cpu())
    assert torch.equal(cpu.grad, gpu.grad.cpu())
    assert torch.equal(raw_cpu.grad, raw_gpu.grad.cpu())

import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticCausalLearningBatch"),
    reason="native causal batch required",
)


def reference(warp, orientation):
    features = st.elliptic_warp_autograd(warp, orientation).double()
    scores = features @ features.transpose(-1, -2) / 3
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
    return (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ features).float()


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 4, 3)])
def test_rust_attention_and_tied_vjp_match_equivalent_torch(shape):
    torch.manual_seed(79)
    x = torch.randn(shape)
    x[..., 0] = 1
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    warp = st.EllipticWarp(1.3, 3, 2)
    actual = st.elliptic_causal_autograd(warp, x)
    expected = reference(warp, other)
    seed = torch.randn_like(actual)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-4, atol=2e-5)


def test_causal_prefix_future_and_batch_boundaries_in_both_directions():
    torch.manual_seed(83)
    warp = st.EllipticWarp(1.0)
    x = torch.randn(2, 4, 3)
    x[..., 0] = 1
    x.requires_grad_()
    full = st.elliptic_causal_autograd(warp, x)
    changed = x.detach().clone()
    changed[0, 2:, 1:] += 3
    changed[1, :, 1:] -= 2
    altered = st.elliptic_causal_autograd(warp, changed)
    assert torch.equal(full[0, :2], altered[0, :2])
    prefix = st.elliptic_causal_autograd(warp, x[:1, :2])
    assert torch.equal(prefix[0], full[0, :2])
    full[0, 1].sum().backward()
    assert x.grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[0, 2:]) == 0
    assert torch.count_nonzero(x.grad[1]) == 0


def test_shape_budget_and_saved_input_version_guards():
    warp = st.EllipticWarp(1.0)
    for x in (
        torch.ones(3),
        torch.ones(2, 3),
        torch.ones(0, 2, 3),
        torch.ones(2, 0, 3),
    ):
        with pytest.raises(ValueError, match="nonempty"):
            st.elliptic_causal_autograd(warp, x)
    for limit in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            st.elliptic_causal_autograd(warp, torch.ones(1, 2, 3), max_pairs=limit)
    with pytest.raises(ValueError, match="budget"):
        st.elliptic_causal_autograd(warp, torch.ones(1, 2, 3), max_pairs=3)
    with pytest.raises(TypeError, match="float32"):
        st.elliptic_causal_autograd(warp, torch.ones(1, 2, 3, dtype=torch.float64))
    x = torch.ones(1, 2, 3, requires_grad=True)
    y = st.elliptic_causal_autograd(warp, x)
    with torch.no_grad():
        x.add_(0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        y.sum().backward()


def test_causal_adapter_trains_both_projections_and_resumes_exactly():
    torch.manual_seed(89)
    adapter = st.EllipticCausalResidualAdapter(4)
    assert sum(p.numel() for p in adapter.parameters()) == 11 * 4 + 2
    x = torch.randn(2, 3, 4)
    assert torch.equal(adapter(x), x)
    target = x + 0.1 * x.cumsum(1)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)

    def step(layer, opt):
        opt.zero_grad()
        loss = (layer(x) - target).square().mean()
        loss.backward()
        opt.step()

    for _ in range(3):
        step(adapter, optimizer)
    assert adapter.orientation.weight.grad.abs().sum() > 0
    assert adapter.readout.weight.grad.abs().sum() > 0
    clone = st.EllipticCausalResidualAdapter(4)
    clone.load_state_dict(copy.deepcopy(adapter.state_dict()))
    resumed = torch.optim.Adam(clone.parameters(), lr=0.01)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(adapter, optimizer)
    step(clone, resumed)
    for a, b in zip(adapter.parameters(), clone.parameters()):
        assert torch.equal(a, b)
    for state, other in zip(optimizer.state.values(), resumed.state.values()):
        for key in state:
            assert torch.equal(state[key], other[key])
    pointwise = st.EllipticResidualAdapter(4)
    with pytest.raises(ValueError, match="incompatible"):
        pointwise.load_state_dict(adapter.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        clone.load_state_dict(pointwise.state_dict())


def test_actual_hf_loss_reaches_native_token_relations_without_base_updates():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(97)
    torch.set_num_threads(2)
    model = (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=16,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    original = model(ids).logits.detach()
    base = {name: p.detach().clone() for name, p in model.named_parameters()}
    adapter = st.EllipticCausalResidualAdapter(8)
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    assert torch.equal(model(ids).logits.detach(), original)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.01)
    for _ in range(4):
        opt.zero_grad()
        loss = model(ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert adapter.orientation.weight.grad.abs().sum() > 0
    assert adapter.readout.weight.grad.abs().sum() > 0
    assert not torch.equal(model(ids).logits.detach(), original)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            old_name = name.replace("mlp.0.", "mlp.")
            assert torch.equal(p, base[old_name]) and p.grad is None


def test_mps_transport_is_not_claimed_resident_execution():
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple GPU")
    warp = st.EllipticWarp(1.0)
    x = torch.tensor([[[1.0, 0.2, 0.3], [1.0, -0.4, 0.5]]], requires_grad=True)
    device_x = x.detach().to("mps").requires_grad_()
    expected = st.elliptic_causal_autograd(warp, x)
    actual = st.elliptic_causal_autograd(warp, device_x)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(device_x.grad.cpu(), x.grad)

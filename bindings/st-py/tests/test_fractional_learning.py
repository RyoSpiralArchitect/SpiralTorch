import copy
import json

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")


def oracle(value, alpha, *, axis, kernel_len, step):
    """Independent dense real GL polynomial; only a test oracle."""
    x, a = value.double().movedim(axis, 0), alpha.double()
    coefficients = [torch.ones_like(a)]
    for k in range(1, kernel_len):
        coefficients.append(coefficients[-1] * (k - 1 - a) / k)
    return torch.stack([
        sum(coefficients[k] * x[t-k] for k in range(min(t + 1, kernel_len)))
        * step ** (-a) for t in range(x.shape[0])
    ]).movedim(0, axis).float()


@pytest.mark.parametrize("alpha", [0.45, 1.0, 1.5])
def test_native_map_and_both_gradients_match_independent_torch(alpha):
    kernel = st.FractionalGlKernel(kernel_len=6, step=0.7)
    x = torch.linspace(-1, 1, 24).reshape(2, 4, 3).requires_grad_()
    a = torch.tensor(alpha, requires_grad=True)
    upstream = torch.cos(torch.arange(24).float()).reshape_as(x)
    y = st.fractional_gl_autograd(x, a, axis=1, kernel=kernel)
    reference = oracle(x, a, axis=1, kernel_len=6, step=0.7)
    torch.testing.assert_close(y, reference, rtol=2e-6, atol=2e-6)
    actual = torch.autograd.grad(y, (x, a), upstream)
    expected = torch.autograd.grad(reference, (x, a), upstream)
    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=2e-5, atol=2e-6)
    assert kernel.execution_backend == "rust_f32_cpu"


def test_native_jvp_uses_shared_order_derivative_and_matches_forward_ad():
    kernel = st.FractionalGlKernel(kernel_len=5, step=0.4)
    x = torch.linspace(-0.7, 0.8, 18).reshape(2, 3, 3)
    a, dx, da = torch.tensor(0.7), x.cos(), torch.tensor(0.2)
    with torch.autograd.forward_ad.dual_level():
        y = st.fractional_gl_autograd(
            torch.autograd.forward_ad.make_dual(x, dx),
            torch.autograd.forward_ad.make_dual(a, da), axis=1, kernel=kernel)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(y)
    eps = 1e-3
    finite_difference = (oracle(x+eps*dx, a+eps*da, axis=1, kernel_len=5, step=0.4)
                         - oracle(x-eps*dx, a-eps*da, axis=1, kernel_len=5, step=0.4)) / (2*eps)
    torch.testing.assert_close(tangent, finite_difference, rtol=2e-3, atol=3e-4)
    torch.testing.assert_close(primal, st.fractional_gl_autograd(x, a, axis=1, kernel=kernel))


def test_causal_prefix_batch_isolation_and_identity_initialization():
    adapter = st.FractionalMemoryAdapter(3, kernel_len=5)
    x = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    assert torch.equal(adapter(x), x)
    with torch.no_grad():
        adapter.gate.fill_(0.7)
    modified = x.clone()
    modified[0, 2:] = 88
    modified[1] = -99
    assert torch.equal(adapter(x)[0, :2], adapter(modified)[0, :2])
    x.requires_grad_()
    gradient = torch.autograd.grad(adapter(x)[0, 1, 0], x)[0]
    assert torch.count_nonzero(gradient[0, 2:]) == 0
    assert torch.count_nonzero(gradient[1]) == 0
    assert torch.count_nonzero(gradient[:, :, 1:]) == 0


def test_recipe_restore_and_invalid_inputs_are_explicit():
    adapter = st.FractionalMemoryAdapter(3, kernel_len=7, step=0.6)
    restored = st.FractionalMemoryAdapter(3, kernel_len=2)
    restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
    assert restored.get_extra_state() == adapter.get_extra_state()
    x = torch.ones(2, 4, 3)
    with pytest.raises(ValueError, match="incompatible"):
        restored.set_extra_state({**adapter.get_extra_state(), "features": 4})
    for alpha in [torch.tensor(0.), torch.tensor(float("nan")), torch.tensor(float("inf"))]:
        with pytest.raises(ValueError):
            st.fractional_gl_autograd(x, alpha, axis=1)
    with pytest.raises(TypeError):
        st.fractional_gl_autograd(x.double(), torch.tensor(0.5), axis=1)
    with pytest.raises(TypeError):
        st.fractional_gl_autograd(x, torch.tensor([0.5]), axis=1)
    with pytest.raises(ValueError):
        st.fractional_gl_autograd(x, torch.tensor(0.5), axis=True)
    with pytest.raises(TypeError, match="Rust FractionalGlKernel"):
        st.fractional_gl_autograd(x, torch.tensor(0.5), axis=1, kernel=object())
    with pytest.raises(ValueError, match="budget"):
        st.FractionalGlKernel(kernel_len=5, max_products=20).forward([1.] * 5, [5], 0, 0.5)
    assert json.loads(adapter._kernel.configuration_json())["kernel_len"] == 7


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_nd_axis_and_noncontiguous_transport(axis):
    x = torch.linspace(-0.8, 0.9, 24).reshape(2, 3, 4).transpose(0, 2).requires_grad_()
    assert not x.is_contiguous()
    a = torch.tensor(0.65, requires_grad=True)
    kernel = st.FractionalGlKernel(kernel_len=4, step=0.8)
    y = st.fractional_gl_autograd(x, a, axis=axis, kernel=kernel)
    expected = oracle(x, a, axis=axis, kernel_len=4, step=0.8)
    torch.testing.assert_close(y, expected, rtol=3e-6, atol=3e-6)
    actual_grad = torch.autograd.grad(y.sum(), (x, a))
    expected_grad = torch.autograd.grad(expected.sum(), (x, a))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def test_tiny_hf_trains_gate_and_order_with_exact_adam_resume_and_frozen_base():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(37)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=12, n_head=3, n_layer=1,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    base = [(p, p.clone()) for p in model.parameters()]
    tokens = torch.arange(12).reshape(2, 6) % 32
    with torch.no_grad():
        original_logits = model(tokens).logits.clone()
    parent = model.transformer.h[0]
    original = parent.mlp
    adapter = st.FractionalMemoryAdapter(12, kernel_len=5, step=0.8)
    initial_alpha = adapter.log_alpha.detach().clone()
    parent.mlp = torch.nn.Sequential(original, adapter)
    with torch.no_grad():
        assert torch.equal(model(tokens).logits, original_logits)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.02)

    def update(opt):
        opt.zero_grad()
        loss = model(tokens, labels=tokens).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    for _ in range(4):
        update(optimizer)
    assert torch.count_nonzero(adapter.gate) > 0
    assert adapter.log_alpha.grad.item() != 0
    assert not torch.equal(adapter.log_alpha.detach(), initial_alpha)
    parameters, moments = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
    update(optimizer)
    restored = st.FractionalMemoryAdapter(12, kernel_len=2)
    restored.load_state_dict(parameters)
    resumed = torch.optim.Adam(restored.parameters(), lr=0.02)
    resumed.load_state_dict(moments)
    parent.mlp = torch.nn.Sequential(original, restored)
    update(resumed)
    assert equal_tree(adapter.state_dict(), restored.state_dict())
    assert equal_tree(optimizer.state_dict(), resumed.state_dict())
    assert all(p.grad is None and torch.equal(p, before) for p, before in base)

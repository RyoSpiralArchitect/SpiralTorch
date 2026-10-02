import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")


def history_oracle(value, alpha, *, axis, kernel_len, step):
    x, a = value.double().movedim(axis, 0), alpha.double()
    coefficients = [torch.ones_like(a)]
    for k in range(1, kernel_len):
        coefficients.append(coefficients[-1] * (k - 1 - a) / k)
    return torch.stack([
        (x[t] * 0 + a * 0 + sum(coefficients[k] * x[t-k]
         for k in range(1, min(t + 1, kernel_len)))) * step ** (-a)
        for t in range(x.shape[0])
    ]).movedim(0, axis).float()


@pytest.mark.parametrize("order", [0.45, 1.0, 1.5])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_history_output_and_input_order_gradients_match_torch_polynomial(order, axis):
    x = torch.linspace(-.8, .9, 24).reshape(2, 3, 4).transpose(0, 2).requires_grad_()
    a = torch.tensor(order, requires_grad=True)
    kernel = st.FractionalGlKernel(kernel_len=5, step=.7)
    upstream = x.detach().cos()
    y = st.fractional_gl_history_autograd(x, a, axis=axis, kernel=kernel)
    expected = history_oracle(x, a, axis=axis, kernel_len=5, step=.7)
    torch.testing.assert_close(y, expected, rtol=3e-6, atol=3e-6)
    observed = torch.autograd.grad(y, (x, a), upstream)
    reference = torch.autograd.grad(expected, (x, a), upstream)
    for actual, wanted in zip(observed, reference):
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-6)


def test_history_forward_ad_matches_joint_tangent_and_has_no_current_dependency():
    x = torch.linspace(-.7, .8, 18).reshape(2, 3, 3)
    a, dx, da = torch.tensor(.7), x.cos(), torch.tensor(.2)
    kernel = st.FractionalGlKernel(kernel_len=5, step=.4)
    with torch.autograd.forward_ad.dual_level():
        y = st.fractional_gl_history_autograd(
            torch.autograd.forward_ad.make_dual(x, dx),
            torch.autograd.forward_ad.make_dual(a, da), axis=1, kernel=kernel)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(y)
    eps = 1e-3
    reference = (history_oracle(x+eps*dx, a+eps*da, axis=1, kernel_len=5, step=.4)
                 - history_oracle(x-eps*dx, a-eps*da, axis=1, kernel_len=5, step=.4)) / (2*eps)
    torch.testing.assert_close(tangent, reference, rtol=2e-3, atol=3e-4)
    changed = x.clone()
    changed[0, 1:] = 80
    changed[1] = -90
    assert torch.equal(primal[0, :2], st.fractional_gl_history_autograd(changed, a, axis=1, kernel=kernel)[0, :2])
    x.requires_grad_()
    y = st.fractional_gl_history_autograd(x, a, axis=1, kernel=kernel)
    gradient = torch.autograd.grad(y[0, 1, 0], x)[0]
    assert gradient[0, 0, 0] != 0
    assert torch.count_nonzero(gradient) == 1
    assert torch.count_nonzero(primal[:, 0]) == 0


def test_native_history_does_not_subtract_large_rounded_current_values():
    kernel = st.FractionalGlKernel(kernel_len=2, step=.7)
    x = [1., torch.finfo(torch.float32).max]
    with pytest.raises(ValueError):
        kernel.forward(x, [2], 0, .5)
    history = kernel.forward_history(x, [2], 0, .5)
    assert history.output[0] == 0
    assert history.output[1] == pytest.approx(-.5 * .7**(-.5), rel=2e-6)
    dx, da = history.vjp([0., 1.])
    assert dx[1] == 0 and da != 0
    with pytest.raises(ValueError):
        kernel.forward_history([float("nan")], [1], 0, .5)


@pytest.mark.parametrize("shape,kernel_len", [((2, 3, 4), 1), ((2, 1, 4), 8)])
def test_empty_history_has_zero_input_and_order_derivatives(shape, kernel_len):
    x = torch.ones(shape, requires_grad=True)
    a = torch.tensor(.6, requires_grad=True)
    y = st.fractional_gl_history_autograd(
        x, a, axis=1, kernel=st.FractionalGlKernel(kernel_len=kernel_len, step=.7))
    assert torch.count_nonzero(y) == 0
    gx, ga = torch.autograd.grad(y.sum(), (x, a))
    assert torch.count_nonzero(gx) == 0 and ga == 0


def test_independent_local_gate_recovers_pointwise_and_history_is_strictly_past():
    adapter = st.FractionalHistoryAdapter(3, kernel_len=5)
    x = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    assert sum(p.numel() for p in adapter.parameters()) == 7
    assert torch.equal(adapter(x), x)
    with torch.no_grad():
        adapter.local_gate.copy_(torch.tensor([.2, -.3, .4]))
    assert torch.equal(adapter(x), x + .1 * adapter.local_gate.tanh() * x)
    adapter(x).square().sum().backward()
    assert adapter.log_alpha.grad == 0
    assert torch.count_nonzero(adapter.gate.grad) > 0
    assert torch.count_nonzero(adapter.local_gate.grad) > 0
    with torch.no_grad():
        adapter.gate.fill_(.3)
    expected = (x + .1 * adapter.local_gate.tanh() * x
                + .1 * adapter.gate.tanh() * history_oracle(
                    x, adapter.log_alpha.exp(), axis=1, kernel_len=5, step=1.))
    torch.testing.assert_close(adapter(x), expected, rtol=2e-6, atol=2e-6)
    changed = x.clone()
    changed[0, 2:] = 90
    changed[1] = -80
    assert torch.equal(adapter(x)[0, :2], adapter(changed)[0, :2])


def test_history_recipe_cannot_silently_load_as_full_gl_and_preserves_controls():
    adapter = st.FractionalHistoryAdapter(3, kernel_len=7, step=.6, strength=.2)
    with torch.no_grad():
        adapter.local_gate.fill_(.4)
        adapter.gate.fill_(-.3)
    state = copy.deepcopy(adapter.state_dict())
    restored = st.FractionalHistoryAdapter(3, kernel_len=2)
    restored.load_state_dict(state)
    assert restored.get_extra_state() == adapter.get_extra_state()
    x = torch.ones(2, 4, 3)
    assert torch.equal(adapter(x), restored(x))
    assert state["_extra_state"]["schema"] == "spiraltorch.fractional_history_adapter.v1"
    full = st.FractionalMemoryAdapter(3)
    with pytest.raises(ValueError, match="incompatible"):
        restored.load_state_dict(full.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        full.load_state_dict(state)
    restored.local_gate = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    with pytest.raises(TypeError, match="float32"):
        restored(x)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def test_tiny_hf_learns_both_gates_and_order_with_exact_adam_resume():
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
    parent, original = model.transformer.h[0], model.transformer.h[0].mlp
    adapter = st.FractionalHistoryAdapter(12, kernel_len=5, step=.8)
    initial_alpha = adapter.log_alpha.detach().clone()
    parent.mlp = torch.nn.Sequential(original, adapter)
    with torch.no_grad():
        assert torch.equal(model(tokens).logits, original_logits)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.02)

    def update(opt):
        opt.zero_grad()
        loss = model(tokens, labels=tokens).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    for _ in range(4):
        update(optimizer)
    for parameter in (adapter.gate, adapter.local_gate):
        assert torch.count_nonzero(parameter) > 0
        assert torch.count_nonzero(parameter.grad) > 0
    assert adapter.log_alpha.grad.item() != 0
    assert not torch.equal(adapter.log_alpha.detach(), initial_alpha)
    parameters, moments = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
    update(optimizer)
    restored = st.FractionalHistoryAdapter(12, kernel_len=2)
    restored.load_state_dict(parameters)
    resumed = torch.optim.Adam(restored.parameters(), lr=.02)
    resumed.load_state_dict(moments)
    parent.mlp = torch.nn.Sequential(original, restored)
    update(resumed)
    assert equal_tree(adapter.state_dict(), restored.state_dict())
    assert equal_tree(optimizer.state_dict(), resumed.state_dict())
    assert all(p.grad is None and torch.equal(p, before) for p, before in base)

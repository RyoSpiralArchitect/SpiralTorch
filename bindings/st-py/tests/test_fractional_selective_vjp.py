import copy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
import spiraltorch.fractional_autograd as bridge


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("need_input,need_alpha", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("buffers", [False, True])
def test_autograd_requests_only_needed_native_components(history, axis, need_input, need_alpha, buffers, monkeypatch):
    if buffers and not bridge._buffer_transport_available():
        pytest.skip("Torch/NumPy buffer interop is not installed")
    monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: buffers)
    kernel = st.FractionalGlKernel(kernel_len=5, step=.7)
    x = torch.linspace(-.8, .9, 24).reshape(2, 3, 4).transpose(0, 2).requires_grad_(need_input)
    alpha = torch.tensor(.6, requires_grad=need_alpha)
    operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
    y = operation(x, alpha, axis=axis, kernel=kernel)
    snapshot = y.grad_fn.snapshot
    upstream = x.detach().cos()
    expected = snapshot.vjp(upstream.reshape(-1).tolist())
    method = "vjp" if need_input and need_alpha else ("vjp_input" if need_input else "vjp_alpha")
    method += "_buffer" if buffers else ""
    calls = []

    class ObservedSnapshot:
        def __getattr__(self, name):
            assert name == method, f"computed unrequested component via {name}"

            def invoke(values):
                calls.append(name)
                return getattr(snapshot, name)(values)

            return invoke

    y.grad_fn.snapshot = ObservedSnapshot()
    arguments = tuple(v for v, needed in ((x, need_input), (alpha, need_alpha)) if needed)
    actual = torch.autograd.grad(y, arguments, upstream)
    wanted = tuple(v for v, needed in (
        (torch.tensor(expected[0]).reshape_as(x), need_input),
        (torch.tensor(expected[1]), need_alpha)) if needed)
    assert calls == [method]
    assert all(torch.equal(a, b) for a, b in zip(actual, wanted))


@pytest.mark.parametrize("history", [False, True])
def test_selected_gradients_do_not_compute_overflowing_unrequested_component(history):
    kernel = st.FractionalGlKernel(kernel_len=2)
    operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
    maximum = torch.finfo(torch.float32).max
    upstream = torch.tensor([0., maximum])
    x = torch.zeros(2)
    alpha = torch.tensor(2., requires_grad=True)
    y = operation(x, alpha, axis=0, kernel=kernel)
    with pytest.raises(ValueError):
        y.grad_fn.snapshot.vjp(upstream.tolist())
    assert torch.autograd.grad(y, alpha, upstream)[0] == 0
    x = torch.tensor([maximum * .5, 0.], requires_grad=True)
    y = operation(x, torch.tensor(1.), axis=0, kernel=kernel)
    with pytest.raises(ValueError):
        y.grad_fn.snapshot.vjp(upstream.tolist())
    assert torch.equal(torch.autograd.grad(y, x, upstream)[0],
                       torch.tensor([-maximum, 0. if history else maximum]))


@pytest.mark.parametrize("kernel_len,shape", [(1, (2, 3)), (8, (6, 1))])
@pytest.mark.parametrize("need_input", [True, False])
def test_selective_empty_history_preserves_zero_map_and_direction_validation(kernel_len, shape, need_input):
    kernel = st.FractionalGlKernel(kernel_len=kernel_len, step=.1)
    x = torch.ones(shape, requires_grad=need_input)
    alpha = torch.tensor(100., requires_grad=not need_input)
    y = st.fractional_gl_history_autograd(x, alpha, axis=1, kernel=kernel)
    snapshot = y.grad_fn.snapshot
    for method in (snapshot.vjp_input, snapshot.vjp_alpha):
        for bad in ([], [1.] * 5, [float("nan")] * 6, [float("inf")] * 6):
            with pytest.raises(ValueError):
                method(bad)
    actual = torch.autograd.grad(y.sum(), x if need_input else alpha)[0]
    assert torch.count_nonzero(y) == torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("history", [False, True])
def test_no_reverse_differentials_are_created_when_neither_input_requires_grad(history):
    operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
    y = operation(torch.ones(2, 4, 3), torch.tensor(.5), axis=1)
    assert not y.requires_grad and y.grad_fn is None


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


@pytest.mark.parametrize("history", [False, True])
def test_hf_frozen_base_selective_and_joint_routes_have_exact_loss_gradients_and_adam(history, monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(149)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=12, n_head=3, n_layer=1,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    base = [p.clone() for p in model.parameters()]
    parent, original = model.transformer.h[0], model.transformer.h[0].mlp
    adapter_type = st.FractionalHistoryAdapter if history else st.FractionalMemoryAdapter
    tokens = torch.arange(48).reshape(8, 6) % 32

    def run():
        adapter = adapter_type(12, kernel_len=5, step=.8)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=.02)
        records = []
        parent.mlp = torch.nn.Sequential(original, adapter)
        try:
            for i in range(8):
                optimizer.zero_grad()
                batch = tokens[[i, (i + 1) % len(tokens)]]
                loss = model(batch, labels=batch).loss
                loss.backward()
                records.append({"loss": float(loss.detach()),
                                "gradients": {n: p.grad.clone() for n, p in adapter.named_parameters()}})
                optimizer.step()
        finally:
            parent.mlp = original
        return records, copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())

    selected = run()

    def old_joint_backward(ctx, upstream):
        value, alpha = ctx.saved_tensors
        dx, da = ctx.snapshot.vjp(bridge._values(upstream))
        return (None, torch.tensor(dx, dtype=value.dtype, device=value.device).reshape_as(value),
                torch.tensor(da, dtype=alpha.dtype, device=alpha.device), None, None)

    with monkeypatch.context() as context:
        context.setattr(bridge._FractionalGlFunction, "backward", staticmethod(old_joint_backward))
        joint = run()
    assert equal_tree(selected, joint)
    assert all(p.grad is None and torch.equal(p, saved) for p, saved in zip(model.parameters(), base))
    assert all(torch.isfinite(g).all() for r in selected[0] for g in r["gradients"].values())
    assert selected[0][0]["gradients"]["log_alpha"] == 0
    assert any(r["gradients"]["log_alpha"] != 0 for r in selected[0][1:])

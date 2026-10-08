import copy
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticLearningBatch"), reason="native batch required"
)


def test_batched_rust_vjp_matches_torch_and_finite_difference():
    warp = st.EllipticWarp(1.5, 4, 2)
    x = torch.tensor([[0.3, 0.4, 0.8], [0.6, -0.3, -0.5]], requires_grad=True)
    seed = torch.linspace(-0.4, 0.7, 18).reshape(2, 9)
    features, telemetry = st.elliptic_warp_autograd(warp, x, return_telemetry=True)
    (features * seed).sum().backward()
    batch = warp.map_orientations_batch(x.detach().flatten().tolist())
    assert features.shape == (2, 9) and len(telemetry) == 2
    expected = torch.tensor(batch.vjp(seed.flatten().tolist())).reshape_as(x)
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    for i in range(x.numel()):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus.flatten()[i] += 1e-3
        minus.flatten()[i] -= 1e-3
        high = (st.elliptic_warp_autograd(warp, plus).double() * seed.double()).sum()
        low = (st.elliptic_warp_autograd(warp, minus).double() * seed.double()).sum()
        numeric = (high - low) / (plus.flatten()[i] - minus.flatten()[i])
        assert float(x.grad.flatten()[i]) == pytest.approx(
            float(numeric), rel=2e-3, abs=1e-4
        )


def test_pole_regression_large_inputs_empty_and_invalid_chart():
    warp = st.EllipticWarp(1.0, 4, 2)
    near = torch.tensor([[1e-4, 2e-4, 1.0]], requires_grad=True)
    features = st.elliptic_warp_autograd(warp, near)
    assert features[0, 0].item() == pytest.approx(0.0002236068, rel=1e-5)
    assert features[0, 6].item() == pytest.approx(-0.0002, rel=1e-5)
    features.sum().backward()
    assert torch.isfinite(near.grad).all()
    large = torch.tensor([[1e20, 2e20, 3e20]], requires_grad=True)
    st.elliptic_warp_autograd(warp, large).sum().backward()
    assert torch.isfinite(large.grad).all() and large.grad.abs().sum() > 0
    empty = torch.empty(2, 0, 3, requires_grad=True)
    y, telemetry = st.elliptic_warp_autograd(warp, empty, return_telemetry=True)
    assert y.shape == (2, 0, 9) and telemetry == [[], []]
    y.sum().backward()
    assert empty.grad.shape == empty.shape
    for bad in (
        [0, 0, 0],
        [0, 0, 1],
        [-1, 0, 0.3],
        [float("nan"), 1, 1],
        [1e-30, 0, 1e20],
    ):
        with pytest.raises(ValueError, match="chart"):
            st.elliptic_warp_autograd(warp, torch.tensor(bad, dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        st.elliptic_warp_autograd(warp, torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="budget"):
        st.elliptic_warp_autograd(warp, torch.ones(65537, 3))


def test_snapshot_recipe_and_saved_input_version_are_bound():
    warp = st.EllipticWarp(1.0)
    x = torch.tensor([0.3, 0.4, 0.8], requires_grad=True)
    expected = warp.map_orientations_batch(x.detach().tolist()).vjp([1.0] * 9)
    y = st.elliptic_warp_autograd(warp, x)
    warp.configure(spin_harmonics=4)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor(expected), rtol=0, atol=0)
    y = st.elliptic_warp_autograd(warp, x)
    with torch.no_grad():
        x.add_(0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        y.sum().backward()
    with pytest.raises(ValueError):
        warp.configure(sheet_count=0, spin_harmonics=3)
    assert warp.spin_harmonics == 4


def test_telemetry_is_local_to_the_calling_context():
    from spiraltorch.elliptic import EllipticWarpFunction

    barrier = Barrier(2)

    def worker(rows):
        st.elliptic_warp_autograd(st.EllipticWarp(1.0), torch.ones(rows, 3))
        barrier.wait(timeout=10)
        return len(EllipticWarpFunction.last_telemetry())

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(worker, [1, 3])) == [1, 3]


@pytest.fixture
def telemetry_calls(monkeypatch):
    import spiraltorch.elliptic as elliptic

    calls = []

    class TrackedBatch:
        def __init__(self, batch):
            self.batch = batch

        @property
        def features(self):
            return self.batch.features

        def vjp(self, upstream):
            return self.batch.vjp(upstream)

        def telemetry(self):
            calls.append(1)
            return self.batch.telemetry()

    class TrackedWarp:
        def __init__(self, warp):
            self.warp = warp

        def map_orientations_batch(self, values):
            return TrackedBatch(self.warp.map_orientations_batch(values))

    monkeypatch.setattr(elliptic, "_EllipticWarp", TrackedWarp)
    token = elliptic._LAST_TELEMETRY.set(None)
    try:
        yield calls, TrackedWarp
    finally:
        elliptic._LAST_TELEMETRY.reset(token)


@pytest.mark.parametrize("shape", [(3,), (2, 3, 3), (2, 0, 3)])
def test_telemetry_is_lazy_cached_and_bound_to_forward(telemetry_calls, shape):
    from spiraltorch.elliptic import EllipticWarpFunction

    calls, tracked = telemetry_calls
    warp = st.EllipticWarp(1.0)
    x = torch.ones(shape, requires_grad=True)
    expected = warp.map_orientations_batch(x.detach().flatten().tolist())
    features = EllipticWarpFunction.apply(tracked(warp), x)
    warp.configure(spin_harmonics=4)
    features.sum().backward()
    assert calls == []
    torch.testing.assert_close(
        x.grad,
        torch.tensor(expected.vjp([1.0] * features.numel())).reshape_as(x),
        rtol=0,
        atol=0,
    )

    telemetry = EllipticWarpFunction.last_telemetry()
    again = EllipticWarpFunction.last_telemetry()
    assert calls == [1]

    def leaves(value):
        if isinstance(value, list):
            return [item for child in value for item in leaves(child)]
        return [value]

    assert all(a is b for a, b in zip(leaves(telemetry), leaves(again)))
    assert [t.as_dict() for t in leaves(telemetry)] == [
        t.as_dict() for t in expected.telemetry()
    ]
    as_dict = EllipticWarpFunction.last_telemetry(as_dict=True)
    assert leaves(as_dict) == [t.as_dict() for t in expected.telemetry()]
    assert calls == [1]
    if shape == (2, 0, 3):
        assert telemetry == [[], []]

    _, requested = st.elliptic_warp_autograd(
        tracked(warp), x.detach(), return_telemetry=True
    )
    assert calls == [1, 1]
    assert leaves(requested) == leaves(EllipticWarpFunction.last_telemetry())


def test_telemetry_requests_do_not_change_training_or_adam_state(telemetry_calls):
    from spiraltorch.elliptic import EllipticWarpFunction

    calls, tracked = telemetry_calls
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(317)
        initial = st.EllipticResidualAdapter(4)
        x = torch.randn(2, 3, 4)
    target = x + 0.1 * x.sin()

    def train(request_telemetry):
        adapter = st.EllipticResidualAdapter(4)
        adapter.load_state_dict(copy.deepcopy(initial.state_dict()))
        adapter._warp = tracked(adapter._warp)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
        trace = []
        for _ in range(4):
            optimizer.zero_grad()
            prediction = adapter(x)
            if request_telemetry:
                assert len(EllipticWarpFunction.last_telemetry()) == 2
            loss = (prediction - target).square().mean()
            loss.backward()
            gradients = [p.grad.clone() for p in adapter.parameters()]
            optimizer.step()
            trace.append((prediction.detach(), loss.detach(), gradients))
        return trace, list(adapter.parameters()), optimizer.state_dict()

    lazy = train(False)
    assert calls == []
    eager = train(True)
    assert calls == [1] * 4

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for first, second in zip(a, b):
                exact(first, second)
        else:
            assert a == b

    exact(lazy, eager)


def test_lazy_telemetry_releases_replaced_or_materialized_batch(telemetry_calls):
    import spiraltorch.elliptic as elliptic

    _, tracked = telemetry_calls
    warp = tracked(st.EllipticWarp(1.0))
    st.elliptic_warp_autograd(warp, torch.ones(3))
    first = weakref.ref(elliptic._LAST_TELEMETRY.get()[1]._batch)
    assert first() is not None
    st.elliptic_warp_autograd(warp, torch.ones(3))
    assert first() is None
    second = weakref.ref(elliptic._LAST_TELEMETRY.get()[1]._batch)
    assert second() is not None
    elliptic.EllipticWarpFunction.last_telemetry()
    assert second() is None


def test_residual_identity_updates_both_projections_and_exact_resume():
    torch.manual_seed(41)
    adapter = st.EllipticResidualAdapter(4, strength=0.3, spin_harmonics=2)
    x = torch.randn(3, 2, 4)
    target = x + 0.1 * torch.sin(x)
    assert torch.equal(adapter(x), x)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)

    def step(layer, opt):
        opt.zero_grad()
        loss = (layer(x) - target).square().mean()
        loss.backward()
        opt.step()

    step(adapter, optimizer)
    step(adapter, optimizer)
    assert adapter.orientation.weight.grad.abs().sum() > 0
    assert adapter.readout.weight.grad.abs().sum() > 0
    clone = st.EllipticResidualAdapter(4)
    clone.load_state_dict(copy.deepcopy(adapter.state_dict()))
    clone_opt = torch.optim.Adam(clone.parameters(), lr=0.01)
    clone_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(adapter, optimizer)
    step(clone, clone_opt)
    for a, b in zip(adapter.parameters(), clone.parameters()):
        assert torch.equal(a, b)
    changed = x.clone()
    changed[:, 1:] += 10
    assert torch.equal(adapter(x)[:, :1], adapter(changed)[:, :1])
    adapter.strength = 0.0
    assert adapter(x) is x


def test_hf_loss_reaches_orientation_and_readout_with_frozen_base():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
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
            )
        )
        .eval()
        .requires_grad_(False)
    )
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    baseline = model(ids).logits.detach()
    adapter = st.EllipticResidualAdapter(8)
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    assert torch.equal(model(ids).logits.detach(), baseline)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.01)
    for _ in range(4):
        opt.zero_grad()
        loss = model(ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert adapter.orientation.weight.grad.abs().sum() > 0
    assert adapter.readout.weight.grad.abs().sum() > 0
    assert not torch.equal(model(ids).logits.detach(), baseline)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_accelerator_transport_with_noncontiguous_orientations():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS transport requires Apple GPU")
    warp = st.EllipticWarp(1.0)
    x = torch.tensor([[1.0, 0.3], [0.2, 0.5], [0.3, 0.8]]).T.requires_grad_()
    assert not x.is_contiguous()
    device_x = x.detach().to("mps").requires_grad_()
    expected = st.elliptic_warp_autograd(warp, x)
    actual = st.elliptic_warp_autograd(warp, device_x)
    assert actual.device.type == "mps"
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(device_x.grad.cpu(), x.grad)

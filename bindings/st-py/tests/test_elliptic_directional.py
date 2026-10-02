"""Native first-order directions, not a Python reconstruction of the map."""

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
fw = torch.autograd.forward_ad


@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("shape", [(3,), (2, 3, 3), (2, 0, 3)])
def test_forward_ad_matches_native_and_preserves_reverse_mode(anchored, shape):
    warp = st.EllipticWarp(1.3, 3, 2)
    x = torch.full(shape, 0.3, requires_grad=True)
    dx = torch.linspace(-0.3, 0.5, x.numel()).reshape(shape)
    gate = torch.tensor(-0.4, requires_grad=True)
    dg = torch.tensor(0.7)
    values = x.detach().flatten().tolist()
    snapshot = (
        warp.map_anchored_batch(values, raw_mix=gate.item())
        if anchored
        else warp.map_orientations_batch(values)
    )
    native = (
        snapshot.jvp(dx.flatten().tolist(), dg.item())
        if anchored
        else snapshot.jvp(dx.flatten().tolist())
    )
    with fw.dual_level():
        dual = fw.make_dual(x, dx)
        output = (
            st.elliptic_anchored_autograd(warp, dual, fw.make_dual(gate, dg))
            if anchored
            else st.elliptic_warp_autograd(warp, dual)
        )
        primal, tangent = fw.unpack_dual(output)
        torch.testing.assert_close(
            tangent, torch.tensor(native).reshape(*shape[:-1], 9), rtol=0, atol=0
        )
        primal.sum().backward()
    gradients = snapshot.vjp([1.0] * primal.numel())
    torch.testing.assert_close(
        x.grad,
        torch.tensor(gradients[0] if anchored else gradients).reshape_as(x),
        rtol=0,
        atol=0,
    )
    if anchored:
        assert gate.grad.item() == gradients[1]


@pytest.mark.parametrize("direction", ["input", "gate"])
def test_joint_jvp_accepts_one_active_input(direction):
    warp = st.EllipticWarp(1.0, 2, 2)
    x, g = torch.tensor([1.0, 0.2, 0.3]), torch.tensor(0.4)
    dx, dg = torch.tensor([0.1, -0.2, 0.4]), torch.tensor(-0.5)
    snapshot = warp.map_anchored_batch(x.tolist(), raw_mix=g.item())
    with fw.dual_level():
        y = st.elliptic_anchored_autograd(
            warp,
            fw.make_dual(x, dx) if direction == "input" else x,
            fw.make_dual(g, dg) if direction == "gate" else g,
        )
        _, actual = fw.unpack_dual(y)
        expected = snapshot.jvp(
            dx.tolist() if direction == "input" else [0.0] * 3,
            dg.item() if direction == "gate" else 0.0,
        )
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)


def test_native_validation_and_snapshot_isolation():
    warp = st.EllipticWarp(1.0, 2, 2)
    snapshot = warp.map_orientations_batch([1.0, 0.2, 0.3])
    expected = snapshot.jvp([0.0, 0.3, -0.2])
    warp.configure(spin_harmonics=5)
    assert snapshot.jvp([0.0, 0.3, -0.2]) == expected
    for invalid in ([], [0.0] * 2, [0.0] * 6, [float("nan")] * 3, [float("inf")] * 3):
        with pytest.raises(ValueError, match="JVP"):
            snapshot.jvp(invalid)
    anchored = warp.map_anchored_batch([1.0, 0.2, 0.3], raw_mix=30.0)
    with pytest.raises(ValueError, match="JVP"):
        anchored.jvp([0.0] * 3, float("nan"))


@pytest.mark.parametrize("anchored", [False, True])
def test_residual_adapter_loss_direction_matches_backward(anchored):
    torch.manual_seed(41)
    kind = (
        st.EllipticAnchoredResidualAdapter if anchored else st.EllipticResidualAdapter
    )
    adapter = kind(8, strength=0.3, spin_harmonics=2)
    x = torch.randn(2, 5, 8, requires_grad=True)
    target = torch.arange(10).reshape(2, 5) % 8
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(
            adapter(x).flatten(0, 1), target.flatten()
        ).backward()
        optimizer.step()
    x.grad = None
    loss = torch.nn.functional.cross_entropy(adapter(x).flatten(0, 1), target.flatten())
    (gradient,) = torch.autograd.grad(loss, x)
    direction = torch.randn_like(x)
    with fw.dual_level():
        output = adapter(fw.make_dual(x.detach(), direction))
        primal, tangent = fw.unpack_dual(
            torch.nn.functional.cross_entropy(output.flatten(0, 1), target.flatten())
        )
    torch.testing.assert_close(primal, loss, rtol=0, atol=0)
    torch.testing.assert_close(
        tangent, (gradient * direction).sum(), rtol=3e-5, atol=1e-6
    )


def test_pullback_fit_example_decreases_loss():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples" / "elliptic_pullback_fit.py"
    spec = importlib.util.spec_from_file_location("elliptic_pullback_fit", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    report = example.run()
    assert report["accepted_updates"] > 0
    assert report["final_loss"] < report["initial_loss"] * 1e-4
    assert report["max_linear_system_residual"] < 1e-4


def test_chart_probe_metrics_and_degenerate_rejection(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples" / "hf_elliptic_chart_probe.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("chart_probe_test", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    features = torch.zeros(4, 9)
    features[:, :2] = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])
    jacobian = torch.zeros(4, 9, 2)
    jacobian[:, 0, 0], jacobian[:, 1, 1] = 2, 1
    metrics = probe.metrics(features, jacobian, torch.zeros(9))
    assert metrics["feature_covariance_effective_rank"] == pytest.approx(2.0)
    assert metrics["jacobian_condition"]["quantiles"]["p50"] == 2.0
    assert metrics["anchor_displacement_l2"]["mean"] == 1.0
    for values in (
        torch.empty(0),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ):
        with pytest.raises(ValueError, match="metric"):
            probe.distribution(values)
    with pytest.raises(ValueError, match="degenerate"):
        probe.metrics(features, torch.zeros_like(jacobian), torch.zeros(9))

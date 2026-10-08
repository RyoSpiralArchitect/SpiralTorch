import copy
import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st


@pytest.fixture
def client(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/hf_fractional_gain_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("angle_chart_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("angle", [-.4, -.24, 0., .4636476, 1.2])
def test_native_scalar_chart_and_first_order_ad(angle, monkeypatch):
    chart = st.FractionalGlAngleChart(angle)
    parameter = torch.tensor(angle, requires_grad=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Torch must not reconstruct the angle chart")
    monkeypatch.setattr(torch.Tensor, "tan", forbidden)
    observed = st.fractional_gl_angle_autograd(parameter)
    assert observed.item() == chart.alpha
    assert chart.alpha == pytest.approx(1 + 2*math.tan(float(parameter.detach())), rel=2e-7)
    observed.backward(torch.tensor(.3))
    assert parameter.grad.item() == chart.vjp(.3)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(parameter.detach(), torch.tensor(-.2))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(st.fractional_gl_angle_autograd(dual))
        assert primal.item() == chart.alpha and tangent.item() == chart.jvp(-.2)
    with pytest.raises((AttributeError, TypeError)):
        chart.alpha = 2.


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), -float("inf"), -.5, 1.6, 6.3])
def test_invalid_angles_do_not_wrap_or_clip(angle):
    with pytest.raises(ValueError, match="angle"):
        st.FractionalGlAngleChart(angle)
    with pytest.raises(ValueError, match="angle"):
        st.fractional_gl_angle_autograd(torch.tensor(angle))


@pytest.mark.parametrize("value", [.2, None, torch.tensor([.2]), torch.tensor(.2, dtype=torch.float64)])
def test_angle_transport_requires_scalar_f32(value):
    with pytest.raises(TypeError, match="scalar float32"):
        st.fractional_gl_angle_autograd(value)


def test_scalar_overflow_mutation_and_second_order_are_not_silently_accepted():
    chart = st.FractionalGlAngleChart(0.)
    for bad in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
        with pytest.raises(ValueError): chart.vjp(bad)
        with pytest.raises(ValueError): chart.jvp(bad)
    angle = torch.tensor(.2, requires_grad=True)
    value = st.fractional_gl_angle_autograd(angle)
    with torch.no_grad(): angle.add_(.1)
    with pytest.raises(RuntimeError): value.backward()
    first, = torch.autograd.grad(st.fractional_gl_angle_autograd(angle), angle, create_graph=True)
    with pytest.raises(RuntimeError): torch.autograd.grad(first, angle)


@pytest.mark.parametrize("angle", [-.4, -.24, 0., .4636476, 1.2])
@pytest.mark.parametrize("sequence_transport", [False, True])
def test_nonzero_gate_input_and_all_parameter_gradients_match_ordinary_short(client, angle, sequence_transport, monkeypatch):
    if sequence_transport:
        import spiraltorch.fractional_autograd as bridge
        monkeypatch.setattr(bridge, "_buffer_transport_available", lambda: False)
    native = st.FractionalAngleGainHistoryAdapter(3, initial_angle=angle, initial_gain=1.7, kernel_len=3)
    ordinary = client.OrdinaryGainShort(3, strength=.1, kernel_len=3, step=1., max_values=1024, max_products=3072)
    with torch.no_grad():
        for model in (native, ordinary):
            model.history_angle.fill_(angle)
            model.log_gain.fill_(math.log(1.7))
            model.gate.copy_(torch.tensor([-.3, .2, .6]))
            model.local_gate.copy_(torch.tensor([.1, -.2, .4]))
    names = ["gate", "local_gate", "history_angle", "log_gain"]
    assert list(dict(native.named_parameters())) == list(dict(ordinary.named_parameters())) == names
    output, gradients = [], []
    for model in (native, ordinary):
        x = torch.linspace(-.7, .9, 48).reshape(2, 8, 3).requires_grad_()
        y = model(x)
        output.append(y)
        gradients.append(torch.autograd.grad(y, (x, *model.parameters()), x.cos()))
    assert torch.allclose(*output, atol=3e-7, rtol=3e-6)
    assert all(torch.allclose(a, b, atol=3e-7, rtol=3e-6) for a, b in zip(*gradients))


def test_joint_input_shape_gain_jvp_matches_vjp_and_finite_difference():
    kernel = st.FractionalGlKernel(kernel_len=8)
    x = torch.linspace(-.7, .8, 48).reshape(2, 8, 3).requires_grad_()
    angle, gain = torch.tensor(-.24, requires_grad=True), torch.tensor(.3, requires_grad=True)
    values, directions = (x, angle, gain), (x.detach().cos(), torch.tensor(.2), torch.tensor(-.3))
    def apply(a, b, c):
        return st.fractional_gl_history_log_gain_autograd(a, st.fractional_gl_angle_autograd(b), c,
                                                        axis=1, kernel=kernel)
    output = apply(*values)
    upstream = x.detach().sin()
    gradients = torch.autograd.grad(output, values, upstream)
    with torch.autograd.forward_ad.dual_level():
        duals = [torch.autograd.forward_ad.make_dual(value.detach(), direction)
                 for value, direction in zip(values, directions)]
        _, tangent = torch.autograd.forward_ad.unpack_dual(apply(*duals))
    assert float((upstream*tangent).sum()) == pytest.approx(
        sum(float((gradient*direction).sum()) for gradient, direction in zip(gradients, directions)), abs=3e-6)
    with torch.no_grad():
        plus = apply(*(v+.001*d for v, d in zip(values, directions)))
        minus = apply(*(v-.001*d for v, d in zip(values, directions)))
    assert torch.allclose(tangent, (plus-minus)/.002, atol=3e-4, rtol=2e-3)


def test_adapter_identity_recipe_and_scalar_state_contracts():
    rng = torch.get_rng_state().clone()
    model = st.FractionalAngleGainHistoryAdapter(3, initial_angle=.4636476090008061, initial_gain=5**.5)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 8 and model.alpha == 2.
    x = torch.linspace(-.3, .7, 24).reshape(2, 4, 3)
    assert torch.equal(model(x), x)
    restored = st.FractionalAngleGainHistoryAdapter(3)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert torch.equal(restored(x), model(x))
    with pytest.raises((ValueError, RuntimeError)):
        st.FractionalGainHistoryAdapter(3).load_state_dict(model.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(st.FractionalGainHistoryAdapter(3).state_dict())
    with pytest.raises(TypeError): st.FractionalAngleGainHistoryAdapter(3, initial_angle=True)
    model.history_angle.data = model.history_angle.data.double()
    with pytest.raises(TypeError): _ = model.alpha
    with pytest.raises(TypeError): model(x)


def tiny_state(client, directory):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(197)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    config = {"features": 8, "steps": 4, "block_size": 6, "batch_size": 2, "seeds": [41],
              "arms": ["short", "full"], "checkpoint_every": 2, "evaluate_every": 2,
              "learning_rate": .01, "strength": .1}
    directory.mkdir()
    plan = {"study_id": "angle-path-test", "config": config,
            "base_parameter_sha256": client.study.pilot.model_digest(model),
            "batch_schedules": {"41": client.study.pilot.schedule(41, 8, 5, 2)}}
    journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
    return model, model.transformer.h[0], model.transformer.h[0].mlp, plan, journal


def angle_factory(arm, config, seed):
    return st.FractionalAngleGainHistoryAdapter(config["features"], initial_angle=.4636476090008061,
                                                initial_gain=5**.5, kernel_len=3 if arm == "short" else 8)


def run_tiny(client, state, path, **options):
    model, parent, original, plan, journal = state
    tokens = torch.arange(48).reshape(8, 6) % 32
    client.study.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan, path, journal,
                              adapter_factory=angle_factory, **options)


def test_tiny_hf_angle_learning_and_interrupted_resume(client, tmp_path):
    full = tiny_state(client, tmp_path / "full")
    run_tiny(client, full, tmp_path / "full")
    interrupted = tiny_state(client, tmp_path / "interrupted")
    def stop(key, cursor):
        if key == "41:full" and cursor == 2:
            raise RuntimeError("controlled interruption")
    with pytest.raises(RuntimeError, match="controlled interruption"):
        run_tiny(client, interrupted, tmp_path / "interrupted", after_checkpoint=stop)
    run_tiny(client, interrupted, tmp_path / "interrupted")
    for key, entry in full[-1]["runs"].items():
        left = client.study.load_checkpoint(tmp_path / "full", entry["checkpoint"], full[-2]["study_id"], key)
        right = client.study.load_checkpoint(tmp_path / "interrupted", interrupted[-1]["runs"][key]["checkpoint"], full[-2]["study_id"], key)
        for field in ("adapter", "optimizer", "records", "development"):
            assert client.study.pilot.equal_state(left[field], right[field])
        records = left["records"]
        assert records[0]["history_angle_gradient"] == 0
        assert all(row["history_angle_gradient"] != 0 for row in records[1:])
        assert records[-1]["alpha_after_update"] > 0
        assert records[-1]["alpha_after_update"] != records[0]["alpha_before_update"]
        assert entry["resume_next_update_equal"] and entry["frozen_base_unchanged"]


@pytest.mark.parametrize("bad", [-.5, 1.6])
def test_post_update_domain_exit_cannot_publish_a_checkpoint(client, tmp_path, bad):
    state = tiny_state(client, tmp_path / "failed")
    def optimizer_factory(adapter, arm, config):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        original = optimizer.step
        def step():
            result = original()
            with torch.no_grad(): adapter.history_angle.fill_(bad)
            return result
        optimizer.step = step
        return optimizer
    with pytest.raises(ValueError, match="fractional history angle"):
        run_tiny(client, state, tmp_path / "failed", optimizer_factory=optimizer_factory)
    assert not state[-1]["runs"]
    assert state[1].mlp is state[2]

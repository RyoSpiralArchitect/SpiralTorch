import copy
import json

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")


def fixture(radius=None):
    model = torch.nn.Sequential(torch.nn.Identity())
    adapter = st.WaveGateAdapter(2, log_radius=radius, curvature=-1.0)
    with torch.no_grad():
        adapter.gate.copy_(torch.tensor([0.3, -0.2]))
        adapter.bias.copy_(torch.tensor([0.1, 0.05]))
    return model, st.GeometryAdapterStack({"0": adapter})


@pytest.mark.parametrize("radius", [None, 0.0, 1.3862944])
def test_actual_forward_hooks_gradients_and_state_are_unchanged(radius):
    model, stack = fixture(radius)
    _, other = fixture(radius)
    other.load_state_dict(copy.deepcopy(stack.state_dict()))
    x = torch.tensor([[0.3, 0.4], [0.1, -0.5]], requires_grad=True)
    y = x.detach().clone().requires_grad_()
    before = copy.deepcopy(stack.state_dict())
    seen = []
    pre = stack.adapters[0].register_forward_pre_hook(lambda *_: seen.append("pre"))
    post = stack.adapters[0].register_forward_hook(lambda *_: seen.append("post"))
    rng = torch.get_rng_state().clone()
    with stack.observe() as observation, stack.attach(model, require_all=True):
        output = model(x)
        output.square().sum().backward()
    with other.attach(model, require_all=True):
        expected = model(y)
        expected.square().sum().backward()
    assert torch.equal(output, expected) and torch.equal(x.grad, y.grad)
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(stack.parameters(), other.parameters())
    )
    assert seen == ["pre", "post"] and torch.equal(rng, torch.get_rng_state())
    for name, value in stack.state_dict().items():
        assert (
            torch.equal(value, before[name])
            if isinstance(value, torch.Tensor)
            else value == before[name]
        )
    report = observation.to_dict()
    assert report["coverage_complete"] and report["calls_seen"] == 1
    event = report["events"][0]
    _, direct = stack.adapters[0].forward_with_conditioning(x.detach())
    assert event["native_reports"] == [direct]
    delta = output.detach().double() - x.detach().double()
    assert event["residual"]["residual_l2"] == torch.linalg.vector_norm(delta).item()
    assert event["residual"]["relative_residual_l2"] == (
        torch.linalg.vector_norm(delta).item()
        / torch.linalg.vector_norm(x.detach().double()).item()
    )
    assert event["grad_enabled"] and event["path"] == "0"
    report["events"].clear()
    assert len(observation.to_dict()["events"]) == 1
    json.dumps(observation.to_dict(), allow_nan=False)
    pre.remove()
    post.remove()


def test_bounded_observation_skips_diagnostics_not_model_calls():
    model, stack = fixture()
    x = torch.ones(2, 2)
    calls = []
    handle = stack.adapters[0].register_forward_hook(lambda *_: calls.append(1))
    with stack.observe(max_events=2, max_values=2) as trace, stack.attach(model):
        model(x)
        model(x[:1])
        model(x[:1])
    report = trace.to_dict()
    assert len(calls) == 3 and report["calls_seen"] == 3
    assert report["events_omitted"] == 1 and not report["coverage_complete"]
    assert [e["status"] for e in report["events"]] == [
        "skipped_value_budget",
        "observed",
    ]
    assert report["events"][0]["native_reports"] == []
    handle.remove()


def test_failures_and_nested_scopes_cleanup_without_leaking_error_text():
    class Failing(torch.nn.Module):
        def forward(self, value):
            raise RuntimeError("private contents must not be recorded")

    model = torch.nn.Sequential(torch.nn.Identity())
    stack = st.GeometryAdapterStack({"0": Failing()})
    with pytest.raises(RuntimeError, match="private contents"):
        with stack.observe() as trace, stack.attach(model):
            with pytest.raises(RuntimeError, match="already observed"):
                with stack.observe():
                    pass
            model(torch.ones(1, 2))
    report = trace.to_dict()
    assert (
        report["status"] == "failed"
        and report["events"][0]["error_type"] == "RuntimeError"
    )
    assert not report["coverage_complete"] and "private contents" not in json.dumps(
        report
    )
    assert stack._observation is None and stack._active_token is None
    assert not model[0]._forward_hooks
    with stack.observe() as next_trace:
        pass
    assert next_trace.to_dict()["calls_seen"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"max_events": 0},
        {"max_events": True},
        {"max_events": 4097},
        {"max_values": 0},
        {"max_values": 1.5},
        {"max_values": 1_048_577},
    ],
)
def test_invalid_observation_budgets_reject_without_attaching(options):
    model, stack = fixture()
    with pytest.raises(ValueError):
        with stack.observe(**options):
            pass
    assert stack._observation is None and not model[0]._forward_hooks


def test_ordinary_and_empty_inputs_have_generic_scalars_without_native_reports():
    model = torch.nn.Sequential(torch.nn.Identity())
    stack = st.GeometryAdapterStack({"0": st.ChannelAffineAdapter(2)})
    with stack.observe() as trace, stack.attach(model):
        model(torch.zeros(1, 2))
        model(torch.empty(0, 2))
    assert trace.to_dict()["coverage_complete"]
    for event in trace.to_dict()["events"]:
        assert not event["native_reports"]
        assert event["residual"]["residual_l2"] == 0.0
        assert event["residual"]["relative_residual_l2"] is None


def test_inplace_custom_adapter_cannot_rewrite_captured_input():
    class Inplace(torch.nn.Module):
        def forward(self, value):
            return value.add_(2.0)

    model = torch.nn.Sequential(torch.nn.Identity())
    stack = st.GeometryAdapterStack({"0": Inplace()})
    with stack.observe() as trace, stack.attach(model), torch.no_grad():
        model(torch.ones(1, 2))
    metric = trace.to_dict()["events"][0]["residual"]
    assert metric["input_l2"] == pytest.approx(2.0**0.5)
    assert metric["residual_l2"] == pytest.approx(8.0**0.5)


def test_native_report_budget_and_explicit_report_mutation_are_isolated():
    class Composite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wave = st.WaveGateAdapter(2)

        def forward(self, value):
            for _ in range(20):
                value, report = self.wave.forward_with_conditioning(value)
                report["rows"] = 999
            return value

    model = torch.nn.Sequential(torch.nn.Identity())
    stack = st.GeometryAdapterStack({"0": Composite()})
    with stack.observe() as trace, stack.attach(model):
        model(torch.ones(1, 2))
    event = trace.to_dict()["events"][0]
    assert len(event["native_reports"]) == 16 and event["native_reports_omitted"] == 4
    assert all(r["rows"] == 1 for r in event["native_reports"])
    assert not trace.to_dict()["coverage_complete"]


def test_nonfinite_custom_output_is_reported_without_changing_return_value():
    class Nonfinite(torch.nn.Module):
        def forward(self, value):
            return value * float("nan")

    model = torch.nn.Sequential(torch.nn.Identity())
    stack = st.GeometryAdapterStack({"0": Nonfinite()})
    with stack.observe() as trace, stack.attach(model):
        output = model(torch.ones(1, 2))
    assert torch.isnan(output).all()
    assert not trace.to_dict()["events"][0]["residual"]["finite"]
    json.dumps(trace.to_dict(), allow_nan=False)


def test_reentrant_native_report_sink_enforces_cap_at_append():
    from spiraltorch._geometry_conditioning_capture import (
        capture_native_conditioning,
        native_conditioning_sink,
    )

    with capture_native_conditioning() as capture:
        for index in range(15):
            native_conditioning_sink()({"index": index})
        outer = native_conditioning_sink()
        native_conditioning_sink()({"index": 15})
        outer({"index": 16})
        for _ in range(3):
            assert native_conditioning_sink() is None
    assert len(capture.reports) == 16 and capture.omitted == 4
    assert [r["index"] for r in capture.reports] == list(range(16))

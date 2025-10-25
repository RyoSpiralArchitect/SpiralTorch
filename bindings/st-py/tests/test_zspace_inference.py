from __future__ import annotations

import math
from typing import Any, Mapping

import pytest

pytest.importorskip("spiraltorch")

from spiraltorch import (
    ZSpaceInferencePipeline,
    ZSpaceInferenceRuntime,
    ZSpacePosterior,
    ZSpacePartialBundle,
    canvas_partial_from_snapshot,
    canvas_coherence_partial,
    coherence_partial_from_diagnostics,
    compile_inference,
    decode_zspace_embedding,
    blend_zspace_partials,
    infer_canvas_snapshot,
    infer_canvas_transformer,
    infer_coherence_diagnostics,
    infer_coherence_from_sequencer,
    infer_canvas_with_coherence,
    infer_with_partials,
    infer_from_partial,
    infer_with_psi,
    weights_partial_from_tensor,
    weights_partial_from_dlpack,
    infer_weights_from_dlpack,
    psi_partial_from_reading,
)


def test_decode_produces_expected_structure():
    vector = [0.12, -0.03, 0.48, -0.2]
    decoded = decode_zspace_embedding(vector)
    assert set(decoded.metrics.keys()) == {"speed", "memory", "stability", "frac", "drs"}
    assert len(decoded.gradient) == len(vector)
    assert math.isclose(sum(decoded.barycentric), 1.0, rel_tol=1e-6)


@pytest.mark.parametrize(
    "partial",
    [
        {"speed": 0.3, "stab": 0.7},
        {"mem": -0.2, "gradient": [0.1, -0.05, 0.0, 0.2]},
    ],
)
def test_infer_from_partial_overrides_requested_metrics(partial):
    vector = [0.2, -0.1, 0.45, -0.05]
    result = infer_from_partial(vector, partial)
    for key, value in partial.items():
        canonical = {
            "mem": "memory",
            "stab": "stability",
        }.get(key, key)
        if canonical == "gradient":
            assert len(result.gradient) == len(vector)
            continue
        assert math.isclose(result.metrics[canonical], float(value))
    assert 0.0 <= result.confidence <= 1.0
    assert math.isclose(sum(result.barycentric), 1.0, rel_tol=1e-6)


def test_posterior_project_matches_helper():
    vector = [0.42, 0.1, -0.25, 0.08]
    partial = {"speed": 0.1, "mem": 0.05}
    posterior = ZSpacePosterior(vector)
    direct = posterior.project(partial)
    helper = infer_from_partial(vector, partial)
    assert math.isclose(direct.residual, helper.residual)
    assert math.isclose(direct.confidence, helper.confidence)
    assert direct.metrics == helper.metrics


def test_runtime_accumulates_partial_updates():
    vector = [0.3, -0.05, 0.22, -0.11]
    runtime = ZSpaceInferenceRuntime(vector, smoothing=0.25)
    first = runtime.update({"speed": 0.45})
    assert math.isclose(first.metrics["speed"], 0.45)
    second = runtime.update({"stab": 0.4})
    assert math.isclose(second.metrics["speed"], 0.45)
    assert math.isclose(second.metrics["stability"], 0.4)
    runtime.clear()
    assert len(runtime.cached_observations) == 0
    reset = runtime.update()
    assert "speed" in reset.metrics


def test_runtime_without_accumulation_replaces_previous_observations():
    vector = [0.15, -0.2, 0.05, 0.18]
    runtime = ZSpaceInferenceRuntime(vector, accumulate=False)
    runtime.update({"speed": 0.6})
    runtime.update({"stab": 0.2})
    assert "speed" not in runtime.cached_observations


def test_blend_partials_supports_weighted_mean_and_gradient():
    blended = blend_zspace_partials(
        [
            {"speed": 0.2, "gradient": [0.1, -0.1]},
            ZSpacePartialBundle({"speed": 0.6, "gradient": [0.5, 0.1]}, weight=2.0),
        ]
    )
    assert math.isclose(blended["speed"], (0.2 + 0.6 * 2.0) / 3.0, rel_tol=1e-6)
    assert math.isclose(blended["gradient"][0], (0.1 + 0.5 * 2.0) / 3.0, rel_tol=1e-6)
    assert len(blended["gradient"]) == 2


def test_infer_with_partials_honours_last_strategy():
    vector = [0.4, -0.12, 0.33, -0.09]
    result = infer_with_partials(
        vector,
        {"speed": 0.15},
        {"speed": 0.9},
        strategy="last",
    )
    assert math.isclose(result.metrics["speed"], 0.9)


def test_pipeline_blends_and_clears_partials():
    vector = [0.22, -0.11, 0.31, -0.07]
    pipeline = ZSpaceInferencePipeline(vector, strategy="mean", smoothing=0.4)
    pipeline.add_partial({"speed": 0.5})
    pipeline.add_partial(ZSpacePartialBundle({"mem": -0.1}, weight=2.0))
    first = pipeline.infer(clear=False)
    assert math.isclose(first.metrics["speed"], 0.5, rel_tol=1e-6)
    assert math.isclose(first.metrics["memory"], -0.1, rel_tol=1e-6)
    second = pipeline.infer()
    assert "speed" in second.metrics
    assert pipeline.posterior is not None


def test_compile_inference_wraps_callable():
    calls: list[dict[str, float]] = []

    def collect(sample: Mapping[str, float]) -> Mapping[str, float]:
        calls.append(dict(sample))
        return sample

    compiled = compile_inference(collect)
    vector = [0.21, -0.04, 0.33, -0.12]
    partial = {"speed": 0.35}
    result = compiled(vector, partial)
    assert calls == [partial]
    assert math.isclose(result.metrics["speed"], 0.35)


def test_compile_inference_supports_decorator_usage():
    @compile_inference(alpha=0.5)
    def produce(speed: float, bias: float = 0.0) -> Mapping[str, float]:
        return {"speed": speed + bias}

    vector = [0.11, 0.02, -0.07, 0.19]
    output = produce(vector, 0.25, bias=0.1)
    assert math.isclose(output.metrics["speed"], 0.35)


def test_compile_inference_validates_return_type():
    @compile_inference
    def invalid(_: Mapping[str, float]):
        return 1

    vector = [0.2, -0.1, 0.05, -0.03]
    with pytest.raises(TypeError):
        invalid(vector, {"speed": 0.1})


def _matrix_stats(matrix: list[list[float]]) -> dict[str, float]:
    flat = [value for row in matrix for value in row]
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean}


class _DummyCanvasSnapshot:
    def __init__(self) -> None:
        self.canvas = [[0.2, -0.1], [0.05, 0.3]]
        self.hypergrad = [[0.08, -0.04], [0.01, -0.02]]
        self.realgrad = [[0.03, -0.01], [0.02, 0.04]]
        self.summary = {
            "hypergrad": _matrix_stats(self.hypergrad),
            "realgrad": _matrix_stats(self.realgrad),
        }
        self.patch = [[0.12, -0.05], [0.02, 0.06]]


class _DummyCanvas:
    def __init__(self, snapshot: _DummyCanvasSnapshot) -> None:
        self._snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self) -> _DummyCanvasSnapshot:
        self.snapshot_calls += 1
        return self._snapshot


def test_canvas_snapshot_inference_derives_augmented_metrics():
    snapshot = _DummyCanvasSnapshot()
    vector = [0.18, -0.07, 0.32, -0.11]
    partial = canvas_partial_from_snapshot(snapshot)
    assert {"canvas_energy", "hypergrad_norm", "realgrad_norm"}.issubset(partial)
    inference = infer_canvas_snapshot(vector, snapshot)
    assert "canvas_energy" in inference.metrics
    assert "hypergrad_norm" in inference.metrics
    assert math.isfinite(inference.metrics["canvas_energy"])


def test_infer_canvas_transformer_uses_snapshot_method():
    snapshot = _DummyCanvasSnapshot()
    canvas = _DummyCanvas(snapshot)
    vector = [0.11, 0.07, -0.05, 0.2]
    inference = infer_canvas_transformer(vector, canvas, smoothing=0.4)
    assert canvas.snapshot_calls == 1
    assert "canvas_mean" in inference.metrics
    assert math.isfinite(inference.metrics["canvas_mean"])


def test_canvas_coherence_partial_combines_sources():
    snapshot = _DummyCanvasSnapshot()
    diagnostics = _DummyDiagnostics()
    partial = canvas_coherence_partial(
        snapshot,
        diagnostics,
        coherence=[0.6, 0.3, 0.1],
        weights=(2.0, 1.0),
    )
    assert "canvas_energy" in partial
    assert "coherence_mean" in partial


class _DummyDiagnostics:
    def __init__(self) -> None:
        self.mean_coherence = 0.62
        self.coherence_entropy = 0.24
        self.energy_ratio = 0.7
        self.z_bias = -0.12
        self.fractional_order = 0.4
        self.normalized_weights = [0.5, 0.3, 0.2]
        self.preserved_channels = 3
        self.discarded_channels = 1
        self.dominant_channel = 0


class _DummyContour:
    def coherence_strength(self) -> float:
        return 0.58

    def prosody_index(self) -> float:
        return 0.41

    def articulation_bias(self) -> float:
        return -0.19


class _DummySequencer:
    def __init__(self, diagnostics: _DummyDiagnostics, contour: _DummyContour) -> None:
        self._diagnostics = diagnostics
        self._contour = contour
        self.calls: list[Any] = []

    def forward_with_diagnostics(self, tensor: object):
        self.calls.append(tensor)
        return ("tensor", [0.62, 0.28, 0.1], self._diagnostics)

    def emit_linguistic_contour(self, tensor: object) -> _DummyContour:
        self.calls.append(("contour", tensor))
        return self._contour


def test_coherence_diagnostics_projection_infers_metrics():
    diagnostics = _DummyDiagnostics()
    partial = coherence_partial_from_diagnostics(diagnostics, coherence=[0.6, 0.3, 0.1])
    assert math.isclose(partial["coherence_mean"], diagnostics.mean_coherence, rel_tol=1e-6)
    assert "coherence_weight_entropy" in partial
    inference = infer_coherence_diagnostics(
        [0.2, -0.05, 0.41, -0.09],
        diagnostics,
        coherence=[0.6, 0.3, 0.1],
        contour=_DummyContour(),
    )
    assert "coherence_strength" in inference.metrics


def test_infer_coherence_from_sequencer_can_return_outputs():
    diagnostics = _DummyDiagnostics()
    contour = _DummyContour()
    sequencer = _DummySequencer(diagnostics, contour)
    vector = [0.22, -0.15, 0.38, -0.04]
    sample = object()
    inference, outputs = infer_coherence_from_sequencer(
        vector,
        sequencer,
        sample,
        include_contour=True,
        return_outputs=True,
    )
    assert len(outputs) == 3
    assert ("contour", sample) in sequencer.calls
    assert "coherence_mean" in inference.metrics


def test_infer_canvas_with_coherence_projects_blended_partial():
    snapshot = _DummyCanvasSnapshot()
    diagnostics = _DummyDiagnostics()
    vector = [0.25, -0.08, 0.34, -0.12]
    inference = infer_canvas_with_coherence(
        vector,
        snapshot,
        diagnostics,
        coherence=[0.7, 0.2, 0.1],
        strategy="mean",
    )
    assert "canvas_energy" in inference.metrics
    assert "coherence_mean" in inference.metrics


def test_weights_partial_from_tensor_produces_expected_stats():
    weights = [[0.5, -0.25], [0.75, 0.0]]
    stats = weights_partial_from_tensor(weights)
    assert math.isclose(stats["weight_count"], 4.0)
    assert math.isclose(stats["weight_mean"], sum(sum(row) for row in weights) / 4, rel_tol=1e-6)
    assert stats["weight_l2"] > 0.0


def test_weights_partial_from_dlpack_invokes_capture(monkeypatch):
    captured: dict[str, object] = {}

    class _StubTensor:
        def __init__(self, data: list[list[float]]) -> None:
            self._data = data

        def tolist(self) -> list[list[float]]:
            return self._data

    def fake_capture(value: object, allow_compat: bool = True) -> _StubTensor:
        captured["value"] = value
        return _StubTensor([[0.1, -0.1], [0.2, -0.2]])

    monkeypatch.setattr(
        "spiraltorch.zspace_inference._capture_tensor_like",
        fake_capture,
        raising=True,
    )
    capsule = object()
    stats = weights_partial_from_dlpack(capsule)
    assert captured["value"] is capsule
    assert math.isclose(stats["weight_count"], 4.0)


def test_infer_weights_from_dlpack_yields_weight_metrics(monkeypatch):
    class _StubTensor:
        def __init__(self, data: list[list[float]]) -> None:
            self._data = data

        def tolist(self) -> list[list[float]]:
            return self._data

    tensor = _StubTensor([[0.2, -0.2], [0.4, 0.1]])
    monkeypatch.setattr(
        "spiraltorch.zspace_inference._capture_tensor_like",
        lambda value, allow_compat=True: tensor,
        raising=True,
    )
    inference = infer_weights_from_dlpack([0.2, -0.1, 0.05], object())
    assert "weight_mean" in inference.metrics
    assert inference.metrics["weight_mean"] != 0.0


class _DummyPsiEvent:
    def __init__(self, up: bool, value: float) -> None:
        self.up = up
        self.value = value


class _DummyAdvisory:
    def __init__(self) -> None:
        self.mu_eff0 = -0.3
        self.alpha3 = 0.8
        self.audit_container_gap = 0.2
        self.audit_cluster = 0.5
        self.container_cluster = 0.4
        self.regime = type("Regime", (), {"name": "Supercritical"})()

    def stability_score(self) -> float:
        return 0.72

    def audit_overbias(self) -> bool:
        return True

    def container_reinforcement(self) -> float:
        return 0.65


class _DummyTuning:
    def __init__(self) -> None:
        self.required_components = ["act", "band"]
        self.weight_increments = {"ACT_DRIFT": 0.2, "BAND_ENERGY": 0.1}
        self.threshold_shifts = {"GRAD_NORM": -0.05}


def test_psi_partial_from_reading_includes_breakdown():
    reading = {
        "total": 0.85,
        "step": 42,
        "breakdown": {"ACT_DRIFT": 0.4, "BAND_ENERGY": 0.3},
    }
    events = [_DummyPsiEvent(True, 0.2), _DummyPsiEvent(False, -0.1)]
    metrics = psi_partial_from_reading(
        reading,
        events=events,
        advisory=_DummyAdvisory(),
        tuning=_DummyTuning(),
    )
    assert math.isclose(metrics["psi_total"], 0.85, rel_tol=1e-6)
    assert metrics["psi_component_act_drift"] == pytest.approx(0.4)
    assert metrics["psi_event_count"] == pytest.approx(2.0)
    assert metrics["psi_spiral_regime"] == pytest.approx(0.0)
    assert metrics["psi_tuning_weight_total"] == pytest.approx(0.3)


def test_infer_with_psi_fetches_latest(monkeypatch):
    reading = {"total": 0.9, "step": 64, "breakdown": {"ACT_DRIFT": 0.5}}
    events = [_DummyPsiEvent(True, 0.1)]
    advisory = _DummyAdvisory()
    tuning = _DummyTuning()

    monkeypatch.setattr(
        "spiraltorch.zspace_inference.fetch_latest_psi_telemetry",
        lambda: (reading, events, advisory, tuning),
        raising=True,
    )
    inference = infer_with_psi([0.3, -0.2, 0.1], {"speed": 0.6}, psi=True)
    assert "psi_total" in inference.metrics
    assert inference.metrics["psi_total"] == pytest.approx(0.9)


def test_pipeline_uses_default_psi_source(monkeypatch):
    reading = {"total": 0.7, "step": 21, "breakdown": {"BAND_ENERGY": 0.25}}
    events = [_DummyPsiEvent(False, -0.2)]

    monkeypatch.setattr(
        "spiraltorch.zspace_inference.fetch_latest_psi_telemetry",
        lambda: (reading, events, _DummyAdvisory(), _DummyTuning()),
        raising=True,
    )
    pipeline = ZSpaceInferencePipeline([0.18, -0.05, 0.31], psi=True)
    pipeline.add_partial({"speed": 0.45})
    inference = pipeline.infer(clear=True)
    assert "psi_total" in inference.metrics
    assert inference.metrics["psi_total"] == pytest.approx(0.7)

from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


class _FakeScaleStack:
    threshold = 0.1
    mode = "scalar"
    samples = [(1.0, 0.25), (2.0, 0.5)]

    def persistence(self):
        return [(0.0, 1.0, 0.25), (1.0, 2.0, 0.25)]

    def interface_density(self):
        return 0.25

    def moment(self, order: int):
        return 0.5 + order

    def boundary_dimension(self, ambient_dim: float, window: int):
        assert ambient_dim == pytest.approx(2.0)
        assert window == 2
        return 1.5

    def coherence_break_scale(self, level: float):
        return 1.0 if level <= 0.25 else 2.0


def test_scale_stack_probe_exported_from_top_level() -> None:
    assert "scale_stack_probe" in st.__all__
    assert "scale_stack_probe_to_zspace_partial" in st.__all__
    assert "scalar_scale_stack_partial" in st.__all__
    assert "scalar_scale_stack_probe" in st.__all__
    assert "semantic_scale_stack_partial" in st.__all__
    assert "semantic_scale_stack_probe" in st.__all__


def test_scale_stack_probe_matches_wasm_payload_shape() -> None:
    probe = st.scale_stack_probe(
        _FakeScaleStack(),
        ambient_dim=2.0,
        dimension_window=2,
        levels=(0.25, 0.5),
    )

    assert probe["kind"] == "spiraltorch.wasm_scale_stack_probe"
    assert probe["source_crate"] == "st-frac::scale_stack"
    assert probe["mode"] == "scalar"
    assert probe["sample_count"] == 2
    assert probe["samples"][0] == {"scale": 1.0, "gate_mean": 0.25}
    assert probe["persistence"][0] == {
        "scale_low": 0.0,
        "scale_high": 1.0,
        "mass": 0.25,
    }
    assert probe["moment_2"] == pytest.approx(2.5)
    assert probe["boundary_dimension"] == pytest.approx(1.5)
    assert probe["coherence_profile"] == [
        {"level": 0.25, "scale": 1.0},
        {"level": 0.5, "scale": 2.0},
    ]


def test_scale_stack_probe_converts_to_zspace_partial() -> None:
    probe = st.scale_stack_probe(
        _FakeScaleStack(),
        ambient_dim=2.0,
        dimension_window=2,
        levels=(0.25, 0.5),
    )

    partial = st.scale_stack_probe_to_zspace_partial(
        probe,
        gradient_dim=6,
        telemetry_prefix="scale",
    )
    metrics = partial.resolved()
    telemetry = partial.telemetry_payload()

    assert isinstance(partial, st.ZSpacePartialBundle)
    assert partial.origin == "scale_stack:scalar"
    assert 0.0 <= metrics["speed"] <= 1.0
    assert 0.0 <= metrics["memory"] <= 1.0
    assert 0.0 <= metrics["stability"] <= 1.0
    assert 0.0 <= metrics["drs"] <= 1.0
    assert len(metrics["gradient"]) == 6
    assert telemetry is not None
    assert telemetry["scale.mode_scalar"] == pytest.approx(1.0)
    assert telemetry["scale.sample_count"] == pytest.approx(2.0)
    assert telemetry["scale.coherence_break_mean"] == pytest.approx(1.5)

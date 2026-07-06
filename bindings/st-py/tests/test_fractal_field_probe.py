from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


class _FakeFractalFieldGenerator:
    octaves = 3
    lacunarity = 2.0
    gain = 0.5
    iterations = 16

    def branching_field(self, log_start: float, log_step: float, length: int):
        assert log_start == pytest.approx(-2.0)
        assert log_step == pytest.approx(0.25)
        return [complex(0.1 * idx, (-0.05) ** idx) for idx in range(length)]


def test_fractal_field_probe_exported_from_top_level_and_frac_module() -> None:
    assert "fractal_field_probe" in st.__all__
    assert "fractal_field_probe_to_zspace_partial" in st.__all__
    assert "fractal_field_partial" in st.__all__
    assert st.frac.fractal_field_probe is st.fractal_field_probe
    assert st.frac.fractal_field_probe_to_zspace_partial is st.fractal_field_probe_to_zspace_partial


def test_fractal_field_probe_matches_wasm_payload_shape() -> None:
    probe = st.fractal_field_probe(
        _FakeFractalFieldGenerator(),
        -2.0,
        0.25,
        6,
        preview_len=3,
    )

    assert probe["kind"] == "spiraltorch.wasm_fractal_field_probe"
    assert probe["source_crate"] == "st-frac::fractal_field"
    assert probe["mode"] == "branching_field"
    assert probe["generator"] == {
        "octaves": 3,
        "lacunarity": 2.0,
        "gain": 0.5,
        "iterations": 16,
    }
    assert probe["log_lattice"]["support"] == [-2.0, -0.75]
    assert probe["sample_count"] == 6
    assert probe["preview_count"] == 3
    assert probe["energy"] > 0.0
    assert 0.0 <= probe["coherence_score"] <= 1.0
    assert probe["samples"][1]["index"] == 1
    assert probe["samples"][1]["log"] == pytest.approx(-1.75)


def test_fractal_field_probe_converts_to_zspace_partial() -> None:
    probe = st.fractal_field_probe(
        _FakeFractalFieldGenerator(),
        -2.0,
        0.25,
        6,
        preview_len=3,
    )

    partial = st.fractal_field_probe_to_zspace_partial(
        probe,
        gradient_dim=7,
        telemetry_prefix="field",
    )
    metrics = partial.resolved()
    telemetry = partial.telemetry_payload()

    assert isinstance(partial, st.ZSpacePartialBundle)
    assert partial.origin == "fractal_field:branching_field"
    assert 0.0 <= metrics["speed"] <= 1.0
    assert 0.0 <= metrics["memory"] <= 1.0
    assert 0.0 <= metrics["stability"] <= 1.0
    assert 0.0 <= metrics["drs"] <= 1.0
    assert len(metrics["gradient"]) == 7
    assert telemetry is not None
    assert telemetry["field.sample_count"] == pytest.approx(6.0)
    assert telemetry["field.generator.octaves"] == pytest.approx(3.0)
    assert telemetry["field.lattice.len"] == pytest.approx(6.0)


def test_fractal_field_partial_uses_native_generator_when_available() -> None:
    if not callable(getattr(st.frac, "FractalFieldGenerator", None)):
        pytest.skip("native FractalFieldGenerator is unavailable")

    partial = st.fractal_field_partial(
        2,
        -1.0,
        0.2,
        6,
        iterations=8,
        gradient_dim=4,
    )
    metrics = partial.resolved()

    assert partial.origin == "fractal_field:branching_field"
    assert len(metrics["gradient"]) == 4

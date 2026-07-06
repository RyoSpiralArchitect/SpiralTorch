from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


def _scale_probe() -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_scale_stack_probe",
        "source_crate": "st-frac::scale_stack",
        "mode": "scalar",
        "threshold": 0.1,
        "sample_count": 2,
        "samples": [
            {"scale": 1.0, "gate_mean": 0.25},
            {"scale": 2.0, "gate_mean": 0.5},
        ],
        "persistence": [{"scale_low": 0.0, "scale_high": 1.0, "mass": 0.25}],
        "interface_density": 0.25,
        "moment_0": 0.5,
        "moment_1": 1.5,
        "moment_2": 2.5,
        "boundary_dimension": 1.5,
        "coherence_profile": [{"level": 0.25, "scale": 1.0}],
    }


def _fractal_probe() -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_fractal_field_probe",
        "source_crate": "st-frac::fractal_field",
        "mode": "branching_field",
        "generator": {"octaves": 3, "lacunarity": 2.0, "gain": 0.5, "iterations": 16},
        "log_lattice": {"log_start": -2.0, "log_step": 0.25, "len": 4},
        "sample_count": 4,
        "preview_count": 2,
        "energy": 0.4,
        "mean_abs": 0.3,
        "max_abs": 0.7,
        "phase_drift": 0.2,
        "total_variation": 0.15,
        "coherence_score": 0.86,
        "samples": [{"index": 0, "re": 0.1, "im": 0.0, "abs": 0.1}],
    }


def _log_z_probe() -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_log_z_series_probe",
        "source_crate": "st-frac::cosmology",
        "mode": "log_z_series",
        "log_lattice": {"log_start": 0.0, "log_step": 0.25, "len": 4},
        "options": {"window": "hann", "normalisation": "l1"},
        "sample_count": 4,
        "sample_stats": {"count": 4, "mean": 2.5, "min": 1.0, "max": 4.0, "energy": 7.5},
        "weight_stats": {"count": 4, "mean": 0.25, "min": 0.0, "max": 0.5, "energy": 0.125},
        "z_count": 2,
        "projection": {
            "count": 2,
            "mean_abs": 1.1,
            "max_abs": 1.4,
            "energy": 1.25,
            "phase_drift": 0.1,
            "stability_score": 0.9,
            "preview_count": 1,
            "preview": [{"index": 0, "re": 1.0, "im": 0.0, "abs": 1.0}],
        },
    }


def test_geometry_context_exports_top_level_and_module() -> None:
    assert "geometry_probe_to_zspace_partial" in st.__all__
    assert "geometry_probe_consensus_partial" in st.__all__
    assert "build_geometry_probe_context" in st.__all__
    assert "geometry" in st.__all__
    assert st.geometry.geometry_probe_to_zspace_partial is st.geometry_probe_to_zspace_partial
    assert st.geometry.geometry_probe_consensus_partial is st.geometry_probe_consensus_partial
    assert st.geometry.build_geometry_probe_context is st.build_geometry_probe_context


def test_geometry_probe_router_accepts_supported_probe_kinds() -> None:
    partials = [
        st.geometry_probe_to_zspace_partial(_scale_probe(), gradient_dim=4),
        st.geometry_probe_to_zspace_partial(_fractal_probe(), gradient_dim=4),
        st.geometry_probe_to_zspace_partial(_log_z_probe(), gradient_dim=4),
    ]

    assert [len(partial.resolved()["gradient"]) for partial in partials] == [4, 4, 4]
    assert partials[0].origin == "scale_stack:scalar"
    assert partials[1].origin == "fractal_field:branching_field"
    assert partials[2].origin == "log_z_series:projection"


def test_build_geometry_probe_context_feeds_api_llm_prompt() -> None:
    partials, metadata = st.build_geometry_probe_context(
        {
            "scale": _scale_probe(),
            "field": _fractal_probe(),
            "logz": _log_z_probe(),
        },
        gradient_dim=5,
    )

    assert len(partials) == 3
    assert metadata["probe_count"] == 3
    assert metadata["families"] == {
        "scale_stack": 1,
        "fractal_field": 1,
        "log_z_series": 1,
    }
    assert metadata["context_origins"] == [
        "geometry:scale",
        "geometry:field",
        "geometry:logz",
    ]

    prompt = st.format_api_llm_context_prompt(
        "Read the browser geometry.",
        partials,
        max_partials=3,
        max_telemetry=12,
    )

    assert "origin=geometry:scale" in prompt
    assert "origin=geometry:field" in prompt
    assert "origin=geometry:logz" in prompt
    assert "geometry.scale_stack.1.sample_count=2" in prompt
    assert "geometry.fractal_field.2.energy=0.4" in prompt
    assert "geometry.log_z_series.3.projection_stability=0.9" in prompt


def test_geometry_probe_consensus_partial_blends_probe_families() -> None:
    consensus, metadata = st.geometry_probe_consensus_partial(
        [_scale_probe(), _fractal_probe(), _log_z_probe()],
        gradient_dim=4,
        consensus_weight=1.7,
    )
    metrics = consensus.resolved()
    telemetry = consensus.telemetry_payload()

    assert consensus.origin == "geometry:consensus"
    assert consensus.weight == pytest.approx(1.7)
    assert metadata["consensus"]["strategy"] == "mean"
    assert metadata["consensus"]["metric_count"] == len(metrics)
    assert {"speed", "memory", "stability", "drs", "gradient"} <= set(metrics)
    assert len(metrics["gradient"]) == 4
    assert telemetry is not None
    assert telemetry["geometry.consensus.probe_count"] == pytest.approx(3.0)
    assert telemetry["geometry.consensus.family_count"] == pytest.approx(3.0)
    assert telemetry["geometry.consensus.family_scale_stack_count"] == pytest.approx(1.0)


def test_build_geometry_probe_context_can_append_consensus_prompt_context() -> None:
    partials, metadata = st.build_geometry_probe_context(
        [_scale_probe(), _fractal_probe(), _log_z_probe()],
        gradient_dim=4,
        include_consensus=True,
    )

    prompt = st.format_api_llm_context_prompt(
        "Read the fused browser geometry.",
        partials,
        max_partials=4,
        max_telemetry=20,
    )

    assert len(partials) == 4
    assert metadata["consensus"]["origin"] == "geometry:consensus"
    assert metadata["context_origins"][-1] == "geometry:consensus"
    assert "origin=geometry:consensus" in prompt
    assert "geometry.consensus.probe_count=3" in prompt
    assert "geometry.consensus.family_log_z_series_count=1" in prompt


def test_geometry_probe_context_artifact_round_trips(tmp_path) -> None:
    path = tmp_path / "geometry-context.json"

    written = st.write_geometry_probe_context_artifact(
        path,
        [_scale_probe(), _fractal_probe()],
        gradient_dim=3,
    )
    partials, metadata = st.load_geometry_probe_context_artifact(written)

    assert str(path) == written
    assert len(partials) == 2
    assert metadata["artifact_schema"] == "spiraltorch.geometry_probe_context.v1"
    assert metadata["probe_count"] == 2
    assert len(partials[0].resolved()["gradient"]) == 3


def test_geometry_probe_router_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported geometry probe kind"):
        st.geometry_probe_to_zspace_partial({"kind": "spiraltorch.unknown"})


def test_geometry_probe_consensus_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unsupported geometry consensus strategy"):
        st.geometry_probe_consensus_partial(_scale_probe(), strategy="median")

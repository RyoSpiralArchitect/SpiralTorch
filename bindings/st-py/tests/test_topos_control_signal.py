from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


def test_topos_control_signal_from_mapping_normalises_pressure() -> None:
    signal = st.topos_control_signal(
        {
            "curvature": -0.9,
            "tolerance": 1e-4,
            "saturation": 2.0,
            "porosity": 0.25,
            "max_depth": 10,
            "max_volume": 100,
        },
        observed_depth=4,
        visited_volume=25,
    )

    assert signal["observed_depth"] == 4
    assert signal["visited_volume"] == 25
    assert signal["remaining_volume"] == 75
    assert signal["depth_pressure"] == pytest.approx(0.4)
    assert signal["volume_pressure"] == pytest.approx(0.25)
    assert signal["closure_pressure"] == pytest.approx(0.4)
    assert signal["openness"] == pytest.approx(0.6)
    assert len(signal["gradient"]) == 6


def test_topos_control_partial_feeds_zspace_inference() -> None:
    partial = st.topos_control_partial(
        {
            "curvature": -1.0,
            "tolerance": 1e-5,
            "saturation": 1.5,
            "porosity": 0.2,
            "max_depth": 8,
            "max_volume": 16,
            "observed_depth": 2,
            "visited_volume": 8,
        },
        gradient_dim=4,
    )
    resolved = partial.resolved()
    telemetry = partial.telemetry_payload()

    assert partial.origin == "topos:control"
    assert resolved["memory"] == pytest.approx(0.5)
    assert resolved["stability"] > 0.0
    assert resolved["frac"] > 0.0
    assert len(resolved["gradient"]) == 4
    assert telemetry is not None
    assert telemetry["topos.closure_pressure"] == pytest.approx(0.5)


def test_topos_control_partial_is_exported_from_top_level() -> None:
    assert "topos_control_signal" in st.__all__
    assert "topos_control_partial" in st.__all__


def test_pipeline_add_topos_control_blends_into_inference() -> None:
    pipeline = st.ZSpaceInferencePipeline([0.18, -0.07, 0.31, -0.14])

    bundle = pipeline.add_topos_control(
        {
            "curvature": -1.0,
            "tolerance": 1e-5,
            "saturation": 1.0,
            "porosity": 0.3,
            "max_depth": 10,
            "max_volume": 20,
        },
        observed_depth=2,
        visited_volume=10,
    )
    inference = pipeline.infer()

    assert bundle.origin == "topos:control"
    assert inference.telemetry is not None
    assert inference.telemetry.payload["topos.volume_pressure"] == pytest.approx(0.5)
    assert inference.metrics["memory"] == pytest.approx(bundle.resolved()["memory"])


def test_topos_control_reaches_zspace_trainer_step() -> None:
    pipeline = st.ZSpaceInferencePipeline([0.18, -0.07, 0.31, -0.14])
    pipeline.add_topos_control(
        {"porosity": 0.2, "max_depth": 8, "max_volume": 20},
        observed_depth=4,
        visited_volume=5,
    )
    inference = pipeline.infer()
    metrics = st.inference_to_zmetrics(inference, include_telemetry=True)
    trainer = st.ZSpaceTrainer(z_dim=4, lam_frac=0.0)

    loss = trainer.step(metrics)

    assert loss > 0.0
    assert metrics.telemetry is not None
    assert metrics.telemetry["topos.closure_pressure"] == pytest.approx(0.5)
    assert trainer.last_telemetry["topos.closure_pressure"] == pytest.approx(0.5)
    assert trainer.last_topos_control["closure_pressure"] == pytest.approx(0.5)


def test_prepared_zmetrics_payload_keeps_topos_telemetry_for_trainer() -> None:
    pipeline = st.ZSpaceInferencePipeline([0.2, -0.08, 0.27, -0.11])
    pipeline.add_topos_control(
        {"max_depth": 6, "max_volume": 12},
        observed_depth=3,
        visited_volume=9,
    )
    inference = pipeline.infer()
    trainer = st.ZSpaceTrainer(z_dim=4, lam_frac=0.0)

    payload = st.prepare_trainer_step_payload(trainer, inference, payload="zmetrics")
    trainer.step(payload)

    assert trainer.last_topos_control["volume_pressure"] == pytest.approx(0.75)


def test_topos_control_gain_can_drive_trainer_without_metric_gradient() -> None:
    metrics = st.z_metrics(
        speed=0.0,
        memory=0.0,
        stability=0.0,
        telemetry={
            "topos": {
                "closure_pressure": 0.75,
                "volume_pressure": 0.75,
                "depth_pressure": 0.25,
                "guard_strength": 0.8,
                "openness": 0.25,
                "exploration_hint": 0.1,
            }
        },
    )
    passive = st.ZSpaceTrainer(z_dim=4, lam_frac=0.0, topos_control_gain=0.0)
    active = st.ZSpaceTrainer(z_dim=4, lam_frac=0.0, topos_control_gain=0.6)

    passive_loss = passive.step(metrics)
    active_loss = active.step(metrics)

    assert passive.state == [0.0, 0.0, 0.0, 0.0]
    assert active.state != passive.state
    assert active_loss > passive_loss
    assert active.last_topos_control["guard_strength"] == pytest.approx(0.8)


def test_z_metrics_partial_preserves_topos_telemetry() -> None:
    metrics = st.z_metrics(
        speed=0.2,
        memory=0.1,
        stability=0.8,
        telemetry={"topos": {"closure_pressure": 0.4}},
    )

    partial = st.z.partial(metrics)

    assert partial.telemetry_payload()["topos.closure_pressure"] == pytest.approx(0.4)


def test_topos_control_context_formats_for_api_llm_prompt() -> None:
    partial = st.topos_control_partial(
        {"max_depth": 10, "max_volume": 20},
        observed_depth=1,
        visited_volume=10,
    )

    prompt = st.format_api_llm_context_prompt(
        "Route with topos control.",
        [partial],
        max_telemetry=32,
    )

    assert "origin=topos:control" in prompt
    assert "topos.closure_pressure" in prompt
    assert "User prompt: Route with topos control." in prompt


def test_open_topos_control_signal_can_override_porosity() -> None:
    guard = st.hypergrad_topos(max_depth=10, max_volume=100)
    signal = st.topos_control_signal(
        guard,
        porosity=0.35,
        observed_depth=5,
        visited_volume=10,
    )

    assert signal["porosity"] == pytest.approx(0.35)
    assert signal["depth_pressure"] == pytest.approx(0.5)
    assert signal["volume_pressure"] == pytest.approx(0.1)
    assert signal["closure_pressure"] == pytest.approx(0.5)


def test_tensor_biome_control_signal_reflects_absorbed_volume() -> None:
    guard = st.hypergrad_topos(max_volume=8)
    biome = st.TensorBiome(guard)
    if not hasattr(biome, "control_signal"):
        pytest.skip("TensorBiome.control_signal requires a freshly built native extension")
    biome.absorb(st.Tensor(1, 2, [0.1, 0.2]))
    biome.absorb(st.Tensor(1, 2, [0.3, 0.4]))

    signal = biome.control_signal()

    assert signal["visited_volume"] == 4
    assert signal["volume_pressure"] == pytest.approx(0.5)
    assert signal["openness"] == pytest.approx(0.5)

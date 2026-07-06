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
    assert signal["step_damping"] == pytest.approx(0.258)
    assert signal["learning_rate_scale"] == pytest.approx(0.8919875)
    assert signal["temperature_scale"] == pytest.approx(0.8533125)
    assert signal["regularization_scale"] == pytest.approx(1.2853125)
    assert signal["sampling_focus"] == pytest.approx(0.668003125)
    assert len(signal["runtime_hints"]) == 5
    assert len(signal["gradient"]) == 6
    assert signal["training_hints"]["gradient_bias_scale"] == pytest.approx(0.0786561)
    assert signal["training_hints"]["clip_scale"] == pytest.approx(0.871)
    assert signal["training_hints"]["momentum_damping"] == pytest.approx(0.2535)
    assert signal["training_plan"]["rate_scale"] == pytest.approx(0.7769211125)
    assert signal["training_plan"]["effective_gradient_bias_scale"] == pytest.approx(
        0.0786561
    )
    assert signal["inference_hints"]["top_p_scale"] == pytest.approx(0.890274375)
    assert signal["inference_hints"]["frequency_penalty_bias"] == pytest.approx(0.28540109375)
    assert signal["inference_hints"]["presence_penalty_bias"] == pytest.approx(-0.038100625)
    assert signal["inference_hints"]["context_weight"] == pytest.approx(0.9225)
    assert signal["inference_plan"]["temperature"] == pytest.approx(0.8533125)
    assert signal["inference_plan"]["top_p"] == pytest.approx(0.890274375)


def test_named_topos_hints_are_exported_for_learning_and_inference() -> None:
    payload = {
        "curvature": -0.9,
        "tolerance": 1e-4,
        "saturation": 2.0,
        "porosity": 0.25,
        "max_depth": 10,
        "max_volume": 100,
    }

    training = st.topos_training_hints(payload, observed_depth=4, visited_volume=25)
    training_plan = st.topos_training_plan(
        payload,
        gain=0.5,
        observed_depth=4,
        visited_volume=25,
    )
    inference = st.topos_inference_hints(payload, observed_depth=4, visited_volume=25)
    inference_plan = st.topos_inference_plan(
        payload,
        gain=0.5,
        observed_depth=4,
        visited_volume=25,
        base_temperature=0.8,
        base_top_p=0.9,
        base_frequency_penalty=0.1,
        base_presence_penalty=0.2,
    )

    assert training["learning_rate_scale"] == pytest.approx(0.8919875)
    assert training["gradient_bias_scale"] == pytest.approx(0.0786561)
    assert len(training["vector"]) == 6
    assert training_plan["raw_rate_scale"] == pytest.approx(0.7769211125)
    assert training_plan["rate_scale"] == pytest.approx(0.88846055625)
    assert training_plan["effective_gradient_bias_scale"] == pytest.approx(0.03932805)
    assert inference["temperature_scale"] == pytest.approx(0.8533125)
    assert inference["top_p_scale"] == pytest.approx(0.890274375)
    assert len(inference["vector"]) == 6
    assert inference_plan["temperature"] == pytest.approx(0.741325)
    assert inference_plan["top_p"] == pytest.approx(0.85062346875)
    assert inference_plan["frequency_penalty"] == pytest.approx(0.242700546875)
    assert inference_plan["presence_penalty"] == pytest.approx(0.1809496875)


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
    assert telemetry["topos.learning_rate_scale"] < 1.0
    assert telemetry["topos.temperature_scale"] > 0.5
    assert telemetry["topos.regularization_scale"] > 1.0
    assert telemetry["topos.training_hints.gradient_bias_scale"] > 0.0
    assert telemetry["topos.inference_hints.context_weight"] > 0.0


def test_topos_control_partial_is_exported_from_top_level() -> None:
    assert "topos_control_signal" in st.__all__
    assert "topos_training_hints" in st.__all__
    assert "topos_training_plan" in st.__all__
    assert "topos_inference_hints" in st.__all__
    assert "topos_inference_plan" in st.__all__
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
                "learning_rate_scale": 0.55,
                "regularization_scale": 1.4,
                "step_damping": 0.6,
                "sampling_focus": 0.7,
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
    assert active.last_topos_control["learning_rate_scale"] == pytest.approx(0.55)


def test_topos_training_hints_reach_trainer_telemetry() -> None:
    signal = st.topos_control_signal(
        {"porosity": 0.25, "max_depth": 10, "max_volume": 100},
        observed_depth=4,
        visited_volume=25,
    )
    metrics = st.z_metrics(speed=0.0, memory=0.0, stability=0.0, telemetry={"topos": signal})
    trainer = st.ZSpaceTrainer(z_dim=4, lam_frac=0.0, topos_control_gain=0.5)

    trainer.step(metrics)

    assert trainer.last_topos_control["training_hints.gradient_bias_scale"] == pytest.approx(
        0.0786561
    )
    assert trainer.last_topos_control["training_hints.clip_scale"] == pytest.approx(0.871)


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
    assert "learning_rate_scale" in signal
    assert "temperature_scale" in signal
    assert "runtime_hints" in signal


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
    assert len(signal["runtime_hints"]) == 5

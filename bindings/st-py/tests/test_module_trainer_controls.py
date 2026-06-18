from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_native() -> types.ModuleType | None:
    _ensure_torch_stub()
    try:
        module = importlib.import_module("spiraltorch")
    except Exception:
        return None

    for native_name in (
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    ):
        try:
            importlib.import_module(native_name)
        except Exception:
            continue
        return module
    return None


def test_module_trainer_prepare_step_zero_and_realgrad_controls() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "ModuleTrainer")
    assert hasattr(st.nn, "Sequential")
    assert hasattr(st.nn, "Linear")
    assert hasattr(st.nn, "MeanSquaredError")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    assert trainer.real_learning_rate is None
    trainer.enable_realgrad(5e-3)
    assert trainer.real_learning_rate == pytest.approx(5e-3)
    trainer.disable_realgrad()
    assert trainer.real_learning_rate is None

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", 2, 1))
    loss = st.nn.MeanSquaredError()
    x = st.Tensor.rand(2, 2, seed=21)
    y = st.Tensor.rand(2, 1, seed=22)

    pred_before = model.forward(x)
    trainer.prepare(model)
    grad_pred = loss.backward(pred_before, y)
    _ = model.backward(x, grad_pred)
    trainer.step(model)
    trainer.zero(model)
    pred_after = model.forward(x)

    assert pred_before.shape() == pred_after.shape()
    assert pred_before.tolist() != pred_after.tolist()


def test_module_trainer_train_epochs_returns_history() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    model = st.nn.Sequential()
    model.add(st.nn.Linear("fit_l1", 2, 1))
    trainer.prepare(model)
    loss = st.nn.MeanSquaredError()
    schedule = trainer.roundtable(
        1,
        1,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    batches = [
        (st.Tensor.rand(1, 2, seed=101), st.Tensor.rand(1, 1, seed=102)),
        (st.Tensor.rand(1, 2, seed=103), st.Tensor.rand(1, 1, seed=104)),
    ]
    validation = [(st.Tensor.rand(1, 2, seed=105), st.Tensor.rand(1, 1, seed=106))]

    eval_stats = trainer.evaluate_epoch(model, loss, validation)
    assert eval_stats.batches == 1

    report = trainer.train_epochs(
        model,
        loss,
        batches,
        schedule,
        epochs=3,
        validation_batches=validation,
        patience=1,
        min_delta=0.0,
        shuffle_seed=123,
        restore_best=True,
    )

    assert report["epochs_run"] >= 1
    assert report["best_epoch"] is not None
    assert report["restored_best"] is True
    assert report["best_score"] == pytest.approx(
        report["history"][report["best_epoch_index"]]["score"]
    )
    assert all("train" in item and "validation" in item for item in report["history"])


def test_module_trainer_curvature_scheduler_metrics_roundtrip() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "CurvatureScheduler")
    assert hasattr(st.nn, "RoundtableConfig")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-1,
        fallback_learning_rate=1e-2,
    )
    scheduler = st.nn.CurvatureScheduler(
        initial=-1.0,
        min_curvature=-2.0,
        max_curvature=-0.2,
        target_pressure=0.0,
        step=0.2,
        tolerance=0.0,
        smoothing=1.0,
    )
    trainer.enable_curvature_scheduler(scheduler)

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", 2, 1))
    trainer.prepare(model)
    loss = st.nn.MeanSquaredError()
    schedule = trainer.roundtable(
        2,
        1,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    x1 = st.Tensor.rand(2, 2, seed=31)
    y1 = st.Tensor.rand(2, 1, seed=32)
    x2 = st.Tensor.rand(2, 2, seed=33)
    y2 = st.Tensor.rand(2, 1, seed=34)
    stats = trainer.train_epoch(model, loss, [(x1, y1), (x2, y2)], schedule)
    assert stats.batches == 2

    metrics = trainer.curvature_metrics()
    assert isinstance(metrics, dict)
    assert "raw_pressure" in metrics
    assert "smoothed_pressure" in metrics
    assert "curvature" in metrics
    assert trainer.curvature == pytest.approx(float(metrics["curvature"]))

    trainer.disable_curvature_scheduler()
    assert trainer.curvature_metrics() is None


def test_curvature_scheduler_exposes_advanced_knobs() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "CurvatureScheduler")

    scheduler = st.nn.CurvatureScheduler(
        initial=-1.0,
        min_curvature=-2.0,
        max_curvature=-0.2,
        target_pressure=0.05,
        step=0.1,
        tolerance=0.02,
        smoothing=0.3,
    )
    scheduler.set_proportional_gain(1.4)
    scheduler.set_stability_threshold(0.002)
    scheduler.set_stability_boost(0.25)
    scheduler.set_dither(0.2, 7)
    scheduler.apply_env_overrides()

    assert scheduler.proportional_gain == pytest.approx(1.4)
    assert scheduler.stability_threshold == pytest.approx(0.002)
    assert scheduler.stability_boost == pytest.approx(0.25)
    assert scheduler.dither_strength == pytest.approx(0.2)
    assert scheduler.dither_period == 7

    _ = scheduler.observe_pressure(0.2)
    _ = scheduler.observe_pressure(0.21)
    _ = scheduler.last_pressure_variance
    _ = scheduler.last_pressure_rel_variance


def test_module_trainer_spectral_and_coherence_bridge_controls() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "ModuleTrainer")
    assert hasattr(st.nn, "SpectralLearningRatePolicy")
    assert hasattr(st, "SpectralLearningRatePolicy")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    policy = st.nn.SpectralLearningRatePolicy(
        smoothing=0.3,
        event_smoothing=0.8,
        turnover_smoothing=0.2,
        phase_gain=0.3,
        stuck_phase_gain=0.4,
        stuck_turnover_threshold=0.1,
        coherence_gain=0.6,
        sheet_gain=0.7,
        spin_gain=0.5,
        radius_gain=0.4,
        energy_gain=0.3,
        lr_bounds=(0.2, 4.0),
        band_bounds=(0.7, 2.2),
        max_lr_step=1.3,
    )
    policy.set_phase_gain(0.35)
    policy.set_coherence_gain(0.55)
    policy.set_lr_bounds(0.3, 3.5)
    policy.apply_env_overrides()

    trainer.enable_spectral_learning_rate(policy)
    assert trainer.spectral_metrics() is None
    trainer.enable_zspace_trace_coherence_bridge()
    trainer.disable_zspace_trace_coherence_bridge()
    trainer.disable_spectral_learning_rate()


def test_module_trainer_spectral_metrics_include_band_snapshot() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    trainer.enable_spectral_learning_rate()

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", 2, 3))
    trainer.prepare(model)
    loss = st.nn.MeanSquaredError()
    schedule = trainer.roundtable(
        1,
        3,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-6),
    )
    batches = [
        (st.Tensor.rand(1, 2, seed=41), st.Tensor.rand(1, 3, seed=42)),
        (st.Tensor.rand(1, 2, seed=43), st.Tensor.rand(1, 3, seed=44)),
    ]

    stats = trainer.train_epoch(model, loss, batches, schedule)
    assert stats.batches == 2

    metrics = trainer.spectral_metrics()
    assert isinstance(metrics, dict)
    assert metrics["source"] == "band_energy"
    assert metrics["adjustment"] is None
    assert metrics["turnover"] == pytest.approx(0.0)
    band = metrics["band"]
    assert isinstance(band, dict)
    assert band["spectral"] is not None
    assert metrics["band_sheet_confidence"] == pytest.approx(
        band["spectral"]["sheet_confidence"]
    )
    assert metrics["band_stability"] >= 0.0


def test_summarize_trainer_trace_events_collects_numeric_metrics(tmp_path) -> None:
    _ensure_torch_stub()
    st = importlib.import_module("spiraltorch")

    trace_path = tmp_path / "trainer_trace.jsonl"
    records = [
        {
            "type": "TrainerStep",
            "payload": {
                "step": 1,
                "metrics": {
                    "step_time_ms": 1.5,
                    "extra": {
                        "band_spin": 0.9,
                        "loss_weighted": 0.3,
                        "optim_state_fallback_lr": 0.01,
                        "optim_state_adapter_avg_energy": 0.2,
                        "optim_accumulator_sync_world_size": 1.0,
                        "optim_accumulator_sync_buffers": 0.0,
                        "coherence_repairs_total": 0.0,
                        "coherence_repaired_detected": 0.0,
                        "coherence_pre_discard_repairs_total": 0.0,
                        "backend_policy_events": 2.0,
                        "backend_policy_wgpu_choices": 1.0,
                        "backend_policy_unison_choices": 1.0,
                        "backend_policy_wasm_tuner_events": 1.0,
                        "backend_policy_wgpu_last_workgroup": 128.0,
                        "backend_policy_wgpu_last_lanes": 16.0,
                        "backend_policy_wgpu_last_compaction_tile": 1024.0,
                        "backend_policy_unison_last_candidate_count": 2.0,
                        "backend_policy_unison_last_best_score": 0.25,
                        "backend_policy_source_wgpu_heuristic_choice_generated": 1.0,
                        "backend_policy_source_unison_rank_choice_fallback": 1.0,
                        "backend_policy_status_kdsl_env_bridge_feature_disabled": 1.0,
                        "backend_policy_status_wasm_tuner_choice_hit": 1.0,
                    },
                },
            },
        },
        {
            "type": "TrainerStep",
            "payload": {
                "step": 2,
                "metrics": {
                    "step_time_ms": 2.5,
                    "extra": {
                        "band_spin": -0.3,
                        "loss_weighted": 0.1,
                        "optim_state_fallback_lr": 0.005,
                        "optim_state_adapter_avg_energy": 0.0,
                        "optim_accumulator_sync_world_size": 2.0,
                        "optim_accumulator_sync_buffers": 3.0,
                        "coherence_repairs_total": 3.0,
                        "coherence_repaired_detected": 1.0,
                        "coherence_pre_discard_repairs_total": 2.0,
                        "backend_policy_events": 3.0,
                        "backend_policy_wgpu_choices": 1.0,
                        "backend_policy_unison_choices": 1.0,
                        "backend_policy_kdsl_env_events": 1.0,
                        "backend_policy_wasm_tuner_events": 1.0,
                        "backend_policy_wgpu_last_workgroup": 256.0,
                        "backend_policy_wgpu_last_lanes": 32.0,
                        "backend_policy_wgpu_last_compaction_tile": 2048.0,
                        "backend_policy_unison_last_candidate_count": 3.0,
                        "backend_policy_unison_last_best_score": 0.75,
                        "backend_policy_source_wgpu_heuristic_choice_generated": 1.0,
                        "backend_policy_source_unison_rank_choice_wgpu_generated": 1.0,
                        "backend_policy_status_kdsl_env_bridge_evaluated": 1.0,
                    },
                },
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    summary = st.summarize_trainer_trace_events(trace_path)
    assert summary["count"] == 2
    assert summary["first_step"] == 1
    assert summary["last_step"] == 2
    assert summary["metrics"]["band_spin"]["first"] == pytest.approx(0.9)
    assert summary["metrics"]["band_spin"]["last"] == pytest.approx(-0.3)
    assert summary["metrics"]["band_spin"]["samples"] == 2
    assert summary["metrics"]["band_spin"]["nonzero"] == 2
    assert summary["metrics"]["loss_weighted"]["mean"] == pytest.approx(0.2)
    assert summary["metrics"]["optim_state_fallback_lr"]["first"] == pytest.approx(0.01)
    assert summary["metrics"]["optim_state_fallback_lr"]["last"] == pytest.approx(0.005)
    assert summary["metrics"]["optim_state_adapter_avg_energy"]["nonzero"] == 1
    assert summary["metrics"]["optim_accumulator_sync_world_size"]["last"] == pytest.approx(2.0)
    assert summary["metrics"]["optim_accumulator_sync_buffers"]["nonzero"] == 1
    assert summary["metrics"]["coherence_repairs_total"]["max"] == pytest.approx(3.0)
    assert summary["metrics"]["coherence_repairs_total"]["nonzero"] == 1
    assert summary["coherence_repairs"]["total_nonzero_steps"] == 1
    assert summary["coherence_repairs"]["detected_steps"] == 1
    assert summary["coherence_repairs"]["max_pre_discard_total"] == pytest.approx(2.0)
    assert summary["metrics"]["backend_policy_events"]["sum"] == pytest.approx(5.0)
    policy = summary["backend_policy"]
    assert policy["counts"]["events"] == pytest.approx(5.0)
    assert policy["counts"]["wgpu_choices"] == pytest.approx(2.0)
    assert policy["counts"]["unison_choices"] == pytest.approx(2.0)
    assert policy["counts"]["kdsl_env_events"] == pytest.approx(1.0)
    assert policy["counts"]["wasm_tuner_events"] == pytest.approx(2.0)
    assert policy["last"]["wgpu_last_workgroup"] == pytest.approx(256.0)
    assert policy["last"]["wgpu_last_lanes"] == pytest.approx(32.0)
    assert policy["last"]["unison_last_candidate_count"] == pytest.approx(3.0)
    assert policy["last"]["unison_last_best_score"] == pytest.approx(0.75)
    assert (
        policy["source_counts"]["wgpu_heuristic_choice_generated"] == pytest.approx(2.0)
    )
    assert (
        policy["source_counts"]["unison_rank_choice_wgpu_generated"] == pytest.approx(1.0)
    )
    assert (
        policy["status_counts"]["kdsl_env_bridge_feature_disabled"] == pytest.approx(1.0)
    )
    assert policy["status_counts"]["kdsl_env_bridge_evaluated"] == pytest.approx(1.0)
    assert policy["status_counts"]["wasm_tuner_choice_hit"] == pytest.approx(1.0)


def test_summarize_trainer_trace_events_reads_plugin_writer_policy_events(tmp_path) -> None:
    _ensure_torch_stub()
    st = importlib.import_module("spiraltorch")

    trace_path = tmp_path / "trainer_trace.jsonl"
    records = [
        {
            "idx": 1,
            "elapsed_ms": 1,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "wasm_tuner_choice",
                        "data": {
                            "backend": "wgpu",
                            "requested_backend": "auto",
                            "status": "miss",
                        },
                    },
                },
            },
        },
        {
            "idx": 2,
            "elapsed_ms": 2,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "unison_rank_choice",
                        "data": {
                            "backend": "wgpu",
                            "requested_backend": "wgpu",
                            "choice_source": "fallback",
                            "candidate_count": 1,
                            "best_score": 1.25,
                            "baseline_score": 1.25,
                            "wgpu_generated_score": 0.0,
                            "wgpu_generated_score_delta": -1.25,
                        },
                    },
                },
            },
        },
        {
            "idx": 3,
            "elapsed_ms": 3,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TrainerStep",
                    "data": {
                        "step": 7,
                        "metrics": {
                            "step_time_ms": 0.5,
                            "extra": {"loss_weighted": 0.125},
                        },
                    },
                },
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    summary = st.summarize_trainer_trace_events(trace_path)
    assert summary["count"] == 1
    assert summary["first_step"] == 7
    assert summary["metrics"]["loss_weighted"]["last"] == pytest.approx(0.125)
    policy = summary["backend_policy"]
    assert policy["counts"]["events"] == pytest.approx(2.0)
    assert policy["counts"]["unison_choices"] == pytest.approx(1.0)
    assert policy["counts"]["wasm_tuner_events"] == pytest.approx(1.0)
    assert policy["last"]["unison_last_candidate_count"] == pytest.approx(1.0)
    assert policy["last"]["unison_last_best_score"] == pytest.approx(1.25)
    assert policy["last"]["unison_last_baseline_score"] == pytest.approx(1.25)
    assert policy["last"]["unison_last_wgpu_generated_score"] == pytest.approx(0.0)
    assert policy["last"]["unison_last_wgpu_generated_score_delta"] == pytest.approx(-1.25)
    assert policy["source_counts"]["unison_rank_choice_fallback"] == pytest.approx(1.0)
    assert policy["status_counts"]["wasm_tuner_choice_miss"] == pytest.approx(1.0)


def test_summarize_trainer_trace_events_recovers_wgpu_runtime_fallbacks(tmp_path) -> None:
    _ensure_torch_stub()
    st = importlib.import_module("spiraltorch")

    trace_path = tmp_path / "trainer_trace.jsonl"
    records = [
        {
            "idx": 1,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "matmul",
                        "data": {
                            "backend": "naive",
                            "requested_backend": "wgpu",
                            "fallback": {
                                "from": "wgpu",
                                "reason": "runtime_unavailable",
                            },
                        },
                    },
                },
            },
        },
        {
            "idx": 2,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "matmul_prepacked_bias",
                        "data": {
                            "backend": "wgpu",
                            "requested_backend": "wgpu",
                        },
                    },
                },
            },
        },
        {
            "idx": 3,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TrainerStep",
                    "data": {
                        "step": 1,
                        "metrics": {
                            "step_time_ms": 0.5,
                        },
                    },
                },
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    summary = st.summarize_trainer_trace_events(trace_path)
    metrics = summary["metrics"]
    assert metrics["tensor_backend_requested_wgpu_runtime_fallbacks"][
        "last"
    ] == pytest.approx(1.0)
    assert metrics["tensor_op_backend_wgpu_runtime_fallback_matmul_naive"][
        "last"
    ] == pytest.approx(1.0)
    assert metrics["tensor_backend_requested_wgpu_hits"]["last"] == pytest.approx(1.0)
    assert metrics["tensor_op_backend_requested_wgpu_hit_matmul_prepacked_bias_wgpu"][
        "last"
    ] == pytest.approx(1.0)


def test_summarize_trainer_trace_events_recovers_lstm_estimated_work(tmp_path) -> None:
    _ensure_torch_stub()
    st = importlib.import_module("spiraltorch")

    trace_path = tmp_path / "trainer_trace.jsonl"
    records = [
        {
            "idx": 1,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_forward",
                        "data": {
                            "estimated_gate_activation_ops": 10,
                        },
                    },
                },
            },
        },
        {
            "idx": 2,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_forward",
                        "data": {
                            "gate_activation_backend": "wgpu",
                            "estimated_gate_activation_ops": 6,
                        },
                    },
                },
            },
        },
        {
            "idx": 3,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_backward",
                        "data": {
                            "bptt_backend": "cpu",
                            "estimated_gate_activation_ops": 7,
                            "estimated_bptt_ops": 100,
                            "estimated_bptt_gate_derivative_ops": 40,
                            "estimated_bptt_cell_recurrence_ops": 20,
                            "estimated_bptt_state_carry_ops": 10,
                            "estimated_bptt_scan_steps": 5,
                        },
                    },
                },
            },
        },
        {
            "idx": 4,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_backward",
                        "data": {
                            "gate_activation_backend": "wgpu",
                            "estimated_gate_activation_ops": 9,
                            "bptt_backend": "wgpu",
                            "estimated_bptt_ops": 80,
                            "estimated_bptt_wgpu_ops": 80,
                            "bptt_scan_runtime_requested": True,
                            "bptt_scan_runtime_available": True,
                        },
                    },
                },
            },
        },
        {
            "idx": 5,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TrainerStep",
                    "data": {
                        "step": 1,
                        "metrics": {
                            "step_time_ms": 0.5,
                        },
                    },
                },
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    summary = st.summarize_trainer_trace_events(trace_path)
    metrics = summary["metrics"]
    assert metrics["lstm_forward_estimated_gate_activation_ops"]["last"] == pytest.approx(
        16.0
    )
    assert metrics["lstm_forward_estimated_gate_activation_cpu_debt_ops"][
        "last"
    ] == pytest.approx(10.0)
    assert metrics["lstm_forward_estimated_gate_activation_wgpu_ops"][
        "last"
    ] == pytest.approx(6.0)
    assert metrics["lstm_backward_estimated_gate_activation_ops"]["last"] == pytest.approx(
        16.0
    )
    assert metrics["lstm_backward_estimated_gate_activation_cpu_debt_ops"][
        "last"
    ] == pytest.approx(7.0)
    assert metrics["lstm_backward_estimated_gate_activation_wgpu_ops"][
        "last"
    ] == pytest.approx(9.0)
    assert metrics["lstm_estimated_gate_activation_ops"]["last"] == pytest.approx(32.0)
    assert metrics["lstm_estimated_gate_activation_cpu_debt_ops"][
        "last"
    ] == pytest.approx(17.0)
    assert metrics["lstm_estimated_gate_activation_wgpu_ops"]["last"] == pytest.approx(
        15.0
    )
    assert metrics["lstm_backward_estimated_bptt_ops"]["last"] == pytest.approx(180.0)
    assert metrics["lstm_backward_estimated_bptt_cpu_debt_ops"]["last"] == pytest.approx(
        100.0
    )
    assert metrics["lstm_backward_estimated_bptt_wgpu_ops"]["last"] == pytest.approx(80.0)
    assert metrics["lstm_estimated_bptt_cpu_debt_ops"]["last"] == pytest.approx(100.0)
    assert metrics["lstm_estimated_bptt_wgpu_ops"]["last"] == pytest.approx(80.0)
    assert metrics["lstm_backward_estimated_bptt_gate_derivative_ops"][
        "last"
    ] == pytest.approx(40.0)
    assert metrics["lstm_backward_estimated_bptt_cell_recurrence_ops"][
        "last"
    ] == pytest.approx(20.0)
    assert metrics["lstm_backward_estimated_bptt_state_carry_ops"][
        "last"
    ] == pytest.approx(10.0)
    assert metrics["lstm_backward_estimated_bptt_scan_steps"]["last"] == pytest.approx(5.0)
    assert metrics["lstm_estimated_cpu_debt_ops"]["last"] == pytest.approx(117.0)


def test_summarize_trainer_trace_events_reports_zero_lstm_debt_for_wgpu_scan(
    tmp_path,
) -> None:
    _ensure_torch_stub()
    st = importlib.import_module("spiraltorch")

    trace_path = tmp_path / "trainer_trace.jsonl"
    records = [
        {
            "idx": 1,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_forward",
                        "data": {
                            "gate_activation_backend": "wgpu",
                            "estimated_gate_activation_ops": 64,
                        },
                    },
                },
            },
        },
        {
            "idx": 2,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TensorOpMeta",
                    "data": {
                        "op_name": "lstm_backward",
                        "data": {
                            "gate_activation_backend": "wgpu",
                            "bptt_backend": "wgpu",
                            "estimated_gate_activation_ops": 48,
                            "estimated_bptt_ops": 96,
                            "estimated_bptt_wgpu_ops": 96,
                            "estimated_bptt_ops_per_scan_step": 32,
                            "bptt_scan_elapsed_us": 123,
                            "bptt_scan_hidden_values": 12,
                            "bptt_scan_gate_values": 48,
                            "bptt_scan_cell_values": 16,
                            "bptt_scan_recurrent_weight_values": 48,
                            "bptt_scan_scratch_values": 8,
                            "bptt_scan_kernel_dispatches": 1,
                            "bptt_scan_serial_steps": 3,
                            "bptt_scan_workgroup_size": 64,
                            "bptt_scan_parallel_lanes": 64,
                            "bptt_scan_runtime_requested": True,
                            "bptt_scan_runtime_available": True,
                        },
                    },
                },
            },
        },
        {
            "idx": 3,
            "event": {
                "kind": "Custom",
                "data": {
                    "event_type": "TrainerStep",
                    "data": {
                        "step": 1,
                        "metrics": {
                            "step_time_ms": 0.25,
                        },
                    },
                },
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    summary = st.summarize_trainer_trace_events(trace_path)
    metrics = summary["metrics"]
    assert metrics["lstm_estimated_cpu_debt_ops"]["last"] == pytest.approx(0.0)
    assert metrics["lstm_estimated_gate_activation_cpu_debt_ops"][
        "last"
    ] == pytest.approx(0.0)
    assert metrics["lstm_estimated_gate_activation_wgpu_ops"]["last"] == pytest.approx(
        112.0
    )
    assert metrics["lstm_estimated_bptt_cpu_debt_ops"]["last"] == pytest.approx(0.0)
    assert metrics["lstm_estimated_bptt_wgpu_ops"]["last"] == pytest.approx(96.0)
    assert metrics["lstm_backward_bptt_scan_elapsed_us"]["last"] == pytest.approx(123.0)
    assert metrics["lstm_backward_bptt_scan_hidden_values"]["last"] == pytest.approx(12.0)
    assert metrics["lstm_backward_bptt_scan_gate_values"]["last"] == pytest.approx(48.0)
    assert metrics["lstm_backward_bptt_scan_cell_values"]["last"] == pytest.approx(16.0)
    assert metrics["lstm_backward_bptt_scan_recurrent_weight_values"][
        "last"
    ] == pytest.approx(48.0)
    assert metrics["lstm_backward_bptt_scan_scratch_values"]["last"] == pytest.approx(8.0)
    assert metrics["lstm_backward_bptt_scan_kernel_dispatches"]["last"] == pytest.approx(
        1.0
    )
    assert metrics["lstm_backward_bptt_scan_serial_steps"]["last"] == pytest.approx(3.0)
    assert metrics["lstm_backward_bptt_scan_workgroup_size"]["last"] == pytest.approx(
        64.0
    )
    assert metrics["lstm_backward_bptt_scan_parallel_lanes"]["last"] == pytest.approx(
        64.0
    )
    assert metrics["lstm_backward_estimated_bptt_ops_per_scan_step"][
        "last"
    ] == pytest.approx(32.0)

from __future__ import annotations

import copy
import math
from pathlib import Path
import runpy
import threading
import time

import pytest

import spiraltorch as st

JS_MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _config(*, dimension: int = 4, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "dimension": dimension,
        "fractional_order": 0.35,
        "weights": {
            "speed": 0.5,
            "memory": 0.3,
            "stability": 0.2,
            "fractional": 0.1,
            "drift_response": 0.0,
        },
        "learning_rate": 1e-2,
        "first_moment_decay": 0.9,
        "second_moment_decay": 0.999,
        "epsilon": 1e-8,
        "topos_control_gain": 0.0,
        "gradient_projection": "tile_or_truncate",
    }
    config.update(overrides)
    return config


def _observation() -> dict[str, object]:
    return {
        "speed": 0.8,
        "memory": 0.5,
        "stability": 0.6,
        "drift_response": 0.0,
        "gradient": [0.1, -0.2, 0.3, -0.1],
        "telemetry": {},
    }


def test_init_returns_rust_owned_versioned_checkpoint() -> None:
    checkpoint = st.zspace_meta_optimizer_init(_config())

    assert checkpoint["contract_version"] == st.ZSPACE_META_OPTIMIZER_CONTRACT_VERSION
    assert checkpoint["kind"] == st.ZSPACE_META_OPTIMIZER_KIND
    assert checkpoint["semantic_owner"] == st.ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER
    assert checkpoint["semantic_backend"] == "rust"
    assert checkpoint["state"] == {
        "z": [0.0] * 4,
        "first_moment": [0.0] * 4,
        "second_moment": [0.0] * 4,
        "step": 0,
    }


def test_step_counter_fails_before_cross_client_precision_diverges() -> None:
    checkpoint = st.zspace_meta_optimizer_init(_config(dimension=2))
    checkpoint["state"]["step"] = JS_MAX_SAFE_INTEGER

    with pytest.raises(ValueError, match="cross-client maximum"):
        st.zspace_meta_optimizer_step(
            config=checkpoint["config"],
            state=checkpoint["state"],
            observation={"gradient": [0.1, -0.2]},
        )


def test_public_trainer_and_direct_transition_are_identical() -> None:
    checkpoint = st.zspace_meta_optimizer_init(_config())
    direct = st.zspace_meta_optimizer_step(
        config=checkpoint["config"],
        state=checkpoint["state"],
        observation=_observation(),
    )
    trainer = st.ZSpaceTrainer(z_dim=4)

    objective = trainer.step(
        {
            "speed": 0.8,
            "memory": 0.5,
            "stability": 0.6,
            "gradient": [0.1, -0.2, 0.3, -0.1],
        }
    )

    assert objective == direct["objective"]["objective_before"]
    assert trainer.state == direct["state_after"]["z"]
    assert trainer.state_dict()["moment"] == direct["state_after"]["first_moment"]
    assert trainer.state_dict()["velocity"] == direct["state_after"]["second_moment"]
    assert trainer.last_optimizer_report == direct
    assert not hasattr(trainer, "_frac_reg")
    assert not hasattr(trainer, "_adam_update")


def test_failed_transition_preserves_python_state_and_cached_report() -> None:
    trainer = st.ZSpaceTrainer(z_dim=3)
    trainer.step({"gradient": [0.2, -0.1, 0.05]})
    state_before = copy.deepcopy(trainer.state_dict())
    report_before = copy.deepcopy(trainer.last_optimizer_report)

    with pytest.raises((TypeError, ValueError), match="finite|JSON"):
        trainer.step({"speed": math.nan, "gradient": [0.1, 0.2, 0.3]})

    assert trainer.state_dict() == state_before
    assert trainer.last_optimizer_report == report_before


def test_tagged_latent_gradient_requires_the_exact_state_dimension() -> None:
    trainer = st.ZSpaceTrainer(z_dim=4)
    state_before = copy.deepcopy(trainer.state_dict())

    with pytest.raises(
        ValueError,
        match="latent-coordinate gradient dimension must match Z state",
    ):
        trainer.step(
            st.ZMetrics(
                speed=0.0,
                memory=0.0,
                stability=0.0,
                gradient=[0.1, -0.2],
                gradient_basis=st.ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS,
            )
        )

    assert trainer.state_dict() == state_before
    assert trainer.last_optimizer_report is None


def test_untagged_legacy_gradient_keeps_configured_projection_behavior() -> None:
    trainer = st.ZSpaceTrainer(z_dim=4)

    trainer.step({"gradient": [0.1, -0.2]})

    assert trainer.state_dict()["step"] == 1
    assert trainer.last_optimizer_report is not None
    assert trainer.last_optimizer_report["gradient"]["projected_normalized"] == (
        pytest.approx(
            [math.tanh(0.1), math.tanh(-0.2), math.tanh(0.1), math.tanh(-0.2)]
        )
    )


def test_malformed_native_checkpoint_is_rejected_before_python_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = st.ZSpaceTrainer(z_dim=3)
    state_before = copy.deepcopy(trainer.state_dict())
    native_step = st.zspace_meta_optimizer_step

    def malformed_transition(**request: object) -> dict[str, object]:
        report = native_step(**request)  # type: ignore[arg-type]
        report["config"]["weights"].pop("speed")
        return report

    monkeypatch.setattr(st, "zspace_meta_optimizer_step", malformed_transition)

    with pytest.raises(KeyError, match="speed"):
        trainer.step({"gradient": [0.2, -0.1, 0.05]})

    assert trainer.state_dict() == state_before
    assert trainer.last_optimizer_report is None


def test_failed_partial_step_preserves_previous_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = st.ZSpaceTrainer(z_dim=3)
    previous = trainer.infer_partial({"speed": 0.2, "memory": 0.3})
    state_before = copy.deepcopy(trainer.state_dict())

    def fail_transition(**_: object) -> dict[str, object]:
        raise ValueError("injected Rust transition failure")

    monkeypatch.setattr(st, "zspace_meta_optimizer_step", fail_transition)

    with pytest.raises(ValueError, match="injected Rust transition failure"):
        trainer.step_partial({"speed": 0.8, "memory": 0.9})

    assert trainer.state_dict() == state_before
    assert trainer.last_inference == previous


def test_trainer_serializes_native_transition_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = st.ZSpaceTrainer(z_dim=2)
    native_step = st.zspace_meta_optimizer_step
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def delayed_transition(**request: object) -> dict[str, object]:
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        try:
            return native_step(**request)  # type: ignore[arg-type]
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(st, "zspace_meta_optimizer_step", delayed_transition)
    threads = [
        threading.Thread(target=trainer.step, args=({"gradient": [0.1, -0.2]},))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum_active == 1
    assert trainer.state_dict()["step"] == 2


def test_restore_is_transactional_and_rejects_negative_second_moment() -> None:
    trainer = st.ZSpaceTrainer(z_dim=2)
    before = trainer.state_dict()

    with pytest.raises(ValueError, match="second_moment"):
        trainer.load_state_dict(
            {
                "z": [0.1, -0.2],
                "moment": [0.0, 0.0],
                "velocity": [0.1, -0.1],
                "step": 4,
            }
        )

    assert trainer.state_dict() == before


def test_restore_rejects_unknown_checkpoint_contract() -> None:
    trainer = st.ZSpaceTrainer(z_dim=2)
    checkpoint = trainer.state_dict()
    checkpoint["contract_version"] = "spiraltorch.zspace_meta_optimizer.v999"

    with pytest.raises(ValueError, match="unsupported.*contract"):
        trainer.load_state_dict(checkpoint)


def test_non_strict_restore_uses_rust_dimension_coercion() -> None:
    trainer = st.ZSpaceTrainer(z_dim=3)
    trainer.load_state_dict(
        {
            "z": [1.0],
            "moment": [2.0, 3.0, 4.0, 5.0],
            "velocity": [],
            "step": 7,
        },
        strict=False,
    )

    state = trainer.state_dict()
    assert state["z"] == [1.0, 0.0, 0.0]
    assert state["moment"] == [2.0, 3.0, 4.0]
    assert state["velocity"] == [0.0, 0.0, 0.0]
    assert state["step"] == 7


def test_exact_gradient_projection_fails_closed() -> None:
    trainer = st.ZSpaceTrainer(z_dim=3, gradient_projection="exact")
    before = trainer.state_dict()

    with pytest.raises(ValueError, match="gradient must contain 3 values"):
        trainer.step({"gradient": [0.1, 0.2]})

    assert trainer.state_dict() == before

    with pytest.raises(ValueError, match="gradient must contain 3 values"):
        trainer.step({})

    assert trainer.state_dict() == before


def test_legacy_mutable_z_view_tracks_native_steps_and_reset() -> None:
    legacy_module = runpy.run_path(
        str(Path(__file__).parents[3] / "tools/python/zspace_trainer.py")
    )
    LegacyZSpaceTrainer = legacy_module["ZSpaceTrainer"]

    trainer = LegacyZSpaceTrainer(z_dim=2)
    z_view = trainer.z
    z_view[0] = 0.25

    trainer.step({"gradient": [0.1, -0.2]})
    assert trainer.z is z_view
    assert z_view == trainer.state

    trainer.reset()
    assert trainer.z is z_view
    assert z_view == [0.0, 0.0]

    trainer.z = [0.4, -0.3]
    assert trainer.z is z_view
    assert z_view == [0.4, -0.3]


def test_topos_hints_control_actual_adam_rate_and_fractional_weight() -> None:
    trainer = st.ZSpaceTrainer(
        z_dim=4,
        lam_frac=0.2,
        topos_control_gain=0.5,
    )
    trainer.load_state_dict(
        {
            "z": [0.2, -0.1, 0.4, -0.3],
            "moment": [0.0] * 4,
            "velocity": [0.0] * 4,
            "step": 0,
        }
    )
    metrics = st.z_metrics(
        speed=0.0,
        memory=0.0,
        stability=0.0,
        gradient=[0.1, -0.2, 0.3, -0.1],
        telemetry={
            "topos": {
                "closure_pressure": 0.75,
                "training_hints": {
                    "learning_rate_scale": 0.5,
                    "regularization_scale": 2.0,
                    "clip_scale": 0.8,
                },
            }
        },
    )

    trainer.step(metrics)
    report = trainer.last_optimizer_report

    assert report is not None
    control = report["topos_control"]
    assert control["active"] is True
    assert control["learning_rate_scale"] == pytest.approx(0.75)
    assert control["effective_learning_rate"] == pytest.approx(0.0075)
    assert control["regularization_scale"] == pytest.approx(1.5)
    assert control["effective_fractional_weight"] == pytest.approx(0.3)
    assert report["adam"]["effective_learning_rate"] == pytest.approx(0.0075)


def test_optimizer_report_property_returns_a_detached_copy() -> None:
    trainer = st.ZSpaceTrainer(z_dim=2)
    trainer.step({"gradient": [0.1, -0.2]})
    report = trainer.last_optimizer_report
    assert report is not None
    report["state_after"]["z"][0] = 999.0

    assert trainer.last_optimizer_report["state_after"]["z"][0] != 999.0


def test_public_optimizer_contract_is_exported_and_stubbed() -> None:
    for name in (
        "ZSPACE_META_OBJECTIVE_FORMULA",
        "ZSPACE_META_OPTIMIZER_CONTRACT_VERSION",
        "ZSPACE_META_OPTIMIZER_KIND",
        "ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND",
        "ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER",
        "zspace_meta_optimizer_init",
        "zspace_meta_optimizer_restore",
        "zspace_meta_optimizer_step",
    ):
        assert name in st.__all__

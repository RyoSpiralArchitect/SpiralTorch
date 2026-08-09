from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import spiraltorch as st
from spiraltorch import hf_cli
from spiraltorch.hf_optimizer_control import (
    HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES,
    HF_ZSPACE_OPTIMIZER_STATE_FILENAME,
    hf_zspace_optimizer_control_callback,
)

BRIDGE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_bridge.py"
)


def _load_bridge_example():
    spec = importlib.util.spec_from_file_location(
        "hf_optimizer_control_bridge_test",
        BRIDGE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256_id(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class _HookHandle:
    def __init__(self, hooks: list[object], hook: object) -> None:
        self.hooks = hooks
        self.hook = hook

    def remove(self) -> None:
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class _FakeOptimizer:
    def __init__(self, learning_rate: float = 1.0e-3) -> None:
        self.param_groups = [{"lr": learning_rate}]
        self.pre_hooks: list[object] = []
        self.post_hooks: list[object] = []
        self.step_learning_rates: list[list[float]] = []

    def register_step_pre_hook(self, hook: object) -> _HookHandle:
        self.pre_hooks.append(hook)
        return _HookHandle(self.pre_hooks, hook)

    def register_step_post_hook(self, hook: object) -> _HookHandle:
        self.post_hooks.append(hook)
        return _HookHandle(self.post_hooks, hook)

    def step(self) -> None:
        args: tuple[object, ...] = ()
        kwargs: dict[str, object] = {}
        for hook in list(self.pre_hooks):
            hook(self, args, kwargs)
        self.step_learning_rates.append(
            [float(group["lr"]) for group in self.param_groups]
        )
        for hook in list(self.post_hooks):
            hook(self, args, kwargs)


class _FakeAcceleratedOptimizer:
    def __init__(self, optimizer: _FakeOptimizer) -> None:
        self.optimizer = optimizer

    def register_step_pre_hook(self, hook: object) -> object:
        raise AttributeError("wrapper hook storage is not initialized")

    def register_step_post_hook(self, hook: object) -> object:
        raise AttributeError("wrapper hook storage is not initialized")


class _FakeFeedbackCore:
    def __init__(self) -> None:
        self.control_calls = 0
        self.observation_calls = 0
        self.restore_calls = 0
        self.default_config = {
            "loss_ema_alpha": 0.2,
            "relative_delta_ema_alpha": 0.5,
            "loss_floor": 1e-8,
            "regression_threshold": 0.01,
            "halt_threshold": 0.05,
            "recovery_threshold": 0.0025,
            "attenuation_rate": 0.25,
            "recovery_rate": 0.125,
            "halt_regression_streak": 2,
            "resume_improvement_streak": 2,
            "warmup_observations": 2,
            "max_stale_updates": 0,
            "maximum_gate": 1.0,
        }

    @staticmethod
    def _state() -> dict[str, object]:
        return {
            "control_step": 0,
            "observation_count": 0,
            "last_observation_step": None,
            "last_loss": None,
            "loss_ema": None,
            "relative_loss_delta_ema": None,
            "gate": 0.0,
            "regression_streak": 0,
            "improvement_streak": 0,
            "halted": False,
        }

    @staticmethod
    def _base(config: dict[str, object]) -> dict[str, object]:
        return {
            "contract_version": st.ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
            "kind": st.ZSPACE_OPTIMIZER_FEEDBACK_KIND,
            "semantic_owner": st.ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
            "semantic_backend": "rust",
            "config": copy.deepcopy(config),
        }

    def init(self, config: dict[str, object] | None = None) -> dict[str, object]:
        resolved = {**self.default_config, **dict(config or {})}
        return {**self._base(resolved), "state": self._state()}

    def restore(
        self,
        *,
        config: dict[str, object],
        state: dict[str, object],
    ) -> dict[str, object]:
        self.restore_calls += 1
        return {**self._base(config), "state": copy.deepcopy(state)}

    def control(
        self,
        *,
        config: dict[str, object],
        state: dict[str, object],
        target_step: int,
        proposed_learning_rate_scale: float,
    ) -> dict[str, object]:
        self.control_calls += 1
        if target_step != int(state["control_step"]) + 1:
            raise ValueError("control target is not the next step")
        gate = float(state["gate"])
        applied_scale = 1.0 + gate * (proposed_learning_rate_scale - 1.0)
        state_after = copy.deepcopy(state)
        state_after["control_step"] = target_step
        return {
            **self._base(config),
            "control_rule": st.ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE,
            "transition_validated": True,
            "target_step": target_step,
            "proposed_learning_rate_scale": proposed_learning_rate_scale,
            "proposed_deviation_from_identity": (proposed_learning_rate_scale - 1.0),
            "feedback_observation_age_updates": (
                None
                if state["last_observation_step"] is None
                else int(state["control_step"]) - int(state["last_observation_step"])
            ),
            "feedback_gate": gate,
            "effective_feedback_gate": gate,
            "applied_learning_rate_scale": applied_scale,
            "applied_deviation_from_identity": applied_scale - 1.0,
            "identity_applied": applied_scale == 1.0,
            "disposition": "active" if gate > 0.0 else "no_feedback",
            "state_before": copy.deepcopy(state),
            "state_after": state_after,
        }

    def observe(
        self,
        *,
        config: dict[str, object],
        state: dict[str, object],
        observation: dict[str, object],
    ) -> dict[str, object]:
        self.observation_calls += 1
        step = int(observation["step"])
        if step != int(state["control_step"]):
            raise ValueError("observation does not match completed control step")
        state_after = copy.deepcopy(state)
        state_after.update(
            {
                "observation_count": int(state["observation_count"]) + 1,
                "last_observation_step": step,
                "last_loss": float(observation["loss"]),
                "loss_ema": float(observation["loss"]),
                "relative_loss_delta_ema": -0.1,
                "gate": 0.5,
                "improvement_streak": 1,
            }
        )
        return {
            **self._base(config),
            "transition_validated": True,
            "observation": copy.deepcopy(observation),
            "projection": {
                "semantic_backend": "rust",
                "loss": float(observation["loss"]),
            },
            "relative_loss_delta": -0.1,
            "relative_loss_delta_ema": -0.1,
            "action": "recover",
            "gate_before": state["gate"],
            "gate_after": 0.5,
            "state_before": copy.deepcopy(state),
            "state_after": state_after,
        }


def _install_fake_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeFeedbackCore:
    core = _FakeFeedbackCore()
    monkeypatch.setattr(st, "zspace_optimizer_feedback_init", core.init)
    monkeypatch.setattr(st, "zspace_optimizer_feedback_restore", core.restore)
    monkeypatch.setattr(st, "zspace_optimizer_feedback_control", core.control)
    monkeypatch.setattr(st, "zspace_optimizer_feedback_observe", core.observe)
    return core


def _fake_transformers() -> types.ModuleType:
    module = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    module.TrainerCallback = TrainerCallback
    return module


def _run_two_steps(
    tmp_path: Path,
    *,
    mode: str,
    control_gain: float = 1.0,
    learning_rate: float = 1.0e-3,
    trajectory_arm: str = "raw",
    trajectory: Path | dict[str, object] | None = None,
) -> tuple[object, _FakeOptimizer]:
    optimizer = _FakeOptimizer(learning_rate=learning_rate)
    args = types.SimpleNamespace(output_dir=str(tmp_path / mode))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(
            mode=mode,
            control_gain=control_gain,
            trajectory_arm=trajectory_arm,
            trajectory=trajectory,
            trace_path=tmp_path / f"{mode}.jsonl",
        )
    callback.on_train_begin(args, state, control, optimizer=optimizer)
    for completed_step in (1, 2):
        callback.on_step_begin(args, state, control, optimizer=optimizer)
        optimizer.step()
        state.global_step = completed_step
        callback.on_step_end(args, state, control, optimizer=optimizer)
    callback.on_train_end(args, state, control, optimizer=optimizer)
    return callback, optimizer


def test_apply_temporarily_scales_optimizer_lr_and_restores_nominal_value(
    tmp_path: Path,
) -> None:
    callback, optimizer = _run_two_steps(tmp_path, mode="apply")
    receipt = callback.receipt()

    assert receipt["status"] == "ready"
    assert receipt["model_update_intervened"] is True
    assert receipt["applied_update_count"] == 2
    assert receipt["restored_update_count"] == 2
    assert receipt["scheduler_nominal_lr_restored"] is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.step_learning_rates[0][0] != pytest.approx(1.0e-3)
    assert optimizer.step_learning_rates[1][0] != pytest.approx(1.0e-3)
    assert optimizer.pre_hooks == []
    assert optimizer.post_hooks == []


def test_zero_gain_records_hook_actuation_without_claiming_an_intervention(
    tmp_path: Path,
) -> None:
    callback, optimizer = _run_two_steps(
        tmp_path,
        mode="apply",
        control_gain=0.0,
    )
    receipt = callback.receipt()

    assert receipt["status"] == "ready"
    assert receipt["applied_update_count"] == 2
    assert receipt["non_identity_update_count"] == 0
    assert receipt["model_update_intervened"] is False
    assert optimizer.step_learning_rates == [[1.0e-3], [1.0e-3]]


def test_recipe_requires_negative_curvature() -> None:
    with pytest.raises(ValueError, match="curvature must be negative"):
        st.hf_zspace_optimizer_recipe_contract(curvature=0.0)


def test_apply_binds_the_inner_optimizer_below_an_accelerate_wrapper(
    tmp_path: Path,
) -> None:
    optimizer = _FakeOptimizer()
    wrapped = _FakeAcceleratedOptimizer(optimizer)
    args = types.SimpleNamespace(output_dir=str(tmp_path / "wrapped"))
    state = types.SimpleNamespace(global_step=0, max_steps=1)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(mode="apply")
    callback.on_train_begin(args, state, control, optimizer=wrapped)
    callback.on_step_begin(args, state, control, optimizer=wrapped)
    optimizer.step()
    state.global_step = 1
    callback.on_step_end(args, state, control, optimizer=wrapped)
    callback.on_train_end(args, state, control, optimizer=wrapped)

    assert callback.receipt()["status"] == "ready"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


def test_observe_and_apply_consume_identical_rust_control_sequences(
    tmp_path: Path,
) -> None:
    observed, observed_optimizer = _run_two_steps(tmp_path, mode="observe")
    applied, _ = _run_two_steps(tmp_path, mode="apply")
    observed_receipt = observed.receipt()
    applied_receipt = applied.receipt()

    assert observed_receipt["status"] == "ready"
    assert observed_receipt["observed_update_count"] == 2
    assert observed_receipt["applied_update_count"] == 0
    assert (
        observed_receipt["control_sequence_id"]
        == applied_receipt["control_sequence_id"]
    )
    assert observed_optimizer.step_learning_rates == [[1.0e-3], [1.0e-3]]


def test_loss_guard_commits_one_rust_control_per_completed_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    feedback = _install_fake_feedback(monkeypatch)
    optimizer = _FakeOptimizer()
    args = types.SimpleNamespace(output_dir=str(tmp_path / "feedback-apply"))
    state = types.SimpleNamespace(global_step=0, max_steps=2, epoch=0.0)
    control = types.SimpleNamespace()
    callback = hf_zspace_optimizer_control_callback(
        mode="apply",
        feedback_mode="loss_guard",
    )

    callback.on_train_begin(args, state, control, optimizer=optimizer)
    callback.on_step_begin(args, state, control, optimizer=optimizer)
    optimizer.step()
    assert feedback.control_calls == 1
    state.global_step = 1
    callback.on_step_end(args, state, control, optimizer=optimizer)
    callback.on_log(
        args,
        state,
        control,
        logs={"loss": 2.0, "learning_rate": 1e-3},
    )

    callback.on_step_begin(args, state, control, optimizer=optimizer)
    optimizer.step()
    assert feedback.control_calls == 2
    state.global_step = 2
    callback.on_step_end(args, state, control, optimizer=optimizer)
    callback.on_train_end(args, state, control, optimizer=optimizer)

    receipt = callback.receipt()
    assert receipt["status"] == "ready"
    assert optimizer.step_learning_rates[0] == [pytest.approx(1e-3)]
    assert callback.schedule_sequence[1]["feedback_learning_rate_scale"] == (
        pytest.approx(
            1.0
            + 0.5
            * (
                float(callback.schedule_sequence[1]["planned_learning_rate_scale"])
                - 1.0
            )
        )
    )
    assert optimizer.step_learning_rates[1][0] == pytest.approx(
        1e-3 * float(callback.schedule_sequence[1]["feedback_learning_rate_scale"])
    )
    assert receipt["feedback_control_count"] == 2
    assert receipt["feedback_observation_count"] == 1
    assert receipt["feedback_active_update_count"] == 1
    assert receipt["feedback_evidence_boundary"] == (
        "within_run_loss_guard_not_counterfactual_efficacy"
    )
    assert receipt["evidence_boundary"] == (
        "receipt proves control derivation and optimizer actuation, including "
        "Rust-owned trajectory, nominal-LR matching, and checkpointed within-run "
        "loss-feedback gating, not learning-quality improvement or counterfactual "
        "efficacy"
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_loss_guard_resume_restores_exact_feedback_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    feedback = _install_fake_feedback(monkeypatch)
    output = tmp_path / "feedback-resume"
    args = types.SimpleNamespace(output_dir=str(output))
    state = types.SimpleNamespace(global_step=0, max_steps=2, epoch=0.0)
    control = types.SimpleNamespace()
    first_optimizer = _FakeOptimizer()
    first = hf_zspace_optimizer_control_callback(
        mode="apply",
        feedback_mode="loss_guard",
    )
    first.on_train_begin(args, state, control, optimizer=first_optimizer)
    first.on_step_begin(args, state, control, optimizer=first_optimizer)
    first_optimizer.step()
    state.global_step = 1
    first.on_step_end(args, state, control, optimizer=first_optimizer)
    first.on_log(args, state, control, logs={"loss": 2.0})
    first.on_save(args, state, control, optimizer=first_optimizer)
    checkpoint = output / "checkpoint-1"
    first.abort(RuntimeError("simulated process handoff"))

    second_optimizer = _FakeOptimizer()
    resumed_state = types.SimpleNamespace(global_step=1, max_steps=2, epoch=0.5)
    second = hf_zspace_optimizer_control_callback(
        mode="apply",
        feedback_mode="loss_guard",
        resume_from_checkpoint=checkpoint,
    )
    assert feedback.restore_calls == 1
    second.on_train_begin(
        args,
        resumed_state,
        control,
        optimizer=second_optimizer,
    )
    second.on_step_begin(
        args,
        resumed_state,
        control,
        optimizer=second_optimizer,
    )
    second_optimizer.step()
    resumed_state.global_step = 2
    second.on_step_end(
        args,
        resumed_state,
        control,
        optimizer=second_optimizer,
    )
    second.on_train_end(
        args,
        resumed_state,
        control,
        optimizer=second_optimizer,
    )

    receipt = second.receipt()
    assert receipt["status"] == "ready"
    assert receipt["resume_state_loaded"] is True
    assert receipt["feedback_control_count"] == 2
    assert receipt["feedback_observation_count"] == 1
    assert second.feedback_state["control_step"] == 2

    state_path = checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered.pop("state_id")
    tampered["feedback_control_sequence"][0]["hf_target_step"] = 2
    tampered["state_id"] = _sha256_id(tampered)
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lineage is not contiguous"):
        hf_zspace_optimizer_control_callback(
            mode="apply",
            feedback_mode="loss_guard",
            resume_from_checkpoint=checkpoint,
        )


def test_bridge_resolves_feedback_defaults_in_the_rust_owned_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_feedback(monkeypatch)
    bridge = _load_bridge_example()

    args = bridge.parse_args(
        [
            "--training-recipe-only",
            "--zspace-optimizer-control",
            "observe",
            "--zspace-optimizer-feedback",
            "loss_guard",
            "--zspace-optimizer-feedback-maximum-gate",
            "0.75",
        ]
    )

    recipe = args._hf_zspace_optimizer_recipe_contract
    assert recipe["feedback_mode"] == "loss_guard"
    assert recipe["feedback_config"]["maximum_gate"] == pytest.approx(0.75)
    assert recipe["feedback_config"]["loss_ema_alpha"] == pytest.approx(0.2)
    assert recipe["feedback_semantic_owner"] == (
        "st-core::runtime::zspace_optimizer_feedback"
    )


def test_bridge_rejects_invalid_feedback_knobs_through_the_rust_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback = _install_fake_feedback(monkeypatch)

    def rejecting_init(config=None):  # type: ignore[no-untyped-def]
        if float(dict(config or {}).get("maximum_gate", 1.0)) > 1.0:
            raise ValueError("maximum_gate must be in (0, 1]")
        return feedback.init(config)

    monkeypatch.setattr(st, "zspace_optimizer_feedback_init", rejecting_init)
    bridge = _load_bridge_example()

    with pytest.raises(SystemExit):
        bridge.parse_args(
            [
                "--training-recipe-only",
                "--zspace-optimizer-control",
                "observe",
                "--zspace-optimizer-feedback",
                "loss_guard",
                "--zspace-optimizer-feedback-maximum-gate",
                "2.0",
            ]
        )


def test_early_stopped_observe_seals_a_realized_trajectory(tmp_path: Path) -> None:
    optimizer = _FakeOptimizer()
    args = types.SimpleNamespace(output_dir=str(tmp_path / "early-observe"))
    state = types.SimpleNamespace(global_step=0, max_steps=3)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(mode="observe")
    callback.on_train_begin(args, state, control, optimizer=optimizer)
    callback.on_step_begin(args, state, control, optimizer=optimizer)
    optimizer.step()
    state.global_step = 1
    callback.on_step_end(args, state, control, optimizer=optimizer)

    callback.on_train_end(args, state, control, optimizer=optimizer)

    receipt = callback.receipt()
    assert receipt["status"] == "blocked"
    assert receipt["planned_update_count"] == 3
    assert receipt["realized_update_count"] == 1
    assert receipt["training_horizon_complete"] is False
    assert receipt["trajectory_horizon_complete"] is True
    assert receipt["trajectory_generated"] is True
    assert receipt["trajectory_step_count"] == 1
    assert receipt["evidence_blockers"] == ["trainer_stopped_before_planned_horizon"]
    assert (Path(args.output_dir) / st.HF_ZSPACE_OPTIMIZER_STATE_FILENAME).is_file()


def test_early_stopped_calibrated_apply_is_blocked_without_raising(
    tmp_path: Path,
) -> None:
    observed, _ = _run_two_steps(tmp_path, mode="observe")
    trajectory = Path(str(observed.receipt()["trajectory_path"]))
    optimizer = _FakeOptimizer()
    args = types.SimpleNamespace(output_dir=str(tmp_path / "early-apply"))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(
            mode="apply",
            trajectory_arm="dose_normalized",
            trajectory=trajectory,
        )
    callback.on_train_begin(args, state, control, optimizer=optimizer)
    callback.on_step_begin(args, state, control, optimizer=optimizer)
    optimizer.step()
    state.global_step = 1
    callback.on_step_end(args, state, control, optimizer=optimizer)

    callback.on_train_end(args, state, control, optimizer=optimizer)

    receipt = callback.receipt()
    assert receipt["status"] == "blocked"
    assert receipt["applied_update_count"] == 1
    assert receipt["restored_update_count"] == 1
    assert receipt["training_horizon_complete"] is False
    assert receipt["trajectory_horizon_complete"] is False
    assert receipt["trajectory_generated"] is False
    assert receipt["trajectory_step_count"] == 2
    assert receipt["evidence_blockers"] == [
        "trainer_stopped_before_planned_horizon",
        "input_trajectory_only_partially_consumed",
    ]
    assert (Path(args.output_dir) / st.HF_ZSPACE_OPTIMIZER_STATE_FILENAME).is_file()


def test_zero_step_early_stop_still_seals_apply_state(tmp_path: Path) -> None:
    optimizer = _FakeOptimizer()
    args = types.SimpleNamespace(output_dir=str(tmp_path / "zero-step-apply"))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(mode="apply")
    callback.on_train_begin(args, state, control, optimizer=optimizer)

    callback.on_train_end(args, state, control, optimizer=optimizer)

    receipt = callback.receipt()
    assert receipt["status"] == "blocked"
    assert receipt["realized_update_count"] == 0
    assert receipt["trajectory_step_count"] is None
    assert receipt["evidence_blockers"] == [
        "no_realized_optimizer_updates",
        "trainer_stopped_before_planned_horizon",
    ]
    assert optimizer.pre_hooks == []
    assert optimizer.post_hooks == []
    assert (Path(args.output_dir) / st.HF_ZSPACE_OPTIMIZER_STATE_FILENAME).is_file()


def test_factorized_trajectory_arms_match_the_intended_integrated_doses(
    tmp_path: Path,
) -> None:
    observed, observed_optimizer = _run_two_steps(tmp_path, mode="observe")
    observed_receipt = observed.receipt()
    trajectory_path = tmp_path / "observe" / st.HF_ZSPACE_OPTIMIZER_TRAJECTORY_FILENAME
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))

    raw, raw_optimizer = _run_two_steps(
        tmp_path,
        mode="apply",
        trajectory_arm="raw",
        trajectory=trajectory_path,
    )
    constant, constant_optimizer = _run_two_steps(
        tmp_path,
        mode="apply",
        trajectory_arm="dose_matched_constant",
        trajectory=trajectory_path,
    )
    normalized, normalized_optimizer = _run_two_steps(
        tmp_path,
        mode="apply",
        trajectory_arm="dose_normalized",
        trajectory=trajectory_path,
    )
    receipts = [
        observed_receipt,
        raw.receipt(),
        constant.receipt(),
        normalized.receipt(),
    ]

    assert observed_optimizer.step_learning_rates == [[1.0e-3], [1.0e-3]]
    assert sum(row[0] for row in raw_optimizer.step_learning_rates) == pytest.approx(
        trajectory["raw_dose"]
    )
    assert sum(
        row[0] for row in constant_optimizer.step_learning_rates
    ) == pytest.approx(trajectory["raw_dose"])
    assert sum(
        row[0] for row in normalized_optimizer.step_learning_rates
    ) == pytest.approx(trajectory["nominal_dose"])
    assert len({receipt["control_sequence_id"] for receipt in receipts}) == 1
    assert len({receipt["nominal_schedule_sequence_id"] for receipt in receipts}) == 1
    assert {receipt["trajectory_id"] for receipt in receipts} == {
        trajectory["trajectory_id"]
    }
    assert observed_receipt["trajectory_generated"] is True
    assert all(receipt["trajectory_validated"] is True for receipt in receipts)
    for arm, callback in (
        ("raw", raw),
        ("dose_matched_constant", constant),
        ("dose_normalized", normalized),
    ):
        receipt = callback.receipt()
        expected_count = trajectory[f"{arm}_non_identity_update_count"]
        assert receipt["non_identity_update_count"] == expected_count
        assert receipt["model_update_intervened"] is (expected_count > 0)
    assert observed_receipt["actuated_learning_rate_dose_ratio"] == pytest.approx(1.0)
    assert raw.receipt()["actuated_learning_rate_dose"] == pytest.approx(
        trajectory["raw_dose"]
    )
    assert constant.receipt()["actuated_learning_rate_dose"] == pytest.approx(
        trajectory["raw_dose"]
    )
    assert normalized.receipt()["actuated_learning_rate_dose"] == pytest.approx(
        trajectory["nominal_dose"]
    )


def test_planned_trajectory_rejects_scheduler_drift_before_optimizer_step(
    tmp_path: Path,
) -> None:
    observed, _ = _run_two_steps(tmp_path, mode="observe")
    trajectory = Path(str(observed.receipt()["trajectory_path"]))
    optimizer = _FakeOptimizer(learning_rate=2.0e-3)
    args = types.SimpleNamespace(output_dir=str(tmp_path / "drift"))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(
            mode="apply",
            trajectory_arm="dose_normalized",
            trajectory=trajectory,
        )
    callback.on_train_begin(args, state, control, optimizer=optimizer)
    callback.on_step_begin(args, state, control, optimizer=optimizer)

    with pytest.raises(RuntimeError, match="scheduler nominal learning rates differ"):
        optimizer.step()

    callback.abort(RuntimeError("expected scheduler drift"))


def test_planned_trajectory_rejects_relative_drift_for_tiny_scheduler_rates(
    tmp_path: Path,
) -> None:
    observed, _ = _run_two_steps(
        tmp_path,
        mode="observe",
        learning_rate=1.0e-200,
    )
    trajectory = Path(str(observed.receipt()["trajectory_path"]))
    optimizer = _FakeOptimizer(learning_rate=2.0e-200)
    args = types.SimpleNamespace(output_dir=str(tmp_path / "tiny-drift"))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(
            mode="apply",
            trajectory_arm="dose_normalized",
            trajectory=trajectory,
        )
    callback.on_train_begin(args, state, control, optimizer=optimizer)
    callback.on_step_begin(args, state, control, optimizer=optimizer)

    with pytest.raises(RuntimeError, match="scheduler nominal learning rates differ"):
        optimizer.step()

    callback.abort(RuntimeError("expected tiny scheduler drift"))


def test_non_raw_apply_requires_a_rust_trajectory_identity() -> None:
    with pytest.raises(ValueError, match="requires trajectory_id"):
        st.hf_zspace_optimizer_recipe_contract(
            mode="apply",
            trajectory_arm="dose_normalized",
        )


def test_hf_bridge_seals_validated_trajectory_identity_into_recipe(
    tmp_path: Path,
) -> None:
    trajectory = st.zspace_parameter_trajectory(
        raw_learning_rate_scales=[0.8, 1.1],
        nominal_learning_rates=[[1e-3], [5e-4]],
    )
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    bridge = _load_bridge_example()

    args = bridge.parse_args(
        [
            "--training-recipe-only",
            "--zspace-optimizer-control",
            "apply",
            "--zspace-optimizer-trajectory-arm",
            "dose_normalized",
            "--zspace-optimizer-trajectory-json",
            str(trajectory_path),
        ]
    )

    assert (
        args._hf_zspace_optimizer_recipe_contract["trajectory_id"]
        == (trajectory["trajectory_id"])
    )
    assert args._hf_zspace_optimizer_recipe_contract["trajectory_arm"] == (
        "dose_normalized"
    )
    assert args._hf_zspace_optimizer_trajectory_report == trajectory


def test_apply_resume_restores_pending_control_and_controller_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resume"
    args = types.SimpleNamespace(output_dir=str(output))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    first_optimizer = _FakeOptimizer()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        first = hf_zspace_optimizer_control_callback(mode="apply")
    first.on_train_begin(args, state, control, optimizer=first_optimizer)
    first.on_step_begin(args, state, control, optimizer=first_optimizer)
    first_optimizer.step()
    state.global_step = 1
    first.on_step_end(args, state, control, optimizer=first_optimizer)
    first.on_save(args, state, control, optimizer=first_optimizer)
    checkpoint = output / "checkpoint-1"
    assert (checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME).is_file()
    first.abort(RuntimeError("simulated process handoff"))

    second_optimizer = _FakeOptimizer()
    resumed_state = types.SimpleNamespace(global_step=1, max_steps=2)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        second = hf_zspace_optimizer_control_callback(
            mode="apply",
            resume_from_checkpoint=checkpoint,
        )
    second.on_train_begin(args, resumed_state, control, optimizer=second_optimizer)
    second.on_step_begin(args, resumed_state, control, optimizer=second_optimizer)
    second_optimizer.step()
    resumed_state.global_step = 2
    second.on_step_end(args, resumed_state, control, optimizer=second_optimizer)
    second.on_train_end(args, resumed_state, control, optimizer=second_optimizer)
    receipt = second.receipt()

    assert receipt["status"] == "ready"
    assert receipt["resume_state_loaded"] is True
    assert receipt["applied_update_count"] == 2
    assert receipt["restored_update_count"] == 2
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    state_path = checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["hf_global_step"] = 0
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        with pytest.raises(RuntimeError, match="resume-state identity mismatch"):
            hf_zspace_optimizer_control_callback(
                mode="apply",
                resume_from_checkpoint=checkpoint,
            )


def test_legacy_v1_resume_preserves_training_without_inventing_schedule_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "legacy-resume"
    args = types.SimpleNamespace(output_dir=str(output))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    first_optimizer = _FakeOptimizer()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        first = hf_zspace_optimizer_control_callback(mode="apply")
    first.on_train_begin(args, state, control, optimizer=first_optimizer)
    first.on_step_begin(args, state, control, optimizer=first_optimizer)
    first_optimizer.step()
    state.global_step = 1
    first.on_step_end(args, state, control, optimizer=first_optimizer)
    first.on_save(args, state, control, optimizer=first_optimizer)
    checkpoint = output / "checkpoint-1"
    state_path = checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME

    legacy = json.loads(state_path.read_text(encoding="utf-8"))
    legacy.pop("state_id")
    legacy["schema"] = "spiraltorch.hf_zspace_optimizer_state.v1"
    for field in (
        "schedule_sequence",
        "schedule_prefix_missing_count",
        "input_trajectory_id",
        "trajectory_id",
        "trajectory_path",
        "feedback_state",
        "feedback_observation_sequence",
        "feedback_control_sequence",
    ):
        legacy.pop(field, None)
    for field in (
        "trajectory_arm",
        "trajectory_id",
        "trajectory_contract",
        "trajectory_semantic_owner",
        "trajectory_semantic_backend",
        "trajectory_required",
        "actuation_scale_source",
        "feedback_mode",
        "feedback_adaptive",
        "feedback_contract",
        "feedback_semantic_owner",
        "feedback_semantic_backend",
        "feedback_config",
        "feedback_control_target",
        "feedback_evidence_boundary",
    ):
        legacy["recipe"].pop(field, None)
    legacy["state_id"] = _sha256_id(legacy)
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    first.abort(RuntimeError("simulated legacy process handoff"))

    second_optimizer = _FakeOptimizer()
    resumed_state = types.SimpleNamespace(global_step=1, max_steps=2)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        second = hf_zspace_optimizer_control_callback(
            mode="apply",
            resume_from_checkpoint=checkpoint,
        )
    second.on_train_begin(args, resumed_state, control, optimizer=second_optimizer)
    second.on_step_begin(args, resumed_state, control, optimizer=second_optimizer)
    second_optimizer.step()
    resumed_state.global_step = 2
    second.on_step_end(args, resumed_state, control, optimizer=second_optimizer)
    second.on_train_end(args, resumed_state, control, optimizer=second_optimizer)
    receipt = second.receipt()

    assert receipt["status"] == "ready"
    assert receipt["applied_update_count"] == 2
    assert receipt["schedule_evidence_complete"] is False
    assert receipt["schedule_prefix_missing_count"] == 1
    assert receipt["trajectory_validated"] is None
    assert receipt["nominal_learning_rate_dose"] is None
    assert receipt["actuated_schedule_sequence_id"] is None
    migrated = json.loads(
        (output / HF_ZSPACE_OPTIMIZER_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert migrated["schema"] == st.HF_ZSPACE_OPTIMIZER_STATE_SCHEMA
    assert migrated["schedule_prefix_missing_count"] == 1


def test_v2_resume_is_accepted_only_when_feedback_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    feedback = _install_fake_feedback(monkeypatch)
    output = tmp_path / "v2-resume"
    args = types.SimpleNamespace(output_dir=str(output))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    optimizer = _FakeOptimizer()
    first = hf_zspace_optimizer_control_callback(mode="apply")
    first.on_train_begin(args, state, control, optimizer=optimizer)
    first.on_step_begin(args, state, control, optimizer=optimizer)
    optimizer.step()
    state.global_step = 1
    first.on_step_end(args, state, control, optimizer=optimizer)
    first.on_save(args, state, control, optimizer=optimizer)
    checkpoint = output / "checkpoint-1"
    state_path = checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME
    legacy = json.loads(state_path.read_text(encoding="utf-8"))
    legacy.pop("state_id")
    legacy["schema"] = "spiraltorch.hf_zspace_optimizer_state.v2"
    for field in (
        "feedback_state",
        "feedback_observation_sequence",
        "feedback_control_sequence",
    ):
        legacy.pop(field)
    for field in (
        "feedback_mode",
        "feedback_adaptive",
        "feedback_contract",
        "feedback_semantic_owner",
        "feedback_semantic_backend",
        "feedback_config",
        "feedback_control_target",
        "feedback_evidence_boundary",
    ):
        legacy["recipe"].pop(field)
    legacy["state_id"] = _sha256_id(legacy)
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    first.abort(RuntimeError("simulated v2 process handoff"))

    resumed = hf_zspace_optimizer_control_callback(
        mode="apply",
        resume_from_checkpoint=checkpoint,
    )
    assert resumed.resume_state_loaded is True

    with pytest.raises(RuntimeError, match="has no feedback lineage"):
        hf_zspace_optimizer_control_callback(
            mode="apply",
            feedback_mode="loss_guard",
            resume_from_checkpoint=checkpoint,
        )
    assert feedback.restore_calls == 0


def test_resume_branches_trace_from_the_verified_checkpoint_prefix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trace-resume"
    trace = output / "control.jsonl"
    args = types.SimpleNamespace(output_dir=str(output))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    first_optimizer = _FakeOptimizer()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        first = hf_zspace_optimizer_control_callback(
            mode="apply",
            trace_path=trace,
        )
    first.on_train_begin(args, state, control, optimizer=first_optimizer)
    first.on_step_begin(args, state, control, optimizer=first_optimizer)
    first_optimizer.step()
    state.global_step = 1
    first.on_step_end(args, state, control, optimizer=first_optimizer)
    first.on_save(args, state, control, optimizer=first_optimizer)
    checkpoint = output / "checkpoint-1"
    checkpoint_state = json.loads(
        (checkpoint / HF_ZSPACE_OPTIMIZER_STATE_FILENAME).read_text(encoding="utf-8")
    )
    first.abort(RuntimeError("simulated crash after checkpoint"))
    parent_with_crash_tail = trace.read_bytes()

    second_optimizer = _FakeOptimizer()
    resumed_state = types.SimpleNamespace(global_step=1, max_steps=2)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        second = hf_zspace_optimizer_control_callback(
            mode="apply",
            trace_path=trace,
            reset_trace=True,
            resume_from_checkpoint=checkpoint,
        )

    segment = Path(second.trace_path)
    assert segment != trace
    assert segment.name.startswith("control.resume-1.")
    assert trace.read_bytes() == parent_with_crash_tail
    segment_start = json.loads(segment.read_text(encoding="utf-8").splitlines()[0])
    assert segment_start["event"] == "trace_segment_started"
    assert segment_start["parent_trace_sha256"] == checkpoint_state["trace_sha256"]
    assert (
        segment_start["parent_trace_size_bytes"] == checkpoint_state["trace_size_bytes"]
    )

    second.on_train_begin(args, resumed_state, control, optimizer=second_optimizer)
    second.on_step_begin(args, resumed_state, control, optimizer=second_optimizer)
    second_optimizer.step()
    resumed_state.global_step = 2
    second.on_step_end(args, resumed_state, control, optimizer=second_optimizer)
    second.on_train_end(args, resumed_state, control, optimizer=second_optimizer)
    receipt = second.receipt()
    assert receipt["status"] == "ready"
    assert receipt["trace_segmented_on_resume"] is True
    assert receipt["trace_parent_sha256"] == checkpoint_state["trace_sha256"]

    trace.write_bytes(b"x" + parent_with_crash_tail[1:])
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        with pytest.raises(
            RuntimeError, match="does not contain the checkpoint prefix"
        ):
            hf_zspace_optimizer_control_callback(
                mode="apply",
                trace_path=trace,
                resume_from_checkpoint=checkpoint,
            )


def test_public_control_surface_is_exported() -> None:
    for name in (
        "HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES",
        "HF_ZSPACE_OPTIMIZER_MODES",
        "HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_STATE_FILENAME",
        "HF_ZSPACE_OPTIMIZER_STATE_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_TRACE_FILENAME",
        "HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS",
        "HF_ZSPACE_OPTIMIZER_TRAJECTORY_FILENAME",
        "HF_ZSPACE_MATCHED_ABLATION_SCHEMA",
        "HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA",
        "HF_ZSPACE_FEEDBACK_ABLATION_SCHEMA",
        "compare_hf_zspace_optimizer_factorized_run_cards",
        "compare_hf_zspace_optimizer_feedback_run_cards",
        "compare_hf_zspace_optimizer_run_cards",
        "hf_zspace_optimizer_control_callback",
        "hf_zspace_optimizer_recipe_contract",
        "write_hf_zspace_optimizer_factorized_ablation_report",
        "write_hf_zspace_optimizer_feedback_ablation_report",
        "write_hf_zspace_optimizer_matched_ablation_report",
    ):
        assert name in st.__all__
    assert HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES == ("off", "loss_guard")


def _identity(label: str) -> dict[str, object]:
    return {
        "status": "ready",
        "identity_verified": True,
        "observed_identity_id": "sha256:"
        + hashlib.sha256(label.encode("utf-8")).hexdigest(),
    }


def _matched_card(
    *,
    mode: str,
    seed: int,
    eval_after: float,
    dataset_identity: str = "dataset",
) -> dict[str, object]:
    recipe = st.hf_zspace_optimizer_recipe_contract(mode=mode)
    count = 4
    return {
        "failure_stage": None,
        "model_saved": True,
        "zspace_optimizer_control_recipe": recipe,
        "zspace_optimizer_control_receipt": {
            "schema": st.HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA,
            "status": "ready",
            "mode": mode,
            "observed_update_count": count if mode == "observe" else 0,
            "applied_update_count": count if mode == "apply" else 0,
            "non_identity_update_count": count if mode == "apply" else 0,
            "restored_update_count": count if mode == "apply" else 0,
            "consumed_control_count": count,
            "model_update_intervened": mode == "apply",
            "control_sequence_id": "sha256:" + "c" * 64,
        },
        "training_recipe_identity": {
            "status": "ready",
            "identity_payload": {
                "schema": "recipe",
                "training_arguments": {"seed": seed, "learning_rate": 1.0e-4},
                "trainer_contract": {
                    "trainer": "transformers.Trainer",
                    "zspace_optimizer_control": recipe,
                },
            },
        },
        "training_input_identity_after_load": _identity("training"),
        "dataset_materialization_identity": _identity(dataset_identity),
        "tokenized_dataset_identity": _identity("tokens"),
        "model_runtime_identity_after_model": _identity("model"),
        "finetune_execution_identity_after_model": _identity("execution"),
        "eval_before_train": {"loss": 2.5},
        "eval_after_train": {"loss": eval_after},
    }


def _factorized_card(
    *,
    arm: str,
    seed: int,
    eval_after: float,
) -> dict[str, object]:
    mode = "observe" if arm == "observe" else "apply"
    trajectory_arm = "raw" if arm == "observe" else arm
    trajectory_id = "sha256:" + "a" * 64
    recipe = st.hf_zspace_optimizer_recipe_contract(
        mode=mode,
        trajectory_arm=trajectory_arm,
        trajectory_id=None if mode == "observe" else trajectory_id,
    )
    card = _matched_card(mode=mode, seed=seed, eval_after=eval_after)
    card["zspace_optimizer_control_recipe"] = recipe
    receipt = card["zspace_optimizer_control_receipt"]
    nominal_dose = 1.0
    raw_dose = 0.8
    actuated_dose = nominal_dose if arm in {"observe", "dose_normalized"} else raw_dose
    receipt.update(
        {
            "recipe": recipe,
            "trajectory_arm": trajectory_arm,
            "trajectory_id": trajectory_id,
            "trajectory_validated": True,
            "trajectory_step_count": 4,
            "trajectory_identity_relative_tolerance": 1.0e-12,
            "trajectory_identity_absolute_tolerance": 1.0e-15,
            "trajectory_raw_non_identity_update_count": 4,
            "trajectory_dose_matched_constant_non_identity_update_count": 4,
            "trajectory_dose_normalized_non_identity_update_count": 4,
            "trajectory_nominal_dose": nominal_dose,
            "trajectory_raw_dose": raw_dose,
            "trajectory_dose_normalized_dose": nominal_dose,
            "trajectory_raw_dose_ratio": raw_dose / nominal_dose,
            "trajectory_dose_normalized_dose_ratio": 1.0,
            "nominal_learning_rate_dose": nominal_dose,
            "actuated_learning_rate_dose": actuated_dose,
            "actuated_learning_rate_dose_ratio": actuated_dose / nominal_dose,
            "schedule_evidence_complete": True,
            "schedule_prefix_missing_count": 0,
            "nominal_schedule_sequence_id": "sha256:" + "d" * 64,
            "actuated_schedule_sequence_id": "sha256:"
            + arm.encode().hex().ljust(64, "0")[:64],
        }
    )
    card["training_recipe_identity"]["identity_payload"]["trainer_contract"][
        "zspace_optimizer_control"
    ] = recipe
    return card


def _feedback_card(
    *,
    arm: str,
    seed: int,
    eval_after: float,
) -> dict[str, object]:
    mode = "observe" if arm == "observe" else "apply"
    trajectory_id = "sha256:" + "a" * 64
    recipe = st.hf_zspace_optimizer_recipe_contract(
        mode=mode,
        trajectory_id=None if mode == "observe" else trajectory_id,
    )
    feedback_config = {
        "loss_ema_alpha": 0.2,
        "relative_delta_ema_alpha": 0.5,
        "loss_floor": 1.0e-8,
        "regression_threshold": 0.01,
        "halt_threshold": 0.05,
        "recovery_threshold": 0.0025,
        "attenuation_rate": 0.25,
        "recovery_rate": 0.125,
        "halt_regression_streak": 2,
        "resume_improvement_streak": 2,
        "warmup_observations": 2,
        "max_stale_updates": 0,
        "maximum_gate": 1.0,
    }
    feedback_mode = "loss_guard" if arm == "raw_loss_guard" else "off"
    if feedback_mode == "loss_guard":
        recipe.update(
            {
                "feedback_mode": feedback_mode,
                "feedback_adaptive": True,
                "feedback_contract": st.ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
                "feedback_semantic_owner": (
                    st.ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
                ),
                "feedback_semantic_backend": "rust",
                "feedback_config": feedback_config,
                "feedback_control_target": (
                    "proposed_learning_rate_scale_deviation_from_identity"
                ),
                "feedback_evidence_boundary": (
                    "within_run_loss_guard_not_counterfactual_efficacy"
                ),
            }
        )
    card = _matched_card(mode=mode, seed=seed, eval_after=eval_after)
    card["zspace_optimizer_control_recipe"] = recipe
    receipt = card["zspace_optimizer_control_receipt"]
    count = 4
    nominal_dose = 1.0
    raw_dose = 0.8
    actuated_dose = {
        "observe": nominal_dose,
        "raw_unguarded": raw_dose,
        "raw_loss_guard": 0.95,
    }[arm]
    feedback_active = 1 if arm == "raw_loss_guard" else 0
    feedback_identity = 3 if arm == "raw_loss_guard" else 0
    receipt.update(
        {
            "recipe": recipe,
            "feedback_mode": feedback_mode,
            "trajectory_arm": "raw",
            "trajectory_id": trajectory_id,
            "trajectory_validated": True,
            "trajectory_step_count": count,
            "planned_update_count": count,
            "realized_update_count": count,
            "training_horizon_complete": True,
            "trajectory_horizon_complete": True,
            "evidence_blockers": [],
            "trajectory_nominal_dose": nominal_dose,
            "trajectory_raw_dose": raw_dose,
            "nominal_learning_rate_dose": nominal_dose,
            "actuated_learning_rate_dose": actuated_dose,
            "actuated_learning_rate_dose_ratio": actuated_dose / nominal_dose,
            "schedule_evidence_complete": True,
            "schedule_prefix_missing_count": 0,
            "nominal_schedule_sequence_id": "sha256:" + "d" * 64,
            "actuated_schedule_sequence_id": "sha256:"
            + hashlib.sha256(arm.encode("utf-8")).hexdigest(),
            "guarded_control_sequence_id": "sha256:"
            + hashlib.sha256(f"guarded-{arm}".encode("utf-8")).hexdigest(),
            "scheduler_nominal_lr_restored": True,
            "observed_update_count": count if arm == "observe" else 0,
            "applied_update_count": 0 if arm == "observe" else count,
            "restored_update_count": 0 if arm == "observe" else count,
            "non_identity_update_count": {
                "observe": 0,
                "raw_unguarded": count,
                "raw_loss_guard": feedback_active,
            }[arm],
            "model_update_intervened": arm != "observe",
            "feedback_control_count": count if arm == "raw_loss_guard" else 0,
            "feedback_observation_count": (count if arm == "raw_loss_guard" else 0),
            "feedback_control_sequence_id": (
                "sha256:" + "e" * 64 if arm == "raw_loss_guard" else None
            ),
            "feedback_observation_sequence_id": (
                "sha256:" + "f" * 64 if arm == "raw_loss_guard" else None
            ),
            "feedback_disposition_counts": (
                {"no_feedback": 1, "warmup": 2, "active": 1}
                if arm == "raw_loss_guard"
                else {}
            ),
            "feedback_active_update_count": feedback_active,
            "feedback_identity_update_count": feedback_identity,
            "feedback_stale_update_count": 0,
            "feedback_halted_update_count": 0,
            "feedback_gate": 0.125 if arm == "raw_loss_guard" else None,
            "feedback_halted": False if arm == "raw_loss_guard" else None,
            "feedback_contract": (
                st.ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
                if arm == "raw_loss_guard"
                else None
            ),
            "feedback_semantic_owner": (
                st.ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
                if arm == "raw_loss_guard"
                else None
            ),
            "feedback_evidence_boundary": (
                "within_run_loss_guard_not_counterfactual_efficacy"
                if arm == "raw_loss_guard"
                else None
            ),
        }
    )
    card["training_recipe_identity"]["identity_payload"]["trainer_contract"][
        "zspace_optimizer_control"
    ] = recipe
    return card


def test_matched_ablation_accepts_the_training_input_identity_schema_alias() -> None:
    observe = _matched_card(mode="observe", seed=13, eval_after=2.0)
    apply = _matched_card(mode="apply", seed=13, eval_after=1.8)
    for card in (observe, apply):
        identity = card["training_input_identity_after_load"]
        identity["observed_input_id"] = identity.pop("observed_identity_id")

    report = st.compare_hf_zspace_optimizer_run_cards([observe, apply])

    assert report["status"] == "ready"


def test_matched_ablation_requires_identical_non_intervention_evidence() -> None:
    report = st.compare_hf_zspace_optimizer_run_cards(
        [
            _matched_card(mode="observe", seed=13, eval_after=2.0),
            _matched_card(mode="apply", seed=13, eval_after=1.8),
        ]
    )

    assert report["status"] == "ready"
    assert report["matched_pair_count"] == 1
    assert report["efficacy_claim_ready"] is False
    pair = report["pairs"][0]
    assert pair["status"] == "ready"
    assert pair["difference_in_differences"] == pytest.approx(-0.2)
    assert pair["apply_win"] is True


def test_matched_ablation_blocks_identity_drift() -> None:
    report = st.compare_hf_zspace_optimizer_run_cards(
        [
            _matched_card(mode="observe", seed=13, eval_after=2.0),
            _matched_card(
                mode="apply",
                seed=13,
                eval_after=1.8,
                dataset_identity="other-dataset",
            ),
        ]
    )

    assert report["status"] == "blocked"
    assert report["pairs"][0]["identity_matches"]["dataset_materialization"] is False
    assert (
        "dataset_materialization identity is missing or mismatched"
        in report["pairs"][0]["errors"]
    )


def test_matched_ablation_blocks_malformed_identity_hash() -> None:
    observe = _matched_card(mode="observe", seed=13, eval_after=2.0)
    apply = _matched_card(mode="apply", seed=13, eval_after=1.8)
    for card in (observe, apply):
        card["training_input_identity_after_load"]["observed_identity_id"] = (
            "sha256:not-a-digest"
        )

    report = st.compare_hf_zspace_optimizer_run_cards([observe, apply])

    assert report["status"] == "blocked"
    assert report["pairs"][0]["identity_matches"]["training_input"] is False


def test_matched_ablation_requires_non_identity_apply_control() -> None:
    observe = _matched_card(mode="observe", seed=13, eval_after=2.0)
    apply = _matched_card(mode="apply", seed=13, eval_after=2.0)
    apply["zspace_optimizer_control_receipt"]["model_update_intervened"] = False
    apply["zspace_optimizer_control_receipt"]["non_identity_update_count"] = 0

    report = st.compare_hf_zspace_optimizer_run_cards([observe, apply])

    assert report["status"] == "blocked"
    assert (
        "apply arm has no non-identity model-update intervention"
        in report["pairs"][0]["errors"]
    )


def test_matched_ablation_rejects_boolean_receipt_counts() -> None:
    observe = _matched_card(mode="observe", seed=13, eval_after=2.0)
    apply = _matched_card(mode="apply", seed=13, eval_after=1.8)
    apply["zspace_optimizer_control_receipt"]["restored_update_count"] = True

    report = st.compare_hf_zspace_optimizer_run_cards([observe, apply])

    assert report["status"] == "blocked"
    assert (
        "apply arm did not restore every nominal learning rate"
        in report["pairs"][0]["errors"]
    )


def test_three_matched_seeds_are_required_for_bounded_efficacy_claim() -> None:
    cards = []
    for seed in (13, 17, 23):
        cards.extend(
            [
                _matched_card(mode="observe", seed=seed, eval_after=2.0),
                _matched_card(mode="apply", seed=seed, eval_after=1.9),
            ]
        )

    report = st.compare_hf_zspace_optimizer_run_cards(cards)

    assert report["status"] == "ready"
    assert report["matched_pair_count"] == 3
    assert report["bounded_trend_ready"] is True
    assert report["bounded_trend_direction"] == "apply_better"
    assert report["efficacy_claim_ready"] is True
    assert report["apply_win_count"] == 3


def test_consistent_regression_is_ready_as_a_negative_trend_not_efficacy() -> None:
    cards = []
    for seed in (13, 17, 23):
        cards.extend(
            [
                _matched_card(mode="observe", seed=seed, eval_after=2.0),
                _matched_card(mode="apply", seed=seed, eval_after=2.1),
            ]
        )

    report = st.compare_hf_zspace_optimizer_run_cards(cards)

    assert report["status"] == "ready"
    assert report["bounded_trend_ready"] is True
    assert report["bounded_trend_direction"] == "apply_worse"
    assert report["efficacy_claim_ready"] is False


def test_factorized_ablation_separates_dose_and_shape_contrasts() -> None:
    report = st.compare_hf_zspace_optimizer_factorized_run_cards(
        [
            _factorized_card(arm="observe", seed=13, eval_after=2.0),
            _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
            _factorized_card(arm="raw", seed=13, eval_after=1.9),
            _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
        ]
    )

    assert report["status"] == "ready"
    assert report["matched_seed_count"] == 1
    seed = report["factorized_seeds"][0]
    assert seed["contrasts"]["dose_effect"] == pytest.approx(0.1)
    assert seed["contrasts"]["shape_effect_at_raw_dose"] == pytest.approx(-0.2)
    assert seed["contrasts"]["dose_normalized_shape_effect"] == pytest.approx(-0.2)
    assert seed["contrasts"]["raw_total_effect"] == pytest.approx(-0.1)
    assert seed["eval_loss_changes"]["dose_normalized"] == pytest.approx(-0.7)
    assert report["efficacy_claim_ready"] is False


def test_factorized_ablation_accepts_identity_dose_matched_constant() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    for card in cards:
        receipt = card["zspace_optimizer_control_receipt"]
        receipt["trajectory_raw_dose"] = 1.0
        receipt["trajectory_raw_dose_ratio"] = 1.0
        receipt["actuated_learning_rate_dose"] = 1.0
        receipt["actuated_learning_rate_dose_ratio"] = 1.0
    constant_receipt = cards[1]["zspace_optimizer_control_receipt"]
    constant_receipt["model_update_intervened"] = False
    constant_receipt["non_identity_update_count"] = 0
    for card in cards:
        card["zspace_optimizer_control_receipt"][
            "trajectory_dose_matched_constant_non_identity_update_count"
        ] = 0

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "ready"
    assert report["factorized_seeds"][0]["errors"] == []


def test_factorized_ablation_accepts_identity_dose_normalized_arm() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    for card in cards:
        card["zspace_optimizer_control_receipt"][
            "trajectory_dose_normalized_non_identity_update_count"
        ] = 0
    normalized_receipt = cards[3]["zspace_optimizer_control_receipt"]
    normalized_receipt["model_update_intervened"] = False
    normalized_receipt["non_identity_update_count"] = 0

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "ready"
    assert report["factorized_seeds"][0]["errors"] == []


def test_factorized_ablation_rejects_unexplained_identity_constant_arm() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    constant_receipt = cards[1]["zspace_optimizer_control_receipt"]
    constant_receipt["model_update_intervened"] = False
    constant_receipt["non_identity_update_count"] = 0

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "blocked"
    assert (
        "dose_matched_constant non-identity update count differs from the Rust trajectory"
        in report["factorized_seeds"][0]["errors"]
    )


def test_two_arm_comparator_rejects_calibrated_non_raw_apply() -> None:
    report = st.compare_hf_zspace_optimizer_run_cards(
        [
            _factorized_card(arm="observe", seed=13, eval_after=2.0),
            _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
        ]
    )

    assert report["status"] == "blocked"
    assert report["matched_pair_count"] == 1
    assert (
        "two-arm comparator requires raw apply; use the factorized comparator "
        "for calibrated trajectory arms" in report["pairs"][0]["errors"]
    )


def test_factorized_ablation_blocks_nominal_scheduler_drift() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    cards[-1]["zspace_optimizer_control_receipt"]["nominal_schedule_sequence_id"] = (
        "sha256:" + "e" * 64
    )

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "blocked"
    assert (
        "factorized arms do not share one nominal LR sequence"
        in report["factorized_seeds"][0]["errors"]
    )


def test_factorized_ablation_blocks_mislabeled_optimizer_dose() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    cards[-1]["zspace_optimizer_control_receipt"]["actuated_learning_rate_dose"] = 0.8

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "blocked"
    assert (
        "dose_normalized measured optimizer dose differs from its trajectory arm"
        in report["factorized_seeds"][0]["errors"]
    )
    assert report["errors"] == [
        "seed 13: dose_normalized measured optimizer dose differs from its trajectory arm"
    ]


def test_factorized_ablation_rejects_relative_dose_drift_at_tiny_scale() -> None:
    cards = [
        _factorized_card(arm="observe", seed=13, eval_after=2.0),
        _factorized_card(arm="dose_matched_constant", seed=13, eval_after=2.1),
        _factorized_card(arm="raw", seed=13, eval_after=1.9),
        _factorized_card(arm="dose_normalized", seed=13, eval_after=1.8),
    ]
    for card in cards:
        receipt = card["zspace_optimizer_control_receipt"]
        receipt["trajectory_nominal_dose"] = 1.0e-200
        receipt["trajectory_raw_dose"] = 8.0e-201
        receipt["trajectory_dose_normalized_dose"] = 1.0e-200
        receipt["nominal_learning_rate_dose"] = 1.0e-200
        if (
            receipt["mode"] == "observe"
            or receipt["trajectory_arm"] == "dose_normalized"
        ):
            receipt["actuated_learning_rate_dose"] = 1.0e-200
            receipt["actuated_learning_rate_dose_ratio"] = 1.0
        else:
            receipt["actuated_learning_rate_dose"] = 8.0e-201
            receipt["actuated_learning_rate_dose_ratio"] = 0.8
    cards[-1]["zspace_optimizer_control_receipt"]["actuated_learning_rate_dose"] = (
        2.0e-200
    )

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "blocked"
    assert report["errors"] == [
        "seed 13: dose_normalized measured optimizer dose differs from its trajectory arm"
    ]


def test_three_factorized_seeds_support_only_a_bounded_normalized_trend() -> None:
    cards = []
    for seed in (13, 17, 23):
        cards.extend(
            [
                _factorized_card(arm="observe", seed=seed, eval_after=2.0),
                _factorized_card(
                    arm="dose_matched_constant", seed=seed, eval_after=2.05
                ),
                _factorized_card(arm="raw", seed=seed, eval_after=1.95),
                _factorized_card(arm="dose_normalized", seed=seed, eval_after=1.9),
            ]
        )

    report = st.compare_hf_zspace_optimizer_factorized_run_cards(cards)

    assert report["status"] == "ready"
    normalized = report["contrasts"]["dose_normalized_shape_effect"]
    assert normalized["bounded_trend_ready"] is True
    assert normalized["bounded_trend_direction"] == "left_arm_better"
    assert report["bounded_improvement_observed"] is True
    assert report["efficacy_claim_ready"] is False


def test_feedback_ablation_separates_guard_benefit_from_absolute_effect() -> None:
    report = st.compare_hf_zspace_optimizer_feedback_run_cards(
        [
            _feedback_card(arm="observe", seed=13, eval_after=2.2),
            _feedback_card(arm="raw_unguarded", seed=13, eval_after=2.4),
            _feedback_card(arm="raw_loss_guard", seed=13, eval_after=2.1),
        ]
    )

    assert report["status"] == "ready"
    seed = report["feedback_seeds"][0]
    assert seed["contrasts"]["unguarded_total_effect"] == pytest.approx(0.2)
    assert seed["contrasts"]["guarded_total_effect"] == pytest.approx(-0.1)
    assert seed["contrasts"]["guard_benefit"] == pytest.approx(-0.3)
    assert seed["feedback_diagnostics"]["active_update_count"] == 1
    assert seed["guarded_dose_fraction_of_raw_deviation"] == pytest.approx(0.25)
    assert report["evidence_scope"] == "single_or_two_seed_feedback_diagnostic"
    assert report["efficacy_claim_ready"] is False


def test_feedback_ablation_blocks_stale_loss_lineage() -> None:
    cards = [
        _feedback_card(arm="observe", seed=13, eval_after=2.2),
        _feedback_card(arm="raw_unguarded", seed=13, eval_after=2.4),
        _feedback_card(arm="raw_loss_guard", seed=13, eval_after=2.1),
    ]
    guarded = cards[-1]["zspace_optimizer_control_receipt"]
    guarded["feedback_disposition_counts"] = {
        "no_feedback": 1,
        "warmup": 2,
        "stale": 1,
    }
    guarded["feedback_active_update_count"] = 0
    guarded["feedback_stale_update_count"] = 1

    report = st.compare_hf_zspace_optimizer_feedback_run_cards(cards)

    assert report["status"] == "blocked"
    assert "seed 13: guarded feedback used stale loss observations" in report["errors"]
    assert "seed 13: guarded feedback gate never became active" in report["errors"]


def test_feedback_ablation_blocks_raw_control_lineage_drift() -> None:
    cards = [
        _feedback_card(arm="observe", seed=13, eval_after=2.2),
        _feedback_card(arm="raw_unguarded", seed=13, eval_after=2.4),
        _feedback_card(arm="raw_loss_guard", seed=13, eval_after=2.1),
    ]
    cards[-1]["zspace_optimizer_control_receipt"]["control_sequence_id"] = (
        "sha256:" + "b" * 64
    )

    report = st.compare_hf_zspace_optimizer_feedback_run_cards(cards)

    assert report["status"] == "blocked"
    assert (
        "seed 13: feedback arms do not share one raw Rust control sequence"
        in report["errors"]
    )


def test_three_feedback_seeds_support_only_bounded_guard_trends() -> None:
    cards = []
    for seed in (13, 17, 23):
        cards.extend(
            [
                _feedback_card(arm="observe", seed=seed, eval_after=2.2),
                _feedback_card(arm="raw_unguarded", seed=seed, eval_after=2.4),
                _feedback_card(arm="raw_loss_guard", seed=seed, eval_after=2.1),
            ]
        )

    report = st.compare_hf_zspace_optimizer_feedback_run_cards(cards)

    assert report["status"] == "ready"
    assert report["evidence_scope"] == (
        "single_model_single_corpus_multi_seed_feedback_ablation"
    )
    assert report["contrasts"]["guard_benefit"]["bounded_trend_ready"] is True
    assert report["contrasts"]["guard_benefit"][
        "population_standard_deviation"
    ] == pytest.approx(0.0)
    assert report["feedback_seeds"][0][
        "guard_recovery_fraction_of_unguarded_harm"
    ] == pytest.approx(1.5)
    recovery = report["guard_recovery_fraction_of_unguarded_harm"]
    assert recovery["applicable"] is True
    assert recovery["bounded_trend_ready"] is True
    assert recovery["all_seeds_reduced_harm"] is True
    assert recovery["count"] == 3
    assert recovery["mean"] == pytest.approx(1.5)
    assert recovery["fraction_of_mean_unguarded_harm"] == pytest.approx(1.5)
    assert report["bounded_guard_benefit_observed"] is True
    assert report["bounded_absolute_improvement_observed"] is True
    assert report["efficacy_claim_ready"] is False


def test_feedback_harm_recovery_is_not_defined_when_raw_control_is_helpful() -> None:
    cards = [
        _feedback_card(arm="observe", seed=13, eval_after=2.2),
        _feedback_card(arm="raw_unguarded", seed=13, eval_after=2.0),
        _feedback_card(arm="raw_loss_guard", seed=13, eval_after=2.1),
    ]

    report = st.compare_hf_zspace_optimizer_feedback_run_cards(cards)

    assert report["status"] == "ready"
    assert report["feedback_seeds"][0][
        "guard_recovery_fraction_of_unguarded_harm"
    ] is None
    recovery = report["guard_recovery_fraction_of_unguarded_harm"]
    assert recovery["applicable"] is False
    assert recovery["values"] == []
    assert recovery["fraction_of_mean_unguarded_harm"] is None


def test_matched_ablation_cli_writes_the_verified_report(tmp_path: Path) -> None:
    cards: list[Path] = []
    for mode, eval_after in (("observe", 2.0), ("apply", 1.9)):
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(_matched_card(mode=mode, seed=13, eval_after=eval_after)),
            encoding="utf-8",
        )
        cards.append(path)
    output = tmp_path / "comparison.json"

    status = hf_cli.zspace_optimizer_compare_main(
        [str(cards[0]), str(cards[1]), "--out", str(output)]
    )

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["matched_pair_count"] == 1


def test_factorized_ablation_cli_writes_the_verified_report(tmp_path: Path) -> None:
    cards: list[Path] = []
    for arm, eval_after in (
        ("observe", 2.0),
        ("dose_matched_constant", 2.1),
        ("raw", 1.9),
        ("dose_normalized", 1.8),
    ):
        path = tmp_path / f"{arm}.json"
        path.write_text(
            json.dumps(_factorized_card(arm=arm, seed=13, eval_after=eval_after)),
            encoding="utf-8",
        )
        cards.append(path)
    output = tmp_path / "factorized.json"

    status = hf_cli.zspace_optimizer_factorized_compare_main(
        [*[str(path) for path in cards], "--out", str(output)]
    )

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["matched_seed_count"] == 1


def test_feedback_ablation_cli_writes_the_verified_report(tmp_path: Path) -> None:
    cards: list[Path] = []
    for arm, eval_after in (
        ("observe", 2.2),
        ("raw_unguarded", 2.4),
        ("raw_loss_guard", 2.1),
    ):
        path = tmp_path / f"{arm}.json"
        path.write_text(
            json.dumps(_feedback_card(arm=arm, seed=13, eval_after=eval_after)),
            encoding="utf-8",
        )
        cards.append(path)
    output = tmp_path / "feedback.json"

    status = hf_cli.zspace_optimizer_feedback_compare_main(
        [*[str(path) for path in cards], "--out", str(output)]
    )

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["matched_seed_count"] == 1

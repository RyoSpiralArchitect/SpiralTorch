from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import spiraltorch as st
from spiraltorch import hf_cli
from spiraltorch.hf_optimizer_control import (
    HF_ZSPACE_OPTIMIZER_STATE_FILENAME,
    hf_zspace_optimizer_control_callback,
)


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
) -> tuple[object, _FakeOptimizer]:
    optimizer = _FakeOptimizer()
    args = types.SimpleNamespace(output_dir=str(tmp_path / mode))
    state = types.SimpleNamespace(global_step=0, max_steps=2)
    control = types.SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
        callback = hf_zspace_optimizer_control_callback(
            mode=mode,
            control_gain=control_gain,
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
        "HF_ZSPACE_OPTIMIZER_MODES",
        "HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_STATE_FILENAME",
        "HF_ZSPACE_OPTIMIZER_STATE_SCHEMA",
        "HF_ZSPACE_OPTIMIZER_TRACE_FILENAME",
        "HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA",
        "HF_ZSPACE_MATCHED_ABLATION_SCHEMA",
        "compare_hf_zspace_optimizer_run_cards",
        "hf_zspace_optimizer_control_callback",
        "hf_zspace_optimizer_recipe_contract",
        "write_hf_zspace_optimizer_matched_ablation_report",
    ):
        assert name in st.__all__


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

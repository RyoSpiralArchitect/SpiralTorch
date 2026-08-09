from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

import spiraltorch as st
from spiraltorch import zspace_optimizer

_NATIVE_FEEDBACK_AVAILABLE = callable(
    getattr(getattr(st, "_rs", None), "_zspace_optimizer_feedback_init", None)
)
requires_native_feedback = pytest.mark.skipif(
    not _NATIVE_FEEDBACK_AVAILABLE,
    reason="local SpiralTorch extension predates the Rust feedback contract",
)


def _control(
    checkpoint: dict[str, object],
    *,
    target_step: int,
    proposed_scale: float = 0.8,
) -> dict[str, object]:
    return st.zspace_optimizer_feedback_control(
        config=checkpoint["config"],  # type: ignore[arg-type]
        state=checkpoint["state"],  # type: ignore[arg-type]
        target_step=target_step,
        proposed_learning_rate_scale=proposed_scale,
    )


def _observe(
    checkpoint: dict[str, object],
    *,
    step: int,
    loss: float,
) -> dict[str, object]:
    return st.zspace_optimizer_feedback_observe(
        config=checkpoint["config"],  # type: ignore[arg-type]
        state=checkpoint["state"],  # type: ignore[arg-type]
        observation={
            "step": step,
            "max_steps": 16,
            "loss": loss,
            "grad_norm": 1.0,
            "learning_rate": 1e-4,
        },
    )


def _checkpoint(report: dict[str, object]) -> dict[str, object]:
    return {"config": report["config"], "state": report["state_after"]}


@requires_native_feedback
def test_feedback_init_is_rust_owned_and_fail_closed() -> None:
    checkpoint = st.zspace_optimizer_feedback_init()

    assert checkpoint["contract_version"] == (
        st.ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
    )
    assert checkpoint["kind"] == st.ZSPACE_OPTIMIZER_FEEDBACK_KIND
    assert checkpoint["semantic_owner"] == (
        "st-core::runtime::zspace_optimizer_feedback"
    )
    assert checkpoint["semantic_backend"] == "rust"
    assert checkpoint["state"]["gate"] == 0.0  # type: ignore[index]

    control = _control(checkpoint, target_step=1, proposed_scale=0.5)
    assert control["disposition"] == "no_feedback"
    assert control["applied_learning_rate_scale"] == 1.0
    assert control["identity_applied"] is True


@requires_native_feedback
def test_feedback_improvement_opens_and_regression_halts_gate() -> None:
    checkpoint = st.zspace_optimizer_feedback_init(
        {
            "warmup_observations": 1,
            "relative_delta_ema_alpha": 1.0,
            "recovery_rate": 0.25,
        }
    )
    first_control = _control(checkpoint, target_step=1)
    first_observation = _observe(
        _checkpoint(first_control),
        step=1,
        loss=2.0,
    )
    second_control = _control(_checkpoint(first_observation), target_step=2)
    improvement = _observe(
        _checkpoint(second_control),
        step=2,
        loss=1.8,
    )

    assert improvement["projection"]["previous_loss"] == 2.0  # type: ignore[index]
    assert improvement["action"] == "recover"
    assert improvement["gate_after"] == pytest.approx(0.25)

    active = _control(_checkpoint(improvement), target_step=3)
    assert active["disposition"] == "active"
    assert active["applied_learning_rate_scale"] == pytest.approx(0.95)
    regression = _observe(_checkpoint(active), step=3, loss=1.9)
    assert regression["action"] == "halt"
    assert regression["state_after"]["halted"] is True  # type: ignore[index]

    halted = _control(_checkpoint(regression), target_step=4)
    assert halted["disposition"] == "halted"
    assert halted["applied_learning_rate_scale"] == 1.0


@requires_native_feedback
def test_feedback_restore_and_step_identity_fail_closed() -> None:
    checkpoint = st.zspace_optimizer_feedback_init()
    first = _control(checkpoint, target_step=1)
    restored = st.zspace_optimizer_feedback_restore(
        config=first["config"],  # type: ignore[arg-type]
        state=first["state_after"],  # type: ignore[arg-type]
    )
    assert restored["state"] == first["state_after"]

    tampered = copy.deepcopy(restored["state"])
    tampered["halted"] = True
    tampered["gate"] = 0.5
    with pytest.raises(ValueError, match="halted gate"):
        st.zspace_optimizer_feedback_restore(
            config=restored["config"],  # type: ignore[arg-type]
            state=tampered,
        )
    with pytest.raises(ValueError, match="next step"):
        _control(restored, target_step=3)


@requires_native_feedback
def test_feedback_rejects_nonfinite_observations_and_unsafe_scales() -> None:
    checkpoint = st.zspace_optimizer_feedback_init()
    with pytest.raises((TypeError, ValueError), match="finite|JSON"):
        st.zspace_optimizer_feedback_observe(
            config=checkpoint["config"],  # type: ignore[arg-type]
            state=checkpoint["state"],  # type: ignore[arg-type]
            observation={"step": 0, "loss": math.nan},
        )
    with pytest.raises(ValueError, match="proposed_learning_rate_scale"):
        _control(checkpoint, target_step=1, proposed_scale=2.0)


def test_feedback_wrapper_rejects_a_forged_native_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = {
        "contract_version": st.ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
        "kind": st.ZSPACE_OPTIMIZER_FEEDBACK_KIND,
        "semantic_owner": st.ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
        "semantic_backend": "rust",
        "control_rule": st.ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE,
        "transition_validated": True,
        "config": {"maximum_gate": 1.0},
        "target_step": 1,
        "proposed_learning_rate_scale": 0.5,
        "effective_feedback_gate": 0.5,
        "applied_learning_rate_scale": 0.1,
        "state_after": {"control_step": 1, "observation_count": 0, "gate": 0.0},
    }
    package = SimpleNamespace(
        _rs=SimpleNamespace(
            _zspace_optimizer_feedback_control=lambda _request: forged,
        )
    )
    monkeypatch.setitem(zspace_optimizer.sys.modules, "spiraltorch", package)

    with pytest.raises(RuntimeError, match="blend invariant"):
        zspace_optimizer.zspace_optimizer_feedback_control(
            config={},
            state={},
            target_step=1,
            proposed_learning_rate_scale=0.5,
        )


def test_feedback_api_is_exported_from_the_package() -> None:
    assert st.zspace_optimizer_feedback_init is (
        zspace_optimizer.zspace_optimizer_feedback_init
    )
    assert "zspace_optimizer_feedback_control" in st.__all__

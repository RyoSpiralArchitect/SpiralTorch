from __future__ import annotations

import types

import pytest

st = pytest.importorskip("spiraltorch")
temperature_control = pytest.importorskip("spiraltorch.temperature_control")


def _config() -> dict[str, float]:
    return {
        "target_entropy": 0.8,
        "eta": 0.2,
        "min_temperature": 0.3,
        "max_temperature": 2.0,
        "z_kappa": 0.4,
        "z_relax": 0.2,
        "scale_gain": 0.6,
        "gradient_decay": 0.5,
    }


def _state() -> dict[str, float]:
    return {
        "temperature": 1.0,
        "z_memory": 0.0,
        "scale_memory": 0.0,
        "gradient_pressure": 12.0,
        "gradient_entropy_bias": 0.2,
    }


def test_temperature_control_exposes_the_rust_owned_transition() -> None:
    contract = st.zspace_temperature_control(
        [0.6, 0.4],
        config=_config(),
        state=_state(),
        feedback={
            "psi_total": 0.1,
            "band_energy": [0.2, 0.1, 0.05],
            "drift": 0.4,
            "z_signal": 0.1,
            "scale_log_radius": -1.3862943611198906,
        },
        gradient_heat=0.5,
    )

    assert contract["kind"] == "spiraltorch.zspace_temperature_control"
    assert contract["contract_version"] == (
        "spiraltorch.zspace_temperature_control.v1"
    )
    assert contract["semantic_owner"] == "st-core::inference::temperature_control"
    assert contract["semantic_backend"] == "rust"
    assert contract["backend"] == "spiraltorch_temperature_control_core"
    assert contract["probability_count"] == 2
    assert contract["entropy_error"] > 0.0
    assert (
        contract["temperature_after_entropy"]
        > contract["previous_state"]["temperature"]
    )
    assert contract["effects"]["z_feedback_applied"] is True
    assert contract["effects"]["scale_feedback_applied"] is True
    assert contract["effects"]["gradient_heat_provided"] is True
    assert contract["effects"]["gradient_feedback_applied"] is True
    assert contract["next_state"]["gradient_pressure"] == pytest.approx(6.0)
    assert contract["temperature"] == contract["next_state"]["temperature"]

    continued = st.zspace_temperature_control(
        [0.55, 0.45],
        config=contract["config"],
        state=contract["next_state"],
    )
    assert continued["previous_state"] == contract["next_state"]
    assert continued["next_state"]["gradient_pressure"] == pytest.approx(6.0)


def test_temperature_control_is_public() -> None:
    assert (
        st.zspace_temperature_control
        is temperature_control.zspace_temperature_control
    )
    assert "zspace_temperature_control" in st.__all__


def test_temperature_control_requires_the_native_semantic_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        temperature_control.zspace_temperature_control(
            [1.0], config=_config(), state=_state()
        )


def test_temperature_control_rejects_an_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_temperature_control=lambda _request: {
                "kind": "spiraltorch.zspace_temperature_control",
                "contract_version": "spiraltorch.zspace_temperature_control.v0",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="untrusted contract"):
        temperature_control.zspace_temperature_control(
            [1.0], config=_config(), state=_state()
        )


def test_temperature_control_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probability mass"):
        st.zspace_temperature_control(
            [0.8, 0.8], config=_config(), state=_state()
        )
    with pytest.raises(ValueError, match="min_temperature <= max_temperature"):
        config = _config()
        config["min_temperature"] = 3.0
        st.zspace_temperature_control([1.0], config=config, state=_state())
    with pytest.raises(ValueError, match="finite"):
        st.zspace_temperature_control(
            [1.0],
            config=_config(),
            state=_state(),
            feedback={"drift": float("nan")},
        )


def test_native_temperature_ingress_rejects_non_object_contract_fields() -> None:
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_temperature_control({"config": []})
    with pytest.raises(ValueError, match="probabilities.*sequence"):
        st._rs._zspace_temperature_control({"probabilities": {}})
    with pytest.raises(ValueError, match="state.*mapping"):
        st._rs._zspace_temperature_control({"state": None})
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._zspace_temperature_control([])

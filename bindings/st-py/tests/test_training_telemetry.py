from __future__ import annotations

import math
import types

import pytest

st = pytest.importorskip("spiraltorch")
training_telemetry = pytest.importorskip("spiraltorch.training_telemetry")


def test_projection_exposes_rust_owned_surrogate_contract() -> None:
    contract = st.training_telemetry_projection(
        step=4,
        max_steps=10,
        epoch=0.4,
        loss=2.0,
        previous_loss=2.5,
        grad_norm=4.0,
        learning_rate=5e-5,
        desire_gain=1.2,
        psi_gain=0.8,
    )

    assert contract["kind"] == "spiraltorch.training_telemetry_projection"
    assert contract["contract_version"] == (
        "spiraltorch.training_telemetry_projection.v1"
    )
    assert contract["semantic_owner"] == "st-core::telemetry::training_projection"
    assert contract["semantic_backend"] == "rust"
    assert contract["signal_source"] == "trainer_log_proxy"
    assert contract["signal_semantics"] == "surrogate"
    assert contract["progress"] == pytest.approx(0.4)
    assert contract["loss_delta"] == pytest.approx(-0.5)
    assert contract["desire"]["pressure"] == pytest.approx(0.8)
    assert contract["desire"]["saturation"] == pytest.approx(0.88)
    assert contract["psi"]["total"] == pytest.approx(0.48)
    assert contract["telemetry"]["psi.total"] == pytest.approx(0.48)
    assert all(math.isfinite(value) for value in contract["telemetry"].values())


def test_projection_is_public() -> None:
    assert st.training_telemetry_projection is training_telemetry.training_telemetry_projection
    assert "training_telemetry_projection" in st.__all__


def test_projection_requires_the_native_semantic_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        training_telemetry.training_telemetry_projection(loss=2.0)


def test_projection_rejects_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _training_telemetry_projection=lambda _request: {
                "kind": "spiraltorch.training_telemetry_projection",
                "contract_version": "spiraltorch.training_telemetry_projection.v0",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="untrusted contract"):
        training_telemetry.training_telemetry_projection(loss=2.0)


def test_projection_rejects_invalid_observations() -> None:
    with pytest.raises(ValueError, match="grad_norm.*non-negative"):
        st.training_telemetry_projection(grad_norm=-1.0)
    with pytest.raises(ValueError, match="finite"):
        st.training_telemetry_projection(psi_gain=math.inf)


def test_native_ingress_rejects_non_mapping_contract_fields() -> None:
    with pytest.raises(ValueError, match="observation.*mapping"):
        st._rs._training_telemetry_projection({"observation": []})
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._training_telemetry_projection([])

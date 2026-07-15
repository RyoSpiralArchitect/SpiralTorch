from __future__ import annotations

import types

import pytest

st = pytest.importorskip("spiraltorch")
free_energy = pytest.importorskip("spiraltorch.free_energy")


def test_free_energy_exposes_the_rust_owned_variational_contract() -> None:
    contract = st.zspace_free_energy(
        reference_loss=0.8,
        candidate_loss=0.5,
        step_time_ms=12.0,
        memory_mb=256.0,
        retry_rate=0.05,
        observation_entropy=0.4,
        external_penalty=0.1,
        band={"above": 0.6, "here": 0.3, "beneath": 0.1},
    )

    assert contract["kind"] == "spiraltorch.variational_free_energy"
    assert contract["contract_version"] == (
        "spiraltorch.variational_free_energy.v1"
    )
    assert contract["semantic_owner"] == "st-core::heur::free_energy"
    assert contract["semantic_backend"] == "rust"
    assert contract["acceptance_rule"] == (
        "P(accept)=1/(1+exp(F_candidate-F_neutral)),F_neutral=0"
    )
    assert contract["utility"] == pytest.approx(-contract["free_energy"])
    assert 0.0 <= contract["acceptance_probability"] <= 1.0
    assert contract["distribution"]["dominant_band"] == "above"
    assert sum(
        contract["distribution"][field]
        for field in ("above", "here", "beneath")
    ) == pytest.approx(1.0)
    assert contract["distribution"]["variational_identity_residual"] < 1e-10
    assert contract["component_sum_residual"] < 1e-10


def test_free_energy_is_public() -> None:
    assert st.zspace_free_energy is free_energy.zspace_free_energy
    assert "zspace_free_energy" in st.__all__


def test_free_energy_zero_band_mass_uses_the_rust_configured_prior() -> None:
    contract = st.zspace_free_energy(
        config={
            "prior": {"above": 0.5, "here": 0.3, "beneath": 0.2},
            "band_potentials": {"above": -0.2, "here": 0.0, "beneath": 0.4},
        }
    )

    distribution = contract["distribution"]
    assert distribution["status"] == "prior_zero_mass"
    assert distribution["above"] == pytest.approx(0.5)
    assert distribution["here"] == pytest.approx(0.3)
    assert distribution["beneath"] == pytest.approx(0.2)
    assert distribution["kl_divergence"] == pytest.approx(0.0)
    assert contract["components"]["band_potential"] == pytest.approx(0.0)


def test_free_energy_requires_the_native_semantic_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "_rs", object())
    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        free_energy.zspace_free_energy()


def test_free_energy_rejects_an_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_free_energy=lambda _request: {
                "kind": "spiraltorch.variational_free_energy",
                "contract_version": "spiraltorch.variational_free_energy.v0",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="untrusted contract"):
        free_energy.zspace_free_energy()


def test_free_energy_preserves_rust_validation() -> None:
    with pytest.raises(ValueError, match="band.above.*non-negative"):
        st.zspace_free_energy(
            band={"above": -0.1, "here": 0.5, "beneath": 0.6}
        )
    with pytest.raises(ValueError, match="step_time_ms.*non-negative"):
        st.zspace_free_energy(step_time_ms=-1.0)
    with pytest.raises(ValueError, match="unknown field"):
        st.zspace_free_energy(config={"mystery_weight": 1.0})


def test_native_free_energy_ingress_requires_a_mapping() -> None:
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._zspace_free_energy([])

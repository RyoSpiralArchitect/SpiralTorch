from __future__ import annotations

import types

import pytest

st = pytest.importorskip("spiraltorch")
concept_diffusion = pytest.importorskip("spiraltorch.concept_diffusion")


def test_concept_diffusion_exposes_true_rust_heat_flow() -> None:
    contract = st.zspace_concept_diffusion(
        ["left", "right"],
        [1.0, 0.0],
        [[0.0, 1.0], [1.0, 0.0]],
        config={"timestep": 0.25},
    )

    assert contract["kind"] == "spiraltorch.zspace_concept_diffusion"
    assert contract["contract_version"] == ("spiraltorch.zspace_concept_diffusion.v1")
    assert contract["semantic_owner"] == "st-core::inference::concept_diffusion"
    assert contract["semantic_backend"] == "rust"
    assert contract["backend"] == "spiraltorch_concept_diffusion_core"
    assert contract["execution_backend"] == "f64_cpu"
    assert contract["next_state"] == pytest.approx([0.75, 0.25])
    assert (
        contract["effects"]["entropy_after_diffusion"]
        > contract["effects"]["entropy_after_bias"]
    )
    assert (
        contract["effects"]["dirichlet_energy_after"]
        < contract["effects"]["dirichlet_energy_before"]
    )
    assert contract["effects"]["substeps"] == 1
    assert contract["output_probability_sum"] == pytest.approx(1.0)

    continued = st.zspace_concept_diffusion(
        contract["tags"],
        contract["next_state"],
        [[0.0, 1.0], [1.0, 0.0]],
        config=contract["config"],
    )
    assert continued["previous_state"] == contract["next_state"]


def test_concept_diffusion_applies_observation_and_z_bias_in_rust() -> None:
    contract = st.zspace_concept_diffusion(
        ["left", "right"],
        [0.8, 0.2],
        [[0.0, 0.0], [0.0, 0.0]],
        z_bias=[0.0, 2.0],
        observation={"probabilities": [0.2, 0.8], "weight": 0.25},
        config={"timestep": 0.5},
    )

    assert contract["state_after_observation"] == pytest.approx([0.65, 0.35])
    assert contract["next_state"][1] > 0.35
    assert contract["effects"]["observation_applied"] is True
    assert contract["effects"]["z_bias_applied"] is True
    assert contract["effects"]["diffusion_applied"] is False


def test_concept_diffusion_is_public() -> None:
    assert st.zspace_concept_diffusion is concept_diffusion.zspace_concept_diffusion
    assert "zspace_concept_diffusion" in st.__all__


def test_concept_diffusion_requires_the_native_semantic_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        concept_diffusion.zspace_concept_diffusion(["only"], [1.0], [[0.0]])


def test_concept_diffusion_rejects_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_concept_diffusion=lambda _request: {
                "kind": "spiraltorch.zspace_concept_diffusion",
                "contract_version": "spiraltorch.zspace_concept_diffusion.v0",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="untrusted contract"):
        concept_diffusion.zspace_concept_diffusion(["only"], [1.0], [[0.0]])


def test_concept_diffusion_rejects_native_invariant_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = st.zspace_concept_diffusion(["only"], [1.0], [[0.0]])
    invalid["next_state"] = [1.1]
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(_zspace_concept_diffusion=lambda _request: invalid),
    )

    with pytest.raises(RuntimeError, match="invalid mass for next_state"):
        concept_diffusion.zspace_concept_diffusion(["only"], [1.0], [[0.0]])


def test_concept_diffusion_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probability mass"):
        st.zspace_concept_diffusion(
            ["left", "right"],
            [0.8, 0.8],
            [[0.0, 1.0], [1.0, 0.0]],
        )
    with pytest.raises(ValueError, match="asymmetric"):
        st.zspace_concept_diffusion(
            ["left", "right"],
            [0.5, 0.5],
            [[0.0, 1.0], [0.5, 0.0]],
        )
    with pytest.raises(ValueError, match="duplicated"):
        st.zspace_concept_diffusion(
            ["same", "same"],
            [0.5, 0.5],
            [[0.0, 1.0], [1.0, 0.0]],
        )


def test_native_concept_diffusion_ingress_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="state.*sequence"):
        st._rs._zspace_concept_diffusion({"state": {}})
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_concept_diffusion({"config": []})
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_concept_diffusion({"config": None})
    with pytest.raises(ValueError, match="diffusion_tensor.*sequence"):
        st._rs._zspace_concept_diffusion({"diffusion_tensor": {}})
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._zspace_concept_diffusion([])

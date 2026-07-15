from __future__ import annotations

import types

import pytest

st = pytest.importorskip("spiraltorch")
schrodinger = pytest.importorskip("spiraltorch.imaginary_time_schrodinger")


def _evolve(**kwargs: object) -> dict[str, object]:
    return st.zspace_imaginary_time_schrodinger(
        ["left", "right"],
        [0.0, 2.0],
        [{"left": 0, "right": 1, "weight": 1.0}],
        **kwargs,
    )


def test_schrodinger_exposes_rust_graph_hamiltonian() -> None:
    contract = _evolve(config={"imaginary_time": 1.0})

    assert contract["kind"] == "spiraltorch.zspace_imaginary_time_schrodinger"
    assert contract["contract_version"] == (
        "spiraltorch.zspace_imaginary_time_schrodinger.v1"
    )
    assert contract["semantic_owner"] == (
        "st-core::inference::imaginary_time_schrodinger"
    )
    assert contract["semantic_backend"] == "rust"
    assert contract["execution_backend"] == "f64_cpu"
    assert contract["route_blocker"] == "f64_sparse_graph_state"
    assert contract["probability"][0] > contract["probability"][1]
    assert contract["log_amplitude_boost"][0] == pytest.approx(0.0)
    assert (
        contract["effects"]["final_rayleigh_energy"]
        < (contract["effects"]["initial_rayleigh_energy"])
    )
    assert contract["effects"]["initial_l2_norm"] == pytest.approx(1.0)
    assert contract["effects"]["final_l2_norm"] == pytest.approx(1.0)
    assert contract["probability_sum"] == pytest.approx(1.0)
    assert contract["probability_sum_tolerance"] > 0.0
    assert contract["effects"]["energy_tolerance"] > 0.0
    assert contract["effects"]["l2_norm_tolerance"] > 0.0


def test_schrodinger_preserves_potential_gauge() -> None:
    baseline = _evolve(config={"imaginary_time": 2.0})
    shifted = st.zspace_imaginary_time_schrodinger(
        ["left", "right"],
        [137.0, 139.0],
        [{"left": 0, "right": 1, "weight": 1.0}],
        config={"imaginary_time": 2.0},
    )

    assert shifted["probability"] == pytest.approx(baseline["probability"])
    assert shifted["effects"]["potential_shift"] == pytest.approx(137.0)


def test_schrodinger_is_public() -> None:
    assert (
        st.zspace_imaginary_time_schrodinger
        is schrodinger.zspace_imaginary_time_schrodinger
    )
    assert "zspace_imaginary_time_schrodinger" in st.__all__


def test_schrodinger_requires_native_semantic_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        _evolve()


def test_schrodinger_rejects_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_imaginary_time_schrodinger=lambda _request: {
                "kind": "spiraltorch.zspace_imaginary_time_schrodinger",
                "contract_version": "spiraltorch.zspace_imaginary_time_schrodinger.v0",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="untrusted contract"):
        _evolve()


def test_schrodinger_rejects_native_invariant_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _evolve()
    invalid["probability"] = [0.8, 0.8]
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_imaginary_time_schrodinger=lambda _request: invalid
        ),
    )

    with pytest.raises(RuntimeError, match="invalid probability mass"):
        _evolve()


def test_schrodinger_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="canonical"):
        st.zspace_imaginary_time_schrodinger(
            ["left", "right"],
            [0.0, 1.0],
            [{"left": 1, "right": 0, "weight": 1.0}],
        )
    with pytest.raises(ValueError, match="initial_amplitude.*positive"):
        _evolve(initial_amplitude=[1.0, 0.0])
    with pytest.raises(ValueError, match="substeps"):
        _evolve(config={"imaginary_time": 100.0, "max_substeps": 1})


def test_native_schrodinger_ingress_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="potential.*sequence"):
        st._rs._zspace_imaginary_time_schrodinger({"potential": {}})
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_imaginary_time_schrodinger({"config": []})
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_imaginary_time_schrodinger({"config": None})
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._zspace_imaginary_time_schrodinger([])

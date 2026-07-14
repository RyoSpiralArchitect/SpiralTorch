from __future__ import annotations

import types

import pytest

st = pytest.importorskip("spiraltorch")
generation_control = pytest.importorskip("spiraltorch.generation_control")


def test_generation_control_exposes_the_rust_owned_contract() -> None:
    contract = st.zspace_generation_control(
        [4.0, 3.5, 1.0],
        [0, 1, 2],
        [0, 0, 0],
        curvature=-1.0,
        temperature=1.0,
        entropy_target=1.0,
        min_temperature=0.5,
        max_temperature=2.0,
        repression_window=4,
        repression_strength=2.0,
        last_token_repression=1.0,
    )

    assert contract["kind"] == "spiraltorch.zspace_generation_control"
    assert contract["contract_version"] == (
        "spiraltorch.zspace_generation_control.v1"
    )
    assert contract["semantic_owner"] == "st-core::inference::generation_control"
    assert contract["semantic_backend"] == "rust"
    assert contract["backend"] == "spiraltorch_generation_control_core"
    assert contract["repression_penalties"] == pytest.approx([7.0, 0.0, 0.0])
    assert contract["adjusted_logits"] == pytest.approx([-3.0, 3.5, 1.0])
    assert contract["before_top_token"] == 0
    assert contract["after_top_token"] == 1
    assert contract["top_token_changed"] is True
    assert sum(contract["probabilities"]) == pytest.approx(1.0)


def test_generation_control_preserves_ngram_decay_semantics() -> None:
    contract = st.zspace_generation_control(
        [1.0],
        [3],
        [1, 2, 3, 1, 2, 3, 1, 2],
        curvature=-1.0,
        repression_window=8,
        repression_strength=0.0,
        last_token_repression=0.0,
        ngram_size=3,
        ngram_repression_strength=2.0,
        ngram_decay=0.5,
    )

    assert contract["ngram_repression_penalties"] == pytest.approx([0.5625])
    assert contract["ngram_repressed_token_count"] == 1


def test_generation_control_is_public() -> None:
    assert st.zspace_generation_control is generation_control.zspace_generation_control
    assert "zspace_generation_control" in st.__all__


def test_generation_control_requires_the_native_semantic_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        generation_control.zspace_generation_control([], [])


def test_generation_control_rejects_an_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        st,
        "_rs",
        types.SimpleNamespace(
            _zspace_generation_control=lambda _request: {
                "kind": "spiraltorch.zspace_generation_control",
                "contract_version": "spiraltorch.zspace_generation_control.v0",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="untrusted contract"):
        generation_control.zspace_generation_control([], [])


def test_generation_control_rejects_invalid_shapes_and_config() -> None:
    with pytest.raises(ValueError, match="candidate lengths differ"):
        st.zspace_generation_control([1.0], [])
    with pytest.raises(ValueError, match="repression_strength.*non-negative"):
        st.zspace_generation_control([], [], repression_strength=-1.0)
    with pytest.raises(ValueError, match="entropy_target.*non-negative"):
        st.zspace_generation_control([], [], entropy_target=-0.1)


def test_native_ingress_rejects_non_object_contract_fields() -> None:
    with pytest.raises(ValueError, match="config.*mapping"):
        st._rs._zspace_generation_control({"config": []})
    with pytest.raises(ValueError, match="logits.*sequence"):
        st._rs._zspace_generation_control({"logits": {}})
    with pytest.raises(ValueError, match="request must be a mapping"):
        st._rs._zspace_generation_control([])

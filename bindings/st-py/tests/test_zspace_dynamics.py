from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import spiraltorch as st


def _forward(*, standard_normal: list[float] | None = None) -> dict[str, object]:
    return st.zspace_stochastic_schrodinger_forward(
        [1.0, 0.25, -0.5, 0.75, 0.1, -0.2],
        [0.2, -0.1, 0.05],
        standard_normal=standard_normal,
        config={
            "time_step": 0.08,
            "hopping_rate": 0.35,
            "loss_rate": 0.02,
            "noise_scale": 0.15,
        },
    )


def test_stochastic_schrodinger_forward_is_rust_owned_and_replayable() -> None:
    receipt = _forward()

    assert (
        receipt["contract_version"]
        == st.ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION
    )
    assert receipt["kind"] == st.ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND
    assert (
        receipt["semantic_owner"]
        == st.ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER
    )
    assert receipt["semantic_backend"] == "rust"
    assert receipt["forward_validated"] is True
    assert receipt["status"] == "ready"
    assert receipt["efficacy_claim_ready"] is False
    assert str(receipt["forward_id"]).startswith("sha256:")
    assert receipt["request"]["rows"] == 2  # type: ignore[index]
    assert receipt["request"]["features"] == 3  # type: ignore[index]
    assert receipt["request"]["standard_normal"] == [0.0] * 6  # type: ignore[index]
    assert st.validate_zspace_stochastic_schrodinger_forward(receipt) == receipt


def test_explicit_noise_is_content_addressed_and_changes_the_transition() -> None:
    zero = _forward()
    noisy = _forward(standard_normal=[0.1, -0.3, 0.2, 0.0, 0.4, -0.2])

    assert zero["forward_id"] != noisy["forward_id"]
    assert zero["step"]["output_real"] != noisy["step"]["output_real"]  # type: ignore[index]
    assert noisy["request"]["standard_normal"] == pytest.approx([  # type: ignore[index]
        0.1,
        -0.3,
        0.2,
        0.0,
        0.4,
        -0.2,
    ])


def test_vjp_revalidates_forward_and_never_trusts_external_phase() -> None:
    forward = _forward(standard_normal=[0.1, -0.3, 0.2, 0.0, 0.4, -0.2])
    receipt = st.zspace_stochastic_schrodinger_vjp(
        forward, [0.2, -0.4, 0.1, 0.3, 0.0, -0.2]
    )

    assert (
        receipt["contract_version"]
        == st.ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION
    )
    assert receipt["kind"] == st.ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND
    assert receipt["vjp_validated"] is True
    assert receipt["forward_id"] == forward["forward_id"]
    assert (
        receipt["gradient_semantics"]
        == st.ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS
    )
    assert receipt["output_observable"] == "real_quadrature"
    assert len(receipt["result"]["grad_input"]) == 6  # type: ignore[index]
    assert len(receipt["result"]["grad_potential"]) == 3  # type: ignore[index]
    assert st.validate_zspace_stochastic_schrodinger_vjp(receipt) == receipt

    tampered_forward = copy.deepcopy(forward)
    tampered_forward["step"]["phase"][0] = 999.0  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical Rust stochastic Schrodinger forward"):
        st.zspace_stochastic_schrodinger_vjp(tampered_forward, [1.0] * 6)


def test_stochastic_schrodinger_replay_rejects_request_and_result_drift() -> None:
    forward = _forward()
    changed_forward = copy.deepcopy(forward)
    changed_forward["step"]["output_real"][0] = 999.0  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical Rust stochastic Schrodinger forward"):
        st.validate_zspace_stochastic_schrodinger_forward(changed_forward)

    vjp = st.zspace_stochastic_schrodinger_vjp(forward, [1.0] * 6)
    changed_vjp = copy.deepcopy(vjp)
    changed_vjp["result"]["grad_potential"][0] = 999.0  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical Rust stochastic Schrodinger VJP"):
        st.validate_zspace_stochastic_schrodinger_vjp(changed_vjp)


def test_stochastic_schrodinger_fails_closed_on_shapes_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="input.*length"):
        st.zspace_stochastic_schrodinger_forward([1.0], [0.0, 0.0])

    with pytest.raises(ValueError, match="grad_output_real.*length"):
        st.zspace_stochastic_schrodinger_vjp(_forward(), [1.0])

    with pytest.raises(ValueError, match="finite"):
        st.zspace_stochastic_schrodinger_forward([float("nan")], [0.0])

    with pytest.raises(ValueError, match="unknown field"):
        st.zspace_stochastic_schrodinger_forward(
            [1.0], [0.0], config={"browser_formula": "trust me"}
        )


def test_stochastic_schrodinger_rejects_active_container_hooks() -> None:
    class ActiveMapping(Mapping[str, object]):
        @property
        def __class__(self) -> type[object]:
            raise AssertionError("custom __class__ must not run")

        def __getitem__(self, _key: str) -> object:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

    class ActiveSequence(Sequence[float]):
        @property
        def __class__(self) -> type[object]:
            raise AssertionError("custom __class__ must not run")

        def __getitem__(self, _index: int) -> float:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

    class HostileList(list[float]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

    class HostileDict(dict[str, object]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

        def items(self):
            raise AssertionError("overridden items must not run")

    receipt = st.zspace_stochastic_schrodinger_forward(
        HostileList([1.0, 0.0]), HostileList([0.0, 0.0])
    )
    assert st.validate_zspace_stochastic_schrodinger_forward(
        HostileDict(receipt)
    ) == receipt

    with pytest.raises(TypeError, match="list or tuple for bounded admission"):
        st.zspace_stochastic_schrodinger_forward(ActiveSequence(), [0.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dict-backed mapping"):
        st.validate_zspace_stochastic_schrodinger_forward(ActiveMapping())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be JSON-like"):
        st._rs._zspace_stochastic_schrodinger_forward_validate(ActiveMapping())


def test_stochastic_schrodinger_surface_is_typed_and_exported() -> None:
    expected = {
        "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION",
        "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION",
        "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER",
        "validate_zspace_stochastic_schrodinger_forward",
        "validate_zspace_stochastic_schrodinger_vjp",
        "zspace_stochastic_schrodinger_forward",
        "zspace_stochastic_schrodinger_vjp",
    }
    assert expected <= set(st.__all__)

    stub = (Path(st.__file__).with_name("__init__.pyi")).read_text(encoding="utf-8")
    for symbol in expected:
        assert symbol in stub
    assert (
        st.zspace_stochastic_schrodinger_forward.__annotations__["input_values"]
        == "list[float] | tuple[float, ...]"
    )

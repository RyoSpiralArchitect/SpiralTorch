from __future__ import annotations

import spiraltorch as st


def test_hypergrad_helper_accepts_tuple_shape() -> None:
    tape = st.hypergrad((1, 4))
    assert tape.shape() == (1, 4)
    assert tape.curvature() == -1.0


def test_hypergrad_helper_accepts_tensor_shape() -> None:
    tensor = st.Tensor((2, 3))
    tape = st.hypergrad(tensor, learning_rate=0.01)
    assert tape.shape() == tensor.shape()
    assert tape.learning_rate() == 0.01


def test_hypergrad_helper_accepts_mapping_topos() -> None:
    tape = st.hypergrad(
        1,
        3,
        curvature=-0.9,
        learning_rate=0.02,
        topos={
            "curvature": -0.9,
            "tolerance": 1e-3,
            "saturation": 0.8,
            "depth": 4,
            "volume": 16,
        },
    )
    guard = tape.topos()
    assert guard.curvature() == -0.9
    assert guard.max_depth() == 4
    assert guard.max_volume() == 16


def test_hypergrad_topos_factory_returns_guard() -> None:
    guard = st.hypergrad_topos(
        curvature=-0.8,
        tolerance=5e-4,
        saturation=0.7,
        max_depth=8,
        max_volume=32,
    )
    assert guard.curvature() == -0.8
    assert guard.tolerance() == 5e-4
    assert guard.max_depth() == 8
    assert guard.max_volume() == 32


def test_z_metrics_aliases_normalise_inputs() -> None:
    metrics = st.z_metrics(
        velocity=0.5,
        mem=0.25,
        stab=0.9,
        drift=0.1,
        grad=[1, -2, 3],
    )
    assert metrics.speed == 0.5
    assert metrics.memory == 0.25
    assert metrics.stability == 0.9
    assert metrics.drs == 0.1
    assert metrics.gradient == [1.0, -2.0, 3.0]


def test_encode_zspace_returns_tensor() -> None:
    tensor = st.encode_zspace("hypergrad keeps z-space grounded", temperature=0.35)
    assert isinstance(tensor, st.Tensor)
    rows, cols = tensor.shape()
    assert rows == 1
    assert cols > 0

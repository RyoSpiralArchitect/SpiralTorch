"""Native Rust ensemble reduction; no Python semantic fallback."""
import struct

import pytest
import spiraltorch as st


def bits(values):
    return struct.pack(f"{len(values)}f", *values)


@pytest.mark.parametrize("count", [1, 3, 17])
@pytest.mark.parametrize("shape", [(1, 1), (3, 341), (5, 205), (37, 71)])
def test_mean_tensors_matches_ordered_f64(count, shape):
    rows, cols = shape
    data = [[float((i * 17 + p * 131) % 4093 - 2046) / 64 for i in range(rows * cols)] for p in range(count)]
    partials = [st.Tensor(rows, cols, values) for values in data]
    for scale in [0.0, -0.0, 1.0, -1.75]:
        expected = []
        for i in range(rows * cols):
            value = 0.0
            for values in data:
                value += values[i]
            expected.append(value / count * scale)
        actual = st.mean_tensors_scaled(partials, scale).tolist()
        assert bits([value for row in actual for value in row]) == bits(expected)
    assert partials[0].tolist() == [data[0][r * cols:(r + 1) * cols] for r in range(rows)]


def test_mean_tensors_preserves_order_and_rejects_nonfinite():
    values = [float(2**60), 1.0, -float(2**60)]
    tensors = [st.Tensor(1, 1, [value]) for value in values]
    assert st.mean_tensors_scaled(tensors).tolist() == [[0.0]]
    assert bits(st.mean_tensors_scaled([tensors[0], tensors[2], tensors[1]]).tolist()[0]) == bits([1 / 3])
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError):
            st.mean_tensors_scaled(tensors, value)
        with pytest.raises(ValueError):
            st.mean_tensors_scaled([st.Tensor(1, 1, [value])], 0.0)
    with pytest.raises(ValueError):
        st.mean_tensors_scaled([])
    with pytest.raises(RuntimeError):
        st.mean_tensors_scaled([st.Tensor(1, 1, [1.0]), st.Tensor(1, 2, [1.0, 2.0])])
    with pytest.raises(ValueError):
        st.mean_tensors_scaled([st.Tensor(1, 1, [3.4028234663852886e38])], 2.0)
    assert st.mean_tensors_scaled([st.Tensor(0, 3, [])]).shape == (0, 3)
    assert "mean_tensors_scaled" in st.__all__

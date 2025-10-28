from __future__ import annotations

from array import array

def test_tensor_factory_methods_shapes(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor

    zeros = Tensor.zeros(2, 3)
    assert zeros.shape() == (2, 3)
    assert zeros.rows == 2
    assert zeros.cols == 3

    random_normal = Tensor.randn(1, 4, seed=123)
    assert random_normal.shape() == (1, 4)

    random_uniform = Tensor.rand(3, 1, seed=456)
    assert random_uniform.shape() == (3, 1)


def test_tensor_python_backend_operations(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    data = array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor = Tensor._from_python_array(2, 3, data)
    assert tensor.backend == "python"

    reshaped = tensor.reshape(3, 2)
    assert reshaped.shape() == (3, 2)
    assert reshaped.backend == "python"
    assert reshaped.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    transposed = tensor.transpose()
    assert transposed.shape() == (3, 2)
    assert transposed.backend == "python"
    assert transposed.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    sums = tensor.sum_axis0()
    assert sums == [5.0, 7.0, 9.0]

    row_sums = tensor.sum_axis1()
    assert row_sums == [6.0, 15.0]


def test_tensor_cat_rows_shapes(stub_spiraltorch) -> None:
    Tensor = stub_spiraltorch.Tensor
    first = Tensor.zeros(1, 2)
    second = Tensor.zeros(2, 2)

    combined = Tensor.cat_rows([first, second])
    assert combined.shape() == (3, 2)

    python_first = Tensor._from_python_array(1, 2, array("d", [1.0, 2.0]))
    python_second = Tensor._from_python_array(1, 2, array("d", [3.0, 4.0]))
    python_combined = Tensor.cat_rows([python_first, python_second])
    assert python_combined.shape() == (2, 2)
    assert python_combined.backend == "python"


def test_tensor_subclass_preserves_type(stub_spiraltorch) -> None:
    class CustomTensor(stub_spiraltorch.Tensor):
        pass

    tensor = CustomTensor(shape=(2, 2), data=[[1.0, 2.0], [3.0, 4.0]], backend="python")

    reshaped = tensor.reshape(1, 4)
    assert type(reshaped) is CustomTensor

    transposed = tensor.transpose()
    assert type(transposed) is CustomTensor

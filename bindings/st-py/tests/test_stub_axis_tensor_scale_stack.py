import pytest


def test_axis_construction_stub(stub_spiraltorch):
    Axis = stub_spiraltorch.Axis
    axis = Axis("batch", 3)
    assert axis.name == "batch"
    assert axis.size == 3
    resized = axis.with_size(5)
    assert resized.name == "batch"
    assert resized.size == 5


def test_tensor_with_axes_stub(stub_spiraltorch):
    row_axis = stub_spiraltorch.Axis("row")
    col_axis = "col"
    labeled = stub_spiraltorch.tensor([[1, 2]], axes=(row_axis, col_axis))
    assert isinstance(labeled, stub_spiraltorch.LabeledTensor)
    assert labeled.shape == (1, 2)
    assert labeled.axes[0].name == "row"
    assert labeled.axes[0].size == 1
    assert labeled.axes[1].name == "col"
    assert labeled.axes[1].size == 2
    assert labeled.tensor.tolist() == [[1.0, 2.0]]


def test_scalar_scale_stack_stub_samples_error(stub_spiraltorch):
    stack = stub_spiraltorch.scalar_scale_stack([1.0, 2.0], (1, 2), [0.5], 0.1)
    assert isinstance(stack, stub_spiraltorch.ScaleStack)
    assert stack.mode == "scalar"
    assert pytest.approx(stack.threshold) == 0.1
    assert stack.meta["shape"] == (1, 2)
    assert stack.meta["scales"] == (0.5,)
    with pytest.raises(RuntimeError, match="maturin develop -m bindings/st-py/Cargo.toml"):
        stack.samples()

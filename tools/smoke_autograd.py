"""Check the installed native graph boundary and a complete nonlinear training loop."""

import importlib
import json
from pathlib import Path
import runpy

import spiraltorch as st


def main():
    native = importlib.import_module("spiraltorch.spiraltorch")
    shifted = st.Tensor(1, 4, [10000, 10001, 10002, 10003])
    normalized, inverse_std = shifted.layer_norm_stats(epsilon=1e-5)
    affine = shifted.layer_norm_affine(
        st.Tensor(1, 4, [1, 1, 1, 1]), st.Tensor(1, 4, [0, 0, 0, 0]), epsilon=1e-5
    )
    expected = [-1.34163542, -0.44721181, 0.44721181, 1.34163542]
    for result in [normalized, affine]:
        assert all(abs(x - y) < 3e-5 for x, y in zip(result.tolist()[0], expected))
    assert abs(inverse_std.tolist()[0][0] - 0.89442361) < 2e-6
    source = st.Tensor(1, 2, [2.0, 3.0])
    assert isinstance(source, native.Tensor)
    leaf = st.AutogradTensor.variable(source)
    assert leaf.value().is_snapshot()
    assert source.storage_token() != leaf.value().storage_token()
    loss = leaf.hadamard(leaf).sum()
    source.add_row_inplace([100.0, 100.0])
    loss.backward()
    assert loss.item() == 13.0
    assert leaf.grad().tolist() == [[4.0, 6.0]]
    gradient = leaf.grad()
    assert gradient.is_snapshot()
    gradient.add_row_inplace([100.0, 100.0])
    assert leaf.grad().tolist() == [[4.0, 6.0]]
    try:
        leaf.value().__dlpack__(max_version=(0, 8), copy=False)
    except BufferError:
        pass
    else:
        raise AssertionError("legacy export exposed a writable graph alias")
    assert st.AutogradSgd is native.AutogradSgd
    maximum = float.fromhex("0x1.fffffep+127")
    last = st.AutogradTensor.variable(st.Tensor(1, 1, [maximum]))
    last.backward(st.Tensor(1, 1, [-maximum]))
    failed = st.AutogradSgd([leaf, last], learning_rate=1.0)
    identities = [parameter.id() for parameter in failed.parameters()]
    try:
        failed.step()
    except ValueError:
        pass
    else:
        raise AssertionError("overflowing grouped SGD step succeeded")
    assert [parameter.id() for parameter in failed.parameters()] == identities
    assert leaf.value().tolist() == [[2.0, 3.0]]
    assert leaf.grad().tolist() == [[4.0, 6.0]]
    assert last.value().tolist() == [[maximum]]
    assert last.grad().tolist() == [[-maximum]]
    optimizer = st.AutogradSgd([leaf], learning_rate=0.5)
    optimizer.step()
    assert optimizer.parameter(0).value().tolist() == [[0.0, 0.0]]
    assert optimizer.parameter(0).grad() is None
    assert optimizer.parameter(0).id() != leaf.id()
    assert leaf.value().tolist() == [[2.0, 3.0]]
    example = Path(__file__).resolve().parents[1] / "examples" / "autograd_xor.py"
    result = runpy.run_path(str(example))["run"]()
    classification = example.with_name("autograd_classification.py")
    classification_result = runpy.run_path(str(classification))["run"]()
    layer_norm = example.with_name("autograd_layer_norm.py")
    layer_norm_result = runpy.run_path(str(layer_norm))["run"]()
    print(
        json.dumps(
            {
                "snapshot_boundary": "ok",
                "atomic_sgd_boundary": "ok",
                "training": result,
                "classification": classification_result,
                "layer_norm": layer_norm_result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

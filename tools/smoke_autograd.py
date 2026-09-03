"""Check the installed native graph boundary and a complete nonlinear training loop."""
import importlib
import json
from pathlib import Path
import runpy

import spiraltorch as st


def main():
    native = importlib.import_module("spiraltorch.spiraltorch")
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
    example = Path(__file__).resolve().parents[1] / "examples" / "autograd_xor.py"
    result = runpy.run_path(str(example))["run"]()
    classification = example.with_name("autograd_classification.py")
    classification_result = runpy.run_path(str(classification))["run"]()
    print(json.dumps({"snapshot_boundary": "ok", "training": result,
                      "classification": classification_result}, sort_keys=True))


if __name__ == "__main__":
    main()

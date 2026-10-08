"""Structural and Torch lifecycle checks, never a speed gate."""

import copy
import importlib.util
import struct
import sys
import weakref
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("benchmark_topos_shared_module_reference.py")
SPEC = importlib.util.spec_from_file_location("shared_module_benchmark", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def receipt(fields=None):
    fields = fields or [[0.2, -0.3, 0.4, -0.1], [0.5, 0.7], [0.1] * 4,
                        [0.0] * 4, [0.0] * 4, [0.0] * 2]
    chunks = [struct.pack("<" + "f" * len(v), *v) for v in fields]
    data = b"".join(chunks)
    return {
        "schema": "spiraltorch.topos_shared_nn_probe.v1", "status": "measured",
        "backend": "cpu", "dtype": "float32", "shape": [2, 2],
        "gate_layout": "shared_rows", "gate_gradient_reduction": "sum_without_additional_mean",
        "gradient_storage": "preallocated_zero_accumulator_for_reference_and_all_rounds",
        "vector_order": list(BENCH.FIELDS),
        "vector_shapes": [[2, 2], [1, 2], [2, 2], [2, 2], [2, 2], [1, 2]],
        "vector_sha256": [BENCH.digest(c) for c in chunks], "vectors_sha256": BENCH.digest(data),
        "config": {"iterations": 5, "coupling": 0.25, "saturation": 1.0, "porosity": 0.2},
        "warmup_per_route": 2, "round_order": [[0, 1], [1, 0]],
        "measurements_ms": {"forward": [0.1, 0.1], "forward_backward": [0.2, 0.2]},
    }, data


def test_valid_shared_shapes_and_byte_counts():
    payload, data = receipt()
    shapes, chunks = BENCH.validate(payload, data)
    assert shapes == payload["vector_shapes"]
    assert list(map(len, chunks)) == [16, 8, 16, 16, 16, 8]


@pytest.mark.parametrize("key,value", [
    ("gate_layout", "elementwise"), ("gate_gradient_reduction", "mean"),
    ("shape", [True, 2]), ("shape", [1048577, 1]),
    ("vector_shapes", [[2, 2]] * 6), ("warmup_per_route", True),
    ("round_order", [[False, True], [True, False]]),
    ("round_order", [[0, 1], [0, 1]]),
    ("measurements_ms", {"forward": [0.1, float("nan")], "forward_backward": [0.2, 0.2]}),
    ("gradient_storage", "none"), ("vectors_sha256", "0" * 64),
])
def test_rejects_changed_shapes_work_and_measurements(key, value):
    payload, data = receipt()
    payload[key] = value
    with pytest.raises(ValueError):
        BENCH.validate(payload, data)


def test_rejects_nonfinite_vectors_even_with_matching_hashes():
    payload, data = receipt([[float("inf")] * 4, [0.0] * 2, [0.0] * 4,
                             [0.0] * 4, [0.0] * 4, [0.0] * 2])
    with pytest.raises(ValueError, match="finite vectors"):
        BENCH.validate(payload, data)


@pytest.fixture
def reference_module(monkeypatch):
    pytest.importorskip("torch")
    path = PATH.with_name("benchmark_topos_module_reference.py")
    spec = importlib.util.spec_from_file_location("benchmark_topos_module_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "benchmark_topos_module_reference", module)
    return module


def test_torch_rejects_structurally_valid_but_wrong_outputs(reference_module):
    payload, data = receipt()
    with pytest.raises(AssertionError):
        BENCH.compare(payload, data)


def test_torch_accumulates_shared_vjp_and_releases_graph_before_next_sample(reference_module, monkeypatch):
    import torch

    payload, _ = receipt()
    fields = [[0.2, -0.3, 0.4, -0.1], [0.5, 0.7], [0.1] * 4]
    x = torch.tensor(fields[0]).reshape(2, 2).requires_grad_()
    gate = torch.tensor(fields[1]).reshape(1, 2).requires_grad_()
    dy = torch.tensor(fields[2]).reshape(2, 2)
    output = reference_module.reference(x, gate, payload["config"])
    dx, dg = torch.autograd.grad(output, (x, gate), dy)
    payload, data = receipt(fields + [t.detach().reshape(-1).tolist() for t in (output, dx, dg)])
    original = reference_module.reference
    previous = []

    def checked_reference(*args):
        assert all(handle() is None for handle in previous)
        result = original(*args)
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(reference_module, "reference", checked_reference)
    report = BENCH.compare(copy.deepcopy(payload), data)
    assert report["status"] == "passed" and report["threads"] == 1
    assert report["max_abs_error"] == {"output": 0.0, "grad_input": 0.0, "grad_gate": 0.0}
    assert len(previous) == 8 and all(handle() is None for handle in previous)

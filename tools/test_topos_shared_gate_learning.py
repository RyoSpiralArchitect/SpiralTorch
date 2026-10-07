"""Bounded receipt guards; these tests do not substitute for a Torch replay."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("check_topos_shared_gate_learning.py")
SPEC = importlib.util.spec_from_file_location("topos_shared_gate_check", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def receipt():
    cases = []
    for porosity in (0.0, 0.3):
        records = []
        for index in range(100):
            rows = (1, 3, 8, 2)[index % 4]
            records.append({"step": index, "shape": [rows, 5],
                            "input_layout": "row_major" if index % 2 == 0 else "col_major",
                            "input": [0.0] * (rows * 5), "target": [0.0] * (rows * 5),
                            "output": [0.0] * (rows * 5), "grad_input": [0.0] * (rows * 5),
                            "grad_gate": [0.0] * 5, "gate_after": [1.0] * 5, "loss": 0.0})
        cases.append({"config": {"iterations": 5, "coupling": 0.2, "saturation": 1.0, "porosity": porosity},
                      "initial_gate": [1.0] * 5, "records": records})
    return {"schema": "spiraltorch.topos_shared_gate_learning.v1", "status": "executed",
            "backend": "cpu", "dtype": "float32", "learning_rate": 0.03,
            "gate_layout": "shared_rows", "gate_gradient_reduction": "sum_without_additional_mean",
            "cases": cases}


def test_bounded_receipt_is_valid_without_torch():
    CHECK.validate_receipt(receipt())


def resident_receipt():
    payload = receipt()
    payload.update(schema="spiraltorch.topos_resident_learning.v1", backend="wgpu",
                   optimizer="resident_subtract_lr_times_gradient",
                   host_readback_during_updates=False, adapter="synthetic guard fixture")
    return payload


def test_resident_receipt_has_a_separate_explicit_contract():
    CHECK.validate_receipt(resident_receipt())


@pytest.mark.parametrize("field,value", [
    ("backend", "cpu"), ("optimizer", "Parameter::apply_step"),
    ("host_readback_during_updates", 0), ("host_readback_during_updates", True),
    ("adapter", ""), ("adapter", None),
    ("schema", "spiraltorch.topos_shared_gate_learning.v1"),
])
def test_resident_receipt_rejects_changed_execution_labels(field, value):
    payload = resident_receipt()
    payload[field] = value
    with pytest.raises(ValueError):
        CHECK.validate_receipt(payload)


@pytest.mark.parametrize("field,value", [
    ("shape", [True, 5]), ("shape", [257, 5]), ("step", False),
    ("input_layout", "unknown"), ("loss", float("nan")), ("loss", -1),
    ("input", [0.0] * 4), ("grad_input", [float("inf")] * 5),
    ("grad_gate", [0.0] * 15), ("gate_after", [True] * 5),
    ("gate_after", [3.5e38] * 5), ("output", ["0"] * 5),
])
def test_record_guard_rejects_malformed_values(field, value):
    payload = receipt()
    payload["cases"][0]["records"][0][field] = value
    with pytest.raises(ValueError):
        CHECK.validate_receipt(payload)


@pytest.mark.parametrize("field,value", [
    ("gate_layout", "elementwise"), ("gate_gradient_reduction", "mean"),
    ("backend", "wgpu"), ("learning_rate", True), ("learning_rate", float("nan")),
])
def test_contract_guard_rejects_changed_meaning(field, value):
    payload = receipt()
    payload[field] = value
    with pytest.raises(ValueError):
        CHECK.validate_receipt(payload)


@pytest.mark.parametrize("change", ["drop_step", "duplicate_step", "duplicate_case", "nonfinite_config"])
def test_trajectory_guard_rejects_incomplete_or_mislabeled_run(change):
    payload = receipt()
    if change == "drop_step":
        payload["cases"][0]["records"].pop()
    elif change == "duplicate_step":
        payload["cases"][0]["records"][1] = copy.deepcopy(payload["cases"][0]["records"][0])
    elif change == "duplicate_case":
        payload["cases"][1] = copy.deepcopy(payload["cases"][0])
    else:
        payload["cases"][0]["config"]["coupling"] = float("nan")
    with pytest.raises(ValueError):
        CHECK.validate_receipt(payload)


def test_review_torch_loss_overflow_cannot_be_certified(monkeypatch):
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("benchmark_topos_module_reference",
                                                PATH.with_name("benchmark_topos_module_reference.py"))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "benchmark_topos_module_reference", reference)
    payload = receipt()
    for case in payload["cases"]:
        case["initial_gate"] = [0.0] * 5
        for record in case["records"]:
            record["target"] = [float(2 ** 70)] * len(record["target"])
            record["loss"] = float(2 ** 140)
            record["gate_after"] = [0.0] * 5
    CHECK.validate_receipt(payload)
    with pytest.raises(ValueError, match="nonfinite Torch loss"):
        CHECK.compare(payload)

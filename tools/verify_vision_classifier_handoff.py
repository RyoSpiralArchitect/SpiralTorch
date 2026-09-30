#!/usr/bin/env python3
"""Replay a public browser ConvNeXt checkpoint and one SGD step in Python.

All model/loss/update math uses the installed Rust bindings. This checks one
bounded cross-runtime handoff, not real-data quality or performance.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


TOLERANCE = 2e-4


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def compare_values(actual, expected, label):
    if not expected or len(actual) != len(expected):
        raise ValueError(f"{label}: empty values or shape mismatch")
    errors = []
    for a, b in zip(actual, expected):
        if (isinstance(a, bool) or isinstance(b, bool)
                or not math.isfinite(a) or not math.isfinite(b)):
            raise ValueError(f"{label}: non-finite or non-numeric value")
        errors.append((abs(a - b), abs(a - b) / (1 + abs(b))))
    absolute = max(e[0] for e in errors)
    scaled = max(e[1] for e in errors)
    if scaled >= TOLERANCE:
        raise ValueError(f"{label}: scaled error {scaled} exceeds {TOLERANCE}")
    return dict(values=len(expected), max_abs_error=absolute, max_scaled_error=scaled)


def compare_checkpoints(actual, expected):
    metadata = [json.loads(payload) for payload in (actual, expected)]
    parameters = [value["backbone"].pop("parameters") + value.pop("head") for value in metadata]
    if metadata[0] != metadata[1] or len(parameters[0]) != len(parameters[1]) or not parameters[0]:
        raise ValueError("checkpoint metadata or parameter count mismatch")
    checks = []
    for a, b in zip(*parameters):
        av, bv = a.pop("values"), b.pop("values")
        if a != b:
            raise ValueError("checkpoint parameter names/shapes/order mismatch")
        checks.append(compare_values(av, bv, a["name"]))
    return dict(parameters=len(checks), values=sum(c["values"] for c in checks),
                max_abs_error=max(c["max_abs_error"] for c in checks),
                max_scaled_error=max(c["max_scaled_error"] for c in checks))


def validate_fixture(value):
    if (value.get("schema") != "spiraltorch.vision.classifier_clients.v1"
            or value.get("status") != "passed"
            or value.get("fixture_request") != "convnext-classifier-clients"
            or value.get("page_errors") != []
            or value.get("within_runtime_resume") != "bitwise"
            or value.get("accepted_revisions") != [1, 2, 3, 4, 6]
            or value.get("rejected_revision") != 5
            or value.get("continued_revision") != 7
            or value.get("continuation_rate") != 0.01
            or value.get("labels") != [0, 1]
            or value.get("parameters") != 24):
        raise ValueError("incomplete, failed or changed browser recipe")
    if value.get("rust_runtime_adapter", {}).get("backend") != "BrowserWebGpu":
        raise ValueError("browser runtime identity missing")
    if not {"frozen_checkpoint", "bitwise_resume", "invalid_labels_preserve_all_weights",
            "valid_retry", "retained_handles"}.issubset(value.get("checks", [])):
        raise ValueError("browser guard evidence missing")
    for name, length in (("normalized_input", 128), ("final_logits", 4), ("continued_logits", 4), ("losses", 4)):
        values = value[name]
        if len(values) != length:
            raise ValueError(f"{name}: fixture length mismatch")
        compare_values(values, values, name)
    compare_values([value["continued_loss"]], [value["continued_loss"]], "continued_loss")


def run(args, report):
    raw = args.browser_report.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("browser report exceeds 32 MiB")
    fixture = json.loads(raw)
    validate_fixture(fixture)
    payload = fixture["checkpoint_json"]
    report.update(browser_report_sha256=sha256(raw), browser_checkpoint_sha256=sha256(payload.encode()),
                  browser_continued_checkpoint_sha256=sha256(fixture["continued_checkpoint_json"].encode()),
                  browser_version=fixture["browser_version"], browser_runtime_adapter=fixture["rust_runtime_adapter"])

    import spiraltorch as st

    device = st.WgpuTensorDevice.create()
    adapter = device.adapter_info()
    if adapter.get("device_type") in (None, "Cpu"):
        raise RuntimeError("native WGPU runtime must not be a CPU fallback")
    report.update(python_runtime_adapter=adapter, spiraltorch_version=st.__version__)
    kind = st.ResidentConvNeXtClassifier
    owner = kind.from_checkpoint_json(device, payload)
    if (owner.attempted_updates != 6 or owner.input_shape != [2, 1, 8, 8]
            or owner.output_shape != [2, 2] or len(owner.parameter_names()) != 24):
        raise ValueError("restored public model metadata mismatch")
    if owner.checkpoint_snapshot().read_json() != payload:
        raise ValueError("imported checkpoint did not roundtrip byte-for-byte")
    report["checkpoint_roundtrip"] = "byte_identical"

    def read(tensor):
        return tensor.snapshot().read_values()

    x = device.upload(owner.input_shape, fixture["normalized_input"])
    forward = owner.forward(x)
    checks = report["checks"]
    checks["restored_logits"] = compare_values(read(forward.prediction_tensor()), fixture["final_logits"], "restored logits")
    host = kind.host_from_checkpoint_json(payload)
    images = [st.ImageTensor(1, 8, 8, fixture["normalized_input"][i:i + 64]) for i in (0, 64)]
    host_logits = [v for row in host.forward(images).tolist() for v in row]
    checks["host_inference_logits"] = compare_values(host_logits, fixture["final_logits"], "host logits")

    y = device.upload([2, 1], fixture["labels"])
    loss = st.nn.CrossEntropyWithLogits().evaluate_resident(forward.prediction_tensor(), y)
    gradients = owner.backward(forward, loss.prediction_gradient_tensor())
    update = owner.sgd(gradients, fixture["continuation_rate"])
    if update.read() != 7 or owner.attempted_updates != 7:
        raise ValueError("continued update was not accepted at revision 7")
    checks["continued_loss"] = compare_values(read(loss.loss_tensor()), [fixture["continued_loss"]], "continued loss")
    continued = owner.checkpoint_snapshot().read_json()
    # Validate the browser's expected checkpoint with the same Rust schema before comparison.
    kind.host_from_checkpoint_json(fixture["continued_checkpoint_json"])
    checks["continued_weights"] = compare_checkpoints(continued, fixture["continued_checkpoint_json"])
    checks["continued_logits"] = compare_values(read(owner.forward(x).prediction_tensor()), fixture["continued_logits"], "continued logits")
    report.update(restored_revision=6, continued_revision=7, native_continued_checkpoint_sha256=sha256(continued.encode()))
    if sha256(args.browser_report.read_bytes()) != report["browser_report_sha256"]:
        raise RuntimeError("browser report changed during verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema="spiraltorch.vision.classifier_handoff.v1", status="error", checks={},
                  tolerance=TOLERANCE,
                  boundary="One synthetic browser-to-native checkpoint and plain-SGD continuation; normalized input reused, no data/RNG resume, quality or timing claim")
    with args.output.open("x") as out:
        try:
            run(args, report)
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=report["status"], checks=len(report["checks"]), error=report.get("error"))))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

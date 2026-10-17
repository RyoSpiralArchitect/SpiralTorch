"""Publish bounded numeric evidence, never private corpus or model values."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


PREVIOUS_SHA = "ddc47f8bcd053391daf265cb1a3f914bdc78bceeb8c4ea4371792772ef80f280"
REQUEST_SHA = "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c"
TORCH_SHA = "96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b"
FIXTURE_SHA = "6d44dd27a675e5d2ffc19c59c9c03c49cc8a0b952917a8c5fc063675d10cef09"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def pinned(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"input identity drift: {path.name}")
    return raw


def validate_rounding(report, fixture):
    if (report["schema"] != "spiraltorch.wide_rounding.validation.v1"
            or report["fixture_sha256"] != FIXTURE_SHA
            or report["case_count"] != len(fixture["cases"])
            or len(report["actual_bits"]) != len(fixture["cases"])
            or len(report["attention"]) != 12):
        raise ValueError("rounding report coverage")
    actual = report["actual_bits"]
    if any(type(v) is not int or not 0 <= v < 2**32 for v in actual):
        raise ValueError("rounding output bits")
    failures = [i for i, (got, case) in enumerate(zip(actual, fixture["cases"]))
                if got != case[4]]
    keys = [(row["sign"], row["high"], row["tail_sign"]) for row in report["attention"]]
    if set(keys) != {(s, h, t) for s in (0, 0x80000000)
                    for h in (0x40800000, 0x40800001) for t in (-1, 0, 1)}:
        raise ValueError("attention case identity")
    attention_passed = []
    for row in report["attention"]:
        increment = int(row["tail_sign"] > 0 or (row["tail_sign"] == 0 and row["high"] & 1))
        expected = (row["high"] - (2 << 23) + increment) ^ row["sign"]
        passed = row["actual_bits"] == expected
        if row["expected_bits"] != expected or row["passed"] is not passed:
            raise ValueError("attention result drift")
        attention_passed.append(passed)
    if (report["mismatch_indices"] != failures
            or report["passed"] is not (not failures and all(attention_passed))):
        raise ValueError("rounding verdict drift")


def publish(repo, validation, previous, reference, output):
    helpers = module(repo / "benchmarks/results/2026-10-16-attention-adamw-numerics/language_probe.py")
    protocol = module(repo / "tools/byte_corpus_study.py")
    diagnostics = module(repo / "benchmarks/results/2026-10-16-language-capacity-budget/diagnose.py")
    request = pinned(validation / "learning/request.json", REQUEST_SHA)
    pinned(previous / "request.json", REQUEST_SHA)
    before = protocol.decode_json(pinned(previous / "native.json", PREVIOUS_SHA))
    expected = protocol.decode_json(pinned(reference, TORCH_SHA))
    result = json.loads((validation / "learning/comparison.json").read_bytes())
    after_raw = pinned(validation / "learning/native.json", result["actual_sha256"])
    pinned(validation / "learning/checkpoint.json", result["checkpoint_sha256"])
    receipt_raw = pinned(validation / "learning/build-receipt.json", result["build_receipt_sha256"])
    receipt = json.loads(receipt_raw)
    if receipt["binary_sha256"] != result["binary_sha256"]:
        raise ValueError("binary receipt drift")
    pinned(validation / "learning/resident_byte_learning", result["binary_sha256"])
    after = helpers.summarize(protocol, diagnostics, request, expected, protocol.decode_json(after_raw))
    if after != result["after"]:
        raise ValueError("stored strict comparison drift")
    fixture = json.loads(pinned(repo / "crates/st-backend-wgpu/tests/fixtures/wide_rounding.json", FIXTURE_SHA))
    reports = {}
    for name, backend in (("native-before.json", "Metal"), ("native-after.json", "Metal"),
                          ("browser-after.json", "BrowserWebGpu")):
        report = json.loads((validation / name).read_bytes())
        validate_rounding(report, fixture)
        if report["backend"] != backend:
            raise ValueError("unexpected execution backend")
        if backend == "BrowserWebGpu" and report["browser_hardware_execution"] != "unverified":
            raise ValueError("unexpected browser hardware claim")
        reports[name] = report
    output.mkdir(exist_ok=False)
    for name, report in reports.items():
        helpers.write(output / name, report)
    helpers.write(output / "language.json", {
        "schema": "spiraltorch.wide_rounding.language_followup.v1",
        "request_sha256": REQUEST_SHA, "torch_sha256": TORCH_SHA,
        "previous_sha256": PREVIOUS_SHA, "actual_sha256": result["actual_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "binary_sha256": result["binary_sha256"],
        "build_receipt_sha256": result["build_receipt_sha256"],
        "execution_wall_seconds": result["execution_wall_seconds"],
        "before": helpers.summarize(protocol, diagnostics, request, expected, before),
        "after": after,
        "scope": "Immediate preceding range repair versus final rounding repair; one seed; unchanged strict gate; no quality or speed claim",
    })
    helpers.write(output / "sources.json", {"compiler": receipt["compiler"],
        "profile": receipt["profile"], "sources": receipt["sources"],
        "scope": "Build-time byte identities, not attestation; local paths and binary omitted"})
    logs = ["build-before.log", "build-after.log", "test-wide.log", "build-wasm.log",
            "test-resident-tensor.log", "test-geometric-integration.log", "browser-attention-after.json"]
    helpers.write(output / "receipts.json", {"sha256": {name: sha((validation / name).read_bytes()) for name in logs},
        "wasm_sha256": {name: sha((repo / path).read_bytes()) for name, path in (
            ("rounding", "target/wide-rounding-web/resident_wide_rounding_browser_bg.wasm"),
            ("attention", "target/attention-vjp-rounding-web/resident_attention_vjp_browser_bg.wasm"))},
        "scope": "Raw logs and complete Attention regression report retained locally"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "validation", "previous", "reference", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    publish(*(getattr(args, key).resolve() for key in ("repo", "validation", "previous", "reference", "output")))


if __name__ == "__main__":
    main()

"""Publish scalar-only paired checks without superseding projective-v1 gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isfinite(a) and struct.pack("!d", a) == struct.pack("!d", b)
    return a == b


def difference(a, b):
    require(len(a) == len(b) and len(a) > 0, "contrast shape mismatch")
    # Match the resident f32 subtraction, not a silently higher-precision gate.
    return [struct.unpack("f", struct.pack("f", x - y))[0] for x, y in zip(a, b)]


def errors(a, b):
    require(len(a) == len(b) and len(a) > 0, "gradient shape mismatch")
    require(all(math.isfinite(x) for x in [*a, *b]), "nonfinite gradient")
    norm = math.sqrt(sum(x * x for x in b))
    error = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return {"max_abs": max(abs(x - y) for x, y in zip(a, b)),
            "reference_norm": norm, "relative_l2": error / norm if norm else None}


def close(a, b):
    result = errors(a, b)
    require(all(abs(x - y) <= 3e-6 + 5e-5 * abs(y) for x, y in zip(a, b)),
            "independent stopped VJP mismatch")
    return result


def relative(a, b):
    result = errors(a, b)
    require(result["reference_norm"] > 1e-8 and any(x != 0. for x in a)
            and result["relative_l2"] <= .002, "inert or inaccurate contrast")
    return result


def validate_runtime(report, fixture, stopped, *, schema="spiraltorch.resident_byte_query_pullback.validation.v1"):
    require(report["schema"] == schema,
            "unexpected paired schema")
    require(len(report["checks"]) == 2 and len(stopped["cases"]) == 2,
            "missing paired cases")
    rows = []
    for actual, reference, detached in zip(report["checks"], fixture["cases"][4:], stopped["cases"]):
        require(actual["name"] == reference["name"] == detached["name"], "case order mismatch")
        for field in ("forward_unchanged", "parameters_unchanged", "full_repeated_exact", "passed"):
            require(actual[field] is True, "failed paired invariant: " + field)
        require(actual["negative_controls_rejected"] == 2, "missing negative controls")
        close(actual["full_embedding_gradient"], reference["embedding_output_gradient"])
        close(actual["stopped_embedding_gradient"], detached["embedding"])
        require(exact(detached["embedding"], reference["query_controls"]["router_detached_embedding_gradient"]),
                "independent stopped reference drift")
        observed = difference(actual["full_embedding_gradient"], actual["stopped_embedding_gradient"])
        expected = difference(reference["embedding_output_gradient"], detached["embedding"])
        paired = relative(observed, expected)
        require(abs(paired["relative_l2"] - actual["paired_relative_l2"]) < 1e-12,
                "reported paired error mismatch")
        legacy_passed = True
        try:
            relative(difference(actual["full_embedding_gradient"], detached["embedding"]), expected)
        except ValueError:
            legacy_passed = False
        require(legacy_passed is actual["legacy_mixed_runtime_passed"], "old gate was relabeled")
        parameter_checks, expected_unchanged = [], []
        require(len(actual["full_parameter_gradients"]) == len(actual["stopped_parameter_gradients"])
                == len(detached["parameters"]) == len(reference["parameters"]), "truncated VJP family")
        for i, descriptor in enumerate(reference["parameters"]):
            name = descriptor["name"]
            full = actual["full_parameter_gradients"][i]
            values = actual["stopped_parameter_gradients"][i]
            cpu = detached["parameters"][i]
            close(full, reference["parameter_gradients"][i])
            error = close(values, cpu)
            if name.startswith("geometry."):
                relative(values, cpu)
            affected = i < 2 or name.startswith(("geometry.projection.", "geometry.raw_decay", "geometry.raw_phase"))
            if not affected:
                require(exact(full, values), "unrelated VJP changed: " + name)
                expected_unchanged.append(i)
            parameter_checks.append({"name": name, **error})
        require(expected_unchanged == actual["unchanged_parameter_slots"], "wrong unchanged-slot claim")
        rows.append({"name": actual["name"], "paired": paired,
            "legacy_mixed_runtime_passed": legacy_passed,
            "stopped_parameter_checks": parameter_checks,
            "embedding_stopped": close(actual["stopped_embedding_gradient"], detached["embedding"]),
            "negative_controls_rejected": 2, "unchanged_parameter_slots": expected_unchanged,
            "full_repeated_exact": True, "forward_unchanged": True, "parameters_unchanged": True})
    require(report["passed"] is True, "paired report failed")
    return {"paired_passed": True, "adapter": report["adapter"], "checks": rows}


def validate_bundle(fixture, stopped, native, native_legacy, old_native, browser, old_browser):
    require(fixture["schema"] == "spiraltorch.resident_byte_geometry_projective.torch_fixture.v1"
            and len(fixture["cases"]) == 6, "wrong frozen fixture")
    require(stopped["schema"] == "spiraltorch.query_stop.torch_reference.v1", "wrong stopped reference")
    require(browser["schema"] == "spiraltorch.query_pullback_bundle.browser.v1", "wrong browser bundle")
    require(exact(native_legacy, old_native), "native full path changed from archived baseline")
    old_browser = {k: v for k, v in old_browser.items() if k != "fixture_sha256"}
    require(exact(browser["legacy"], old_browser), "browser full path changed from archived baseline")
    result = {"schema": "spiraltorch.query_pullback_paired.scalar_results.v1",
        "scope": "paired correctness only; legacy six-case qualification NOT superseded; no corpus quality or speed claim"}
    for name, paired, legacy in [("native", native, native_legacy), ("browser", browser["paired"], browser["legacy"])]:
        result[name] = validate_runtime(paired, fixture, stopped)
        result[name].update({"legacy_entire_report_exact": True, "legacy_qualification_passed": legacy["passed"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Private original projective-v1 final directory")
    parser.add_argument("current", type=Path, help="Private current directory with native-paired.json, native-legacy.json, browser.json and diagnostic/")
    parser.add_argument("output", type=Path, help="Fresh public-safe JSON output; never overwritten")
    args = parser.parse_args()
    paths = [args.baseline / "reference.json", args.current / "diagnostic/stopped-reference.json",
             args.current / "native-paired.json", args.current / "native-legacy.json",
             args.baseline / "native-observed.json", args.current / "browser.json",
             args.baseline / "browser.json"]
    raw = [path.read_bytes() for path in paths]
    hashes = [hashlib.sha256(value).hexdigest() for value in raw]
    require(hashes[0] == "07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358", "frozen fixture hash changed")
    values = [json.loads(value) for value in raw]
    require(values[5]["fixture_sha256"] == hashes[0], "browser fixture mismatch")
    result = validate_bundle(*values)
    result["input_hashes"] = [{"role": role, "sha256": sha} for role, sha in zip(
        ["fixture", "stopped_reference", "native_paired", "native_legacy", "baseline_native", "browser_bundle", "baseline_browser"], hashes)]
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"native_paired": True, "browser_paired": True, "legacy_reports_exact": True}))


if __name__ == "__main__":
    main()

"""Scalar-only publication of wave batch-sum and unchanged QKV gate outcomes."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct

from validate_resident_query_pullback import exact, require, validate_runtime


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def scalars(source, fields):
    result = {key: source[key] for key in fields.split()}
    text = set("schema adapter scope name gate error".split())
    flags = set("passed logical_shapes all_inputs_and_cotangents_strided good_bad_good retained_gradients late_reduction_whole_family large_state_chart future_gradient_zero sensitivity_control retained_output f64_analytic_reference large_exponent_nonzero all_four_vjps observation_after_all_updates analytic_reference exact_f32 finite_family opposite_partials_cancel same_sign_whole_family_rejected".split())
    signed = {"expected_phase_gradient", "expected", "cpu", "gpu"}
    for key, value in result.items():
        if key in text:
            require(type(value) is str, "invalid text field: " + key)
        elif key in flags:
            require(type(value) is bool, "invalid flag: " + key)
        else:
            require(type(value) in (int, float) and math.isfinite(value)
                    and (key in signed or value >= 0), "invalid numeric field: " + key)
    return result


def wave(report):
    require(report["schema"] == "spiraltorch.causal_zspace_wave.validation.v1" and report["passed"] is True, "wave failed")
    require(len(report["checks"]) == 10 and len(report["radial"]["cases"]) == 9
            and len(report["learning"]["trace"]) == 16, "missing existing wave cases")
    controls = report["batch_reduction"]
    require(controls["analytic_reference"] is True and len(controls["checks"]) == 36, "missing analytic cases")
    expected_cases = set(itertools.product((24, 48, 96), itertools.permutations(range(3)), (0, 1)))
    observed = []
    for c in controls["checks"]:
        require(all(type(v) is int for v in [c["exponent"], c["terminal_channel"], *c["order"]]), "invalid case identity")
        key = (c["exponent"], tuple(c["order"]), c["terminal_channel"])
        observed.append(key)
        expected = f32(f32(.99) * .25) if key[2] == 0 else f32(f32(f32(.99) * .5) * f32(math.pi))
        require(c["exact_f32"] is True and c["finite_family"] is True
                and c["expected"] == expected and c["cpu"] == expected and c["gpu"] == expected,
                "analytic batch sum mismatch")
    require(len(set(observed)) == 36 and set(observed) == expected_cases, "duplicate or replaced analytic case")
    require(len(controls["overflow"]) == 2 and {c["terminal_channel"] for c in controls["overflow"]} == {0, 1}, "missing overflow cases")
    for c in controls["overflow"]:
        require(type(c["terminal_channel"]) is int, "invalid overflow identity")
        require(all(c[k] is True for k in ("opposite_partials_cancel", "finite_family", "same_sign_whole_family_rejected")), "overflow guard failed")
    checks = []
    for i, c in enumerate(report["checks"]):
        suffix = "features" if i == 0 else "terminal" if i == 1 else "both"
        require(c["name"] == f"case{i}_{suffix}" and c["logical_shapes"] is True
                and c["all_inputs_and_cotangents_strided"] is True, "invalid existing wave case")
        checks.append({**scalars(c, "name cpu_max_abs_error logical_shapes all_inputs_and_cotangents_strided"),
            "errors": scalars(c["errors"], "features final_state drive_vjp raw_decay_vjp raw_phase_vjp initial_state_vjp")})
    guards = scalars(report["guards"], "forward_operand_guards cotangent_guards atomic_rejections rejection_negative_controls good_bad_good retained_gradients late_reduction_whole_family large_state_chart")
    for key, count in {"forward_operand_guards": 4, "cotangent_guards": 2,
                       "atomic_rejections": 2, "rejection_negative_controls": 4}.items():
        require(guards[key] == count, "missing guard control")
    require(all(guards[k] is True for k in ("good_bad_good", "retained_gradients", "late_reduction_whole_family", "large_state_chart")), "failed guard")
    streaming = scalars(report["streaming"], "feature_max_abs_error state_max_abs_error vjp_max_abs_error prefix_max_abs_error future_gradient_zero sensitivity_control retained_output")
    require(all(streaming[k] is True for k in ("future_gradient_zero", "sensitivity_control", "retained_output")), "failed streaming control")
    radial = scalars(report["radial"], "f64_analytic_reference one_ulp_controls large_exponent_nonzero")
    require(radial == {"f64_analytic_reference": True, "one_ulp_controls": 2, "large_exponent_nonzero": True}, "missing radial controls")
    names = ["axis", "diagonal", "oblique", "oblique_up", "oblique_down", "oblique_negative", "large_exponent", "oblique_phase", "mixed_exponents"]
    radial["cases"] = [scalars(c, "name all_four_vjps cpu_and_gpu_max_abs_error expected_phase_gradient") for c in report["radial"]["cases"]]
    require([c["name"] for c in radial["cases"]] == names
            and all(c["all_four_vjps"] is True for c in radial["cases"]), "missing radial VJP")
    learning = scalars(report["learning"], "steps observation_after_all_updates")
    require(learning == {"steps": 16, "observation_after_all_updates": True}, "incomplete learning")
    learning["trace"] = [scalars(s, "revision feature_loss terminal_loss parameter_and_gradient_max_abs_error") for s in report["learning"]["trace"]]
    require([s["revision"] for s in learning["trace"]] == list(range(1, 17)), "missing update")
    # Explicit fields keep future tensor/checkpoint additions out of publication.
    return {**scalars(report, "schema passed adapter scope"), "checks": checks,
        "guards": guards, "streaming": streaming, "radial": radial, "learning": learning,
        "batch_reduction": {**scalars(controls, "analytic_reference scope"),
            "checks": [{**scalars(c, "exponent terminal_channel expected cpu gpu exact_f32 finite_family"), "order": list(c["order"])} for c in controls["checks"]],
            "overflow": [scalars(c, "terminal_channel opposite_partials_cancel finite_family same_sign_whole_family_rejected") for c in controls["overflow"]]}}


def failure_signatures(case):
    signatures = {(f["gate"], f["step"], f["geometry_slot"], f["threshold"])
                  for f in case["qualification_failures"]}
    if "query_controls_error" in case:
        error = case["query_controls_error"]
        require(type(error) is str, "invalid query error payload")
        # Only this initial embedding-contrast gate is inherited in the frozen
        # fixture. A router-update/shape/guard failure must not hide behind it.
        match = re.fullmatch(r"geometry derivative is inert or inaccurate: norm=([\d.eE+\-]+), actual=([\d.eE+\-]+), relative=([\d.eE+\-]+)", error)
        require(match is not None, "different query failure kind or location")
        values = [float(v) for v in match.groups()]
        require(all(math.isfinite(v) and v >= 0 for v in values), "invalid query failure measurement")
        signatures.add(("legacy_query_geometry_relative_l2", "initial_embedding_contrast", None, .002))
    return signatures


def model(report, baseline):
    require(report["schema"] == baseline["schema"] == "spiraltorch.resident_byte_geometry_projective.validation.v1", "wrong model schema")
    require(len(report["checks"]) == len(baseline["checks"]) == 6, "missing model cases")
    rows = []
    for new, old in zip(report["checks"], baseline["checks"]):
        require(new["name"] == old["name"] and exact(new["output"], old["output"]), "initial forward changed")
        require(failure_signatures(new).issubset(failure_signatures(old)), "new qualification failure")
        require(new["resume_trajectory"]["exact_parameters"] is True, "resume is not exact")
        new_trace, old_trace = new["learning"]["trace"], old["learning"]["trace"]
        require(len(new_trace) == len(old_trace) == 16, "truncated training trajectory")
        trace = []
        for n, o in zip(new_trace, old_trace):
            scalars(n, "loss")
            scalars(o, "loss")
            require(len(n["parameters"]) == len(o["parameters"]) > 0, "changed parameter family")
            delta = 0.
            for a, b in zip(n["parameters"], o["parameters"]):
                require(len(a) == len(b) > 0 and all(type(x) in (int, float) and math.isfinite(x) for x in [*a, *b]), "invalid parameter tensor")
                delta = max(delta, max(abs(x-y) for x, y in zip(a, b)))
            trace.append({"loss": n["loss"], "baseline_loss": o["loss"], "max_abs_parameter_change_from_baseline": delta})
        rows.append({"name": new["name"], "qualified": not failure_signatures(new),
            "qualification_failures": [scalars(f, "gate step geometry_slot threshold error") for f in new["qualification_failures"]],
            "baseline_qualification_failures": [scalars(f, "gate step geometry_slot threshold error") for f in old["qualification_failures"]],
            "query_controls_error": new.get("query_controls_error"), "baseline_query_controls_error": old.get("query_controls_error"),
            "exact_resume": True, "initial_forward_unchanged": True, "trace": trace})
    require(report["passed"] is all(c["qualified"] for c in rows), "incorrect overall qualification flag")
    return {"qualified_cases": sum(c["qualified"] for c in rows), "total_cases": 6, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Prior paired-v1 reviewed raw directory")
    parser.add_argument("projective", type=Path, help="Original projective-v1 final raw directory")
    parser.add_argument("current", type=Path, help="Current wave experiment raw directory")
    parser.add_argument("output", type=Path, help="Fresh public scalar JSON path")
    args = parser.parse_args()
    receipts = []
    def read(path, role):
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        receipts.append({"role": role, "sha256": sha})
        if role == "fixture":
            require(sha == "07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358", "fixture changed")
        return json.loads(data)
    fixture = read(args.projective / "reference.json", "fixture")
    stopped = read(args.baseline / "diagnostic/stopped-reference.json", "stopped_reference")
    require(fixture["schema"] == "spiraltorch.resident_byte_geometry_projective.torch_fixture.v1"
            and len(fixture["cases"]) == 6, "wrong frozen fixture")
    require(stopped["schema"] == "spiraltorch.query_stop.torch_reference.v1", "wrong stopped reference")
    old_native = read(args.baseline / "native-legacy.json", "baseline_native")
    old_browser = read(args.baseline / "browser.json", "baseline_browser")["legacy"]
    browser = read(args.current / "browser-lm.json", "browser_lm")
    require(browser["schema"] == "spiraltorch.query_pullback_bundle.browser.v1", "wrong browser bundle")
    require(browser["fixture_sha256"] == receipts[0]["sha256"], "browser fixture changed")
    result = {"schema": "spiraltorch.wave_batch_adjoint.scalar_results.v1",
        "scope": "batch-gradient correctness; old QKV gates retained; no speed, corpus quality or CUDA claim"}
    for runtime, current, prior, paired in [
        ("native", read(args.current / "native-projective.json", "native_lm"), old_native, read(args.current / "native-paired.json", "native_paired")),
        ("browser", browser["legacy"], old_browser, browser["paired"]),
    ]:
        result[runtime] = {"wave": wave(read(args.current / (runtime + "-wave.json"), runtime + "_wave")),
            "model": model(current, prior), "paired_query": validate_runtime(paired, fixture, stopped)}
    result["input_hashes"] = receipts
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"wave_passed": True, "no_new_model_failures": True, "native_model_qualified": result["native"]["model"]["qualified_cases"], "browser_model_qualified": result["browser"]["model"]["qualified_cases"]}))


if __name__ == "__main__":
    main()

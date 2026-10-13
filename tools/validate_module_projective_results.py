"""Verify complete v2 passes and publish scalars, never model arrays/checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

from byte_projective_diagnostic_contract import finite, require, validate_report
from validate_resident_query_pullback import close, exact, relative, validate_runtime


CONTRACT = {"residual": "projection_bias_before_residual.v1",
            "router_coordinate_vjp": "same_runtime_full_minus_stop.v1"}
SCHEMA = "spiraltorch.resident_byte_geometry_projective.validation.v2"


def flags(source, names):
    require(all(source.get(n) is True for n in names), "missing or failed control")
    return {n: True for n in names}


def paired_controls(report, fixture, stopped):
    result = validate_runtime(report, fixture, stopped,
                              schema="spiraltorch.resident_byte_query_pullback.validation.v2")
    # Adapter metadata is not part of the mathematical claim and is not a
    # public payload channel for arbitrary objects from imported receipts.
    return {k: result[k] for k in ("paired_passed", "checks")}


def causality_controls(control):
    result = flags(control, ["sensitivity_control", "suffix_gradient_zero"])
    for name, length in [("prefix", 512), ("other_document", 1024), ("extension", 1024)]:
        witness = control["numeric_witnesses"][name]
        require(len(witness["actual"]) == len(witness["expected"]) == length, "incomplete causality witness")
        error = close(witness["actual"], witness["expected"])
        key = name + "_max_abs_error"
        require(finite(control[key]) and abs(control[key] - error["max_abs"]) < 1e-12,
                "misreported causality error")
        result[key] = error["max_abs"]
    return result


def numeric_case(case, actual):
    descriptors = case["parameters"]
    geometry_end = 2 + len(case["learning"]["trace"][0]["geometry_gradients"])
    require(len(actual["parameter_gradients"]) == len(descriptors), "incomplete initial VJPs")
    gradients = []
    for slot, (p, a, e) in enumerate(zip(descriptors, actual["parameter_gradients"], case["parameter_gradients"], strict=True)):
        error = close(a, e)
        if 2 <= slot < geometry_end:
            relative(a, e)
        gradients.append({"name": p["name"], **error})
    steps = []
    require(len(actual["learning"]["trace"]) == len(case["learning"]["trace"]) == 16, "incomplete learning trace")
    for step, (a, e) in enumerate(zip(actual["learning"]["trace"], case["learning"]["trace"], strict=True), 1):
        require(a["revision"] == step and len(a["parameters"]) == len(descriptors), "incomplete parameter updates")
        params = [close(x, y) for x, y in zip(a["parameters"], e["parameters"], strict=True)]
        geometry = []
        require(len(a["geometry_gradients"]) == geometry_end - 2, "incomplete geometry VJPs")
        for slot, (x, y, reported) in enumerate(zip(a["geometry_gradients"], e["geometry_gradients"], a["geometry_gradient_relative_l2"], strict=True)):
            close(x, y)
            error = relative(x, y)
            require(finite(reported) and abs(reported - error["relative_l2"]) < 1e-12, "geometry error misreported")
            geometry.append({"name": descriptors[slot+2]["name"], **error})
        expected_weights = e.get("mixture_weights", [])
        observed_weights = a.get("mixture_weights", [])
        require(len(observed_weights) == len(expected_weights), "missing mixture update")
        mixtures = [close(x, y) for x, y in zip(observed_weights, expected_weights, strict=True)]
        steps.append({"step": step, "loss": a["loss"], "reference_loss": e["loss"],
                      "loss_error": close([a["loss"]], [e["loss"]]),
                      "parameter_count": len(params), "parameter_max_abs_error": max(p["max_abs"] for p in params),
                      "geometry_gradients": geometry, "mixture_weights": mixtures})
    require(actual["qualification_failures"] == [] and "query_controls_error" not in actual,
            "failed v2 condition; retain the raw report, do not publish it as a pass")
    return {"name": case["name"], "output": close(actual["output"], case["output"]),
            "parameter_gradients": gradients, "steps": steps}


def resume_controls(actual):
    resume = actual["resume_trajectory"]
    result = flags(resume, ["exact_losses", "exact_parameters", "exact_geometry_gradients",
                            "exact_embedding_gradients", "fresh_owner"])
    require(resume["split_revision"] == 7 and resume["final_revision"] == resume["verified_steps"] == 16,
            "incomplete resume interval")
    trace = actual["learning"]["trace"]
    require(len(resume["trace"]) == len(trace) == 16, "incomplete resumed trace")
    for a, b in zip(resume["trace"], trace, strict=True):
        for name in ["revision", "loss", "parameters", "geometry_gradients", "embedding_output_gradient"]:
            require(exact(a[name], b[name]), "resumed values differ: " + name)
        if "mixture_weights" in b:
            require(exact(a["mixture_weights"], b["mixture_weights"]), "resumed mixture differs")
            require(resume.get("exact_mixture_weights") is True, "missing mixture resume gate")
    result.update({"split_revision": 7, "verified_steps": 16, "final_revision": 16})
    for field in ["checkpoint_json", "final_checkpoint_json"]:
        require(isinstance(resume[field], str), "missing checkpoint")
        result[field + "_sha256"] = hashlib.sha256(resume[field].encode()).hexdigest()
    return result


def verify(fixture, report):
    require(fixture["schema"] == "spiraltorch.resident_byte_geometry_projective.torch_fixture.v2"
            and fixture["execution_contract"] == report["execution_contract"] == CONTRACT
            and len(fixture["cases"]) == 6, "wrong v2 fixture contract")
    require(fixture["tolerance"] == {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002}, "threshold drift")
    validate_report(fixture, report, schema=SCHEMA)
    stopped = {"cases": [{"name": c["name"],
                "embedding": c["query_controls"]["router_detached_embedding_gradient"],
                "parameters": c["query_controls"]["router_detached_parameter_gradients"]}
                for c in fixture["cases"][4:]]}
    paired = paired_controls(report["paired_query"], fixture, stopped)
    rows = []
    for index, (case, actual) in enumerate(zip(fixture["cases"], report["checks"], strict=True)):
        row = numeric_case(case, actual)
        row["embedding"] = close(actual["embedding_output_gradient"], case["embedding_output_gradient"])
        row["bias_gradients"] = [close(a, b) for a, b in zip(actual["bias_gradients"], case["bias_gradients"], strict=True)]
        row["initial_mixture_weights"] = [close(a, b) for a, b in zip(actual.get("mixture_weights", []), case.get("mixture_weights", []), strict=True)]
        row["causality"] = causality_controls(actual["causality"])
        row["tapes"] = flags(actual["tapes"], ["atomic_rejection", "foreign_tape", "good_bad_good",
            "invalid_bias_preserves_tape", "recovery", "retained_gradients", "retained_output",
            "stale_gradient", "superseded_tape"])
        row["checkpoint"] = flags(actual["checkpoint"], ["exact_logits_loss", "exact_parameter_gradients",
            "exact_parameters", "foreign_gradients", "foreign_tape", "immutable_capture",
            "nonzero_update_control", "rejected_update_resume"])
        row["resume"] = resume_controls(actual)
        control = actual["metric_controls"]
        require(control["relative_checker_negative_controls"] == 3
                and control["qk_scores_zero"] is (index % 2 == 0), "missing geometry controls")
        require(control["off_output_separation"] > 1e-6 and control["detached_embedding_gradient_separation"] > 1e-8,
                "inert geometry controls")
        row["geometry_controls"] = {k: control[k] for k in ["off_output_separation",
            "detached_embedding_gradient_separation", "output_contrast_relative_l2", "pullback_contrast_relative_l2"]}
        require(all(finite(v) for v in row["geometry_controls"].values())
                and 0 <= control["output_contrast_relative_l2"] <= .002
                and 0 <= control["pullback_contrast_relative_l2"] <= .002, "failed geometry contrast")
        if index >= 4:
            query = actual["query_controls"]
            require(query["detached_router_rejected"] is True and query["zero_initialization"] is False
                    and query["frozen_update_rejections"] == 48 * (index - 3), "missing query update controls")
            require(0 <= query["router_update_relative_l2"] <= .002
                    and query["shared_output_max_difference"] > 1e-6
                    and abs(query["router_coordinate_vjp_relative_l2"] - paired["checks"][index-4]["paired"]["relative_l2"]) < 1e-12,
                    "query control mismatch")
            row["query_controls"] = {k: query[k] for k in ["frozen_update_rejections", "router_update_relative_l2",
                "shared_output_max_difference", "router_coordinate_vjp_relative_l2"]}
        rows.append(row)
    guard = report["query_guard"]
    flags(guard, ["all_parameter_gradients_guarded", "embedding_gradient_guarded", "finite_control"])
    require(guard["atomic_rejections"] == 2 and guard["fault_block"] == 1, "missing query guard")
    boundary = report["projective_boundary"]
    require(boundary["cases"] == 8 and boundary["analytic_cancellation_cases"] == 5
            and boundary["nonzero_endpoint_components"] == 30
            and finite(boundary["max_relative_error"]) and 0 <= boundary["max_relative_error"] <= 8e-5, "missing projective boundary")
    require(report["passed"] is True, "v2 report did not qualify")
    return {"qualified_cases": 6, "checks": rows, "paired_query": paired,
            "projective_boundary": {k: boundary[k] for k in ["cases", "analytic_cancellation_cases", "nonzero_endpoint_components", "max_relative_error"]},
            "atomic_query_rejections": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hashes = []
    def read(name):
        raw = (args.current / name).read_bytes()
        hashes.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    fixture = read("reference.json")
    identity = read("fixture-identity.json")
    require(hashes[0]["sha256"] == identity["fixture_sha256"]
            and identity["legacy_fixture_sha256"] == "07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358", "fixture identity mismatch")
    native, browser = read("native.json"), read("browser.json")
    require(browser["schema"] == "spiraltorch.module_projective_bundle.browser.v2"
            and browser["fixture_sha256"] == hashes[0]["sha256"], "browser identity mismatch")
    result = {"schema": "spiraltorch.module_projective.scalar_results.v2",
              "scope": "v2 matched-graph qualification only; original v1 remains 4/6; no quality or speed claim",
              "execution_contract": CONTRACT, "native": verify(fixture, native),
              "browser": verify(fixture, browser["report"]), "input_hashes": hashes}
    with args.output.open("x") as output:
        json.dump(result, output, allow_nan=False)
        output.write("\n")
    print(json.dumps({"native_qualified": 6, "browser_qualified": 6, "original_v1_unchanged": True}))


if __name__ == "__main__":
    main()

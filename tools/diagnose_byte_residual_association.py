"""Compare explicit projection/residual boundaries; never rewrite v1 evidence."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("same_state", Path(__file__).with_name("diagnose_resident_byte_projective.py"))
same = importlib.util.module_from_spec(spec)
spec.loader.exec_module(same)
spec = importlib.util.spec_from_file_location("byte_contract", Path(__file__).with_name("byte_projective_diagnostic_contract.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def compare(actual, expected):
    a, b = same.tensor(actual), same.tensor(expected)
    contract.require(a.shape == b.shape and a.numel() > 0, "comparison shape mismatch")
    contract.require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "non-finite comparison")
    return {**same.errors(a, b), "absolute_gate": bool(torch.all((a-b).abs() <= 3e-6 + 5e-5*b.abs()))}


def geometry_check(actual, expected):
    error = compare(actual, expected)
    error["relative_gate"] = (error["reference_norm"] > 1e-8 and any(v != 0. for v in actual)
                              and error["relative_l2"] is not None and error["relative_l2"] <= .002)
    return error


def replay(case, grouped):
    values = [p["values"] for p in case["parameters"]]
    initial = same.evaluate(case, values, torch.float32, module_residual=grouped)
    if not grouped:
        contract.require(torch.equal(initial["output"], same.tensor(case["output"])), "legacy output replay mismatch")
        for g, expected in zip(initial["gradients"], case["parameter_gradients"], strict=True):
            contract.require(torch.equal(g, same.tensor(expected)), "legacy initial gradient replay mismatch")
    trace = []
    for step in range(16):
        result = same.evaluate(case, values, torch.float32, "ce", module_residual=grouped)
        values = [(same.tensor(p).float() - case["learning"]["rate"] * g.float()).tolist()
                  for p, g in zip(values, result["gradients"], strict=True)]
        geometries = [g.tolist() for g in result["gradients"][2:result["geometry_end"]]]
        if not grouped:
            frozen = case["learning"]["trace"][step]
            contract.require(values == frozen["parameters"] and geometries == frozen["geometry_gradients"],
                             f"legacy update replay mismatch: {case['name']} step {step+1}")
        trace.append({"parameters": values, "geometry_gradients": geometries, "loss": result["loss"]})
    return initial, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture, reports, hashes = contract.load_inputs(args.fixture, args.current)
    rows = []
    for i, case in enumerate(fixture["cases"]):
        replay(case, False)
        initial, trace = replay(case, True)
        runtimes = []
        for name, report in reports.items():
            actual = report["checks"][i]
            steps = []
            for step, (observed, ref) in enumerate(zip(actual["learning"]["trace"], trace, strict=True)):
                g = [geometry_check(a, b) for a, b in zip(observed["geometry_gradients"], ref["geometry_gradients"], strict=True)]
                parameters = [compare(a, b) for a, b in zip(observed["parameters"], ref["parameters"], strict=True)]
                steps.append({"step": step+1, "observed_loss": observed["loss"], "grouped_loss": ref["loss"],
                    "geometry_gradients": g, "parameter_max_abs_error": max(e["max_abs"] for e in parameters),
                    "all_parameter_absolute_gates": all(e["absolute_gate"] for e in parameters)})
            initial_gradients = [compare(g, expected.tolist()) for g, expected in zip(actual["parameter_gradients"], initial["gradients"], strict=True)]
            failures, query = contract.inherited_failures(actual)
            runtimes.append({"runtime": name, "initial_output": compare(actual["output"], initial["output"].tolist()),
                "initial_parameter_gradients": initial_gradients, "steps": steps,
                "inherited_qualification_failures": failures, "inherited_query_error": query})
        rows.append({"name": case["name"], "legacy_exact_replay": True, "runtimes": runtimes})
    result = {"schema": "spiraltorch.residual_association.diagnostic.v1", "fixture_sha256": hashes[0]["sha256"],
        "input_hashes": hashes, "torch": torch.__version__,
        "scope": "alternate module-grouped CPU evaluation; original six-case qualification unchanged", "cases": rows}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    for row in rows:
        print(json.dumps({"name": row["name"], "runtimes": [{"runtime": r["runtime"],
            "geometry_failures": [{"step": s["step"], "slot": j, "relative": e["relative_l2"]}
                for s in r["steps"] for j, e in enumerate(s["geometry_gradients"]) if not e["relative_gate"]],
            "step12_phase_relative": r["steps"][11]["geometry_gradients"][3]["relative_l2"]}
            for r in row["runtimes"]]}))


if __name__ == "__main__":
    main()

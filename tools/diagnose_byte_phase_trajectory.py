"""Locate phase-VJP trajectory sensitivity without changing frozen model gates."""
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


def phase(case, values, dtype):
    return same.evaluate(case, values, dtype, "ce")["gradients"][5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture, reports, hashes = contract.load_inputs(args.fixture, args.current)
    case = fixture["cases"][1]
    if case["name"] != "projective_causal_geometry_blocks2_metric_onlyFalse" or len(case["parameters"]) != 37:
        raise ValueError("unexpected case")
    native = reports["native"]["checks"][1]
    browser = reports["browser"]["checks"][1]
    initial = [p["values"] for p in case["parameters"]]
    frozen_trace = case["learning"]["trace"]
    if len(frozen_trace) != 16:
        raise ValueError("incomplete reference")
    for step, row in enumerate(frozen_trace):
        values = initial if step == 0 else frozen_trace[step - 1]["parameters"]
        if not torch.equal(phase(case, values, torch.float32), same.tensor(row["geometry_gradients"][3])):
            raise ValueError(f"CPU-f32 reference not reproduced at step {step + 1}")
    reference_state = frozen_trace[10]["parameters"]
    ref64 = phase(case, reference_state, torch.float64)
    results = []
    for runtime, report in [("native", native), ("browser", browser)]:
        trace = report["learning"]["trace"]
        if report["name"] != case["name"] or len(trace) != 16:
            raise ValueError("incomplete runtime")
        rows = []
        for step, row in enumerate(trace):
            values = initial if step == 0 else trace[step - 1]["parameters"]
            cpu = same.evaluate(case, values, torch.float32, "ce")
            f64 = phase(case, values, torch.float64)
            frozen = same.tensor(frozen_trace[step]["geometry_gradients"][3])
            observed = same.tensor(row["geometry_gradients"][3])
            updates = []
            for descriptor, before, grad, actual, reference in zip(case["parameters"], values, cpu["gradients"], row["parameters"], frozen_trace[step]["parameters"], strict=True):
                predicted = (same.tensor(before).float() - case["learning"]["rate"] * grad.float()).double()
                actual = same.tensor(actual)
                updates.append({"name": descriptor["name"],
                    "next_value_vs_same_state_cpu_update": same.errors(actual, predicted),
                    "trajectory_value_vs_frozen": same.errors(actual, same.tensor(reference))})
            rows.append({"step": step + 1, "gpu_vs_frozen": same.errors(observed, frozen),
                "gpu_vs_same_state_f32": same.errors(observed, cpu["gradients"][5]),
                "gpu_vs_same_state_f64": same.errors(observed, f64),
                "same_state_f32_vs_f64": same.errors(cpu["gradients"][5], f64), "updates": updates})
        runtime_state = trace[10]["parameters"]
        runtime64 = phase(case, runtime_state, torch.float64)
        attributions = []
        for slot, descriptor in enumerate(case["parameters"]):
            inject, repair = list(reference_state), list(runtime_state)
            inject[slot], repair[slot] = runtime_state[slot], reference_state[slot]
            injected, repaired = phase(case, inject, torch.float64), phase(case, repair, torch.float64)
            attributions.append({"slot": slot, "name": descriptor["name"],
                "parameter_drift": same.errors(same.tensor(runtime_state[slot]), same.tensor(reference_state[slot])),
                "inject_one_tensor_vs_reference_phase": same.errors(injected, ref64),
                "repair_one_tensor_vs_reference_phase": same.errors(repaired, ref64)})
        results.append({"runtime": runtime, "trace": rows,
            "step12_state_effect_f64": same.errors(runtime64, ref64), "step12_all_tensor_attributions": attributions})
    result = {"schema": "spiraltorch.phase_trajectory_origin.diagnostic.v1", "case": case["name"],
        "scope": "state attribution and same-state diagnostics; original qualification failures retained; not an adoption gate",
        "torch": torch.__version__, "frozen_f32_phase_exact_steps": 16, "runtimes": results, "input_hashes": hashes}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    for runtime in results:
        top = sorted(runtime["step12_all_tensor_attributions"], key=lambda c: c["inject_one_tensor_vs_reference_phase"]["absolute_l2"], reverse=True)[:5]
        print(json.dumps({"runtime": runtime["runtime"], "state_effect": runtime["step12_state_effect_f64"], "top_sensitivity": top}))


if __name__ == "__main__":
    main()

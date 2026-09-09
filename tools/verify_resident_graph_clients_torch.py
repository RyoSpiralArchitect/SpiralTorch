#!/usr/bin/env python3
"""Independent eager Torch replay of public graph clients and weight-only resume.

Reuses the Torch graph oracle, never SpiralTorch math. No timing/quality claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def checked_plan(text, sha, topology=None, parameters=None):
    if digest(text) != sha:
        raise ValueError("plan digest mismatch")
    plan = json.loads(text)
    if plan["schema"] != "spiraltorch.nn.inference_plan.v2":
        raise ValueError("graph schema mismatch")
    if topology is not None:
        if (plan["input_shape"] != topology["input_shape"] or plan["stages"] != topology["stages"]
                or [(p["role"], p["shape"]) for p in plan["parameters"]]
                != [(p["role"], p["shape"]) for p in topology["parameters"]]):
            raise ValueError("resume changed graph topology or parameter IDs")
    if parameters is not None:
        # Rust JSON prints shortest roundtripping f32; client snapshots use f64 JSON.
        # Compare the actual transported f32 bits, including signed zero, not decimals.
        def bits(value):
            return (value["role"], value["shape"], [struct.pack("<f", x) for x in value["values"]])
        if [bits(p) for p in plan["parameters"]] != [bits(p) for p in parameters]:
            raise ValueError("exported weights differ from the captured state")
    return plan


def load_inputs(python_path, browser_path, return_path):
    data, sources = [], []
    for path in (python_path, browser_path, return_path):
        raw = path.read_bytes()
        data.append(json.loads(raw))
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
    python, browser, returned = data
    for value, schema in zip(data, ("spiraltorch.nn.graph_client_fixture.v1",
                                  "spiraltorch.nn.graph_browser_fixture.v1", "spiraltorch.nn.graph_handoff_return.v1")):
        if value.get("status") != "passed" or value.get("schema") != schema or len(value.get("cases", [])) != 24:
            raise ValueError("incomplete or failed client fixture")
    if (browser.get("cpu_only") is not False or browser.get("rollback_recovery_cases") != 2
            or browser.get("analytic_policy_cases") != 2 or browser.get("parameterless_cases") != 1
            or browser["asset_sha256"]["/fixture.json"] != sources[0]["sha256"]
            or returned["fixture_sha256"] != sources[0]["sha256"] or returned["browser_sha256"] != sources[1]["sha256"]):
        raise ValueError("handoff lineage or guard evidence missing")
    expected = {(seed, shape, policy, kernel, accumulation) for seed in (17, 29)
                for shape in ((2,), (3, 2), (2, 129, 2)) for policy in ("exact", "module_compatible")
                for kernel, accumulation in (("scalar", "sequential"), ("register_2x2", "compensated"))}
    recipes = [(c["seed"], tuple(c["shape"]), c["policy"], c["kernel"], c["accumulation"]) for c in python["cases"]]
    if len(recipes) != len(set(recipes)) or set(recipes) != expected:
        raise ValueError("fixture recipe matrix differs")
    for index, (case, remote, back) in enumerate(zip(python["cases"], browser["cases"], returned["cases"])):
        if remote["index"] != index or back["index"] != index or remote["plan_sha256"] != case["plan_sha256"]:
            raise ValueError("case order/lineage differs")
        if case["learning_rate"] != .03125 or len(case["states"]) != 4 or len(remote["states"]) != 4 or len(remote["resumed_states"]) != 2:
            raise ValueError("SGD recipe differs")
        for capture in (case, remote, back):
            if capture["adapter"].get("device_type") in (None, "Cpu"):
                raise ValueError("missing device identity or CPU fallback")
        plan = checked_plan(case["plan_json"], case["plan_sha256"])
        if plan["input_shape"] != case["shape"]:
            raise ValueError("N-D shape drift")
        checked_plan(case["half_plan_json"], case["half_plan_sha256"], plan, case["states"][1]["parameters"])
        checked_plan(case["final_plan_json"], case["final_plan_sha256"], plan, case["states"][-1]["parameters"])
        checked_plan(remote["final_plan_json"], remote["final_plan_sha256"], plan, remote["states"][-1]["parameters"])
        for states in (case["states"], remote["states"], remote["resumed_states"], [back["state"]]):
            for step, state in enumerate(states, 1):
                if (state["submitted_step"] != step or state["batch_generation"] != 1
                        or state["gradient_policy"] != case["policy"] or state["input_shape"] != case["shape"]
                        or state["output_shape"] != case["shape"] or state["stage_count"] != 5
                        or [(p["role"], p["shape"]) for p in state["parameters"]]
                        != [(p["role"], p["shape"]) for p in plan["parameters"]]):
                    raise ValueError("state metadata/ownership drift")
    return data, sources


def replay_case(case, plan, states):
    return dict(plan=json.loads(plan), policy={"exact": "Exact", "module_compatible": "ModuleCompatible"}[case["policy"]],
        seed=case["seed"], input_shape=case["shape"], input=case["input"], target=case["target"],
        rates=[case["learning_rate"]] * len(states), steps=[dict(loss=s["loss"], prediction=s["prediction"],
            input_gradient=s["input_gradient"], parameters=[p["values"] for p in s["parameters"]],
            raw_gradients=[p["raw"] for p in s["parameters"]], effective_gradients=[p["effective"] for p in s["parameters"]]) for s in states])


def run(args, report):
    (python, browser, returned), sources = load_inputs(args.python_report, args.browser_report, args.return_report)
    report["sources"] = sources
    spec = importlib.util.spec_from_file_location("_graph_torch_oracle", Path(__file__).with_name("validate_resident_graph_training_vs_torch.py"))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch = reference.torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    report["torch"] = torch.__version__
    if "mps" in args.devices and (not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"):
        raise RuntimeError("MPS requires a real device and explicit disabled fallback")
    for device in args.devices:
        for index, (case, remote, back) in enumerate(zip(python["cases"], browser["cases"], returned["cases"])):
            for lane, plan, states in (("python", case["plan_json"], case["states"]),
                    ("browser", case["plan_json"], remote["states"]),
                    ("python_to_browser_resume", case["half_plan_json"], remote["resumed_states"]),
                    ("browser_to_python_resume", remote["final_plan_json"], [back["state"]])):
                checked = reference.replay(replay_case(case, plan, states), device, expected_steps=len(states))
                report["cases"].append(dict(index=index, lane=lane, **checked))
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("fixture changed during validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-report", required=True, type=Path)
    parser.add_argument("--browser-report", required=True, type=Path)
    parser.add_argument("--return-report", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = dict(schema="spiraltorch.nn.graph_clients.torch_validation.v1", status="error", cases=[],
        boundary="float32 tanh GELU/Relu, mean-MSE VJP and plain SGD; weight-only resume; correctness, not throughput or model quality")
    with args.output.open("x") as out:
        try:
            run(args, report)
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=report["status"], cases=len(report["cases"]), error=report.get("error"),
        max_abs_error=max((c["max_abs_error"] for c in report["cases"]), default=None))))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

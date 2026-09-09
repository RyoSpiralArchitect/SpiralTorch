"""Public Python -> browser -> Python weight-only graph training handoff.

Captures real GPU states, not a timing or training-quality claim. Validate with
tools/verify_resident_graph_clients_torch.py in an independent PyTorch process.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random

import spiraltorch as st


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def model(seed):
    rng = random.Random(seed)
    net = st.nn.Sequential()
    net.add(st.nn.Scaler.from_gain("input_gain", st.Tensor(1, 2, [.75, 1.25])))
    net.add(st.nn.Linear(2, 3, name="up"))
    net.add(st.nn.Gelu())
    net.add(st.nn.Relu())
    net.add(st.nn.Scaler.from_gain("output_gain", st.Tensor(1, 3, [1., .5, 1.5])))
    net.add(st.nn.Linear(3, 2, name="down"))
    net.load_state_dict([
        ("input_gain::gain", st.Tensor(1, 2, [.75, 1.25])),
        ("output_gain::gain", st.Tensor(1, 3, [1., .5, 1.5])),
        ("up::weight", st.Tensor(2, 3, [rng.uniform(-.5, .5) for _ in range(6)])),
        ("up::bias", st.Tensor(1, 3, [.125, -.25, .5])),
        ("down::weight", st.Tensor(3, 2, [rng.uniform(-.5, .5) for _ in range(6)])),
        ("down::bias", st.Tensor(1, 2, [.125, -.25])),
    ])
    return net


def state_payload(state):
    return dict(loss=state.loss, input_shape=list(state.input_shape), output_shape=list(state.output_shape),
                submitted_step=state.submitted_step, batch_generation=state.batch_generation,
                gradient_policy=state.gradient_policy, stage_count=state.stage_count,
                prediction=state.prediction_values(), input_gradient=state.input_gradient_values(),
                parameters=[dict(role=state.parameter_role(i), shape=list(state.parameter_shape(i)),
                                 values=state.parameter_values(i), raw=state.parameter_gradient_values(i),
                                 effective=state.effective_gradient_values(i))
                            for i in range(state.parameter_count)])


def require_gpu(gpu):
    info = gpu.adapter_info()
    if info.get("device_type") in (None, "Cpu"):
        raise RuntimeError("missing adapter identity or CPU fallback in GPU evidence")
    return info


def capture(report):
    for seed in (17, 29):
        for shape in ([2], [3, 2], [2, 129, 2]):
            plan = model(seed).inference_plan(shape)
            original = plan.to_json()
            rows = math.prod(shape[:-1])
            x = [((i * 7 + seed) % 31 - 15) / 16 for i in range(rows * 2)]
            y = [((i * 3 + seed) % 19 - 9) / 32 for i in range(rows * 2)]
            for policy in ("exact", "module_compatible"):
                for kernel, accumulation in (("scalar", "sequential"), ("register_2x2", "compensated")):
                    gpu = plan.compile_graph_training_wgpu(gradient_policy=policy, kernel=kernel, accumulation=accumulation)
                    adapter = require_gpu(gpu)
                    gpu.upload_batch_values(x, y)
                    snapshots, losses = [], []
                    for step in range(1, 5):
                        gpu.step(.03125)
                        snapshots.append(gpu.state_snapshot())
                        losses.append(gpu.loss_snapshot())
                        if step == 2:
                            half = gpu.parameter_snapshot()
                    final = gpu.parameter_snapshot()
                    del gpu
                    gc.collect()
                    states = [state_payload(s.read_state()) for s in snapshots]
                    loss_values = [s.read() for s in losses]
                    if loss_values != [s["loss"] for s in states] or plan.to_json() != original:
                        raise RuntimeError("snapshot loss or source-plan ownership drifted")
                    half_json, final_json = half.read_plan().to_json(), final.read_plan().to_json()
                    report["cases"].append(dict(seed=seed, shape=shape, policy=policy, kernel=kernel,
                        accumulation=accumulation, input=x, target=y, learning_rate=.03125,
                        plan_json=original, plan_sha256=sha(original), states=states, adapter=adapter,
                        half_plan_json=half_json, half_plan_sha256=sha(half_json),
                        final_plan_json=final_json, final_plan_sha256=sha(final_json)))


def receive(fixture_path, browser_path, report):
    fixture_raw, browser_raw = fixture_path.read_bytes(), browser_path.read_bytes()
    fixture, browser = json.loads(fixture_raw), json.loads(browser_raw)
    if (fixture.get("schema") != "spiraltorch.nn.graph_client_fixture.v1" or fixture.get("status") != "passed"
            or browser.get("schema") != "spiraltorch.nn.graph_browser_fixture.v1"
            or browser.get("status") != "passed" or browser.get("cpu_only") is not False
            or len(fixture["cases"]) != 24 or len(browser["cases"]) != 24
            or browser["asset_sha256"]["/fixture.json"] != hashlib.sha256(fixture_raw).hexdigest()):
        raise ValueError("expected a complete successful browser handoff of this exact fixture")
    report["fixture_sha256"] = hashlib.sha256(fixture_raw).hexdigest()
    report["browser_sha256"] = hashlib.sha256(browser_raw).hexdigest()
    for index, (item, returned) in enumerate(zip(fixture["cases"], browser["cases"])):
        if (returned["index"] != index or returned["plan_sha256"] != item["plan_sha256"]
                or sha(returned["final_plan_json"]) != returned["final_plan_sha256"]):
            raise ValueError("handoff identity mismatch")
        plan = st.nn.InferencePlan.from_json(returned["final_plan_json"])
        gpu = plan.compile_graph_training_wgpu(gradient_policy=item["policy"], kernel=item["kernel"],
                                               accumulation=item["accumulation"])
        adapter = require_gpu(gpu)
        if (gpu.submitted_steps, gpu.batch_generation) != (0, 0):
            raise RuntimeError("weight-only resume inherited runtime counters")
        gpu.upload_batch_values(item["input"], item["target"])
        gpu.step(item["learning_rate"])
        state = gpu.state_snapshot()
        del gpu
        report["cases"].append(dict(index=index, state=state_payload(state.read_state()), adapter=adapter))
    if fixture_path.read_bytes() != fixture_raw or browser_path.read_bytes() != browser_raw:
        raise RuntimeError("handoff evidence changed during replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if bool(args.browser_report) != bool(args.fixture):
        parser.error("return handoff requires both --fixture and --browser-report")
    report = dict(schema="spiraltorch.nn.graph_handoff_return.v1" if args.browser_report else
                  "spiraltorch.nn.graph_client_fixture.v1", status="error", cases=[],
                  boundary="public GPU client capture; independent correctness validation is separate; no timing/quality claim")
    with args.output.open("x") as out:
        try:
            if args.browser_report:
                receive(args.fixture, args.browser_report, report)
            else:
                capture(report)
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=report["status"], cases=len(report["cases"]), error=report.get("error"))))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Existing NN modules -> Rust GPU VJP/SGD -> portable weight-only handoff.

Produces 18 VJP cases and three 128-update runs through the public installed
Python package. No Python optimizer or derivative implementation is used here.
Validate numerical correctness independently with the companion PyTorch tool.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct

import spiraltorch as st


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def model(seed):
    net = st.nn.Sequential()
    net.add(st.nn.Linear(4, 7, name="up"))
    net.add(st.nn.Gelu())
    net.add(st.nn.Linear(7, 3, name="down"))
    parameters = []
    for name, rows, cols in (("up::weight", 4, 7), ("up::bias", 1, 7),
                             ("down::weight", 7, 3), ("down::bias", 1, 3)):
        values = [((i * 17 % 23) - 11) / 32 for i in range(seed, seed + rows * cols)]
        seed += rows * cols
        parameters.append((name, st.Tensor(rows, cols, values)))
    net.load_state_dict(parameters)
    return net


def batch(rows):
    x = [f32(f32(((r * 7 + c * 11) % 29) / 16.) - f32(.8))
         for r in range(rows) for c in range(4)]
    y = [f32(f32(f32(f32(.4) * x[r*4+c]) - f32(f32(.2) * x[r*4+3])) + f32(c / 10.))
         for r in range(rows) for c in range(3)]
    return x, y


def layers(plan):
    return [dict(inner=p["inner"], cols=p["cols"], weights=p["weight"], bias=p["bias"], gelu=p["gelu"])
            for p in json.loads(plan.to_json())["stages"]]


def state_payload(state):
    flat = lambda t: [v for row in t.tolist() for v in row]
    return dict(loss=state.loss, prediction=state.prediction_values(),
                input_gradient=state.input_gradient_values(),
                parameters=layers(state.to_plan()), parameter_gradients=[
                    dict(weights=flat(state.weight_gradient(i)), bias=flat(state.bias_gradient(i)))
                    for i in range(state.stage_count)],
                submitted_step=state.submitted_step, batch_generation=state.batch_generation)


def checked_state(gpu, rate):
    gpu.step(rate)
    return gpu.state_snapshot().read_state()


def run(report):
    report["python_build_info"] = st.build_info()
    for seed, shape in ((17, [4]), (29, [3, 4]), (43, [2, 47, 4])):
        net = model(seed)
        plan = net.inference_plan(shape)
        x, y = batch(math.prod(shape[:-1]))
        for kernel in ("scalar", "register_2x2"):
            for accumulation in ("sequential", "tiled", "compensated"):
                gpu = plan.compile_training_wgpu(kernel=kernel, accumulation=accumulation)
                report["adapter"] = gpu.adapter_info()
                if report["adapter"]["device_type"] == "Cpu":
                    raise RuntimeError("CPU fallback is not GPU evidence")
                gpu.upload_batch_values(x, y)
                gpu.step(0.)
                probe = gpu.state_snapshot()
                gpu.step(.125)
                updated = gpu.state_snapshot()
                del gpu
                gc.collect()
                probe, updated = probe.read_state(), updated.read_state()
                if (probe.submitted_step, updated.submitted_step, updated.batch_generation) != (1, 2, 1):
                    raise RuntimeError("snapshot metadata drifted")
                if list(updated.input_shape) != shape or list(updated.output_shape) != shape[:-1] + [3]:
                    raise RuntimeError("N-D shape drifted")
                report["vjps"].append(dict(seed=seed, shape=shape, kernel=kernel, accumulation=accumulation,
                    input=x, target=y, plan_json=plan.to_json(), initial_parameters=layers(plan),
                    learning_rate=.125, probe=state_payload(probe), updated=state_payload(updated)))
    for seed in (17, 29, 43):
        shape = [2, 16, 4]
        plan = model(seed).inference_plan(shape)
        original = plan.to_json()
        gpu = plan.compile_training_wgpu()
        x, y = batch(32)
        gpu.upload_batch(st.Tensor(32, 4, x), st.Tensor(32, 3, y))
        initial = checked_state(gpu, 0.)
        receipts = []
        half_plan = None
        for step in range(1, 129):
            gpu.step(.2)
            receipts.append(gpu.loss_snapshot())
            if step == 64:
                half_plan = gpu.parameter_snapshot()
        final = checked_state(gpu, 0.)
        losses = [snapshot.read() for snapshot in receipts]
        half_json = half_plan.read_plan().to_json()
        final_json = final.to_plan().to_json()
        if final.loss >= initial.loss or plan.to_json() != original:
            raise RuntimeError("learning did not improve or changed the source plan")
        report["learning"].append(dict(seed=seed, shape=shape, input=x, target=y,
            initial_parameters=layers(plan), plan_json=original, steps=128, learning_rate=.2,
            initial=state_payload(initial), final=state_payload(final), losses=losses,
            half_plan_json=half_json, half_plan_sha256=hashlib.sha256(half_json.encode()).hexdigest(),
            final_plan_json=final_json))
    report["status"] = "passed"


def receive(path, report):
    raw = path.read_bytes()
    browser = json.loads(raw)
    if (browser.get("schema") != "spiraltorch.nn.training_browser_fixture.v1" or
            browser.get("status") != "passed" or browser.get("cpu_only") is not False or
            len(browser["learning"]) != 3):
        raise ValueError("expected a successful GPU browser handoff")
    report["browser_sha256"] = hashlib.sha256(raw).hexdigest()
    for item in browser["learning"]:
        if hashlib.sha256(item["plan_json"].encode()).hexdigest() != item["plan_sha256"]:
            raise ValueError("returned plan bytes changed")
        plan = st.nn.InferencePlan.from_json(item["plan_json"])
        if plan.to_json() != item["plan_json"] or plan.input_shape != (2, 16, 4):
            raise ValueError("returned graph changed")
        x, _ = batch(32)
        gpu = plan.compile_wgpu()
        gpu.upload_values(x)
        gpu.dispatch()
        output = gpu.snapshot().read_values()
        expected = item["final"]["prediction"]
        if len(output) != len(expected) or any(not math.isfinite(a) or not math.isfinite(b) or
                abs(a-b) > 1e-5 + 1e-4*abs(b) for a,b in zip(output, expected)):
            raise ValueError("reimported inference differs from browser's final zero-rate probe")
        report["learning"].append(dict(seed=item["seed"], plan_sha256=item["plan_sha256"],
            output=output, max_abs_error=max(abs(a-b) for a,b in zip(output, expected)), adapter=gpu.adapter_info()))
    if path.read_bytes() != raw:
        raise RuntimeError("browser report changed during replay")
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path, help="reimport browser-trained weights and run native inference")
    args = parser.parse_args()
    report = dict(schema="spiraltorch.nn.training_client_fixture.v1", status="error", vjps=[], learning=[],
                  boundary="public Python client capture; independent numerical validation is separate; no timing or quality claim")
    if args.browser_report:
        report["schema"] = "spiraltorch.nn.training_handoff_return.v1"
        report["boundary"] = "browser-trained weights reimported for Python resident inference, not optimizer resume"
    with args.output.open("x") as output:
        try:
            if args.browser_report:
                receive(args.browser_report, report)
            else:
                run(report)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=report["status"], error=report.get("error"), output=str(args.output))))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()

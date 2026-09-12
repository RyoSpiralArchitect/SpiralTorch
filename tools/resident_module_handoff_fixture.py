"""Source-owned fixture for browser training -> original Python/Rust Module."""
import json
import math
from pathlib import Path
import sys
import spiraltorch as st

SHAPE = [2, 2, 2]
INPUT = [0.25] * 8
TARGET = [1.0] * 8


def model():
    result = st.nn.Sequential()
    result.add(st.nn.Linear("up", 2, 3))
    result.add(st.nn.Gelu())
    result.add(st.nn.Scaler.from_gain("hidden", st.Tensor(1, 3, [1.] * 3)))
    result.add(st.nn.Relu())
    result.add(st.nn.Linear("down", 3, 2))
    result.add(st.nn.Scaler.from_gain("output", st.Tensor(1, 2, [1.] * 2)))
    return result


def flat(tensor):
    return [value for row in tensor.tolist() for value in row]


def fixture():
    return dict(schema="spiraltorch.resident_module_handoff_fixture.v1",
        baseline_plan=model().inference_plan(SHAPE).to_json(), input=INPUT, target=TARGET,
        steps=16, learning_rate=0.05)


def apply_report(report):
    if report.get("status") != "passed" or report.get("fixture") != fixture():
        raise ValueError("browser fixture differs from this source")
    if report["adapter"]["backend"] != "BrowserWebGpu" or report["adapter"]["device_type"] == "Cpu":
        raise ValueError("browser WebGPU required")
    expected = {(p, f) for p in ("exact", "module_compatible") for f in (False, True)}
    if len(report["cases"]) != 4 or {(c["policy"], c["fused"]) for c in report["cases"]} != expected:
        raise ValueError("missing or duplicated browser cases")
    rows = []
    for case in report["cases"]:
        if case["updates"] != 16 or case["submitted_steps"] != 17 or len(case["losses"]) != 16:
            raise ValueError("incomplete browser updates")
        if not all(math.isfinite(v) and v >= 0 for v in case["losses"]):
            raise ValueError("invalid guarded browser losses")
        host = model()
        base = host.inference_plan(SHAPE)
        x = st.Tensor(4, 2, INPUT)
        original = flat(host.forward(x))  # Warm the original parameter packs.
        updated = st.nn.InferencePlan.from_json(case["trained_plan"])
        count = base.apply_parameters_to(host, updated)
        if count != 6:
            raise ValueError("incomplete parameter handoff")
        after = flat(host.forward(x))
        prediction = case["prediction"]
        if len(after) != len(prediction) or after == original:
            raise ValueError("missing or unchanged host prediction")
        errors = [abs(a-b) for a,b in zip(after, prediction)]
        if not all(math.isfinite(e) and e < 2e-5 for e in errors):
            raise ValueError("host/browser prediction mismatch")
        current = host.inference_plan(SHAPE)
        if json.loads(current.to_json())["parameters"] != json.loads(updated.to_json())["parameters"]:
            raise ValueError("host parameters differ from the received plan")
        trainer = st.nn.ModuleTrainer(backend="cpu")
        trainer.prepare(host)
        host.backward(x, st.Tensor(4, 2, [0.25] * 8))
        trainer.step(host)
        continued = host.inference_plan(SHAPE).to_json()
        if continued == current.to_json():
            raise ValueError("new ModuleTrainer phase did not change weights")
        if not all(math.isfinite(v) for v in flat(host.forward(x))):
            raise ValueError("new ModuleTrainer phase is nonfinite")
        rows.append(dict(policy=case["policy"], fused=case["fused"], parameters=count,
            max_abs_error=max(errors), new_module_trainer_phase="passed"))
    return dict(status="passed", schema="spiraltorch.resident_module_handoff.v1", cases=rows,
        boundary="Weight handoff, not optimizer-state continuation. Browser model/SGD and the new host ModuleTrainer phase have distinct explicit optimizer semantics.")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "export" and len(args) == 1:
        value, output = fixture(), Path(args[0])
    elif mode == "apply" and len(args) == 2:
        value, output = apply_report(json.loads(Path(args[0]).read_text())), Path(args[1])
    else:
        raise ValueError("export OUTPUT or apply BROWSER_REPORT OUTPUT")
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="passed", output=str(output))))

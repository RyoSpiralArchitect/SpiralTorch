"""Execute an existing NN in Python, then export the same Rust plan for WASM."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import spiraltorch as st


def build_model():
    net = st.nn.Sequential()
    net.add(st.nn.Linear(2, 3, name="up"))
    net.add(st.nn.Gelu())
    net.add(st.nn.Linear(3, 2, name="down"))
    net.load_state_dict([
        ("up::weight", st.Tensor(2, 3, [0.5, -0.25, 0.125, -0.5, 0.25, 0.75])),
        ("up::bias", st.Tensor(1, 3, [0.125, -0.25, 0.5])),
        ("down::weight", st.Tensor(3, 2, [0.25, -0.5, 0.125, 0.75, -0.25, 0.5])),
        ("down::bias", st.Tensor(1, 2, [0.125, -0.25])),
    ])
    return net


def oracle(values):
    output = []
    for i in range(0, len(values), 2):
        a, b = values[i:i+2]
        h = [0.5*a - 0.5*b + 0.125, -0.25*a + 0.25*b - 0.25, 0.125*a + 0.75*b + 0.5]
        h = [0.5*y*(1 + math.tanh(0.7978845834732056*(y + 0.044715*y**3))) for y in h]
        output.extend([0.25*h[0] + 0.125*h[1] - 0.25*h[2] + 0.125,
                       -0.5*h[0] + 0.75*h[1] + 0.5*h[2] - 0.25])
    return output


def close(actual, expected):
    if len(actual) != len(expected) or not all(
            math.isfinite(a) and abs(a-b) <= 1e-5 + 1e-4*abs(b) for a, b in zip(actual, expected)):
        raise ValueError("resident/module output differs from independent fixture oracle")


def run():
    net = build_model()
    cases = []
    for shape in ([2], [2, 2], [2, 3, 2]):
        plan = net.inference_plan(shape)
        payload = plan.to_json()
        gpu = plan.compile_wgpu()
        if gpu.adapter_info()["device_type"] == "Cpu":
            raise RuntimeError("real GPU required, no CPU fallback")
        values = [(i - 3) / 8 for i in range(math.prod(shape))]
        expected = oracle(values)
        host = net(st.Tensor(math.prod(shape) // 2, 2, values)).tolist()
        close([value for row in host for value in row], expected)
        gpu.upload_values(values)
        gpu.dispatch()
        snapshot = gpu.snapshot()
        if snapshot.shape != tuple(shape):
            raise ValueError("logical output shape changed")
        replay_input = [v / 2 for v in values]
        replay_expected = oracle(replay_input)
        gpu.upload_values(replay_input)
        gpu.dispatch()
        replay = gpu.snapshot()
        output = snapshot.read_values()
        close(output, expected)
        replay_output = replay.read_values()
        close(replay_output, replay_expected)
        cases.append(dict(shape=shape, plan_json=payload,
                          plan_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                          input=values, expected=expected, python_output=output,
                          replay_input=replay_input, replay_expected=replay_expected,
                          replay_python_output=replay_output,
                          adapter=gpu.adapter_info()))
    return dict(schema="spiraltorch.nn.client_fixture.v1", status="passed", cases=cases,
                boundary="existing Python Sequential -> Rust plan -> resident WGPU; same serialized plan is imported by WASM; independent small f64 oracle; correctness, not timing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], cases=len(report["cases"]))))

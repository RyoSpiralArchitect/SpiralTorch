"""Same-gradient AdamW oracles, independent of Attention and SpiralTorch kernels."""

import json
import math

import torch


def f32(value):
    return float(torch.tensor(value, dtype=torch.float32))


def fixture():
    torch.set_num_threads(1)
    initial = [0.0, 1.0, -1.0, 0.125, -0.375, 1e-6, -1e-6, 0.0]
    initial = list(map(f32, initial))
    scales = [0.0, 1.0, 0.125, 0.001, 1e-6, 1e-9, 1e-9, 0.01]
    periodic = [
        [f32(((step * (2 * col + 3) + col * 7) % 31 - 15) / 16 * scale)
         for col, scale in enumerate(scales)]
        for step in range(31)
    ]
    cases = []
    for name, betas, gradients in [
        ("default_constant", (0.9, 0.999), [periodic[0]]),
        ("default_periodic", (0.9, 0.999), periodic),
        ("near_one_periodic", (0.9999, 1.0 - 2.0**-24), periodic),
    ]:
        config = list(map(f32, (*betas, 1e-5)))
        rate, decay = map(f32, (0.001, 0.01))
        beta1, beta2, epsilon = config
        parameter = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [parameter], lr=rate, betas=(beta1, beta2), eps=epsilon,
            weight_decay=decay, foreach=False, fused=False,
        )
        values = initial.copy()
        first = [0.0] * len(initial)
        second = [0.0] * len(initial)
        trajectory = []
        for step in range(1, 1025):
            gradient = gradients[(step - 1) % len(gradients)]
            parameter.grad = torch.tensor(gradient, dtype=torch.float32)
            optimizer.step()
            correction1 = -math.expm1(step * math.log(beta1))
            correction2 = -math.expm1(step * math.log(beta2))
            for i, g in enumerate(gradient):
                first[i] = beta1 * first[i] + (1.0 - beta1) * g
                second[i] = beta2 * second[i] + (1.0 - beta2) * g * g
                values[i] = values[i] * (1.0 - rate * decay) - rate * (
                    first[i] / correction1
                ) / (math.sqrt(second[i] / correction2) + epsilon)
            if step in (1, 2, 8, 32, 128, 256, 512, 768, 1024):
                state = optimizer.state[parameter]
                trajectory.append({
                    "steps": step,
                    "torch": {"values": parameter.detach().tolist(),
                              "first": state["exp_avg"].tolist(),
                              "second": state["exp_avg_sq"].tolist()},
                    "f64": {"values": values.copy(), "first": first.copy(),
                            "second": second.copy()},
                })
        cases.append({"name": name, "config": config, "rate": rate,
                      "decay": decay, "initial": initial,
                      "gradient_period": gradients, "trajectory": trajectory})
    return {
        "schema": "spiraltorch.resident_adamw.numerics_fixture.v1",
        "torch": torch.__version__, "device": "cpu", "dtype": "float32",
        "torch_policy": {"foreach": False, "fused": False},
        "updates": 1024, "atol": 3e-6, "rtol": 5e-5,
        "f64_scope": "Independent scalar AdamW on the same float32 controls and gradients",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2, allow_nan=False))

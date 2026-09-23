"""Matched LayerNorm + MSE SGD control for the Rust residency benchmark."""

import json
import math
import os
import struct
import time

import torch
import torch.nn.functional as functional


STEPS = 32
WARMUP = 3
ITERATIONS = 9
TOLERANCE = 5e-4
SHAPES = [(2, 3), (8, 257), (32, 256), (64, 768), (128, 1025), (256, 256)]


def values(length, multiplier, modulus):
    return [(((i * multiplier + 17) % modulus) - modulus // 2) / 64.0
            for i in range(length)]


def digest(numbers):
    value = 0xCBF29CE484222325
    for number in numbers:
        for byte in struct.pack("<f", number):
            value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def evaluate(device, host_input, host_target, cols, rate):
    if device == "cpu":
        x, target = host_input, host_target
    else:
        x = host_input.detach().to(device=device, copy=True).requires_grad_(True)
        target = host_target.to(device=device, copy=True)
    gamma = torch.ones(cols, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.zeros(cols, device=device, dtype=torch.float32, requires_grad=True)
    learning_rate = torch.tensor(rate, device=device, dtype=torch.float32)
    for _ in range(STEPS):
        prediction = functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
        loss = functional.mse_loss(prediction, target)
        _, dg, db = torch.autograd.grad(loss, (x, gamma, beta))
        gamma = (gamma + learning_rate * dg).detach().requires_grad_(True)
        beta = (beta + learning_rate * db).detach().requires_grad_(True)
    return [tensor.detach().to(device="cpu", copy=True).reshape(-1).tolist()
            for tensor in (loss, gamma, beta, dg, db)]


def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    if not torch.backends.mps.is_available():
        raise RuntimeError("a real MPS device is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cases = []
    valid = True
    for rows, cols in SHAPES:
        input_values = values(rows * cols, 37, 257)
        target_values = values(rows * cols, 11, 67)
        host_input = torch.tensor(input_values, dtype=torch.float32).reshape(rows, cols)
        host_input.requires_grad_(True)
        host_target = torch.tensor(target_values, dtype=torch.float32).reshape(rows, cols)
        rate = torch.tensor(-0.1, dtype=torch.float32).item() * cols
        rate = torch.tensor(rate, dtype=torch.float32).item()
        initial_prediction = functional.layer_norm(
            host_input, (cols,), torch.ones(cols), torch.zeros(cols), 1e-5)
        initial_loss = functional.mse_loss(initial_prediction, host_target).item()
        oracle = evaluate("cpu", host_input, host_target, cols, rate)
        errors = {"cpu": 0.0, "mps": 0.0}
        intervals = []
        final_outputs = {}
        for iteration in range(WARMUP + ITERATIONS):
            routes = ("cpu", "mps") if iteration % 2 == 0 else ("mps", "cpu")
            for route in routes:
                start = time.perf_counter_ns()
                output = evaluate(route, host_input, host_target, cols, rate)
                ms = (time.perf_counter_ns() - start) / 1e6
                for actual, expected in zip(output, oracle):
                    if len(actual) != len(expected):
                        raise RuntimeError("output shape differs from CPU oracle")
                    for value, reference in zip(actual, expected):
                        scaled = abs(value - reference) / (TOLERANCE * (1.0 + abs(reference)))
                        errors[route] = max(errors[route], scaled)
                        if not (math.isfinite(value) and scaled <= 1.0):
                            valid = False
                if iteration >= WARMUP:
                    intervals.append({"iteration": iteration - WARMUP, "route": route, "ms": ms})
                final_outputs[route] = output
        cases.append({"rows": rows, "cols": cols, "initial_loss": initial_loss,
                      "input_fnv64": digest(input_values),
                      "target_fnv64": digest(target_values),
                      "max_scaled_error": errors, "intervals": intervals,
                      "final_outputs": final_outputs})
    report = {"schema": "spiraltorch.layer_norm.training_residency_torch.v1",
              "status": "passed" if valid else "numerical_mismatch",
              "torch_version": torch.__version__, "devices": ["cpu", "mps"],
              "mps_fallback": False, "compiled": False,
              "intra_op_threads": 4, "inter_op_threads": 1,
              "steps": STEPS, "warmup": WARMUP, "iterations": ITERATIONS,
              "epsilon": 1e-5, "rate": "-0.1 * cols", "scaled_tolerance": TOLERANCE,
              "scope": "32-step forward+MSE+all VJPs+SGD; CPU inputs already host-owned, MPS includes initial upload and final CPU-owned loss, parameters and affine gradients",
              "cases": cases}
    print(json.dumps(report, allow_nan=False))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

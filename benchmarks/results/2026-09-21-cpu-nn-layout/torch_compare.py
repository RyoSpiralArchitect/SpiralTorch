"""Matched CPU eager nn.Linear/GELU forwards; no SpiralTorch validation overhead is added."""
import json
import sys
import time
sys.path.insert(0, sys.argv[1])
import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def fixture(length, modulus, divisor):
    return ((np.arange(length) % modulus).astype(np.float32) - modulus // 2) / np.float32(divisor)

def linear(inner, cols):
    weights = fixture(inner * cols, 17, 256).reshape(inner, cols)
    bias = fixture(cols, 7, 128)
    model = torch.nn.Linear(inner, cols, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(weights.T.copy()))
        model.bias.copy_(torch.from_numpy(bias))
    return model, weights.astype(np.float64), bias.astype(np.float64)

cases = []
with torch.inference_mode():
    for rows, inner, cols in [(1, 64, 64), (8, 768, 3072), (32, 768, 3072), (64, 256, 1024), (17, 137, 195), (65, 1025, 97)]:
        values = fixture(rows * inner, 13, 64).reshape(rows, inner)
        x = torch.from_numpy(values)
        for mlp in [False, True]:
            first, weights, bias = linear(inner, cols)
            expected = values.astype(np.float64) @ weights + bias
            model = first
            if mlp:
                second, weights, bias = linear(cols, inner)
                model = torch.nn.Sequential(first, torch.nn.GELU(approximate="tanh"), second)
                hidden = 0.5 * expected * (1 + np.tanh(np.sqrt(2 / np.pi) * (expected + 0.044715 * expected**3)))
                expected = hidden @ weights + bias
            model.eval()
            for _ in range(3):
                model(x)
            elapsed = []
            for _ in range(9):
                start = time.perf_counter_ns()
                for _ in range(2):
                    model(x)
                elapsed.append((time.perf_counter_ns() - start) / 2)
            actual = model(x).numpy().astype(np.float64)
            error = np.abs(actual - expected)
            valid = bool(np.isfinite(actual).all() and np.all(error <= 1e-4 + 1e-4 * np.abs(expected)))
            assert valid
            cases.append({"rows": rows, "inner": inner, "cols": cols, "operation": "mlp" if mlp else "linear", "valid": valid,
                          "max_abs": float(error.max()), "elapsed_ns": elapsed})
print(json.dumps({"torch": torch.__version__, "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                  "schema": "spiraltorch.cpu_nn_layout.torch.v1", "cases": cases, "warmups": 3, "intervals": 9, "repetitions": 2,
                  "boundary": "CPU eager nn.Linear/GELU inference; contiguous output-by-input weights; output allocation and free included; no per-call Rust validation or cache invalidation equivalent; NumPy f64 reference outside timing"}))

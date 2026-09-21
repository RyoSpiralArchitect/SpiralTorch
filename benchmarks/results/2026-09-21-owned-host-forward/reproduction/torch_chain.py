"""Eager Torch CPU chain control; Python dispatch is not Rust Module dispatch."""
import hashlib
import json
import math
import platform
import sys
import time

import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
reports = []
for round_id in ["warm", "a", "b"]:
    cases = []
    for rows, cols in [(1, 1), (1, 8), (1, 32), (1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)]:
        values = [(i % 257) / 32 - 4 for i in range(rows * cols)]
        x = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(rows, cols)
        for depth in [1, 4, 16]:
            model = torch.nn.Sequential(*(torch.nn.GELU(approximate="tanh") for _ in range(depth)))
            expected = []
            for value in values:
                for _ in range(depth):
                    value = 0.5 * value * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
                expected.append(value)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                elapsed = []
                for _ in range(15):
                    start = time.perf_counter_ns()
                    for _ in range(8):
                        model(x)
                    elapsed.append((time.perf_counter_ns() - start) / 8)
                output = model(x).flatten()
            actual = output.tolist()
            assert len(actual) == len(expected)
            assert all(math.isfinite(a) and abs(a - b) <= 2e-6 * (1 + abs(b)) for a, b in zip(actual, expected))
            assert x.flatten().tolist() == values
            cases.append({"rows": rows, "cols": cols, "depth": depth, "valid": True,
                          "elapsed_ns": elapsed, "max_abs": max(abs(a - b) for a, b in zip(actual, expected)),
                          "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()})
    reports.append({"round": round_id, "cases": cases})
print(json.dumps({"torch": torch.__version__, "torch_file": torch.__file__, "python": sys.version,
                  "platform": platform.platform(), "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                  "warmups": 3, "intervals": 15, "repetitions": 8, "reports": reports,
                  "boundary": "CPU eager no-grad GELU chain, allocation/free included; Python entry differs from Rust Module dispatch; no compile, fusion, model training or fastest-PyTorch claim"}, indent=2))

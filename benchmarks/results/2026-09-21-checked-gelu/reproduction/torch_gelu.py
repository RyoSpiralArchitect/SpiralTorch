"""Eager Torch CPU comparison; forward excludes autograd recording, VJP includes traversal."""
import hashlib
import json
import math
import platform
import sys
import time

import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
layer = torch.nn.GELU(approximate="tanh")
reports = []
for round_id in ["warm", "a", "b"]:
    cases = []
    for rows, cols in [(1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)]:
        values = [(i % 257) / 32 - 4 for i in range(rows * cols)]
        seeds = [(i % 29) / 16 - 0.5 for i in range(rows * cols)]
        x = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(rows, cols).requires_grad_()
        g = torch.tensor(seeds, dtype=torch.float32, device="cpu").reshape(rows, cols)
        recorded = layer(x)
        for backward in [False, True]:
            expected = []
            for v, seed in zip(values, seeds):
                c = math.sqrt(2 / math.pi)
                t = math.tanh(c * (v + 0.044715 * v * v * v))
                expected.append((0.5 * (1 + t) + 0.5 * v * (1 - t * t) * c * (1 + 3 * 0.044715 * v * v)) * seed
                                if backward else 0.5 * v * (1 + t))
            def run():
                if backward:
                    return torch.autograd.grad(recorded, x, grad_outputs=g, retain_graph=True)[0]
                return layer(x)
            with torch.no_grad():
                for _ in range(3):
                    run()
                elapsed = []
                for _ in range(15):
                    start = time.perf_counter_ns()
                    for _ in range(8):
                        run()
                    elapsed.append((time.perf_counter_ns() - start) / 8)
                output = run().flatten()
            actual = output.tolist()
            assert len(actual) == len(expected)
            valid = all(math.isfinite(a) and abs(a - b) <= 2e-6 * (1 + abs(b)) for a, b in zip(actual, expected))
            assert valid
            cases.append({"rows": rows, "cols": cols, "backward": backward, "elapsed_ns": elapsed,
                          "valid": valid, "max_abs": max(abs(a - b) for a, b in zip(actual, expected)),
                          "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()})
    reports.append({"round": round_id, "cases": cases})
print(json.dumps({"torch": torch.__version__, "torch_file": torch.__file__, "python": sys.version,
                  "platform": platform.platform(), "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                  "warmups": 3, "intervals": 15, "repetitions": 8, "reports": reports,
                  "boundary": "CPU eager calls and allocation/free included; Python entry and autograd traversal are not Rust Module dispatch; no fastest-PyTorch claim"}, indent=2))

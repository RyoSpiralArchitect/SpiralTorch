"""Eager vectorized CPU encoding control; no compile or fastest-Torch claim."""
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
    for rows in [1, 32, 1024]:
        for cols in [3, 6]:
            for bands in [0, 4, 10]:
                for residual in [False, True]:
                    values = [(i % 257) / 64 - 2 for i in range(rows * cols)]
                    x = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
                    frequencies = (2.0 ** torch.arange(bands, dtype=torch.float32)).reshape(1, bands, 1)
                    def encode():
                        phase = x[:, None, :] * frequencies
                        encoded = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
                        return torch.cat((x, encoded), dim=1) if residual else encoded
                    expected = []
                    for row in range(rows):
                        v = values[row * cols:(row + 1) * cols]
                        if residual:
                            expected.extend(v)
                        for band in range(bands):
                            for value in v:
                                expected.extend([math.sin(value * 2**band), math.cos(value * 2**band)])
                    with torch.no_grad():
                        for _ in range(3):
                            encode()
                        elapsed = []
                        for _ in range(15):
                            start = time.perf_counter_ns()
                            for _ in range(8):
                                encode()
                            elapsed.append((time.perf_counter_ns() - start) / 8)
                        output = encode().flatten()
                    actual = output.tolist()
                    assert len(actual) == len(expected)
                    assert all(math.isfinite(a) and abs(a-b) <= 2e-6 for a, b in zip(actual, expected))
                    assert x.flatten().tolist() == values
                    cases.append({"rows": rows, "cols": cols, "bands": bands, "residual": residual,
                        "valid": True, "elapsed_ns": elapsed,
                        "max_abs": max((abs(a-b) for a, b in zip(actual, expected)), default=0),
                        "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()})
    reports.append({"round": round_id, "cases": cases})
print(json.dumps({"torch": torch.__version__, "torch_file": torch.__file__, "python": sys.version,
    "platform": platform.platform(), "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
    "warmups": 3, "intervals": 15, "repetitions": 8, "reports": reports,
    "boundary": "Vectorized eager CPU no-grad sin/cos encoding, allocation/free included. Python dispatch differs from Rust and WASM. No torch.compile, GPU, model quality or fastest-Torch claim."}, indent=2))

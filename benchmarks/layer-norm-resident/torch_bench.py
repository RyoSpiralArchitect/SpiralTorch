"""Exploratory ATen LayerNorm + all VJPs, with the same host-to-host boundary."""
import json
import os
import time

import torch


def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("MPS fallback must remain disabled")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("real MPS device required")
    cases = []
    for rows, cols in [(2, 3), (8, 257), (32, 256), (64, 768), (128, 1025), (256, 256)]:
        def values(n, multiplier, modulus, divisor):
            return [(((i * multiplier + 17) % modulus) - modulus // 2) / divisor for i in range(n)]
        host = [torch.tensor(values(rows * cols, 37, 257, 64.)).reshape(rows, cols),
                torch.tensor(values(cols, 19, 127, 64.)), torch.tensor(values(cols, 7, 31, 64.)),
                torch.tensor(values(rows * cols, 29, 193, 128.)).reshape(rows, cols)]

        def evaluate(device, dtype):
            x, gamma, beta, seed = [v.to(device=device, dtype=dtype, copy=True) for v in host]
            for v in (x, gamma, beta):
                v.requires_grad_(True)
            y = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)
            dx, dg, db = torch.autograd.grad(y, (x, gamma, beta), seed)
            dg, db = dg * 0.5, db * 0.5
            return [v.detach().to(device="cpu", copy=True) for v in (y, dx, dg, db)]

        oracle = evaluate("cpu", torch.float64)
        intervals = []
        max_scaled = {"cpu": 0., "mps": 0.}
        for iteration in range(21):
            order = ("cpu", "mps") if iteration % 2 == 0 else ("mps", "cpu")
            for device in order:
                start = time.perf_counter_ns()
                output = evaluate(device, torch.float32)
                ms = (time.perf_counter_ns() - start) / 1e6
                for actual, expected in zip(output, oracle):
                    scaled = (actual.double() - expected).abs() / (2e-5 * (1. + expected.abs()))
                    maximum = scaled.max().item()
                    if not torch.isfinite(actual).all() or maximum > 1.:
                        raise RuntimeError(f"{device} {rows}x{cols}: scaled error {maximum}")
                    max_scaled[device] = max(max_scaled[device], maximum)
                if iteration >= 3:
                    intervals.append(dict(iteration=iteration - 3, route=device, ms=ms))
        cases.append(dict(rows=rows, cols=cols, intervals=intervals, max_scaled_error=max_scaled))
    print(json.dumps(dict(schema="spiraltorch.layer_norm.exploratory_torch.v1", status="passed",
                         scope="host-to-host; four input copies + affine forward + all VJPs + four owning CPU outputs",
                         torch_version=torch.__version__, devices=["cpu", "mps"], compiled=False,
                         intra_op_threads=4, inter_op_threads=1, warmup=3, iterations=18,
                         epsilon=1e-5, parameter_gradient_scale=0.5, cases=cases), allow_nan=False))


if __name__ == "__main__":
    main()

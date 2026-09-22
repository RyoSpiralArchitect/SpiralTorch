"""Eager CPU/MPS reusable-output controls; scalar and Torch-f64 oracle checks."""
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softmax_protocol as protocol


def prepare(case, device):
    import torch
    rows, cols, mode = protocol.key(case)
    x = torch.tensor(case["input"], dtype=torch.float32, device=device).reshape(rows, cols)
    flat = torch.empty(rows * cols * (2 if mode == 4 else 1), dtype=torch.float32, device=device)
    output = flat[:rows * cols].view(rows, cols)
    mask = flat[rows * cols:].view(rows, cols) if mode == 4 else None
    maxima = torch.empty((rows, 1), dtype=torch.float32, device=device)
    peaks = torch.empty((rows, cols), dtype=torch.bool, device=device)
    return x, flat, output, mask, maxima, peaks


def compute(prepared):
    import torch
    x, flat, output, mask, maxima, peaks = prepared
    torch.softmax(x, dim=1, out=output)
    if mask is not None:
        torch.amax(x, dim=1, keepdim=True, out=maxima)
        torch.eq(x, maxima, out=peaks)
        mask.copy_(peaks)
    return flat


def run(report):
    import torch
    cases = protocol.admit(report, "native")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("real MPS without fallback required")
    records = []
    with torch.inference_mode():
        for ident, case in cases.items():
            rows, cols, mode = ident
            x = torch.tensor(case["input"], dtype=torch.float64).reshape(rows, cols)
            reference = torch.softmax(x, 1).flatten()
            if mode == 4:
                reference = torch.cat((reference, (x == x.amax(1, keepdim=True)).flatten()))
            reference = reference.to(torch.float32).tolist()
            protocol.close(reference, protocol.oracle(case["input"], rows, cols, mode))
            prepared = [prepare(case, device) for device in ("cpu", "mps")]
            for p in prepared:
                protocol.close(compute(p).to("cpu", copy=True).tolist(), reference)
            intervals, outputs = [], [None, None]
            for burst in (1, 4):
                for block in range(12):
                    order = [0, 1] if (block + rows + cols + mode) % 2 == 0 else [1, 0]
                    for route in order:
                        start = time.perf_counter_ns()
                        for _ in range(burst):
                            result = compute(prepared[route])
                        owned = result.to("cpu", copy=True)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        values = owned.tolist()
                        error, _ = protocol.close(values, reference)
                        outputs[route] = values
                        if block >= 3:
                            intervals.append(dict(block=block-3, burst=burst, route=("cpu", "mps")[route],
                                                  order=order, elapsed_ms=elapsed, max_abs_error=error))
            records.append(dict(rows=rows, cols=cols, mode=mode, input=case["input"],
                                reference=reference, last_outputs=outputs, intervals=intervals))
    result = dict(schema=protocol.TORCH, input_protocol=protocol.SCHEMA, status="passed",
                  torch_version=torch.__version__, devices=["cpu", "mps"], compiled=False,
                  preallocated_outputs=True, intra_op_threads=torch.get_num_threads(),
                  inter_op_threads=torch.get_num_interop_threads(), warmup=3, blocks=9,
                  bursts=[1, 4], cases=records)
    protocol.admit(result, "torch")
    return result


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), allow_nan=False))

"""Eager reusable-output CPU/MPS controls for the same pair/raw-consensus math."""
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consensus_protocol as protocol


def prepare(case, device, dtype=None):
    import torch
    rows, cols, count = protocol.key(case)
    dtype = dtype or torch.float32
    x = torch.tensor(case["input"], dtype=dtype, device=device).reshape(rows, cols)
    size = rows * cols
    flat = torch.empty(2 * size if count == 2 else 3 * size + rows * 4, dtype=dtype, device=device)
    output = flat[:size].view(rows, cols)
    mask = flat[size:2 * size].view(rows, cols)
    maxima = torch.empty((rows, 1), dtype=dtype, device=device)
    peaks = torch.empty_like(x, dtype=torch.bool)
    scratch = None
    if count == 4:
        spiral = flat[2 * size:3 * size].view(rows, cols)
        metrics = flat[3 * size:].view(rows, 4)
        scratch = (spiral, metrics, torch.empty_like(x), torch.empty(rows, dtype=dtype, device=device),
                   torch.empty(rows, dtype=dtype, device=device), torch.empty(rows, dtype=dtype, device=device),
                   torch.empty(rows, dtype=torch.bool, device=device))
    return x, flat, output, mask, maxima, peaks, scratch, protocol.parameters(cols)


def compute(prepared):
    import torch
    x, flat, soft, mask, maxima, peaks, scratch, params = prepared
    torch.softmax(x, dim=1, out=soft)
    torch.amax(x, dim=1, keepdim=True, out=maxima)
    torch.eq(x, maxima, out=peaks)
    mask.copy_(peaks)
    if scratch is not None:
        spiral, metrics, work, geodesic, tmp, denominator, gate = scratch
        entropy, mass, enrichment, coherence = metrics.unbind(1)
        phi, conjugate, bias, leech, ratio, inv_cols, epsilon = params
        torch.clamp(soft, min=epsilon, out=work)
        torch.log(work, out=work)
        work.mul_(soft)
        torch.sum(work, dim=1, out=entropy)
        entropy.neg_()
        torch.sum(mask, dim=1, out=mass)
        torch.mul(entropy, ratio, out=geodesic)
        geodesic.add_(mass, alpha=phi)
        torch.mul(geodesic, leech, out=enrichment)
        torch.abs(geodesic, out=tmp)
        torch.le(tmp, epsilon, out=gate)
        enrichment.masked_fill_(gate, 0.)
        torch.add(entropy, 1., out=denominator)
        torch.div(entropy, denominator, out=coherence)
        coherence.clamp_(0., 1.)
        torch.mul(mass, inv_cols, out=tmp)
        coherence.add_(tmp.clamp_(0., 1.))
        torch.abs(enrichment, out=denominator)
        denominator.add_(1.)
        torch.div(enrichment, denominator, out=tmp)
        coherence.add_(tmp.clamp_(0., 1.)).div_(3.)
        torch.mul(soft, conjugate, out=spiral)
        spiral.add_(mask, alpha=bias)
        torch.add(enrichment, 1., out=tmp)
        spiral.mul_(tmp[:, None])
    return flat


def run(report):
    import torch
    cases = protocol.admit(report, "native")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("real MPS without fallback required")
    records = []
    with torch.inference_mode():
        for ident, case in cases.items():
            rows, cols, count = ident
            reference = compute(prepare(case, "cpu", torch.float64)).float().tolist()
            protocol.close(reference, protocol.oracle(case["input"], rows, cols, count))
            prepared = [prepare(case, device) for device in ("cpu", "mps")]
            for p in prepared:
                protocol.close(compute(p).to("cpu", copy=True).tolist(), reference)
            intervals, outputs = [], [None, None]
            for burst in (1, 4):
                for block in range(12):
                    order = protocol.order(case, block, True)
                    for route in order:
                        start = time.perf_counter_ns()
                        for _ in range(burst):
                            result = compute(prepared[route])
                        owned = result.to("cpu", copy=True)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        values = owned.tolist()
                        absolute, scaled = protocol.close(values, reference)
                        outputs[route] = values
                        if block >= 3:
                            intervals.append(dict(block=block-3, burst=burst, route=("cpu", "mps")[route], order=order,
                                                  elapsed_ms=elapsed, max_abs_error=absolute, max_scaled_error=scaled))
            records.append(dict(rows=rows, cols=cols, count=count, input=case["input"], reference=reference,
                                last_outputs=outputs, intervals=intervals,
                                **({"order_scheme": case["order_scheme"]} if "order_scheme" in case else {})))
    result = dict(schema=protocol.TORCH, input_protocol=protocol.SCHEMA, status="passed",
                  torch_version=torch.__version__, devices=["cpu", "mps"], compiled=False, preallocated_outputs=True,
                  intra_op_threads=torch.get_num_threads(), inter_op_threads=torch.get_num_interop_threads(),
                  warmup=3, blocks=9, bursts=[1, 4], cases=records)
    protocol.admit(result, "torch")
    return result


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), allow_nan=False))

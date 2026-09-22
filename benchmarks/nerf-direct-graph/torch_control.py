"""Independent CPU numerical oracle and eager CPU/MPS matched-boundary timings.

Not a kernel-only comparison: eager tensor orchestration, guard reductions and
an owning CPU copy are included. No torch.compile, autograd or training claim.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import admit, close


def prepare(case, device, reference=False, *, input_prelude=None):
    import torch

    if input_prelude not in (None, "relu") or case.get("input_prelude") != input_prelude:
        raise ValueError("Torch input prelude differs from the explicit comparison")
    rows, count = case["rays"], case["samples"]
    dtype = torch.float64 if reference else torch.float32
    rays = torch.tensor(case["ray_inputs"], dtype=torch.float32, device=device).to(dtype)
    parameters = [torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"])
                  for p in case["parameters"]]
    return (rays, parameters, torch.arange(rows, dtype=torch.int64, device=device)[:, None],
            torch.arange(count, dtype=torch.int64, device=device)[None, :], case["seed"], reference,
            input_prelude)


def render(prepared):
    import torch

    rays, parameters, row, sample, seed, reference, input_prelude = prepared
    count, dtype = sample.shape[1], rays.dtype
    # The u32 counter hash is independently expressed as masked integer tensor
    # operations. It is recomputed within every timed render, on that device.
    bits = seed ^ ((row * 0x9E3779B9 + sample * 0x85EBCA6B) & 0xFFFFFFFF)
    bits = ((bits ^ (bits >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    bits = ((bits ^ (bits >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    bits = bits ^ (bits >> 16)
    offset = (bits >> 8).to(dtype) / 16777216
    width = (rays[:, 7:8] - rays[:, 6:7]) / count
    t = rays[:, 6:7] + (sample.to(dtype) + offset) * width
    positions = (rays[:, None, :3] + t[:, :, None] * rays[:, None, 3:6]).to(torch.float32)
    field = positions
    guards = [torch.isfinite(positions).all()]
    if input_prelude == "relu":
        field = field.relu()
    for i in range(0, len(parameters), 2):
        field = field @ parameters[i] + parameters[i + 1]
        guards.append(torch.isfinite(field).all())
        if i + 2 < len(parameters):
            field = field.relu()
    tau = field[:, :, 0].to(dtype).clamp_min(0) * width
    prefix = torch.cat([torch.zeros_like(tau[:, :1]), tau[:, :-1].cumsum(1)], dim=1)
    alpha = -torch.expm1(-tau)
    if not reference:
        # The observed MPS expm1 error accumulated past the frozen tolerance at
        # 256 samples. Keep the independent f64 oracle; stabilize only controls.
        thin = tau * (1 + tau * (-0.5 + tau * (1/6 - tau / 24)))
        alpha = torch.where(tau < 0.01, thin, alpha)
    mass = (-prefix).exp() * alpha
    rgb = (mass[:, :, None] * field[:, :, 1:].to(dtype)).sum(1)
    rgba = torch.cat([rgb, mass.sum(1, keepdim=True)], dim=1).to(torch.float32)
    guards.extend([torch.isfinite(tau).all(), torch.isfinite(rgba).all()])
    valid = torch.stack(guards).all().to(torch.float32).reshape(1)
    return torch.cat([rgba.flatten(), valid])


def read(output):
    # copy=True matters for the CPU control too: neither route returns an alias.
    values = output.to(device="cpu", copy=True).tolist()
    if values[-1] != 1:
        raise ValueError("Torch finite guard failed")
    return values[:-1]


def run(source, *, input_prelude=None):
    import torch

    if input_prelude not in (None, "relu") or source.get("input_prelude") != input_prelude:
        raise ValueError("Torch report prelude differs from the explicit comparison")
    cases = admit(source, "wgpu")
    if not torch.backends.mps.is_available():
        raise RuntimeError("this CPU/MPS comparison requires real MPS; no CPU fallback")
    if __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS fallback would invalidate the comparison")
    records = []
    with torch.inference_mode():
        for case in cases.values():
            reference = read(render(prepare(case, "cpu", reference=True, input_prelude=input_prelude)))
            close(case["reference"], reference)
            prepared = [prepare(case, d, input_prelude=input_prelude) for d in ["cpu", "mps"]]
            for p in prepared:
                close(read(render(p)), reference)
            intervals, outputs = [], [None, None]
            for burst in [1, 4]:
                for block in range(3 + 9):
                    order = [0, 1] if (block + case["rays"] + case["hidden"]) % 2 == 0 else [1, 0]
                    for route in order:
                        start = time.perf_counter_ns()
                        for _ in range(burst):
                            output = render(prepared[route])
                        values = read(output)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        error = close(values, reference)
                        outputs[route] = values
                        if block >= 3:
                            intervals.append({"block":block-3, "burst":burst, "route":["cpu", "mps"][route],
                                              "order":order, "elapsed_ms":elapsed, "max_abs_error":error})
            records.append({**{k:case[k] for k in ["rays", "samples", "hidden", "seed", "ray_inputs", "parameters"]},
                            "reference":reference, "last_outputs":outputs, "intervals":intervals})
    result = {"schema":"spiraltorch.nerf_torch_bench.v1", "status":"passed", "torch_version":torch.__version__,
              "devices":["cpu", "mps"], "intra_op_threads":torch.get_num_threads(),
              "inter_op_threads":torch.get_num_interop_threads(), "compiled":False, "thin_alpha":"fourth_order_below_0.01",
              "warmup":3, "blocks":9, "bursts":[1, 4], "cases":records,
              "boundary":"Prepared f32 rays/parameters. Each render recomputes counter RNG, positions, f32 NN, f32 prefix integration and finite guards. Last output and validity copied to owning CPU memory per 1/4-render interval. Setup, oracle, comparisons and serialization excluded. Oracle uses CPU f64 geometry/integration with f32 NN. Eager CPU/MPS application-path timings, not matched shader arithmetic, kernel timing, training, or torch.compile comparison."}
    admit(result, "torch")
    if input_prelude is not None:
        result["input_prelude"] = input_prelude
        for case in records:
            case["input_prelude"] = input_prelude
    return result


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), allow_nan=False))

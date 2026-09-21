"""Independent eager CPU PyTorch control for the real resident NeRF fixture.

Consumes emitted f32 ray/parameter inputs, not Rust-computed references. This
checks numerical agreement; no timing, training or universal speed claim.
"""
import hashlib
import itertools
import json
import math
import struct
import sys

GUARDS = {
    "shape_rejected", "zero_rejected", "subnormal_width_rejected",
    "inherited_error_rejected", "retained_version",
}


def f32_bytes(values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite f32 payload")
    try:
        return struct.pack("<" + "f" * len(values), *values)
    except OverflowError as error:
        raise ValueError("unrepresentable f32 payload") from error


def key(case):
    return case["rays"], case["samples"], case["varying"], case["seed"]


def admit(report):
    expected = set(itertools.product([1, 65, 256], [1, 8, 64], [False, True], [None, 17]))
    if report["status"] != "passed" or len(report["cases"]) != len(expected):
        raise ValueError("incomplete NeRF fixture")
    if set(report["guards"]) != GUARDS or not all(value is True for value in report["guards"].values()):
        raise ValueError("NeRF guard failure")
    cases = {key(case): case for case in report["cases"]}
    if set(cases) != expected:
        raise ValueError("unexpected or duplicate NeRF conditions")
    if len(report["weights"]) != 12 or len(report["bias"]) != 4:
        raise ValueError("field parameter shape")
    f32_bytes(report["weights"])
    f32_bytes(report["bias"])
    for case in cases.values():
        if len(case["ray_inputs"]) != case["rays"] or any(len(ray) != 8 for ray in case["ray_inputs"]):
            raise ValueError("ray input shape")
        if len(case["rgba"]) != case["rays"] * 4:
            raise ValueError("render output shape")
        for ray in case["ray_inputs"]:
            f32_bytes(ray)
            if ray[7] < ray[6]:
                raise ValueError("reversed ray bounds")
        f32_bytes(case["rgba"])
        mode = "Midpoint" if case["seed"] is None else "Stratified { seed: 17 }"
        if case["mode"] != mode:
            raise ValueError("sampling mode mismatch")
    return cases


def offsets(rows, count, seed):
    import torch

    if seed is None:
        return torch.full((rows, count), 0.5, dtype=torch.float64)
    values = []
    for row in range(rows):
        for sample in range(count):
            bits = seed ^ ((row * 0x9E3779B9 + sample * 0x85EBCA6B) & 0xFFFFFFFF)
            bits = ((bits ^ (bits >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
            bits = ((bits ^ (bits >> 15)) * 0x846CA68B) & 0xFFFFFFFF
            bits ^= bits >> 16
            values.append((bits >> 8) / 16777216)
    return torch.tensor(values, dtype=torch.float64).reshape(rows, count)


def render(case, weights, bias):
    import torch

    rows, count = case["rays"], case["samples"]
    # Decode source scalars as f32 before promoting coordinates/integration.
    rays = torch.tensor(case["ray_inputs"], dtype=torch.float32).to(torch.float64)
    width = (rays[:, 7:8] - rays[:, 6:7]) / count
    t = rays[:, 6:7] + (torch.arange(count, dtype=torch.float64) + offsets(rows, count, case["seed"])) * width
    positions = (rays[:, None, :3] + t[:, :, None] * rays[:, None, 3:6]).to(torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32).reshape(3, 4)
    if not case["varying"]:
        weights.zero_()
    field = positions @ weights + torch.tensor(bias, dtype=torch.float32)
    tau = field[:, :, 0].to(torch.float64).clamp_min(0) * width
    prefix = torch.cat([torch.zeros((rows, 1), dtype=torch.float64), tau[:, :-1].cumsum(1)], dim=1)
    mass = (-prefix).exp() * -torch.expm1(-tau)
    rgb = (mass[:, :, None] * field[:, :, 1:].to(torch.float64)).sum(1)
    return torch.cat([rgb, mass.sum(1, keepdim=True)], dim=1).to(torch.float32).flatten().tolist()


def compare(native, browser):
    import torch

    a, b = admit(native), admit(browser)
    for field in ["weights", "bias"]:
        if f32_bytes(native[field]) != f32_bytes(browser[field]):
            raise ValueError("native/browser parameter mismatch")
    records = []
    with torch.no_grad():
        for ident, case in a.items():
            other = b[ident]
            inputs = f32_bytes([v for ray in case["ray_inputs"] for v in ray])
            if inputs != f32_bytes([v for ray in other["ray_inputs"] for v in ray]):
                raise ValueError("native/browser ray mismatch")
            reference = render(case, native["weights"], native["bias"])
            f32_bytes(reference)
            errors = {}
            for name, values in [("native", case["rgba"]), ("browser", other["rgba"])]:
                f32_bytes(values)
                deltas = [abs(x-y) for x, y in zip(values, reference)]
                if any(d > 4e-7 + 4e-6*abs(y) for d, y in zip(deltas, reference)):
                    raise ValueError(f"{name} mismatch at {ident}: {max(deltas)}")
                errors[name] = max(deltas)
            records.append({"rays":ident[0], "samples":ident[1], "varying":ident[2], "seed":ident[3],
                            "input_sha256":hashlib.sha256(inputs).hexdigest(), "max_abs_errors":errors,
                            "torch_rgba":reference, "torch_sha256":hashlib.sha256(f32_bytes(reference)).hexdigest()})
    return {"status":"passed", "torch_version":torch.__version__, "device":"cpu",
            "intra_op_threads":torch.get_num_threads(), "inter_op_threads":torch.get_num_interop_threads(),
            "cases":records, "boundary":"Numerical control only; no performance or training comparison"}


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with open(sys.argv[1]) as stream:
        native = json.load(stream)
    with open(sys.argv[2]) as stream:
        browser = json.load(stream)
    print(json.dumps(compare(native, browser), allow_nan=False))

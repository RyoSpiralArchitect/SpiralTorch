"""Admission of complete, matched forward-only NeRF benchmark reports."""
import hashlib
import itertools
import math
import struct

SHAPES = [(1, 1), (1, 64), (65, 64), (256, 64), (1024, 64), (256, 256)]
KEYS = {(r, n, h) for r, n in SHAPES for h in (0, 32)}
ATOL, RTOL = 4e-7, 4e-6


def key(case):
    values = tuple(case[k] for k in ("rays", "samples", "hidden"))
    if any(type(v) is not int for v in values):
        raise ValueError("non-integer condition")
    return values


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("non-finite or non-numeric payload")
    return value


def f32(values):
    try:
        return struct.pack("<" + "f" * len(values), *(finite(v) for v in values))
    except (OverflowError, struct.error) as error:
        raise ValueError("unrepresentable f32 payload") from error


def close(actual, expected):
    if len(actual) != len(expected) or not expected:
        raise ValueError("output shape")
    f32(actual)
    f32(expected)
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    if any(e > ATOL + RTOL * abs(b) for e, b in zip(errors, expected)):
        raise ValueError(f"numerical mismatch: {max(errors)}")
    return max(errors)


def inputs(case):
    rows, count, hidden = key(case)
    if (rows, count, hidden) not in KEYS or type(case["seed"]) is not int or case["seed"] != 17:
        raise ValueError("unexpected condition/seed")
    rays = case["ray_inputs"]
    if len(rays) != rows or any(len(r) != 8 for r in rays):
        raise ValueError("ray shape")
    if any(r[7] <= r[6] for r in rays):
        raise ValueError("ray bounds")
    data = f32([v for r in rays for v in r])
    dims = [3, hidden, 4] if hidden else [3, 4]
    shapes = []
    for a, b in zip(dims, dims[1:]):
        shapes.extend([("weight", [a, b]), ("bias", [b])])
    if len(case["parameters"]) != len(shapes):
        raise ValueError("parameter count")
    for p, (role, shape) in zip(case["parameters"], shapes):
        if p["role"] != role or p["shape"] != shape or len(p["values"]) != math.prod(shape):
            raise ValueError("parameter layout")
        data += f32(p["values"])
    return hashlib.sha256(data).hexdigest()


def admit(report, kind):
    if kind not in ("wgpu", "torch"):
        raise ValueError("unknown report kind")
    schema = "spiraltorch.nerf_direct_bench.v1" if kind == "wgpu" else "spiraltorch.nerf_torch_bench.v1"
    if report["schema"] != schema or report["status"] != "passed":
        raise ValueError("failed/unrecognized report")
    if report["warmup"] != 3 or report["blocks"] != 9 or report["bursts"] != [1, 4]:
        raise ValueError("timing protocol")
    routes = ("staged", "direct") if kind == "wgpu" else ("cpu", "mps")
    if kind == "wgpu":
        native_gpu = "IntegratedGpu" in report["adapter"] or "DiscreteGpu" in report["adapter"]
        # Browser WGPU 0.20 withholds the device class/name. Preserve that limit:
        # the separately recorded non-fallback probe is not runtime attestation.
        browser_unknown = ("device_type: Other" in report["adapter"] and "backend: BrowserWebGpu" in report["adapter"]
                           and report.get("browser_adapter_probe", {}).get("is_fallback_adapter") is False)
        if report["guard_cases"] != 12 or not (native_gpu or browser_unknown):
            raise ValueError("GPU/guard admission")
        if report["kernel"] != "register_2x2" or report["accumulation"] != "sequential":
            raise ValueError("kernel mismatch")
    elif report["devices"] != ["cpu", "mps"] or report["intra_op_threads"] != 4 or report["inter_op_threads"] != 1 or report["compiled"] is not False or report["thin_alpha"] != "fourth_order_below_0.01":
        raise ValueError("Torch control mismatch")
    if len(report["cases"]) != len(KEYS):
        raise ValueError("incomplete case grid")
    cases = {key(c): c for c in report["cases"]}
    if set(cases) != KEYS:
        raise ValueError("missing/duplicate condition")
    expected = set(itertools.product((1, 4), range(9), routes))
    for case in cases.values():
        inputs(case)
        if len(case["reference"]) != case["rays"] * 4 or len(case["last_outputs"]) != 2:
            raise ValueError("output shape")
        for output in case["last_outputs"]:
            close(output, case["reference"])
        intervals = case["intervals"]
        if len(intervals) != len(expected):
            raise ValueError("missing timing intervals")
        seen = set()
        for interval in intervals:
            burst, block, route = (interval[k] for k in ("burst", "block", "route"))
            if type(burst) is not int or type(block) is not int:
                raise ValueError("timing condition type")
            seen.add((burst, block, route))
            order = [0, 1] if (block + 3 + case["rays"] + case["hidden"]) % 2 == 0 else [1, 0]
            if interval["order"] != order or finite(interval["elapsed_ms"]) <= 0:
                raise ValueError("invalid timing/order")
            if finite(interval["max_abs_error"]) < 0 or interval["max_abs_error"] > ATOL + RTOL * max(abs(v) for v in case["reference"]):
                raise ValueError("unaccepted interval result")
        if seen != expected:
            raise ValueError("duplicate/unknown timing condition")
    return cases

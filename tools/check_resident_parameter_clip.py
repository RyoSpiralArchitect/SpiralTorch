"""Independent CPU Torch norm oracle for the complete synthetic GPU clipping report.

Use ST's documented small-norm/near-one no-ops, not torch.clip_grad_norm_'s
additive denominator epsilon. Evaluate the norm in float64 from the exact f32
inputs so overflow of a float32 sum of squares is not mistaken for a zero update.
This is numerical validation, not a throughput or language-quality comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


NAMES = ["global", "zero", "norm_floor", "near_one", "wide_norm", "tiny_scale",
         "frozen_excluded", "all_frozen", "many_partials", "floor_below", "floor_equal",
         "floor_above_9", "floor_above_33", "floor_above_257", "subnormal_noop"]


def from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def inputs(torch):
    maximum = torch.finfo(torch.float32).max
    return [
        ([[3., -0.], [4.]], [True, True], 1.),
        ([[0., -0.], [0.]], [True, True], 1.),
        ([[1e-8], [-1e-8]], [True, True], 1e-30),
        ([[1.], [0.]], [True, True], 1. - 2**-24),
        ([[maximum, -maximum], [maximum]], [True, True], 1.),
        ([[maximum, -maximum], [maximum]], [True, True], 1e-30),
        ([[3., 4.], [maximum]], [True, False], 1.),
        ([[3., -0.], [maximum]], [False, False], .1),
        ([[1.] * 1031, [-2.] * 515], [True, True], 2.),
        ([[from_bits(0x332aaaaa)] * 9], [True], 1e-30),
        ([[from_bits(0x33000000)] * 16], [True], 1e-30),
        ([[from_bits(0x332aaaab)] * 9], [True], 1e-30),
        ([[from_bits(0x32b2416a)] * 33], [True], 1e-30),
        ([[from_bits(0x31ff8060)] * 257], [True], 1e-30),
        ([[from_bits(1), from_bits(0x007fffff)]], [True], 1e-30),
    ]


def check(report):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if (report.get("schema") != "spiraltorch.resident_parameter_clip.v1"
            or report.get("passed") is not True
            or [case["name"] for case in report["cases"]] != NAMES):
        raise ValueError("incomplete clipping report")
    rows = []
    for case, (raw_expected, active_expected, limit_expected) in zip(report["cases"], inputs(torch), strict=True):
        active = case["active"]
        tensors = [torch.tensor(v, dtype=torch.float32).double() for v in case["inputs"]]
        if (len(tensors) != len(raw_expected) or active != active_expected
                or any(type(value) is not bool for value in active)):
            raise ValueError("invalid active coverage")
        for actual, expected in zip(tensors, raw_expected, strict=True):
            if not torch.equal(actual.float().view(torch.int32), torch.tensor(expected, dtype=torch.float32).view(torch.int32)):
                raise ValueError("synthetic input changed")
        selected = [v for v, enabled in zip(tensors, active) if enabled]
        norm = torch.linalg.vector_norm(torch.cat(selected)).item() if selected else 0.0
        limit = torch.tensor(case["limit"], dtype=torch.float32).item()
        if limit != torch.tensor(limit_expected, dtype=torch.float32).item():
            raise ValueError("clipping limit changed")
        ratio = min(1.0, limit / norm) if norm > torch.finfo(torch.float32).eps else 1.0
        if abs(torch.tensor(ratio, dtype=torch.float32).item() - 1.0) <= torch.finfo(torch.float32).eps:
            ratio = 1.0
        maximum = 0.0
        if len(case["clipped"]) != len(tensors):
            raise ValueError("incomplete clipped outputs")
        for expected, actual, enabled in zip(tensors, case["clipped"], active, strict=True):
            expected = (expected * ratio if enabled else expected).float()
            actual = torch.tensor(actual, dtype=torch.float32)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise ValueError("invalid clipped values")
            torch.testing.assert_close(actual, expected, atol=0, rtol=3e-6)
            if not enabled or ratio == 1.0:
                if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                    raise ValueError("frozen or no-op output changed bits")
            error = (actual.double() - expected.double()).abs()
            relative = error / expected.double().abs().clamp_min(1e-300)
            maximum = max(maximum, relative.max().item())
        rows.append(dict(name=case["name"], norm_f64=norm, max_relative_error=maximum, passed=True))
    return dict(schema="spiraltorch.resident_parameter_clip.torch.v1", passed=True,
                torch=torch.__version__, device="cpu", cases=rows, quality_claim=False,
                scope="same clipping policy, float64 Torch norm over exact f32 operands; no speed claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = check(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(passed=True, cases=len(result["cases"]))))


if __name__ == "__main__":
    main()

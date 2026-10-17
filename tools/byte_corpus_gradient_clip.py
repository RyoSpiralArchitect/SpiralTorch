"""Independent clipping oracle and receipt validation, never a runtime fallback.

Rust owns execution. Torch uses a float64 norm and a single float64 scaling of
its own float32 VJPs, not the GPU's norm reduction or factor decomposition.
"""
import math
import struct

SCHEMA = "spiraltorch.byte_corpus.request.v10"
DISABLED = "disabled.v1"
ACTIVE = "active_global_l2.v1"


def recipe(request):
    if request["schema"] != SCHEMA:
        if "gradient_clip" in request:
            raise ValueError("gradient_clip metadata belongs to v10 only")
        return {"kind": DISABLED}
    value = request.get("gradient_clip")
    if type(value) is not dict:
        raise ValueError("v10 requires explicit gradient_clip")
    if value == {"kind": DISABLED}:
        return dict(value)
    if value.get("kind") != ACTIVE or value.keys() != {"kind", "max_norm"}:
        raise ValueError("invalid clipping recipe")
    maximum = value["max_norm"]
    if type(maximum) not in (int, float) or not math.isfinite(maximum):
        raise ValueError("clipping maximum must be finite numeric float32")
    try:
        maximum = struct.unpack("<f", struct.pack("<f", maximum))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("clipping maximum overflows float32") from exc
    if not math.isfinite(maximum) or maximum < 2**-126:
        raise ValueError("clipping maximum must be positive normal float32")
    return {"kind": ACTIVE, "max_norm": maximum}


def metadata(request, report):
    expected = recipe(request)
    if request["schema"] == SCHEMA:
        actual = report.get("gradient_clip")
        if (type(actual) is not dict or actual.keys() != expected.keys()
                or actual.get("kind") != expected["kind"]
                or (expected["kind"] == ACTIVE and
                    (type(actual["max_norm"]) not in (int, float)
                     or actual["max_norm"] != expected["max_norm"]))):
            raise ValueError("clipping report identity differs")
    elif "gradient_clip" in report:
        raise ValueError("foreign clipping report metadata")


def torch_gradients(torch, controls, gradients, active):
    if controls["kind"] == DISABLED:
        return gradients
    if len(gradients) != len(active) or any(type(value) is not bool for value in active):
        raise ValueError("invalid active gradient policy")
    # Inactive gradients do not dilute the budget, but cannot hide invalid data.
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("nonfinite clipping input")
    with torch.no_grad():
        squared = [gradient.double().square().sum() for gradient, enabled in zip(gradients, active) if enabled]
        norm = torch.stack(squared).sum().sqrt().item() if squared else 0.
        ratio = min(1., controls["max_norm"] / norm) if norm > 2**-23 else 1.
        rounded = struct.unpack("<f", struct.pack("<f", ratio))[0]
        if abs(rounded - 1.) <= 2**-23:
            return gradients
        return tuple((gradient.double() * ratio).to(gradient.dtype) if enabled else gradient
                     for gradient, enabled in zip(gradients, active))

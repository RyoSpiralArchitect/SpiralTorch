"""Independent optimizer oracle/receipt checks, not a runtime implementation.

Rust owns the public study contract. This module deliberately uses PyTorch's
optimizer rather than reproducing the Rust update formula.
"""
import math
import struct

SCHEMA = "spiraltorch.byte_corpus.request.v9"
CLIPPED_SCHEMA = "spiraltorch.byte_corpus.request.v10"
SGD = "stateless_sgd.v1"
ADAMW = "adamw_zero_rate_freeze.v1"
FIELDS = {"beta1", "beta2", "epsilon", "weight_decay", "geometry_weight_decay"}
MOMENT_RTOL = .002


def scalar(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("optimizer scalar must be finite numeric float32")
    try:
        narrowed = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("optimizer scalar overflows float32") from exc
    if not math.isfinite(narrowed) or (value != 0 and abs(narrowed) < 2**-126):
        raise ValueError("optimizer scalar must be zero or normal float32")
    return narrowed


def recipe(request):
    if request["schema"] not in (SCHEMA, CLIPPED_SCHEMA):
        if "optimizer" in request or "design_schema" in request:
            raise ValueError("optimizer/design metadata belongs to v9/v10 only")
        return {"kind": SGD}
    if request.get("design_schema") not in {
        f"spiraltorch.byte_corpus.request.v{i}" for i in range(1, 9)
    }:
        raise ValueError("v9/v10 require a v1-v8 geometry design")
    value = request.get("optimizer")
    if type(value) is not dict:
        raise ValueError("v9/v10 require an explicit optimizer")
    if value.get("kind") == SGD and set(value) == {"kind"}:
        return {"kind": SGD}
    if value.get("kind") != ADAMW or set(value) != FIELDS | {"kind"}:
        raise ValueError("invalid optimizer recipe")
    result = {"kind": ADAMW, **{name: scalar(value[name]) for name in sorted(FIELDS)}}
    if (not 0 <= result["beta1"] < 1 or not 0 <= result["beta2"] < 1
            or result["epsilon"] <= 0 or result["weight_decay"] < 0
            or result["geometry_weight_decay"] < 0 or scalar(request["rate"]) <= 0):
        raise ValueError("invalid AdamW controls")
    return result


def design(request):
    if request["schema"] not in (SCHEMA, CLIPPED_SCHEMA):
        return request
    recipe(request)
    result = {key: value for key, value in request.items()
              if key not in ("optimizer", "design_schema")}
    if request["schema"] == CLIPPED_SCHEMA:
        result.pop("gradient_clip", None)
    result["schema"] = request["design_schema"]
    return result


def metadata(request, report):
    expected = recipe(request)
    if request["schema"] in (SCHEMA, CLIPPED_SCHEMA):
        actual = report.get("optimizer")
        if (report.get("design_schema") != request["design_schema"] or type(actual) is not dict
                or actual.keys() != expected.keys() or actual.get("kind") != expected["kind"]
                or any(type(actual[key]) not in (int, float) or actual[key] != value
                       for key, value in expected.items() if key != "kind")):
            raise ValueError("optimizer/design report identity differs")
    elif "optimizer" in report or "design_schema" in report:
        raise ValueError("foreign optimizer/design report metadata")


def torch_optimizer(torch, controls, rate, descriptors, parameters):
    if controls["kind"] == SGD:
        return None
    groups = [{"params": [parameter], "weight_decay": controls[
        "geometry_weight_decay" if descriptor["name"].startswith("geometry.") else "weight_decay"]}
        for descriptor, parameter in zip(descriptors, parameters)]
    return torch.optim.AdamW(groups, lr=scalar(rate), betas=(controls["beta1"], controls["beta2"]),
                             eps=controls["epsilon"], foreach=False, fused=False)


def torch_state(optimizer, controls, descriptors, parameters):
    if optimizer is None:
        return None
    slots = []
    for descriptor, parameter in zip(descriptors, parameters):
        state = optimizer.state.get(parameter, {})
        slots.append(dict(shape=descriptor["shape"],
                          first=state["exp_avg"].reshape(-1).tolist() if state else [0.0] * parameter.numel(),
                          second=state["exp_avg_sq"].reshape(-1).tolist() if state else [0.0] * parameter.numel(),
                          steps=int(state["step"].item()) if state else 0))
    return {**{name: controls[name] for name in ("beta1", "beta2", "epsilon")}, "slots": slots}


def validate_state(case, state, controls, updates, frozen):
    if controls["kind"] == SGD:
        if state is not None:
            raise ValueError("stateless SGD must not carry AdamW state")
        return
    if (type(state) is not dict or set(state) != {"beta1", "beta2", "epsilon", "slots"}
            or type(state["slots"]) is not list or len(state["slots"]) != len(case["parameters"])):
        raise ValueError("incomplete optimizer state")
    for name in ("beta1", "beta2", "epsilon"):
        if scalar(state[name]) != controls[name]:
            raise ValueError("optimizer configuration mismatch")
    for parameter, slot in zip(case["parameters"], state["slots"]):
        if (type(slot) is not dict or set(slot) != {"shape", "first", "second", "steps"}
                or type(slot["shape"]) is not list
                or any(type(dim) is not int for dim in slot["shape"])
                or slot["shape"] != parameter["shape"]):
            raise ValueError("optimizer slot shape mismatch")
        clock = 0 if frozen(case, parameter["name"]) else updates
        if type(slot["steps"]) is not int or slot["steps"] != clock:
            raise ValueError("optimizer accepted clock differs from update/freeze policy")
        for moment in ("first", "second"):
            values = slot[moment]
            if type(values) is not list or len(values) != len(parameter["values"]):
                raise ValueError("optimizer moment shape mismatch")
            for value in values:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("nonfinite optimizer moment")
                try:
                    bits = struct.pack("<f", value)
                except (OverflowError, struct.error) as exc:
                    raise ValueError("optimizer moment overflows float32") from exc
                if (moment == "second" and value < 0) or (clock == 0 and bits != bytes(4)):
                    raise ValueError("invalid second or frozen/unstarted moment")


def compare_states(case, expected, actual, controls, updates, frozen, close):
    for state in (expected, actual):
        validate_state(case, state, controls, updates, frozen)
    if controls["kind"] == SGD:
        return {}
    result = {}
    for moment in ("first", "second"):
        maximum, relative = 0., 0.
        for a, e in zip(actual["slots"], expected["slots"]):
            maximum = max(maximum, close(a[moment], e[moment], case["name"] + "." + moment))
            norm = math.sqrt(math.fsum(x*x for x in e[moment]))
            error = math.sqrt(math.fsum((x-y)**2 for x, y in zip(a[moment], e[moment])))
            # No absolute floor: small nonzero history cannot be erased and
            # promoted merely because ordinary weight tolerances are larger.
            ratio = error / norm if norm else (0. if error == 0 else math.inf)
            if ratio > MOMENT_RTOL:
                raise ValueError("unqualified optimizer moment: " + case["name"] + "." + moment)
            relative = max(relative, ratio)
        result[moment + "_max_error"] = maximum
        result[moment + "_max_relative_l2_error"] = relative
    return result

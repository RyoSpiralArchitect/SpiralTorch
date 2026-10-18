"""Fixed study policy and scalar aggregation, not a new geometric operator."""

import copy
import math

ARMS = ["fixed_r1", "learned_r1", "fixed_r4", "learned_r4"]
SITES = [f"transformer.h.{index}.mlp" for index in (8, 9, 10, 11)]


def cases_for(plan):
    expected = {"schema": "spiraltorch.hf_residual_calibration_plan.v1", "seeds": [41, 43, 47],
                "geometry_arms": ARMS, "case_count": 27, "reference_cases": 15,
                "calibrated_ordinary_cases": 12, "train_windows": 1047, "development_windows": 159,
                "calibration_windows": 64, "calibration_relative_tolerance": .001,
                "context_length": 128, "eval_batch_size": 2, "max_values": 1_048_576,
                "device": "mps", "dtype": "float32", "attention": "eager", "kv_cache": False,
                "optimizer_updates": 0}
    if any(type(plan.get(key)) is not type(value) or plan[key] != value for key, value in expected.items()):
        raise ValueError("unsupported calibration declaration")
    cases = []
    for seed in plan["seeds"]:
        cases.append({"arm": "ordinary", "seed": seed, "mode": "reference"})
        for arm in ARMS:
            cases.extend([{"arm": arm, "seed": seed, "mode": mode}
                          for mode in ("reference", "calibrated_ordinary")])
    return cases


def calibration_indices(size, count):
    if type(size) is not int or type(count) is not int or not 0 < count <= size:
        raise ValueError("calibration needs distinct valid training windows")
    return [index * size // count for index in range(count)]


def fitted_strength(strength, target, current):
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (strength, target, current)) or strength > 1:
        raise ValueError("calibration requires positive finite strength and RMS")
    result = strength * (target / current)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError("calibrated strength is outside the existing recipe domain")
    return result


def without_strength(record):
    result = copy.deepcopy(record)
    if [p["path"] for p in result["placements"]] != SITES:
        raise ValueError("calibration requires the pinned forward layer order")
    for placement in result["placements"]:
        del placement["recipe"]["strength"]
    return result


def public_layers(aggregates):
    if [layer["path"] for layer in aggregates] != SITES:
        raise ValueError("observation site coverage differs")
    result = [{"path": layer["path"], "calls": layer["calls"], "rows": layer["rows"],
               "relative_residual_rms": layer["relative_residual_energy"]} for layer in aggregates]
    if any(type(layer["relative_residual_rms"]) not in (int, float)
           or not math.isfinite(layer["relative_residual_rms"])
           or layer["relative_residual_rms"] <= 0 for layer in result):
        raise ValueError("residual calibration requires finite nonzero observations")
    return result


def mismatch(actual, target):
    if ([row["path"] for row in actual] != SITES
            or [row["path"] for row in target] != SITES):
        raise ValueError("comparison needs every ordered site")
    errors = []
    for left, right in zip(actual, target):
        a, b = left["relative_residual_rms"], right["relative_residual_rms"]
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (a, b)):
            raise ValueError("comparison needs finite nonzero RMS")
        errors.append({"path": left["path"], "relative_rms_error": a / b - 1})
    return errors


def public_record(record):
    result = copy.deepcopy(record)
    # Target/current calibration RMS would reconstruct each fitted strength.
    # Keep both the reference calibration and intermediate fit operands private.
    stages = [*result["calibration_stages"], result["calibration"]]
    if result["mode"] == "calibrated_ordinary":
        # First-layer ordinary inputs/weights are identical on ANY split:
        # matched/reference development RMS would also reveal its strength.
        stages.append(result["development"])
    for stage in stages:
        stage["layers"] = [{key: layer[key] for key in ("path", "calls", "rows")}
                           for layer in stage["layers"]]
    for key in ("calibration_mismatch", "development_mismatch"):
        if result.get(key) is not None:
            result[key] = [{"path": row["path"],
                            "within_0_1_percent": abs(row["relative_rms_error"]) <= .001,
                            "within_1_percent": abs(row["relative_rms_error"]) <= .01,
                            "within_5_percent": abs(row["relative_rms_error"]) <= .05}
                           for row in result[key]]
    return result

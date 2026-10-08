#!/usr/bin/env python3
"""Summarize sealed receipts; optional checkpoint verification imports Torch, never scores."""

import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_gate(raw):
    """Descriptive tanh, fixed at 12 decimal places without platform libm."""
    require(math.isfinite(raw), "nonfinite raw gate")
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        value = Decimal.from_float(float(raw))
        if abs(value) >= 20:
            return -1.0 if value < 0 else 1.0
        decay = (-2 * abs(value)).exp()
        mix = ((1 - decay) / (1 + decay)).copy_sign(value)
        # Normalize signed zero as well as the final decimal representation.
        return float(mix.quantize(Decimal("1e-12"))) or 0.0


def causal_factorial_contrasts(config, runs, measured, sets):
    gated = config.get("schema") == "spiraltorch.elliptic_gated_protocol.v1"
    prefix = "gated" if gated else "causal"
    tangent, elliptic = f"{prefix}_tangent", f"{prefix}_elliptic"
    arms = {"tangent", "elliptic", tangent, elliptic}
    require(set(config["arms"]) == arms, "incomplete causal factorial design")
    for seed in config["seeds"]:
        rows = {arm: runs[f"{seed}:{arm}"] for arm in arms}
        hashes = {
            row.get("initial_projection_sha256" if gated and arm.startswith("gated_")
                    else "initial_parameter_sha256")
            for arm, row in rows.items()
        }
        require(
            len(hashes) == 1
            and all(isinstance(value, str) and len(value) == 64 for value in hashes),
            "factorial initial parameters are not paired",
        )
        require(
            all(row["parameter_count"] == 11 * config["features"] + 2
                + int(gated and arm.startswith("gated_"))
                for arm, row in rows.items()),
            "factorial parameter counts differ",
        )
        if gated:
            gated_hashes = {rows[arm].get("initial_parameter_sha256") for arm in (tangent, elliptic)}
            require(
                len(gated_hashes) == 1
                and all(isinstance(value, str) and len(value) == 64 for value in gated_hashes),
                "gated initial parameters are not paired",
            )
    contrasts = {
        "geometry_pointwise": {"elliptic": 1, "tangent": -1},
        f"geometry_{prefix}": {elliptic: 1, tangent: -1},
        "mixing_tangent": {tangent: 1, "tangent": -1},
        "mixing_elliptic": {elliptic: 1, "elliptic": -1},
        "interaction": {
            elliptic: 1,
            tangent: -1,
            "elliptic": -1,
            "tangent": 1,
        },
    }
    return contrast_report(config, measured, sets, contrasts)


def anchored_factorial_contrasts(config, runs, measured, sets):
    arms = {"anchored_tangent", "anchored_elliptic", "gated_tangent", "gated_elliptic"}
    require(
        set(config["arms"]) == arms and config.get("reference_arm") == "anchored_tangent",
        "incomplete anchored factorial design",
    )
    for seed in config["seeds"]:
        rows = [runs[f"{seed}:{arm}"] for arm in arms]
        for field in ("initial_parameter_sha256", "initial_projection_sha256"):
            hashes = {row.get(field) for row in rows}
            require(
                len(hashes) == 1
                and all(isinstance(value, str) and len(value) == 64 for value in hashes),
                "anchored factorial initial parameters are not paired",
            )
        require(
            all(row["parameter_count"] == 11 * config["features"] + 3 for row in rows),
            "anchored factorial parameter counts differ",
        )
    contrasts = {
        "geometry_anchored": {"anchored_elliptic": 1, "anchored_tangent": -1},
        "geometry_gated": {"gated_elliptic": 1, "gated_tangent": -1},
        "anchor_tangent": {"anchored_tangent": 1, "gated_tangent": -1},
        "anchor_elliptic": {"anchored_elliptic": 1, "gated_elliptic": -1},
        "interaction": {
            "anchored_elliptic": 1,
            "anchored_tangent": -1,
            "gated_elliptic": -1,
            "gated_tangent": 1,
        },
    }
    return contrast_report(config, measured, sets, contrasts)


def contrast_report(config, measured, sets, contrasts):
    report = {}
    for name in sets:
        report[name] = {}
        for label, weights in contrasts.items():
            values = [
                sum(
                    weight * measured[f"{seed}:{arm}"][name]
                    for arm, weight in weights.items()
                )
                for seed in config["seeds"]
            ]
            report[name][label] = {
                "weights": weights,
                "mean_ce_difference": statistics.fmean(values),
                "paired_sample_sd": statistics.stdev(values)
                if len(values) > 1
                else None,
                "negative_seeds": sum(value < 0 for value in values),
                "per_seed": [
                    {"seed": seed, "ce_difference": value}
                    for seed, value in zip(config["seeds"], values)
                ],
            }
    return report


def order_trajectory(row, initial, learned):
    records = row["records"]
    before = [r.get("log_alpha_before_update") for r in records]
    after = [r.get("log_alpha_after_update") for r in records]
    alpha_before = [r.get("alpha_before_update") for r in records]
    alpha_after = [r.get("alpha_after_update") for r in records]
    final_log, final_alpha = row.get("final_log_alpha"), row.get("final_alpha")
    require(bool(records) and all(type(v) in (int, float) and math.isfinite(v)
                for v in before + after + alpha_before + alpha_after + [final_log, final_alpha]),
            "missing or nonfinite order trajectory")
    for log_alpha, alpha in zip(before + after, alpha_before + alpha_after):
        try:
            expected = math.exp(log_alpha)
        except OverflowError:
            expected = math.inf
        require(alpha > 0 and math.isfinite(expected)
                and math.isclose(alpha, expected, rel_tol=2e-6, abs_tol=2**-149),
                "alpha does not match positive f32 order")
    require(math.isclose(alpha_before[0], initial, rel_tol=2e-6)
            and before[1:] == after[:-1] and alpha_before[1:] == alpha_after[:-1]
            and final_log == after[-1] and final_alpha == alpha_after[-1],
            "order trajectory or endpoint differs")
    require(all(r.get("log_alpha_trainable") is learned for r in records),
            "order training mode differs")
    gradients = [r.get("log_alpha_gradient") for r in records]
    if learned:
        require(all(type(v) in (int, float) and math.isfinite(v) for v in gradients)
                and gradients[0] == 0, "invalid learned order gradients")
    else:
        require(all(v is None for v in gradients) and all(v == before[0] for v in before + after),
                "fixed fractional order changed")
    return {
        "learnable_order": learned, "initial_alpha": alpha_before[0],
        "final_alpha": final_alpha, "final_log_alpha": final_log,
        "min_alpha": min(alpha_before + alpha_after),
        "max_alpha": max(alpha_before + alpha_after),
        "nonzero_order_gradient_steps": sum(v not in (None, 0) for v in gradients),
    }


def fractional_report(config, runs, measured, sets):
    history = config.get("schema") == "spiraltorch.fractional_history_protocol.v1"
    fixed, learned_arm = (("history_fixed", "history_learned") if history
                          else ("fractional_fixed", "fractional_learned"))
    arms = {"pointwise", fixed, learned_arm} | ({"ema_learned"} if history else set())
    require(set(config["arms"]) == arms and config.get("reference_arm") == "pointwise",
            "incomplete fractional design")
    require(type(config.get("features")) is int and config["features"] > 0,
            "invalid fractional feature count")
    initial = config.get("initial_alpha")
    require(type(initial) in (int, float) and math.isfinite(initial) and initial > 0,
            "invalid initial alpha")
    trajectories = {}
    for seed in config["seeds"]:
        hashes = {runs[f"{seed}:{arm}"].get("initial_parameter_sha256")
                  for arm in (fixed, learned_arm)}
        require(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes),
                "fractional initial parameters are not paired")
        for arm in config["arms"]:
            row = runs[f"{seed}:{arm}"]
            learned = arm == learned_arm
            gate_count = config["features"] * (2 if history and arm != "pointwise" else 1)
            require(type(row.get("parameter_count")) is int
                    and type(row.get("trainable_parameter_count")) is int
                    and row["parameter_count"] == gate_count + int(arm != "pointwise")
                    and row["trainable_parameter_count"] == gate_count + int(learned or arm == "ema_learned"),
                    "fractional parameter counts differ")
            if arm not in (fixed, learned_arm):
                continue
            trajectories[f"{seed}:{arm}"] = order_trajectory(row, initial, learned)
    contrasts = {"fixed_minus_pointwise": {fixed: 1, "pointwise": -1},
                 "learned_minus_fixed": {learned_arm: 1, fixed: -1},
                 "learned_minus_pointwise": {learned_arm: 1, "pointwise": -1}}
    if history:
        contrasts.update(learned_minus_ema={learned_arm: 1, "ema_learned": -1},
                         ema_minus_pointwise={"ema_learned": 1, "pointwise": -1})
    return contrast_report(config, measured, sets, contrasts), trajectories


def fractional_lag_report(config, runs, measured, sets):
    two_lag = config.get("schema") == "spiraltorch.fractional_two_lag_protocol.v1"
    ordinary_arm = "lag2" if two_lag else "lag1"
    fixed_arm = "history_fixed_two" if two_lag else "history_fixed_one"
    paired_arm = "history_learned_two" if two_lag else "history_learned_one"
    initial_orders = ({"history_fixed_two": 2.0, "history_learned_two": 2.0,
                       "history_learned_one": 1.0} if two_lag else
                      {"history_fixed_one": 1.0, "history_learned_one": 1.0,
                       "history_learned_half": 0.5})
    arms = {ordinary_arm, *initial_orders}
    require(set(config["arms"]) == arms and config.get("reference_arm") == ordinary_arm,
            "incomplete lag design")
    require(config.get("initial_orders") == initial_orders
            and all(type(v) in (int, float) for v in config["initial_orders"].values()),
            "lag initial orders differ")
    require(type(config.get("features")) is int and config["features"] > 0,
            "invalid lag feature count")
    kernel = config.get("kernel", {})
    require(type(kernel.get("step")) in (int, float) and kernel["step"] == 1
            and type(kernel.get("kernel_len")) is int and kernel["kernel_len"] > (2 if two_lag else 1),
            "lag kernel differs")
    trajectories, parity = {}, {}
    common_fields = ("loss", "gate_gradient_l2", "local_gate_gradient_l2",
                     "gate_before_update_l2", "local_gate_before_update_l2")
    for seed in config["seeds"]:
        hashes = {runs[f"{seed}:{arm}"].get("initial_parameter_sha256")
                  for arm in (fixed_arm, paired_arm)}
        require(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes),
                "integer-order initial parameters are not paired")
        for arm in config["arms"]:
            row = runs[f"{seed}:{arm}"]
            learned = arm.startswith("history_learned_")
            gate_count = 2 * config["features"]
            require(type(row.get("parameter_count")) is int
                    and type(row.get("trainable_parameter_count")) is int
                    and row["parameter_count"] == gate_count + int(arm != ordinary_arm)
                    and row["trainable_parameter_count"] == gate_count + int(learned),
                    "lag parameter counts differ")
            require(bool(row["records"]) and all(
                type(r.get(k)) in (int, float) and math.isfinite(r[k]) and r[k] >= 0
                for r in row["records"] for k in common_fields), "invalid lag update receipt")
            if arm in initial_orders:
                trajectories[f"{seed}:{arm}"] = order_trajectory(row, initial_orders[arm], learned)
        ordinary, fixed = (runs[f"{seed}:{arm}"] for arm in (ordinary_arm, fixed_arm))
        parity[str(seed)] = {
            "update_receipts_equal":
                [[r[k] for k in common_fields] for r in ordinary["records"]]
                == [[r[k] for k in common_fields] for r in fixed["records"]],
            "development_equal": ordinary["development"] == fixed["development"],
            "endpoint_block_losses_equal": {
                name: ordinary["scores"][name]["block_losses"] == fixed["scores"][name]["block_losses"]
                for name in sets
            },
        }
    contrasts = {
        "fixed_two_minus_lag2": {"history_fixed_two": 1, "lag2": -1},
        "learned_two_minus_fixed_two": {"history_learned_two": 1, "history_fixed_two": -1},
        "learned_one_minus_learned_two": {"history_learned_one": 1, "history_learned_two": -1},
        "learned_one_minus_lag2": {"history_learned_one": 1, "lag2": -1},
    } if two_lag else {
        "fixed_one_minus_lag1": {"history_fixed_one": 1, "lag1": -1},
        "learned_one_minus_fixed_one": {"history_learned_one": 1, "history_fixed_one": -1},
        "learned_half_minus_learned_one": {"history_learned_half": 1, "history_learned_one": -1},
        "learned_half_minus_lag1": {"history_learned_half": 1, "lag1": -1},
    }
    return contrast_report(config, measured, sets, contrasts), trajectories, parity


def history_factorial_report(config, runs, measured, sets):
    arms = {"history_raw_short", "history_raw_full", "history_l2_short", "history_l2_full"}
    require(set(config["arms"]) == arms and config.get("reference_arm") == "history_raw_full",
            "incomplete history factorial design")
    require(type(config.get("features")) is int and config["features"] > 0,
            "invalid history feature count")
    require(type(config.get("initial_alpha")) in (int, float) and config["initial_alpha"] == 2,
            "history initial order differs")
    require(type(config.get("history_l2_gain")) in (int, float)
            and config["history_l2_gain"] == math.sqrt(5), "history initial filter energy differs")
    kernel = config.get("kernel", {})
    require(type(config.get("short_kernel_len")) is int and config["short_kernel_len"] == 3
            and type(kernel.get("kernel_len")) is int and kernel["kernel_len"] > 3
            and type(kernel.get("step")) in (int, float) and kernel["step"] == 1,
            "history factorial kernels differ")
    trajectories, initial_parity = {}, {}
    fields = ("loss", "gate_gradient_l2", "local_gate_gradient_l2",
              "gate_before_update_l2", "local_gate_before_update_l2")
    for seed in config["seeds"]:
        hashes = {runs[f"{seed}:{arm}"].get("initial_parameter_sha256") for arm in arms}
        require(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes),
                "history initial parameters are not paired")
        for arm in config["arms"]:
            row = runs[f"{seed}:{arm}"]
            require(type(row.get("parameter_count")) is int
                    and type(row.get("trainable_parameter_count")) is int
                    and row["parameter_count"] == row["trainable_parameter_count"] == 2*config["features"]+1,
                    "history parameter counts differ")
            require(bool(row["records"]) and all(
                type(r.get(k)) in (int, float) and math.isfinite(r[k]) and r[k] >= 0
                for r in row["records"] for k in fields), "invalid history update receipt")
            require(row["records"][0]["gate_before_update_l2"] == 0
                    and row["records"][0]["local_gate_before_update_l2"] == 0,
                    "history gates are not identity initialized")
            trajectories[f"{seed}:{arm}"] = order_trajectory(row, 2., True)
        first = runs[f"{seed}:history_raw_full"]["records"][0]
        equal = all(runs[f"{seed}:{arm}"]["records"][0][field] == first[field]
                    for arm in arms for field in fields)
        # Keep an observed mismatch visible instead of turning it into a quality win.
        initial_parity[str(seed)] = {"first_update_receipts_equal": equal,
                                     "status": "passed" if equal else "failed"}
    contrasts = {
        "raw_full_minus_raw_short": {"history_raw_full": 1, "history_raw_short": -1},
        "l2_full_minus_l2_short": {"history_l2_full": 1, "history_l2_short": -1},
        "l2_short_minus_raw_short": {"history_l2_short": 1, "history_raw_short": -1},
        "l2_full_minus_raw_full": {"history_l2_full": 1, "history_raw_full": -1},
        "length_by_normalization_interaction": {
            "history_l2_full": 1, "history_l2_short": -1,
            "history_raw_full": -1, "history_raw_short": 1},
    }
    return contrast_report(config, measured, sets, contrasts), trajectories, initial_parity


def two_lag_state_parity(plan, runs, checkpoint_dir):
    """Read hash-bound final states, not scalar norms or validation-update states."""
    if checkpoint_dir is None:
        return {str(seed): {"status": "unverified", "saved_gates_equal": None,
                            "saved_named_adam_equal": None}
                for seed in plan["config"]["seeds"]}
    import torch

    config, directory = plan["config"], Path(checkpoint_dir)
    gates = ("gate", "local_gate")

    def tensor_receipt(value, shape):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.layout == torch.strided and value.dtype == torch.float32
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                "invalid checkpoint tensor")
        raw = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        return {"dtype": "float32", "shape": list(shape),
                "sha256": hashlib.sha256(raw).hexdigest()}

    def read_state(seed, arm):
        key = f"{seed}:{arm}"
        row, names = runs[key], ["gate", "local_gate"]
        if arm == "history_fixed_two":
            names = ["gate", "log_alpha", "local_gate"]
        receipt = row["checkpoint"]
        name = receipt["filename"]
        require(isinstance(name, str) and Path(name).name == name
                and "/" not in name and "\\" not in name
                and name.startswith("checkpoint-") and name.endswith(".pt"),
                "invalid checkpoint path")
        path = directory / name
        require(not path.is_symlink(), "checkpoint must not be a symlink")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt["sha256"],
                "checkpoint hash mismatch")
        # Deserialize the exact bytes just hashed, with no arbitrary pickle objects.
        saved = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        require(saved["study_id"] == plan["study_id"] and saved["run_key"] == key
                and saved["cursor"] == config["steps"]
                and saved.get("frozen_base_verified") is True
                and saved["initial_parameter_sha256"] == row["initial_parameter_sha256"]
                and saved["records"] == row["records"]
                and saved["development"] == row["development"],
                "checkpoint identity or endpoint receipts differ")
        adapter, optimizer = saved["adapter"], saved["optimizer"]
        # These study schemas serialize parameters in registration order. Adam's
        # numeric IDs differ across arms because the frozen order occupies a slot.
        require(list(adapter) == [*names, "_extra_state"], "checkpoint parameter names differ")
        extra = adapter["_extra_state"]
        require(extra.get("features") == config["features"]
                and extra.get("strength") == config["strength"]
                and all(extra.get("kernel", {}).get(k) == v for k, v in config["kernel"].items()),
                "checkpoint recipe differs")
        if arm == "lag2":
            require(extra.get("schema") == "spiraltorch.ordinary_two_lag_control.v1"
                    and extra.get("history_coefficients") == [-2.0, 1.0]
                    and extra.get("accumulation_dtype") == "float64", "checkpoint recipe differs")
        else:
            require(extra.get("study_schema") == "spiraltorch.fractional_two_lag_control.v1"
                    and extra.get("arm") == arm and extra.get("initial_alpha") == 2.0
                    and extra.get("learnable_alpha") is False, "checkpoint recipe differs")
            tensor_receipt(adapter["log_alpha"], ())
            require(float(adapter["log_alpha"]) == row["final_log_alpha"]
                    and float(adapter["log_alpha"].exp()) == row["final_alpha"] == 2.0,
                    "checkpoint fixed order differs")
        group_list = optimizer["param_groups"]
        require(len(group_list) == 1, "checkpoint Adam groups differ")
        group = group_list[0]
        ids = group["params"]
        require(len(ids) == len(names) and all(type(i) is int for i in ids)
                and len(set(ids)) == len(ids), "checkpoint Adam parameter IDs differ")
        by_name = dict(zip(names, ids))
        require(set(optimizer["state"]) == {by_name[n] for n in gates},
                "checkpoint Adam states missing or frozen order has state")
        require(group.get("lr") == config["learning_rate"] and group.get("amsgrad") is False,
                "checkpoint Adam configuration differs")
        parameters, adam = {}, {}
        for field in gates:
            parameters[field] = tensor_receipt(adapter[field], (config["features"],))
            state = optimizer["state"][by_name[field]]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "checkpoint Adam fields differ")
            adam[field] = {name: tensor_receipt(value, () if name == "step" else (config["features"],))
                           for name, value in state.items()}
            require(float(state["step"]) == config["steps"], "checkpoint Adam cursor differs")
        metadata = {k: v for k, v in group.items() if k != "params"}
        metadata_raw = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        return {"checkpoint_sha256": receipt["sha256"], "gates": parameters, "named_adam": adam,
                "adam_group_sha256": hashlib.sha256(metadata_raw).hexdigest()}

    report = {}
    for seed in config["seeds"]:
        left, right = (read_state(seed, arm) for arm in ("lag2", "history_fixed_two"))
        gates_equal = left["gates"] == right["gates"]
        adam_equal = (left["named_adam"] == right["named_adam"]
                      and left["adam_group_sha256"] == right["adam_group_sha256"])
        report[str(seed)] = {
            "status": "passed" if gates_equal and adam_equal else "failed",
            "saved_gates_equal": gates_equal, "saved_named_adam_equal": adam_equal,
            "states": {"lag2": left, "history_fixed_two": right},
        }
    return report


def ema_trajectories(config, runs):
    initial = config.get("initial_decay")
    require(type(initial) in (int, float) and math.isfinite(initial) and 0 < initial < 1,
            "invalid initial decay")

    def sigmoid(value):
        z = math.exp(-abs(value))
        return 1 / (1 + z) if value >= 0 else z / (1 + z)

    report = {}
    for seed in config["seeds"]:
        key = f"{seed}:ema_learned"
        row = runs[key]
        records = row["records"]
        before = [r.get("logit_decay_before_update") for r in records]
        after = [r.get("logit_decay_after_update") for r in records]
        decay = [r.get("decay_after_update") for r in records]
        gradients = [r.get("logit_decay_gradient") for r in records]
        final_logit, final_decay = row.get("final_logit_decay"), row.get("final_decay")
        require(bool(records) and all(type(v) in (int, float) and math.isfinite(v)
                for v in before + after + decay + gradients + [final_logit, final_decay]),
                "missing or nonfinite EMA trajectory")
        require(math.isclose(sigmoid(before[0]), initial, rel_tol=2e-6)
                and before[1:] == after[:-1] and gradients[0] == 0
                and final_logit == after[-1] and final_decay == decay[-1],
                "EMA trajectory or endpoint differs")
        require(all(0 < d < 1 and math.isclose(d, sigmoid(v), rel_tol=2e-6, abs_tol=2**-149)
                    for v, d in zip(after, decay)), "EMA decay differs from logit")
        report[key] = {"initial_decay": initial, "final_decay": final_decay,
                       "final_logit_decay": final_logit,
                       "nonzero_decay_gradient_steps": sum(v != 0 for v in gradients)}
    return report


def chart_step_report(config, runs, measured, sets):
    arms = {"adam_tangent", "adam_elliptic", "chart_tangent", "chart_elliptic"}
    require(set(config["arms"]) == arms and config.get("reference_arm") == "adam_tangent", "incomplete chart factorial design")
    damping = config.get("relative_damping")
    require(type(damping) in (int, float) and math.isfinite(damping) and 1e-6 <= damping <= 1.0, "invalid relative damping")
    for seed in config["seeds"]:
        rows = [runs[f"{seed}:{arm}"] for arm in arms]
        for field in ("initial_parameter_sha256", "initial_projection_sha256"):
            hashes = {row.get(field) for row in rows}
            require(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes), "chart initial parameters are not paired")
        require(all(row["parameter_count"] == 11 * config["features"] + 3 for row in rows), "chart parameter counts differ")
    trajectories = {}
    for key, row in runs.items():
        enabled = key.split(":", 1)[1].startswith("chart_")
        receipts = [record.get("optimizer_step") for record in row["records"]]
        require(all(isinstance(r, dict) and r.get("enabled") is enabled for r in receipts), "missing or mismatched chart-step receipt")
        if not enabled:
            continue
        for receipt in receipts:
            fields = ("proposal_l2", "step_l2", "applied_step_l2", "damped_condition", "gradient_dot_proposal", "gradient_dot_applied_step")
            require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) for k in fields), "invalid chart-step scalar")
            require(all(receipt[k] >= 0 for k in fields[:3]), "negative step norm")
            require(math.isclose(receipt["proposal_l2"], receipt["step_l2"], rel_tol=1e-6, abs_tol=1e-40), "native chart step changed proposal budget")
            # No absolute floor: a vanished tiny update is not a preserved budget.
            require(math.isclose(receipt["proposal_l2"], receipt["applied_step_l2"], rel_tol=2e-5, abs_tol=0.0), "applied chart step changed proposal budget")
            require(receipt["damped_condition"] >= 1, "invalid damped condition")
            metric = receipt.get("metric")
            require(isinstance(metric, list) and len(metric) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in metric), "invalid chart metric")
            require(metric[0] >= 0 and metric[3] >= 0 and metric[0] + metric[3] > 0 and metric[1] == metric[2], "invalid chart metric")
            # Normalize before products to avoid overflowing/underflowing det(G).
            scale = max(metric[0], metric[3])
            a, b, c = metric[0] / scale, metric[1] / scale, metric[3] / scale
            require(math.isfinite(b) and a * c - b * b >= -64 * math.ulp(1.0), "chart metric is not positive semidefinite")
            trace = a + c
            a, b, c = 2 * a / trace + damping, 2 * b / trace, 2 * c / trace + damping
            determinant = a * c - b * b
            largest = ((a + c) + math.hypot(a - c, 2 * b)) * 0.5
            require(determinant > 0, "invalid damped chart metric")
            expected_condition = largest * largest / determinant
            # The native configuration crosses an f32 boundary; metrics use f64.
            require(math.isclose(receipt["damped_condition"], expected_condition, rel_tol=1e-6), "chart condition differs from metric and damping")
            cosine = receipt.get("cosine")
            require((receipt["proposal_l2"] == 0 and cosine is None and receipt["applied_step_l2"] == 0) or (receipt["proposal_l2"] > 0 and type(cosine) in (int, float) and math.isfinite(cosine) and 0 <= cosine <= 1), "invalid step direction cosine")
        active = [r for r in receipts if r["proposal_l2"] > 0]
        trajectories[key] = {
            "nonzero_proposals": len(active),
            "mean_direction_cosine": statistics.fmean(r["cosine"] for r in active) if active else None,
            "mean_damped_condition": statistics.fmean(r["damped_condition"] for r in receipts),
            "max_applied_norm_relative_error": max((abs(r["applied_step_l2"] / r["proposal_l2"] - 1) for r in active), default=0),
            "positive_gradient_dot_proposal_steps": sum(r["gradient_dot_proposal"] > 0 for r in receipts),
            "positive_gradient_dot_applied_steps": sum(r["gradient_dot_applied_step"] > 0 for r in receipts),
        }
    contrasts = {
        "chart_elliptic": {"chart_elliptic": 1, "adam_elliptic": -1},
        "chart_tangent": {"chart_tangent": 1, "adam_tangent": -1},
        "geometry_adam": {"adam_elliptic": 1, "adam_tangent": -1},
        "geometry_chart": {"chart_elliptic": 1, "chart_tangent": -1},
        "interaction": {"chart_elliptic": 1, "adam_elliptic": -1, "chart_tangent": -1, "adam_tangent": 1},
    }
    return contrast_report(config, measured, sets, contrasts), trajectories


def gated_trajectories(config, runs, arms=("gated_tangent", "gated_elliptic")):
    report = {}
    for seed in config["seeds"]:
        for arm in arms:
            key = f"{seed}:{arm}"
            row = runs[key]
            before = [r.get("raw_mix_before_update") for r in row["records"]]
            after = [r.get("raw_mix_after_update") for r in row["records"]]
            gradients = [r.get("raw_mix_gradient") for r in row["records"]]
            values = before + after + gradients + [row.get("final_raw_mix")]
            require(
                all(type(v) in (int, float) and math.isfinite(v) for v in values),
                "missing or nonfinite gate trajectory",
            )
            require(
                before[0] == 0.0 and after[:-1] == before[1:]
                and after[-1] == row["final_raw_mix"],
                "gate trajectory is not continuous or endpoint differs",
            )
            report[key] = {
                "initial_raw_mix": before[0],
                "final_raw_mix": after[-1],
                "final_mix": canonical_gate(after[-1]),
                "min_raw_mix": min(before + after),
                "max_raw_mix": max(before + after),
                "nonzero_gradient_steps": sum(v != 0 for v in gradients),
            }
    return report


def summarize(plan, result, journal, result_sha256, *, checkpoint_dir=None):
    require(
        result["status"] == journal["status"] == "completed",
        "study is not completed",
    )
    require(
        result["study_id"] == journal["study_id"] == plan["study_id"],
        "study identity differs",
    )
    require(journal["results_sha256"] == result_sha256, "result hash differs")
    config = plan["config"]
    require(checkpoint_dir is None or config.get("schema") == "spiraltorch.fractional_two_lag_protocol.v1",
            "checkpoint parity is supported only for the two-lag protocol")
    seeds, arms = config["seeds"], config["arms"]
    reference = config.get("reference_arm", "tangent")
    require(
        len(set(seeds)) == len(seeds) > 0
        and len(set(arms)) == len(arms)
        and reference in arms,
        "invalid paired design",
    )
    expected = {f"{seed}:{arm}" for seed in seeds for arm in arms}
    runs = {row["run_key"]: row for row in result["runs"]}
    require(
        set(runs) == set(journal["runs"]) == expected
        and len(runs) == len(result["runs"]),
        "missing or duplicate run",
    )
    sets = plan["data"]["evaluation_block_hashes"]
    require(bool(sets), "no evaluation sets")

    def scores(payload):
        require(set(payload) == set(sets), "evaluation sets differ")
        means = {}
        for name, hashes in sets.items():
            values = payload[name]["block_losses"]
            require(len(values) == len(hashes) > 0, "evaluation block count differs")
            require(
                all(
                    type(x) in (int, float) and math.isfinite(x) and x >= 0
                    for x in values
                ),
                "invalid block loss",
            )
            mean = statistics.fmean(values)
            require(
                math.isclose(payload[name]["mean"], mean, rel_tol=1e-12, abs_tol=1e-12),
                "reported mean differs from block losses",
            )
            means[name] = mean
        return means

    baseline = scores(result["baseline"])
    measured = {}
    for key, row in runs.items():
        entry = journal["runs"][key]
        require(
            entry["status"] == "completed"
            and entry["cursor"] == config["steps"]
            and entry["frozen_base_unchanged"] is True
            and entry["resume_next_update_equal"] is True
            and row["resume_next_update_equal"] is True
            and row["checkpoint"] == entry["checkpoint"],
            "unverified endpoint",
        )
        records = row["records"]
        schedule = plan["batch_schedules"][key.split(":", 1)[0]]
        require(
            len(records) == config["steps"]
            and len(schedule) == config["steps"] + 1
            and all(
                record["step"] == i + 1 and record["batch_indices"] == schedule[i]
                for i, record in enumerate(records)
            ),
            "training cursor or batch history differs",
        )
        measured[key] = scores(row["scores"])

    comparisons = {}
    for name, hashes in sets.items():
        rows = {}
        for arm in arms:
            values = [measured[f"{seed}:{arm}"][name] for seed in seeds]
            delta = [
                value - measured[f"{seed}:{reference}"][name]
                for seed, value in zip(seeds, values)
            ]
            rows[arm] = {
                "mean_ce": statistics.fmean(values),
                "mean_delta_vs_baseline": statistics.fmean(values) - baseline[name],
                f"mean_delta_vs_{reference}": statistics.fmean(delta),
                "paired_delta_sample_sd": statistics.stdev(delta)
                if len(delta) > 1
                else None,
                f"seeds_better_than_{reference}": sum(value < 0 for value in delta),
                "per_seed": [
                    {"seed": seed, "ce": value, f"delta_vs_{reference}": difference}
                    for seed, value, difference in zip(seeds, values, delta)
                ],
            }
        comparisons[name] = {
            "blocks": len(hashes),
            "baseline_ce": baseline[name],
            "arms": rows,
        }
    comparison_notes = config.get(
        "comparison_notes",
        [
            "Seeds vary paired sample order, not the frozen model or identity initialization.",
            "The learned-radius arm has one extra scalar; radius 4 was selected by an earlier pilot.",
        ],
    )
    require(
        isinstance(comparison_notes, list)
        and all(isinstance(x, str) for x in comparison_notes),
        "invalid comparison notes",
    )
    summary = {
        "schema": config.get(
            "summary_schema", "spiraltorch.wave_gate_long_horizon_summary.v1"
        ),
        "study_id": plan["study_id"],
        "steps_per_run": config["steps"],
        "runs": len(runs),
        "primary_updates": config["steps"] * len(runs),
        "comparisons": comparisons,
        "interpretation": [
            "Negative cross-entropy differences favor the named arm; no significance claim.",
            comparison_notes[0]
            if comparison_notes
            else "No seed interpretation supplied.",
            "Seeds share evaluation blocks; blocks are not independent experimental replicas.",
            *comparison_notes[1:],
            ("This summary also checks hash-bound final same-math checkpoint states, not process termination."
             if checkpoint_dir is not None else
             "This summary checks published receipts, not checkpoint contents or process termination."),
            "No speed or pristine-corpus generalization claim.",
        ],
    }
    if config.get("schema") in {
        "spiraltorch.elliptic_causal_protocol.v1", "spiraltorch.elliptic_gated_protocol.v1"
    }:
        summary["paired_factorial_contrasts"] = causal_factorial_contrasts(
            config, runs, measured, sets
        )
    if config.get("schema") == "spiraltorch.elliptic_gated_protocol.v1":
        summary["gate_trajectories"] = gated_trajectories(config, runs)
    if config.get("schema") == "spiraltorch.elliptic_anchored_protocol.v1":
        summary["reference_arm"] = reference
        summary["paired_factorial_contrasts"] = anchored_factorial_contrasts(
            config, runs, measured, sets
        )
        summary["gate_trajectories"] = gated_trajectories(config, runs, arms)
    if config.get("schema") in {"spiraltorch.fractional_memory_protocol.v1",
                                "spiraltorch.fractional_history_protocol.v1"}:
        summary["reference_arm"] = reference
        summary["paired_fractional_contrasts"], summary["order_trajectories"] = fractional_report(
            config, runs, measured, sets
        )
        if config["schema"] == "spiraltorch.fractional_history_protocol.v1":
            summary["decay_trajectories"] = ema_trajectories(config, runs)
    if config.get("schema") in ("spiraltorch.fractional_lag_protocol.v1",
                               "spiraltorch.fractional_two_lag_protocol.v1"):
        summary["reference_arm"] = reference
        (summary["paired_fractional_contrasts"], summary["order_trajectories"],
         summary["same_math_receipt_parity"]) = fractional_lag_report(config, runs, measured, sets)
        if config["schema"] == "spiraltorch.fractional_two_lag_protocol.v1":
            states = two_lag_state_parity(plan, runs, checkpoint_dir)
            summary["same_math_state_parity"] = states
            receipts_equal = all(row["update_receipts_equal"] and row["development_equal"]
                                 and all(row["endpoint_block_losses_equal"].values())
                                 for row in summary["same_math_receipt_parity"].values())
            summary["same_math_parity_status"] = (
                "failed" if not receipts_equal or any(row["status"] == "failed" for row in states.values())
                else "unverified" if checkpoint_dir is None else "passed")
    if config.get("schema") == "spiraltorch.fractional_history_factorial_protocol.v1":
        summary["reference_arm"] = reference
        (summary["paired_factorial_contrasts"], summary["order_trajectories"],
         summary["initial_filter_receipt_parity"]) = history_factorial_report(config, runs, measured, sets)
        summary["initial_filter_receipt_status"] = (
            "passed" if all(row["status"] == "passed" for row in summary["initial_filter_receipt_parity"].values())
            else "failed")
    if config.get("schema") == "spiraltorch.elliptic_chart_step_protocol.v1":
        summary["reference_arm"] = reference
        summary["paired_factorial_contrasts"], summary["chart_step_trajectories"] = chart_step_report(config, runs, measured, sets)
        summary["gate_trajectories"] = gated_trajectories(config, runs, arms)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "results", "journal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Read final two-lag checkpoints for gate/named-Adam parity (requires Torch)")
    args = parser.parse_args()
    raw = {}
    for name in ("plan", "results", "journal"):
        path = getattr(args, name)
        value = path.read_bytes()
        raw[name] = gzip.decompress(value) if path.suffix == ".gz" else value
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    parsed = {name: json.loads(value) for name, value in raw.items()}
    summary = summarize(
        parsed["plan"], parsed["results"], parsed["journal"], hashes["results"],
        checkpoint_dir=args.checkpoint_dir,
    )
    summary["input_sha256"] = hashes
    # Never replace an input or a previous derived record by accident.
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

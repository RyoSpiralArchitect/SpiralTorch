#!/usr/bin/env python3
"""Summarize a completed, hash-sealed study without importing Torch or scoring again."""

import argparse
import gzip
import hashlib
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


def chart_step_report(config, runs, measured, sets):
    arms = {"adam_tangent", "adam_elliptic", "chart_tangent", "chart_elliptic"}
    require(set(config["arms"]) == arms and config.get("reference_arm") == "adam_tangent", "incomplete chart factorial design")
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
            require(receipt["damped_condition"] >= 1, "invalid damped condition")
            metric = receipt.get("metric")
            require(isinstance(metric, list) and len(metric) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in metric), "invalid chart metric")
            require(metric[0] >= 0 and metric[3] >= 0 and metric[0] + metric[3] > 0 and metric[1] == metric[2], "invalid chart metric")
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


def summarize(plan, result, journal, result_sha256):
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
            "This summary checks published receipts, not checkpoint contents or process termination.",
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
    if config.get("schema") == "spiraltorch.elliptic_chart_step_protocol.v1":
        summary["reference_arm"] = reference
        summary["paired_factorial_contrasts"], summary["chart_step_trajectories"] = chart_step_report(config, runs, measured, sets)
        summary["gate_trajectories"] = gated_trajectories(config, runs, arms)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "results", "journal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    raw = {}
    for name in ("plan", "results", "journal"):
        path = getattr(args, name)
        value = path.read_bytes()
        raw[name] = gzip.decompress(value) if path.suffix == ".gz" else value
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    parsed = {name: json.loads(value) for name, value in raw.items()}
    summary = summarize(
        parsed["plan"], parsed["results"], parsed["journal"], hashes["results"]
    )
    summary["input_sha256"] = hashes
    # Never replace an input or a previous derived record by accident.
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

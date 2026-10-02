#!/usr/bin/env python3
"""Summarize a completed, hash-sealed study without importing Torch or scoring again."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def causal_factorial_contrasts(config, runs, measured, sets):
    arms = {"tangent", "elliptic", "causal_tangent", "causal_elliptic"}
    require(set(config["arms"]) == arms, "incomplete causal factorial design")
    for seed in config["seeds"]:
        rows = [runs[f"{seed}:{arm}"] for arm in arms]
        hashes = {row.get("initial_parameter_sha256") for row in rows}
        require(
            len(hashes) == 1
            and all(isinstance(value, str) and len(value) == 64 for value in hashes),
            "factorial initial parameters are not paired",
        )
        require(
            all(row["parameter_count"] == 11 * config["features"] + 2 for row in rows),
            "factorial parameter counts differ",
        )
    contrasts = {
        "geometry_pointwise": {"elliptic": 1, "tangent": -1},
        "geometry_causal": {"causal_elliptic": 1, "causal_tangent": -1},
        "mixing_tangent": {"causal_tangent": 1, "tangent": -1},
        "mixing_elliptic": {"causal_elliptic": 1, "elliptic": -1},
        "interaction": {
            "causal_elliptic": 1,
            "causal_tangent": -1,
            "elliptic": -1,
            "tangent": 1,
        },
    }
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
    require(
        len(set(seeds)) == len(seeds) > 0
        and len(set(arms)) == len(arms)
        and "tangent" in arms,
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
                value - measured[f"{seed}:tangent"][name]
                for seed, value in zip(seeds, values)
            ]
            rows[arm] = {
                "mean_ce": statistics.fmean(values),
                "mean_delta_vs_baseline": statistics.fmean(values) - baseline[name],
                "mean_delta_vs_tangent": statistics.fmean(delta),
                "paired_delta_sample_sd": statistics.stdev(delta)
                if len(delta) > 1
                else None,
                "seeds_better_than_tangent": sum(value < 0 for value in delta),
                "per_seed": [
                    {"seed": seed, "ce": value, "delta_vs_tangent": difference}
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
    if config.get("schema") == "spiraltorch.elliptic_causal_protocol.v1":
        summary["paired_factorial_contrasts"] = causal_factorial_contrasts(
            config, runs, measured, sets
        )
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

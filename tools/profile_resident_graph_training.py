#!/usr/bin/env python3
"""Source-bound diagnostic GPU timestamps, deliberately separate from A/B timing."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_resident_training_vs_torch import Native, audit, recipes


def validate(row):
    if (row.get("status") != "passed" or len(row.get("profiles", [])) != 12
            or len(row.get("controls", [])) != 12
            or not row.get("guards") or any(v is not True for v in row["guards"].values())):
        raise ValueError("incomplete profiling fixture")
    for index, profile in enumerate(row["profiles"]):
        if (profile.get("schema") != "spiraltorch.graph_training_gpu_profile.v1"
                or profile.get("accepted") is not True or profile.get("instrumented") is not True
                or profile.get("warmup") != (index < 3)
                or profile.get("submitted_step") != str(index+1)
                or profile.get("batch_generation") != "1"):
            raise ValueError("profile acceptance/counter mismatch")
        period = profile["timestamp_period_ns"]
        if not math.isfinite(period) or period <= 0 or not profile["passes"]:
            raise ValueError("invalid timestamp period/passes")
        totals = {}
        zeros = 0
        for item in profile["passes"]:
            for key in ("start_tick", "end_tick"):
                if not isinstance(item[key], str) or not item[key].isdigit() or not 0 <= int(item[key]) < 2**64:
                    raise ValueError("ticks must be u64 decimal strings")
            delta = int(item["end_tick"]) - int(item["start_tick"])
            if delta < 0 or not math.isfinite(item["elapsed_ns"]) or item["elapsed_ns"] != delta*period:
                raise ValueError("invalid timestamp interval")
            zeros += delta == 0
            totals[item["phase"]] = totals.get(item["phase"], 0.) + item["elapsed_ns"]
        if profile["zero_intervals"] != zeros or totals != profile["phase_totals_ns"]:
            raise ValueError("invalid phase totals")
        span = int(profile["passes"][-1]["end_tick"])-int(profile["passes"][0]["start_tick"])
        if profile["gpu_span_ns"] != (span*period if span >= 0 else None):
            raise ValueError("invalid GPU span")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="error", cases=[], boundary=__doc__, gpu_contention="UNKNOWN")
    worker = None
    with args.output.open("x") as out, Path(str(args.output)+".stderr").open("xb") as stderr:
        try:
            before = audit.source_identity()
            if before["tracked_dirty"] or audit.git_bytes("ls-files", "--others", "--exclude-standard"):
                raise ValueError("freeze the harness before profiling")
            worker = Native(args.binary, args.source, stderr)
            result.update(harness_source=before, binary=dict(file=worker.file, identity=worker.identity, binding=worker.binding))
            for config in recipes(True, "standard") + recipes(True, "wide"):
                fixture = worker.request(dict(op="init", config=config))
                policies = ("exact", "module_compatible") if config["shape"] == [2, 16, 32] else ("exact",)
                for policy in policies:
                    row = worker.request(dict(op="profile", policy=policy))
                    result["cases"].append(row)
                    validate(row)
                    if row["config"] != config or row["policy"] != policy or row["adapter"] != fixture["adapter"]:
                        raise ValueError("profile recipe/adapter drift")
                    print(json.dumps(dict(config=config, policy=policy, max_abs_error=row["max_abs_error"])), flush=True)
            if audit.source_identity() != before:
                raise ValueError("profiling source changed")
            result["status"] = "passed"
        except BaseException as error:
            result["error"] = repr(error)
            raise
        finally:
            try:
                if worker:
                    worker.close()
            except BaseException as error:
                result.update(status="error", worker_error=repr(error))
                raise
            finally:
                json.dump(result, out, separators=(",", ":"))
                out.write("\n")


if __name__ == "__main__":
    main()

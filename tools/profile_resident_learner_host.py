#!/usr/bin/env python3
"""Opt-in host/enqueue diagnosis, never GPU phase timing or throughput evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_resident_training_vs_torch as bench

PHASES = ("forward_enqueue", "quadratic_seed_enqueue", "quadratic_vjp_enqueue",
          "quartic_seed_enqueue", "quartic_vjp_enqueue", "update_enqueue",
          "receipt_capture_enqueue", "receipt_wait")


def validate(value, cadence, config):
    plain = dict(value)
    profile = plain.pop("host_profile")
    bench.validate_sample(plain, cadence, config["steps"], learner=True,
                          learner_optimizer=config.get("learner_optimizer"))
    if (profile.get("schema") != "spiraltorch.learner_host_phases.v1"
            or profile.get("instrumented") is not True
            or profile.get("clock_domain") != "host_wall"
            or profile.get("gpu_phase_attribution") is not False):
        raise ValueError("host profile clock/measurement boundary differs")
    rows = profile["phases"]
    if tuple(row["name"] for row in rows) != PHASES:
        raise ValueError("host phase list differs")
    durations = [row["ms"] for row in rows] + [profile["unattributed_ms"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError("invalid host duration")
    if any(type(row["count"]) is not int or row["count"] != config["steps"] for row in rows):
        raise ValueError("missing host phase measurement")
    if not math.isclose(sum(durations), value["elapsed_ms"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("host phases do not partition the instrumented interval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = bench.audit.source_identity()
    if before["tracked_dirty"] or bench.audit.git_bytes("ls-files", "--others", "--exclude-standard"):
        raise ValueError("clean harness source required")
    report = dict(status="running", source=bench.source_for(args.source), harness_source=before,
                  boundary="Host wall/enqueue phases on the ordinary device; no timestamp queries or extra GPU synchronization. Instrumented intervals are not ordinary throughput evidence. Uninstrumented reset controls check state identity. Host exclusivity UNKNOWN; no GPU phase attribution.",
                  warmup_blocks=2, retained_blocks=8, cases=[])
    worker = None
    with args.output.open("x") as output, Path(str(args.output)+".stderr").open("x") as stderr:
        try:
            worker = bench.Native(args.binary, args.source, stderr)
            report.update(binary=worker.file, identity=worker.identity)
            for mode in (None, "topos_ema", "clipped_topos_ema"):
                for config in bench.recipes(True, "standard", mode):
                    fixture = worker.request(dict(op="init", config=config))
                    if fixture["adapter"]["device_type"] == "Cpu":
                        raise ValueError("GPU required")
                    case = dict(config=config, fixture=fixture, samples=[], medians={})
                    report["cases"].append(case)
                    expected = None
                    for cadence in ("immediate", "deferred"):
                        for block in range(10):
                            order = ("learn", "learn_host_profile") if block % 2 == 0 else ("learn_host_profile", "learn")
                            for op in order:
                                result = worker.request(dict(op=op, cadence=cadence, capture=block == 0))
                                if op == "learn_host_profile":
                                    validate(result, cadence, config)
                                else:
                                    bench.validate_sample(result, cadence, config["steps"], learner=True, learner_optimizer=mode)
                                if expected is None:
                                    expected = result["state_sha256"]
                                if result["state_sha256"] != expected:
                                    raise ValueError("instrumentation or reset changed numerical state")
                                case["samples"].append(dict(op=op, block=block, warmup=block < 2, result=result))
                        selected = [s["result"] for s in case["samples"] if not s["warmup"]
                                    and s["op"] == "learn_host_profile" and s["result"]["cadence"] == cadence]
                        case["medians"][cadence] = {
                            name: statistics.median(v["host_profile"]["phases"][i]["ms"] for v in selected)
                            for i, name in enumerate(PHASES)}
                    print(json.dumps(dict(config=config, medians=case["medians"])), flush=True)
            worker.close()
            worker = None
            if bench.audit.source_identity() != before:
                raise ValueError("harness source changed")
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="error", error=repr(error))
            raise
        finally:
            try:
                if worker is not None:
                    worker.close()
            except BaseException as error:
                report.update(status="error", shutdown_error=repr(error))
                raise
            finally:
                output.write(json.dumps(report)+"\n")


if __name__ == "__main__":
    main()

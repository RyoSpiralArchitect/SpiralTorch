#!/usr/bin/env python3
"""Check retained convolution diagnostics and export path/weight-free results."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check_steps(steps, batch, ids, order, profiled):
    geometry = None
    compact = []
    for index, step in enumerate(steps):
        require(step["step"] == index + 1, "step coverage")
        labels = [str(ids[i]) for i in order[index * batch:(index + 1) * batch]]
        require(len(labels) == batch and step["labels"] == labels, "input order")
        host = step["host_step_ns"]
        require(type(host) is int and host > 0, "host duration")
        times = []
        if profiled:
            gpu = step["gpu"]
            require(gpu["schema"] == "spiraltorch.convolution_vjp_gpu_profile.v1", "GPU schema")
            operations = gpu["operations"]
            require([op["kind"] for op in operations] == ["depthwise", "dense", "depthwise", "dense"], "VJP coverage")
            current_geometry = [{key: value for key, value in op.items() if key != "passes"} for op in operations]
            if geometry is None:
                geometry = current_geometry
            require(current_geometry == geometry, "geometry changed within run")
            previous_end = None
            for op in operations:
                period = op["timestamp_period_ns"]
                require(math.isfinite(period) and period > 0, "timestamp period")
                require(op["input"][0] == batch and op["upstream"][0] == batch, "GPU batch")
                require([p["phase"] for p in op["passes"]] == ["input_vjp", "weight_vjp", "bias_vjp"], "pass coverage")
                for sample in op["passes"]:
                    start, end = int(sample["start_tick"]), int(sample["end_tick"])
                    elapsed = sample["elapsed_ns"]
                    require(0 <= start <= end < 2**64, "timestamp interval")
                    require(previous_end is None or start >= previous_end, "timestamp order")
                    require(math.isfinite(elapsed) and elapsed >= 0, "GPU duration")
                    require(math.isclose(elapsed, (end - start) * period, rel_tol=1e-12, abs_tol=1e-6), "timestamp decode")
                    times.append(elapsed)
                    previous_end = end
        else:
            require(step["gpu"] is None, "control unexpectedly profiled")
        compact.append(dict(step=index + 1, host_step_ns=host, convolution_pass_ns=times))
    require(bool(compact), "empty route")
    return geometry, compact


def verify(root, timing, grid, source_ref):
    summary = read(root / "summary.json")
    require(summary["schema"] == "spiraltorch.vision.gpu_profile_sweep.v1" and summary["passed"] is True, "incomplete sweep")
    require(summary["requested"] == grid, "requested grid")
    source = subprocess.check_output(["git", "show", f"{source_ref}:tools/profile_vision_trainer_gpu.py"])
    require(sha(source) == summary["launcher_sha256"], "launcher source identity")
    expected = list(itertools.product(grid["seeds"], grid["batches"], grid["modes"], range(grid["repeats"])))
    observed = [(r["seed"], r["batch_size"], r["mode"], r["repeat"]) for r in summary["records"]]
    require(observed == expected, "case coverage or order")
    hashes, records = {}, []

    def tracked(path, name):
        blob = path.read_bytes()
        hashes[name] = sha(blob)
        return blob

    tracked(root / "summary.json", "profile/summary.json")
    for row in summary["records"]:
        seed, batch, mode, repeat = row["seed"], row["batch_size"], row["mode"], row["repeat"]
        name = f"seed-{seed}-batch-{batch}-{mode}-{repeat}"
        require(row["case"] == name, "case identity")
        recipe_blob = tracked(root / f"{name}.json", f"profile/{name}.json")
        recipe = json.loads(recipe_blob)
        result = json.loads(tracked(root / name / "result.json", f"profile/{name}/result.json"))
        require(result == row["result"], "summary differs from worker result")
        require(result["schema"] == "spiraltorch.vision.trainer_gpu_profile.v1" and result["passed"] is True, "worker completion")
        require(result["case_sha256"] == sha(recipe_blob), "recipe fixity")
        require(result["batch_size"] == batch and recipe["seed"] == seed, "recipe case")
        require(recipe["profile_first"] is bool(repeat % 2), "route ordering")
        retained_name = f"seed-{seed}-batch-{batch}-{mode}-0-candidate"
        retained = timing / retained_name
        reference = json.loads(tracked(retained / "result.json", f"timing/{retained_name}/result.json"))
        require(reference["status"] == "passed", "unverified timing reference")
        initial_blob = tracked(retained / "initial.json", f"timing/{retained_name}/initial.json")
        final_blob = tracked(retained / "spiraltorch-final.json", f"timing/{retained_name}/spiraltorch-final.json")
        initial = json.loads(initial_blob)
        require(sha(initial_blob) == reference["contract"]["initial_checkpoint_sha256"] == recipe["initial_sha256"] == result["initial_sha256"], "initial checkpoint")
        require(sha(final_blob) == reference["final_checkpoint"]["sha256"] == recipe["expected_final_sha256"] == result["expected_final_sha256"], "reference final checkpoint")
        require(result["exact_retained_checkpoint"] is True, "checkpoint result")
        require(result["dataset_id"] == recipe["dataset_id"] == reference["contract"]["dataset_sha256"], "dataset identity")
        data = reference["contract"]["data"]
        pixels = tracked(root / "pixels.u8", "profile/pixels.u8")
        require(sha(pixels) == recipe["dataset"]["pixels_sha256"] == data["train_pixels_sha256"], "pixel identity")
        require(recipe["dataset"]["ids"] == data["train_indices"], "sample identities")
        input_state = initial["input"]
        require(input_state["position"] == 0 and input_state["batch_size"] == batch, "initial input position")
        require(recipe["steps"] == reference["recipe"]["steps"] == 16, "step horizon")
        require(recipe["warmup"] == reference["recipe"]["warmup"] == 3, "warmup horizon")
        require([r["profiled"] for r in result["records"]] == [recipe["profile_first"], not recipe["profile_first"]], "two routes required")
        compact_routes = []
        geometry = None
        for route in result["records"]:
            profiled = route["profiled"]
            filename = "profiled-final.json" if profiled else "control-final.json"
            checkpoint = tracked(root / name / filename, f"profile/{name}/{filename}")
            require(checkpoint == final_blob and route["final_sha256"] == sha(checkpoint), "full checkpoint parity")
            require(len(route["steps"]) == recipe["steps"], "route horizon")
            current_geometry, steps = check_steps(route["steps"], batch, recipe["dataset"]["ids"], input_state["order"], profiled)
            if profiled:
                geometry = current_geometry
            compact_routes.append(dict(profiled=profiled, steps=steps))
        records.append(dict(case=name, seed=seed, batch_size=batch, mode=mode, repeat=repeat,
                            adapter=result["adapter"], initial_sha256=sha(initial_blob),
                            final_sha256=sha(final_blob), operations=geometry, routes=compact_routes))
    table = []
    for batch, mode in itertools.product(grid["batches"], grid["modes"]):
        group = [r for r in records if r["batch_size"] == batch and r["mode"] == mode]
        host, conv, control = [], [], []
        for row in group:
            for route in row["routes"]:
                mean_host = statistics.mean(s["host_step_ns"] for s in route["steps"]) / 1e6
                if route["profiled"]:
                    host.append(mean_host)
                    conv.append(statistics.mean(sum(s["convolution_pass_ns"]) for s in route["steps"]) / 1e6)
                else:
                    control.append(mean_host)
        table.append(dict(batch_size=batch, mode=mode, cases=len(group),
                          median_profiled_host_step_ms=statistics.median(host),
                          median_control_host_step_ms=statistics.median(control),
                          median_convolution_vjp_ms=statistics.median(conv),
                          min_convolution_vjp_ms=min(conv), max_convolution_vjp_ms=max(conv)))
    verification = dict(schema="spiraltorch.vision.gpu_profile_verification.v1", passed=True,
                        boundary="Retained-file replay, not hardware attestation or a full-step timing decomposition",
                        source_ref=source_ref, binary_sha256=summary["binary_sha256"],
                        launcher_sha256=summary["launcher_sha256"], requested=grid,
                        workers=len(records), exact_checkpoint_comparisons=2 * len(records),
                        profiled_steps=16 * len(records), convolution_passes=16 * 12 * len(records),
                        retained_file_sha256=hashes)
    return verification, records, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid = dict(seeds=[17, 29, 43], batches=[1, 16, 64], modes=["plain", "feedback"], repeats=3)
    verification, records, table = verify(args.root, args.timing_root, grid, args.source_ref)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [("verification.json", verification), ("measurements.json", records), ("table.json", table)]:
        with (args.output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "workers": len(records), "table": table}))


if __name__ == "__main__":
    main()

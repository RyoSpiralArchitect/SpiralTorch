#!/usr/bin/env python3
"""Offline verification of retained timing cases, checkpoints and Torch weights.

Does not rerun training, attest a device or reproduce wall-clock durations.
"""
import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vision_timing", HERE / "bench_vision_trainer_vs_torch.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
require = timing.require


def checkpoint(directory, receipt):
    name = receipt["file"]
    require(isinstance(name, str) and Path(name).name == name, "invalid receipt path")
    data = (directory / name).read_bytes()
    require(timing.sha(data) == receipt["sha256"] and len(data) == receipt["bytes"], "file fixity differs")
    return json.loads(data)


def compare_parameters(actual, reference):
    require(actual and len(actual) == len(reference)
            and len({p["name"] for p in actual}) == len(actual), "parameter coverage differs")
    checks = []
    for left, right in zip(actual, reference, strict=True):
        require(left["name"] == right["name"] and left["shape"] == right["shape"], "parameter role/shape differs")
        a, b = left["values"], right["values"]
        require(len(a) == len(b) == math.prod(left["shape"]) > 0, "parameter length differs")
        require(all(math.isfinite(v) for v in [*a, *b]), "non-finite retained parameter")
        absolute = max(abs(x - y) for x, y in zip(a, b, strict=True))
        scaled = max(abs(x - y) / (1 + abs(y)) for x, y in zip(a, b, strict=True))
        require(scaled < 2e-4, "retained parameter numerical bound")
        checks.append(dict(name=left["name"], max_abs_error=absolute, max_scaled_error=scaled, values=len(a)))
    return checks


def verify_coverage(saved):
    require(saved["status"] == "passed" and saved["schema"] == "spiraltorch.vision.training_timing_sweep.v1",
            "wrong/incomplete sweep")
    request = saved["requested"]
    require(type(request["repeats"]) is int and request["repeats"] > 0, "invalid repeat count")
    for key in ("seeds", "batches", "modes", "runtimes"):
        require(request[key] and len(set(request[key])) == len(request[key]), "duplicate/empty requested cases")
    expected = set(itertools.product(request["seeds"], request["batches"], request["modes"],
                                     range(request["repeats"]), request["runtimes"]))
    observed = []
    for record in saved["records"]:
        c = record["recipe"]
        observed.append((c["seed"], c["batch_size"], c["mode"], c["repeat"], c["runtime"]))
        require(all(c[key] == request[key] for key in ("steps", "warmup", "profile")), "requested boundary differs")
    require(len(observed) == len(expected) and set(observed) == expected, "missing/duplicate/extra cases")


def verify(directory, source_ref):
    saved = json.loads((directory / "summary.json").read_text())
    verify_coverage(saved)
    checks = timing.summarize(saved["records"])
    require(all(saved[key] == value for key, value in checks.items()), "summary differs from intervals")
    sources = saved["records"][0]["contract"]["source_sha256"]
    for name, digest in sources.items():
        require(Path(name).name == name, "source basename required")
        captured = subprocess.check_output(["git", "show", f"{source_ref}:tools/{name}"], cwd=HERE.parent)
        require(timing.sha(captured) == digest, "captured harness source differs")
    cases = []
    for row in saved["records"]:
        c = row["recipe"]
        name = f"seed-{c['seed']}-batch-{c['batch_size']}-{c['mode']}-{c['repeat']}-{c['runtime']}"
        root = directory / name
        require(json.loads((root / "result.json").read_text()) == row, "worker record differs")
        recipe = json.loads((root / "recipe.json").read_text())
        require({k: v for k, v in recipe.items() if k != "data_root"} == c, "worker recipe differs")
        require(row["contract"]["source_sha256"] == sources, "harness changed during measurement")
        initial_bytes = (root / "initial.json").read_bytes()
        require(timing.sha(initial_bytes) == row["contract"]["initial_checkpoint_sha256"], "initial checkpoint fixity")
        initial = json.loads(initial_bytes)
        final = checkpoint(root, row["final_checkpoint"])
        reference = checkpoint(root, row["torch_final"])
        require(final["input"]["order"] == initial["input"]["order"]
                and final["input"]["position"] == c["steps"] * c["batch_size"]
                and final["trainer"]["accepted_updates"] == c["steps"]
                and final["trainer"]["rejected_updates"] == 0, "retained input/update clock differs")
        actual = final["model"]["backbone"]["parameters"] + final["model"]["head"]
        recomputed = compare_parameters(actual, reference)
        require(row["final_comparison"] == recomputed, "reported parameter parity differs")
        for measured in row["measurements"].values():
            require(math.isfinite(measured["last_loss"]), "invalid retained loss")
        a, b = (row["measurements"][n]["last_loss"] for n in ("spiraltorch", "torch"))
        require(abs(a - b) / (1 + abs(b)) < 2e-4, "retained loss parity")
        cases.append(dict(case=name, values=sum(p["values"] for p in recomputed),
                          max_scaled_error=max(p["max_scaled_error"] for p in recomputed)))
    return dict(schema="spiraltorch.vision.training_timing_verification.v1", status="passed",
                scope="Retained coverage, fixity, weight parity and exact A/B checkpoints; not new timing or device attestation",
                source_ref=source_ref, summary_sha256=timing.sha((directory / "summary.json").read_bytes()),
                cases=cases, exact_pairs=len(checks["pairs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory, args.source_ref)
    timing.write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]), exact_pairs=result["exact_pairs"])))


if __name__ == "__main__":
    main()

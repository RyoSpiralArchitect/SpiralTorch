"""Byte integrity, source links, complete grids and aggregation; not numerical replay."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, raw=False, source_root=None):
    manifest = json.loads((root / "manifest.json").read_text())
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert files == set(manifest["files"])
    for name, digest in manifest["files"].items():
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts and not (root / path).is_symlink()
        assert sha(root / path) == digest, name
    results = json.loads((root / "results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    source = json.loads((root / "source.json").read_text())
    assert results["commit"] == summary["measured_commit"] == source["commit"]
    assert source["status"] == ""
    assert len(results["results"]) == 18
    grid = list(itertools.product([1, 32, 256], [1, 8, 64], [False, True]))
    expected_orders = list(itertools.permutations(["native", "wasm", "torch"]))
    assert len(summary["conditions"]) == 18
    assert summary["records"] == 324 and summary["preconditioning_records"] == 54
    assert summary["intervals_per_record"] == 9 and summary["iterations_per_interval"] == 4
    raw_manifest = json.loads((root / "local-raw-manifest.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    assert [r["worker"] for r in preflight] == ["native", "wasm", "torch"]
    preflight_cases = {}
    for record in preflight:
        assert raw_manifest[record["raw_path"]] == record["raw_sha256"]
        assert len(record["cases"]) == 18
        for key, case in zip(grid, record["cases"]):
            assert (case["batch"], case["samples"], case["varying"]) == key
            assert len(case["elapsed_ns"]) == 9
            assert all(math.isfinite(t) and t > 0 for t in case["elapsed_ns"])
            assert case["median_ns"] == statistics.median(case["elapsed_ns"])
            assert 0 <= case["max_abs_vs_native_preflight"] <= 3e-6
            preflight_cases[(record["worker"], key)] = case
        assert 0 <= record["contracts"]["one_step_parameter_max_abs_vs_native"] <= 3e-6
    for path, digest in results["workers"].items():
        assert raw_manifest[path] == digest
    seen = set()
    grouped = {}
    for record in results["results"]:
        block, worker = record["block"], record["worker"]
        assert (block, worker) not in seen
        seen.add((block, worker))
        assert tuple(record["order"]) == expected_orders[block]
        assert record["order"][record["position"]] == worker
        assert raw_manifest[record["raw_path"]] == record["raw_sha256"]
        receipt = json.loads((root / "measurement-receipts" / f"block-{block}-{worker}" / "receipt.json").read_text())
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
        assert receipt["source"] == source
        assert len(record["cases"]) == 18
        for key, case in zip(grid, record["cases"]):
            assert (case["batch"], case["samples"], case["varying"]) == key
            timings = case["elapsed_ns"]
            assert len(timings) == 9 and all(math.isfinite(t) and t > 0 for t in timings)
            assert case["median_ns"] == statistics.median(timings)
            assert 0 <= case["max_abs_vs_native_preflight"] <= 3e-6
            reference = preflight_cases[(worker, key)]
            assert case["input_metadata_sha256"] == reference["input_metadata_sha256"]
            assert case["output_f32le_sha256"] == reference["output_f32le_sha256"]
            grouped[(block, worker, key)] = case
        assert 0 <= record["contracts"]["one_step_parameter_max_abs_vs_native"] <= 3e-6
    assert seen == set(itertools.product(range(6), ["native", "wasm", "torch"]))
    def gm(values):
        return math.exp(statistics.mean(math.log(x) for x in values))
    for key, condition in zip(grid, summary["conditions"]):
        assert (condition["batch"], condition["samples"], condition["varying"]) == key
        for worker in ["native", "wasm", "torch"]:
            rows = [grouped[(b, worker, key)] for b in range(6)]
            assert len({c["output_f32le_sha256"] for c in rows}) == 1
            assert condition["median_ns"][worker] == statistics.median(c["median_ns"] for c in rows)
            if worker == "torch":
                continue
            ratios = [grouped[(b, "torch", key)]["median_ns"] / grouped[(b, worker, key)]["median_ns"] for b in range(6)]
            assert condition["torch_over_runtime"][worker] == {"geomean": gm(ratios), "min": min(ratios), "max": max(ratios)}
    for worker in ["native", "wasm"]:
        ratios = [c["torch_over_runtime"][worker]["geomean"] for c in summary["conditions"]]
        assert summary["aggregate"][worker] == {"geomean": gm(ratios), "favorable_conditions": sum(x > 1 for x in ratios)}
    stages = json.loads((root / "validation/stages.json").read_text())
    assert [s["stage"] for s in stages] == ["format", "fixture-format", "clippy", "vision-tests",
        "vision-no-default", "vision-wasm", "fixture-native", "fixture-wasm", "fixture-bindgen",
        "native-preflight", "wasm-preflight", "torch-preflight"]
    assert len(stages) == 12 and all(s["exit_code"] == 0 for s in stages)
    for stage in stages:
        receipt = json.loads((root / "validation" / stage["stage"] / "receipt.json").read_text())
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
        assert receipt["source"] == source
    assert summary["max_render_abs_vs_native"] == max(c["max_abs_vs_native_preflight"] for r in results["results"] for c in r["cases"])
    assert summary["max_one_step_parameter_abs_vs_native"] == max(r["contracts"]["one_step_parameter_max_abs_vs_native"] for r in results["results"])
    if raw:
        for name, digest in raw_manifest.items():
            assert sha(Path(name)) == digest
    if source_root:
        for name, digest in source["files"].items():
            assert sha(source_root / name) == digest, name
    return len(files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="Also hash-check local payload/worker paths; not numerical replay")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    count = verify(Path(__file__).parent, args.raw, args.source_root)
    print(f"Verified {count} archive files, 324 measured records and 12 validation stages; numerical replay not performed")

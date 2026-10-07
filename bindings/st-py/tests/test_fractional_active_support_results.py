"""Recompute published timing summaries; not an independent native execution."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "benchmarks/results/2026-10-07-fractional-active-support"


def read(name):
    data = (DIRECTORY / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def test_closed_public_inventory_and_validation_counts():
    inventory = {}
    for line in (DIRECTORY / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in inventory
        path = DIRECTORY / name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        inventory[name] = expected
    assert set(inventory) == {p.name for p in DIRECTORY.iterdir() if p.name != "SHA256SUMS"}
    assert not any(p.suffix in {".pt", ".so", ".wasm", ".safetensors", ".log"} for p in DIRECTORY.iterdir())
    validation = read("validation.json")
    assert validation["status"] == "passed"
    assert validation["native_benchmark_processes"] == 96
    assert all(code == 0 for code in validation["final_exit_codes"].values())
    assert validation["rust_tests"] == 151
    assert validation["geometry_python_tests"] == 848
    assert validation["benchmark_python_tests"] == 84
    assert validation["old_runtime_files_verified"] == 71


def test_all_paired_native_receipts_rebuild_including_initial_regressions():
    summary = read("summary.json")
    identities = read("build-identities.json")
    for variant in ("initial", "sliced", "tiled"):
        archive = read(f"native-{variant}.json.gz")
        assert len(archive["runs"]) == 32
        groups = {}
        for row in archive["runs"]:
            key = f'{row["alpha"]:g}/{row["window"][0]}:{row["window"][1]}/' + ("joint" if row["input_requires_grad"] else "parameters")
            groups.setdefault(key, []).append(row)
            assert row["status"] == "measured" and row["native_profile_declared"] == "release"
            assert row["shape"] == [2, 128, 768] and row["kernel_len"] == 32
            assert row["normalization"] == "full_declared_kernel_before_window"
            assert row["correctness_tolerance"] == {"rtol": 3e-5, "atol": 3e-5}
        assert len(groups) == 8
        for result in summary["native"][variant]:
            rows = groups[result["key"]]
            assert len(rows) == 4
            for row in rows:
                route = "before" if row["file"].endswith("before.json") else variant
                assert row["native_sha256"] == identities[route]["native_sha256"]
                assert row["bridge_sha256"] == identities[route]["bridge_sha256"]
                for field in ("input_sha256", "upstream_sha256", "benchmark_sha256", "shape", "window",
                              "kernel_len", "alpha", "log_gain", "seed", "threads", "input_requires_grad", "torch"):
                    assert row[field] == rows[0][field]
                for name in ("rust_buffer", "torch_window_conv1d"):
                    assert row["correctness"][name]["sha256"] == rows[0]["correctness"][name]["sha256"]
                    times = row["measurements_ms"][name]
                    assert len(times) == 12 and all(math.isfinite(v) and v > 0 for v in times)
            def median(route, implementation):
                return statistics.median(value for row in rows if row["file"].endswith(route + ".json")
                                         for value in row["measurements_ms"][implementation])
            before, after = median("before", "rust_buffer"), median("after", "rust_buffer")
            assert result["before_ms"] == before and result["after_ms"] == after
            assert result["speedup"] == before / after
            assert result["torch_before_ms"] == median("before", "torch_window_conv1d")
            assert result["torch_after_ms"] == median("after", "torch_window_conv1d")
    # The unsuccessful earlier full-history input-VJP timings remain visible.
    for variant in ("initial", "sliced"):
        full = next(r for r in summary["native"][variant] if r["key"] == "0.09/1:32/joint")
        assert full["speedup"] < 1


def test_wasm_contract_receipts_and_learning_are_separate_from_native_timing():
    for variant in ("initial", "sliced", "hoisted", "tiled"):
        report = read(f"wasm-{variant}.json")
        assert report["status"] == "passed" and len(report["correctness"]) == 60
        assert len(report["cases"]) == 4
        for row in report["correctness"]:
            assert set(row["sha256"]) == {"output", "parameters", "input", "joint_parameters", "selective_input", "jvp"}
            assert all(len(h) == 64 for h in row["sha256"].values())
        for row in report["cases"]:
            assert row["median_ms"] == [statistics.median(v) for v in row["measurements_ms"]]
    learning = read("wasm-learning.json")
    for row in learning["runs"]:
        assert row["updates"] == 100
        assert 0 <= row["final_loss"] < row["initial_loss"]
        assert row["nonzero_angle_steps"] > 0 and row["nonzero_gain_steps"] > 0

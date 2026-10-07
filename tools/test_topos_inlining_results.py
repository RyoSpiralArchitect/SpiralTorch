"""Validate frozen inlining receipts without rerunning learning or benchmarks."""

import gzip
import hashlib
import json
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-saturation-inlining"


def read(name):
    data = (DATA / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def test_closed_archive_and_no_private_data():
    hashes = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(hashes) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    for name, expected in hashes.items():
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        assert not any(m in raw for m in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))


def test_all_runs_and_predeclared_paired_order_remain_visible():
    measurements = read("measurements.json.gz")
    runs, plan = measurements["runs"], measurements["pair_plan"]
    assert len(runs) == 45 and len(plan) == 32
    assert len({p["file"] for p in plan}) == 32
    assert set(runs) == {p["file"] for p in plan} | {"screen-after-4-large.json"} | {
        f"before-{i}-{s}-{r}.json" for i in (4, 16) for s in ("small", "large") for r in (1, 2, 3)
    }
    verification = read("verification.json.gz")
    for file, run in runs.items():
        assert run["status"] == "measured" and run["all_inputs_require_grad"]
        assert run["benchmark_sha256"] == verification["client_sha256"]["benchmark_topos_learning.py"]
        candidate = file == "screen-after-4-large.json" or file.endswith("-after.json")
        manifest = verification["runtime_after" if candidate else "runtime_before"]
        assert run["native_sha256"] == manifest["files"]["spiraltorch/spiraltorch.abi3.so"]
        assert run["bridge_sha256"] == manifest["files"]["spiraltorch/geometry_autograd.py"]
        for name, samples in run["measurements_ms"].items():
            assert len(samples) == 12 and min(samples) > 0
            assert run["median_ms"][name] == statistics.median(samples)
            for pos in range(4):
                assert sum(order[pos] == name for order in run["round_order"]) == 3
    summaries = read("summary.json")["rows"]
    assert len(summaries) == 4
    assert {(s["iterations"], s["size"]) for s in summaries} == {
        (i, s) for i in (4, 16) for s in ("small", "large")
    }
    for summary in summaries:
        iterations, size = summary["iterations"], summary["size"]
        selected = [p for p in plan if p["iterations"] == iterations and p["size"] == size]
        phases = [p["phase"] for p in selected]
        assert phases == ["before", "after", "after", "before", "after", "before", "before", "after"]
        before, after = [], []
        for pair in (1, 2, 3, 4):
            a, b = (runs[f"paired-{iterations}-{size}-{pair}-{phase}.json"] for phase in ("before", "after"))
            for key in ("shape", "config", "seed", "threads", "torch", "machine", "input_sha256",
                        "upstream_sha256", "benchmark_sha256", "bridge_sha256", "transport_sha256"):
                assert a[key] == b[key]
            for run in (a, b):
                for route in ("rust_list", "rust_public", "rust_buffer_recomputed"):
                    assert len(run["correctness"][route]) == 3
                    for name, value in run["correctness"][route].items():
                        assert value["sha256"] == a["correctness"]["rust_list"][name]["sha256"]
            before.append(a["median_ms"]["rust_public"])
            after.append(b["median_ms"]["rust_public"])
        assert summary["before_ms"] == statistics.median(before)
        assert summary["after_ms"] == statistics.median(after)
        assert summary["paired_ratios"] == [a / b for a, b in zip(before, after)]
        for prefix, route in (("bulk", "rust_buffer_recomputed"), ("torch", "torch_reference")):
            for phase in ("before", "after"):
                medians = [runs[p["file"]]["median_ms"][route] for p in selected if p["phase"] == phase]
                assert summary[f"{prefix}_{phase}_ms"] == statistics.median(medians)


def test_native_only_change_and_exact_wasm_checkpoint_replay():
    v = read("verification.json.gz")
    before, after = v["runtime_before"], v["runtime_after"]
    assert set(before) == set(after) == {"source_revision", "files"}
    assert before["files"].keys() == after["files"].keys()
    assert {n for n in before["files"] if before["files"][n] != after["files"][n]} == {"spiraltorch/spiraltorch.abi3.so"}
    a, b = v["wasm_before"], v["wasm_after"]
    assert a["cases"] == b["cases"] and len(a["cases"]) == 24
    assert a["learning"] == b["learning"] and b["learning"]["next_update_exact"]
    assert a["capture_required"] and b["capture_required"] and a["guard_checks"] == b["guard_checks"] == 50
    replay = v["replay"]
    assert replay["runtime_manifest_sha256"] == v["runtime_after_manifest_sha256"]
    assert replay["native_sha256"] == after["files"]["spiraltorch/spiraltorch.abi3.so"]
    assert replay["status"] == "bitwise_exact" and replay["auxiliary_updates"] == 1
    assert replay["raw_gradients_adapter_adam_rng_exact"] and replay["base_unchanged"]
    assert len(replay["records"]) == 1 and len(replay["records"][0]["gradients"]) == 12
    assert not replay["heldout_scoring"] and not replay["timing_evidence"]
    assert v["validation"]["strict_wasm_clippy"]["exit_code"] == 0
    assert v["validation"]["no_cleanup_performed"]

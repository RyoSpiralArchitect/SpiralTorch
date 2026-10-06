"""Frozen numeric receipts only; never rerun a study or benchmark in CI."""

import gzip
import hashlib
import json
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parents[3] / "benchmarks/results/2026-10-07-topos-captured-vjp"


def read(name):
    raw = (DATA / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def test_closed_archive_has_no_private_paths_or_weights():
    hashes = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(hashes) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    for name, expected in hashes.items():
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        assert not name.endswith((".pt", ".so", ".wasm", ".safetensors"))
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        assert not any(m in raw for m in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))


def test_matched_inputs_all_gradients_and_route_positions():
    runs = {r["file"]: r["report"] for r in read("benchmark-runs.json.gz")}
    assert len(runs) == 24
    summary = read("summary.json")
    for iterations in (4, 16):
        for size in ("small", "large"):
            ref = runs[f"before-{iterations}-{size}-1.json"]
            new = []
            for phase in ("before", "after"):
                for repeat in (1, 2, 3):
                    run = runs[f"{phase}-{iterations}-{size}-{repeat}.json"]
                    assert run["status"] == "measured" and run["all_inputs_require_grad"]
                    for key in ("shape", "config", "seed", "threads", "torch", "machine",
                                "input_sha256", "upstream_sha256", "benchmark_sha256"):
                        assert run[key] == ref[key]
                    names = set(run["measurements_ms"])
                    assert names == ({"rust_list", "rust_public", "torch_reference"}
                                     | ({"rust_buffer_recomputed"} if phase == "after" else set()))
                    for route in names:
                        assert len(run["measurements_ms"][route]) == 12
                        assert min(run["measurements_ms"][route]) > 0
                        assert run["median_ms"][route] == statistics.median(run["measurements_ms"][route])
                        for position in range(len(names)):
                            assert sum(order[position] == route for order in run["round_order"]) == 12 // len(names)
                        if route != "torch_reference":
                            assert set(run["correctness"][route]) == {"output", "input_gradient", "gate_gradient"}
                            for name, value in run["correctness"][route].items():
                                assert value["sha256"] == ref["correctness"]["rust_list"][name]["sha256"]
                    if phase == "after":
                        new.append(run)
            row = next(r for r in summary["rows"] if r["iterations"] == iterations and r["shape"] == ref["shape"])
            assert row["new_captured_ms"] == statistics.median(r["median_ms"]["rust_public"] for r in new)
            assert row["native_old_new_all_tensor_hashes_equal"]


def test_wasm_and_pretrained_replay_are_exact_and_bounded():
    before, after = read("wasm-before.json"), read("wasm-after.json")
    assert before["status"] == after["status"] == "passed"
    assert not before["captured"] and after["captured"] and after["guard_checks"] == 50
    assert len(before["cases"]) == 24 and before["cases"] == after["cases"]
    assert before["learning"] == after["learning"] and after["learning"]["next_update_exact"]
    assert after["learning"]["updates"] == 240 and after["learning"]["legacy_trajectory_exact"]
    replay = read("replay.json")
    assert replay["schema"] == "spiraltorch.geometry_stack_native_replay.v2"
    assert replay["status"] == "bitwise_exact" and replay["auxiliary_updates"] == 1
    assert replay["raw_gradients_adapter_adam_rng_exact"] and replay["base_unchanged"]
    assert replay["topos_capture_required"] and replay["parameter_count"] == 12294
    assert len(replay["records"]) == 1 and replay["records"][0]["step"] == 3
    assert len(replay["records"][0]["gradients"]) == 12
    assert all(g["nonzero"] > 0 for g in replay["records"][0]["gradients"].values())
    assert replay["bulk_transport_shapes"] == [[2, 128, 768], [768], [768]] + [[2, 128, 768]] * 4
    assert not replay["heldout_scoring"] and not replay["timing_evidence"]


def test_runtime_and_client_hashes_bind_the_published_evidence():
    before, after = read("runtime-before-sha256.json"), read("runtime-sha256.json")
    assert set(before) == set(after) == {"source_revision", "files"}
    assert len(before["files"]) == len(after["files"]) == 73
    changed = {name for name in before["files"].keys() | after["files"].keys()
               if before["files"].get(name) != after["files"].get(name)}
    assert changed == {"spiraltorch/__init__.py", "spiraltorch/__init__.pyi",
                       "spiraltorch/geometry_autograd.py", "spiraltorch/spiraltorch.abi3.so"}
    replay, validation = read("replay.json"), read("validation.json")
    assert replay["runtime_manifest_sha256"] == hashlib.sha256((DATA / "runtime-sha256.json").read_bytes()).hexdigest()
    assert replay["native_sha256"] == after["files"]["spiraltorch/spiraltorch.abi3.so"]
    assert replay["replay_client_sha256"] == validation["client_sha256"]["replay_geometry_stack_update.py"]
    for row in read("benchmark-runs.json.gz"):
        manifest = before if row["file"].startswith("before-") else after
        run = row["report"]
        assert run["native_sha256"] == manifest["files"]["spiraltorch/spiraltorch.abi3.so"]
        assert run["bridge_sha256"] == manifest["files"]["spiraltorch/geometry_autograd.py"]
        assert run["benchmark_sha256"] == validation["client_sha256"]["benchmark_topos_learning.py"]
    assert validation["setup_failure_preserved"]["before_test_bodies"]
    assert validation["no_cleanup_performed"]

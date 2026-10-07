"""Public receipt integrity only; does not rerun native learning or timings."""

import gzip
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "benchmarks/results/2026-10-07-wave-gate-buffer-transport"


def read(name):
    raw = (DATA / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def test_closed_public_archive_and_no_private_paths():
    hashes = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(hashes) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    for file, expected in hashes.items():
        raw = (DATA / file).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        if file.endswith(".gz"):
            raw = gzip.decompress(raw)
        assert not any(marker in raw for marker in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))


def test_all_native_conditions_preserve_output_and_every_gradient():
    runs = read("benchmark-runs.json.gz")
    assert len(runs) == 24
    for mode in ("legacy", "radius"):
        for size in ("small", "large"):
            reference = runs[f"before-{mode}-{size}-1.json"]
            for phase in ("before", "after"):
                for repeat in (1, 2, 3):
                    run = runs[f"{phase}-{mode}-{size}-{repeat}.json"]
                    assert run["status"] == "measured" and run["all_inputs_require_grad"]
                    assert run["benchmark_sha256"] == reference["benchmark_sha256"]
                    assert run["input_sha256"] == reference["input_sha256"]
                    assert run["upstream_sha256"] == reference["upstream_sha256"]
                    assert run["shape"] == reference["shape"] and run["log_radius"] == reference["log_radius"]
                    for route in ("rust_list", "rust_public"):
                        assert len(run["correctness"][route]) == (4 if mode == "legacy" else 5)
                        for name, value in run["correctness"][route].items():
                            assert value["sha256"] == reference["correctness"]["rust_list"][name]["sha256"]
                    assert all(len(samples) == 12 and min(samples) > 0 for samples in run["measurements_ms"].values())


def test_wasm_learning_and_real_checkpoint_replay_are_separate():
    before, after = read("wasm-before.json"), read("wasm-after.json")
    assert before["status"] == after["status"] == "passed"
    assert len(before["cases"]) == 16 and before["cases"] == after["cases"]
    assert before["learning"] == after["learning"] and after["learning"]["next_update_equal"]
    replay = read("replay.json")
    assert replay["status"] == "bitwise_exact" and replay["auxiliary_updates"] == 1
    assert replay["parameter_count"] == 12294 and replay["base_unchanged"]
    assert replay["raw_gradients_adapter_adam_rng_exact"]
    assert len(replay["records"]) == 1 and len(replay["records"][0]["gradients"]) == 12
    assert all(row["nonzero"] > 0 for row in replay["records"][0]["gradients"].values())
    assert replay["bulk_transport_shapes"] == [[2, 128, 768], [768], [768], [2, 128, 768]]
    assert not replay["heldout_scoring"] and not replay["timing_evidence"]


def test_runtime_changes_and_incomplete_strict_lint_are_explicit():
    old, new = read("runtime-before-sha256.json"), read("runtime-sha256.json")
    changed = {name for name in old["files"].keys() | new["files"].keys()
               if old["files"].get(name) != new["files"].get(name)}
    assert changed == {"spiraltorch/spiraltorch.abi3.so", "spiraltorch/__init__.pyi",
                       "spiraltorch/geometry_autograd.py", "spiraltorch/fractional_autograd.py",
                       "spiraltorch/_torch_transport.py"}
    replay = read("replay.json")
    assert replay["native_sha256"] == new["files"]["spiraltorch/spiraltorch.abi3.so"]
    assert replay["runtime_manifest_sha256"] == hashlib.sha256((DATA / "runtime-sha256.json").read_bytes()).hexdigest()
    validation = read("validation.json")
    assert validation["strict_clippy"]["exit_code"] == 101
    assert validation["strict_clippy"]["diagnostics"] == 23
    assert validation["strict_clippy"]["diagnostic_files_unchanged_from_parent"]
    assert len(validation["setup_failures_preserved"]) == 2
    assert validation["no_cleanup_performed"]

"""Public numeric receipt checks only, not a rerun of private model training."""

import gzip
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "benchmarks/results/2026-10-07-fractional-multi-adapter-replay"


def read(name):
    return json.loads((DATA / name).read_bytes())


def reports():
    raw = json.loads(gzip.decompress((DATA / "run-reports.json.gz").read_bytes()))
    for value in raw.values():
        assert hashlib.sha256(value["report_json"].encode()).hexdigest() == value["sha256"]
    return {name: json.loads(value["report_json"]) for name, value in raw.items()}


def test_closed_inventory_hashes_and_runtime_difference():
    hashes = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(hashes) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    assert len(hashes) == 7
    for name, expected in hashes.items():
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        text = gzip.decompress(raw) if name.endswith(".gz") else raw
        assert not any(value in text for value in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))
    manifests = read("runtime-sha256.json")
    old, new = (manifests[key]["files"] for key in ("before", "after"))
    assert old.keys() == new.keys() and len(old) == 71
    assert [name for name in old if old[name] != new[name]] == ["spiraltorch/spiraltorch.abi3.so"]


def test_all_six_runs_and_native_paths_are_accounted_for():
    runs, summary = reports(), read("summary.json")
    assert runs.keys() == summary["runs"].keys()
    assert len(runs) == summary["completed_processes"] == 6
    assert sum(r["updates_executed"] for r in runs.values()) == summary["auxiliary_updates"] == 20
    assert summary["unique_auxiliary_trajectory_updates"] == 8 and summary["replay_only_updates"] == 12
    assert summary["primary_study_updates"] == 0 and summary["trainable_parameters"] == 3076
    for name, run in runs.items():
        receipt = summary["runs"][name]
        assert run["status"] == "completed" and run["base_unchanged"]
        assert not run["heldout_scoring"] and not run["timing_evidence"]
        assert run["state"] == receipt["state"] and len(run["records"]) == 4
        assert run["binding"]["config"]["blocks"] == ["transformer.h.0.mlp", "transformer.h.1.mlp"]
        assert [row["batch_indices"] for row in run["records"]] == run["binding"]["schedule"]
        for index, row in enumerate(run["records"]):
            assert row["step"] == index + 1 and len(row["gradients"]) == len(row["parameters"]) == 8
            assert row["loss"]["sha256"] == hashlib.sha256(struct.pack("<f", row["loss_value"])).hexdigest()
            assert [(e["site"], e["method"]) for e in row["native_calls"]] == [
                ("site1", "vjp_buffer"), ("site0", "vjp_parameters_buffer")]
            assert (row["native_calls"][0]["input_vjp"]["nonzero"] > 0) == (index > 0)
            if index:
                assert all(g["nonzero"] > 0 for g in row["gradients"].values())


def test_pairing_checkpoint_lineage_and_scope_are_preserved():
    runs, summary = reports(), read("summary.json")
    for window in ("full", "short"):
        old = runs[f"{window}-before" + ("-v2" if window == "full" else "")]
        fresh, resumed = runs[f"{window}-after"], runs[f"{window}-resumed"]
        comparison = read(f"{window}-comparison.json")
        assert old["binding"] == fresh["binding"] == resumed["binding"]
        assert comparison["records"] == old["records"] == fresh["records"] == resumed["records"]
        assert comparison["status"] == "bitwise_exact" and comparison["updates_executed"] == 10
        assert resumed["start_cursor"] == resumed["updates_executed"] == 2
        assert resumed["resume_source_sha256"] == old["midpoint"]["sha256"]
        assert comparison["old_native"] == old["runtime"]
        assert comparison["new_native"] == fresh["runtime"] == resumed["runtime"]
        assert comparison["old_native"]["native_sha256"] != comparison["new_native"]["native_sha256"]
    validation = read("validation.json")
    assert validation["failure_preserved"]["stage"] == "manifest admission before model load or training"
    assert validation["no_cleanup_performed"] and validation["raw_private_records_retained"]
    assert not summary["heldout_scoring"] and not summary["timing_evidence"]

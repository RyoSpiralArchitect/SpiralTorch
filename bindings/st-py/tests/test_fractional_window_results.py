"""Public numeric/hash consistency, not independent checkpoint execution."""

import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "benchmarks/results/2026-10-07-fractional-window-diagnostic"
ORIGINAL = ROOT / "benchmarks/results/2026-10-05-fractional-angle-study"
CANDIDATE = ROOT / "benchmarks/results/2026-10-05-fractional-history-window"


def read(name, directory=DIRECTORY):
    raw = (directory / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_inventory_and_separate_native_identities():
    inventory = {}
    for line in (DIRECTORY / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in inventory
        assert not (DIRECTORY / name).is_symlink() and sha(DIRECTORY / name) == expected
        inventory[name] = expected
    assert set(inventory) == {p.name for p in DIRECTORY.iterdir() if p.name != "SHA256SUMS"}
    assert not any(p.suffix in {".pt", ".safetensors", ".so", ".log"} for p in DIRECTORY.iterdir())
    result, validation, plan = read("results.json"), read("validation.json"), read("plan.json")
    assert result["status"] == "completed" and plan["status"] == "prepared"
    assert validation["status"] == "passed" and validation["diagnostic_process_exit_code"] == 0
    assert validation["results_sha256"] == sha(DIRECTORY / "results.json")
    assert result["frozen_manifest_inputs_preserved"] and result["frozen_base_unchanged"]
    assert {k: result[k] for k in plan if k != "status"} == {k: v for k, v in plan.items() if k != "status"}
    assert result["training_updates"] == result["optimizer_steps"] == 0
    assert result["environment"]["frozen_files_verified"] == {
        "study": 76, "client": 8, "original_runtime": 71, "candidate_runtime": 71}
    for source, name in (("diagnostic", "diagnose_fractional_history_windows.py"),
                         ("helper", "verify_fractional_history_factorial.py")):
        assert result["source_sha256"][source] == sha(DIRECTORY / name)
    original, candidate = read("frozen-runtime-sha256.json", ORIGINAL), read("frozen-runtime-sha256.json", CANDIDATE)
    assert result["environment"]["original_native_sha256"] == original["files"]["spiraltorch/spiraltorch.abi3.so"]
    assert result["environment"]["candidate_native_sha256"] == candidate["files"]["spiraltorch/spiraltorch.abi3.so"]
    assert result["environment"]["original_native_sha256"] != result["environment"]["candidate_native_sha256"]
    for field, directory, name in (("original_runtime_manifest", ORIGINAL, "frozen-runtime-sha256.json"),
                                    ("candidate_runtime_manifest", CANDIDATE, "frozen-runtime-sha256.json"),
                                    ("client_manifest", ORIGINAL, "client-sha256.json")):
        assert result["manifest_sha256"][field] == sha(directory / name)


def test_every_original_full_score_and_parameter_hash_reproduced():
    result = read("results.json")
    original = {row["run_key"]: row for row in read("results.json.gz", ORIGINAL)["runs"]}
    verification = read("checkpoint-verification.json", ORIGINAL)
    original_plan = read("plan.json.gz", ORIGINAL)
    assert result["source_study_id"] == original_plan["study_id"]
    assert result["seeds"] == original_plan["config"]["seeds"] == [41, 43, 47]
    assert result["evaluation_block_hashes"] == original_plan["data"]["evaluation_block_hashes"]
    assert set(result["runs"]) == {f"{seed}:history_angle_full" for seed in result["seeds"]}
    assert result["gate"] == {"all_seeds_before_interventions": True, "rtol": 0., "atol": 0.}
    for key, row in result["runs"].items():
        assert row["checkpoint"] == original[key]["checkpoint"]
        assert row["modes"]["full"]["scores"] == original[key]["scores"]
        assert row["full_replay"] == {"exact": True, "rtol": 0., "atol": 0.,
                                    "max_abs_block_error": {"pride_unused_tail": 0., "alice_transfer": 0.}}
        assert set(row["modes"]) == set(result["windows"])
        for mode, value in row["modes"].items():
            receipt = value["receipt"]
            assert receipt["lag_window"] == result["windows"][mode]
            assert receipt["normalization_kernel_len"] == 32 and receipt["parameter_bits_preserved"]
            assert receipt["parameter_sha256"] == verification["runs"][key]["final_parameter_sha256"]
    assert verification["paired_short_state_status"] == "failed"


def test_all_block_deltas_and_seed_averages_rebuild_without_model_imports():
    result, summary = read("results.json"), read("summary.json")
    assert summary["diagnostic_id"] == result["diagnostic_id"]
    assert summary["results_sha256"] == sha(DIRECTORY / "results.json")
    assert summary["runs"] == 3
    assert summary["blocks_per_seed"] == {"pride_unused_tail": 120, "alice_transfer": 32}
    assert result["windows"] == {"full": None, "retained_short": [1, 3], "retained_tail": [3, 32], "local_only": [1, 1]}
    for mode, labels in summary["modes"].items():
        for label, numbers in labels.items():
            means, contrasts = [], []
            for row in result["runs"].values():
                value = row["modes"][mode]
                scores = value["scores"][label]
                losses = scores["block_losses"]
                full = row["modes"]["full"]["scores"][label]
                assert len(losses) == summary["blocks_per_seed"][label]
                assert all(math.isfinite(loss) for loss in losses)
                assert sum(losses) / len(losses) == scores["mean"]
                means.append(scores["mean"])
                contrasts.append(scores["mean"] - full["mean"])
                if mode != "full":
                    expected = [a - b for a, b in zip(losses, full["block_losses"])]
                    assert value["delta_from_full"][label] == {"mean": sum(expected) / len(expected), "block_deltas": expected}
            assert numbers == {"mean_ce": math.fsum(means) / len(means),
                               "mean_delta": math.fsum(contrasts) / len(contrasts), "per_seed": contrasts}


@pytest.mark.parametrize("bytecode_enabled", [False, True])
def test_archived_and_current_summary_rebuild_identically_without_model_imports(monkeypatch, bytecode_enabled):
    import builtins
    original_import = builtins.__import__
    def no_models(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_models)
    monkeypatch.setattr(sys, "dont_write_bytecode", not bytecode_enabled)
    for directory in (ROOT / "tools", DIRECTORY):
        inventory_before = {path.name for path in directory.iterdir()}
        spec = importlib.util.spec_from_file_location("numeric_window_summary", directory / "summarize_fractional_window_diagnostic.py")
        module = importlib.util.module_from_spec(spec)
        with monkeypatch.context() as imports:
            imports.setattr(sys, "dont_write_bytecode", True)
            spec.loader.exec_module(module)
        assert sys.dont_write_bytecode == (not bytecode_enabled)
        assert {path.name for path in directory.iterdir()} == inventory_before
        rebuilt = module.summarize((DIRECTORY / "results.json").read_bytes())
        assert (json.dumps(rebuilt, indent=2, allow_nan=False) + "\n").encode() == (DIRECTORY / "summary.json").read_bytes()

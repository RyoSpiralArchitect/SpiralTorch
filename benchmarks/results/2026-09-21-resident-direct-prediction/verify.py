"""Verify published bytes, the full recipe grid and timing aggregation.
Optional --raw-root verifies retained local bytes, not GPU/numerical replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def checked_files(root, records):
    seen = set()
    for row in records:
        path = (root / row["path"]).resolve(strict=True)
        assert path.is_relative_to(root.resolve()) and path.is_file()
        assert str(path.relative_to(root.resolve())) == row["path"] and row["path"] not in seen
        seen.add(row["path"])
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], row["path"]
    return seen


def main():
    if not __debug__:
        raise SystemExit("Run without -O: the archive checks require assertions.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_bytes())
    seen = checked_files(root, manifest["files"])
    assert seen == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    summary = json.loads((root / "summary.json").read_bytes())
    verified = json.loads((root / "verification/receipt.json").read_bytes())
    final = json.loads((root / "verification-final/receipt.json").read_bytes())
    followup = json.loads((root / "test-cfg-followup.json").read_bytes())
    measured = json.loads((root / "measurement-receipt.json").read_bytes())
    baseline = json.loads((root / "baseline-verification.json").read_bytes())
    assert summary["baseline_source"] == measured["baseline_source"] == baseline["source"]
    assert measured["baseline_verification_receipt_sha256"] == sha(root / "baseline-verification.json")
    assert measured["baseline_products"] == baseline["products"]
    assert summary["source"] == manifest["source"] == verified["source"] == measured["source"] == measured["worker_source"]
    assert followup["measured_source"] == summary["source"]
    assert followup["source"] == final["source"] != summary["source"]
    assert followup["changed"] == ["crates/st-backend-wgpu/src/resident_tensor.rs"]
    assert [line for line in followup["diff"].splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))] == [
        "-    #[cfg(test)]", '+    #[cfg(all(test, not(target_arch = "wasm32")))]']
    assert followup["old_sha256"] != followup["new_sha256"]
    assert all(len(followup[key]) == 64 and set(followup[key]) <= set("0123456789abcdef")
        for key in ["old_sha256", "new_sha256"])
    for receipt, steps in [(verified, 28), (final, 28), (measured, 18)]:
        assert receipt["status"] == "passed" and "active" not in receipt
        assert len(receipt["steps"]) == steps and len({s["name"] for s in receipt["steps"]}) == steps
        assert all(s["exit_code"] == 0 for s in receipt["steps"])
    assert [s["name"] for s in final["steps"]] == [s["name"] for s in verified["steps"]]
    old_wasm = next(s["command"] for s in verified["steps"] if s["name"] == "backend-wasm-clippy")
    new_wasm = next(s["command"] for s in final["steps"] if s["name"] == "backend-wasm-clippy")
    assert "--lib" in old_wasm and "--all-targets" not in old_wasm
    assert new_wasm == ["--all-targets" if arg == "--lib" else arg for arg in old_wasm]
    assert measured["verification_receipt_sha256"] == sha(root / "verification/receipt.json")
    assert measured["runner_sha256"] == sha(root / "reproduction/measure.py")
    assert measured["products"] == verified["products"]
    rows = json.loads((root / "intervals.json").read_bytes())["cases"]
    expected, observed, ratios = set(), set(), []
    shapes = [([2,16,32],2), ([2,129,32],4), ([4,32,64],8)]
    modes = ["plain", "topos_ema", "clipped_topos_ema"]
    for round_name in ["a", "b"]:
        for mode in modes:
            for client in ["native", "browser"]:
                for shape, depth in shapes:
                    for seed in [17,29,43]:
                        config = dict(shape=shape,depth=depth,seed=seed,steps=8,graph=True)
                        if mode != "plain":
                            config["learner_optimizer"] = mode
                        expected.add((round_name+"-"+mode,client,json.dumps(config,sort_keys=True)))
    total, retained = 0, 0
    for row in rows:
        key = (row["run"],row["client"],json.dumps(row["config"],sort_keys=True))
        assert key in expected and key not in observed
        observed.add(key)
        lanes = ["baseline","candidate"] + (["torch"] if row["client"] == "native" else [])
        assert len(row["samples"]) == 20
        for i, sample in enumerate(row["samples"]):
            cadence, block = ("immediate" if i < 10 else "deferred"), i % 10
            rotate = (block + row["config"]["seed"]) % len(lanes)
            assert sample["order"] == lanes[rotate:] + lanes[:rotate]
            assert sample["cadence"] == cadence and type(sample["block"]) is int and sample["block"] == block
            assert sample["warmup"] is (block < 2)
            assert set(sample["times_ms"]) == set(lanes)
            assert all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in sample["times_ms"].values())
            assert "pointwise_cotangent_routes" not in sample
            assert sample["fused_learner_seeds"] == {lane:False for lane in lanes}
            assert sample["learner_optimizers"] == {lane:row["config"].get("learner_optimizer") for lane in lanes}
            total += len(lanes)
            retained += len(lanes) * (not sample["warmup"])
        for cadence in ["immediate", "deferred"]:
            medians = {lane: statistics.median(s["times_ms"][lane] for s in row["samples"]
                if s["cadence"] == cadence and not s["warmup"]) for lane in lanes}
            ratios.append(dict(run=row["run"],client=row["client"],config=row["config"],cadence=cadence,
                baseline_over_candidate=medians["baseline"]/medians["candidate"],
                **({"torch_over_candidate":medians["torch"]/medians["candidate"]} if "torch" in lanes else {})))
    assert observed == expected and total == summary["measured_intervals"] == 5400
    assert retained == summary["retained_intervals"] == 4320
    assert len(observed) // 2 == summary["recipes"] == 54
    assert ratios == summary["ratios"]
    expected_groups = {(mode, client, cadence) for mode in modes
        for client in ["native", "browser"] for cadence in ["immediate", "deferred"]}
    assert len(summary["groups"]) == len(expected_groups)
    assert {(g["optimizer"], g["client"], g["cadence"]) for g in summary["groups"]} == expected_groups
    pairs = [dict(run=r["run"], client=r["client"], config=r["config"],
        paired_fingerprints_equal=r["fingerprints"]["baseline"] == r["fingerprints"]["candidate"]) for r in rows]
    assert json.loads((root / "paired_fingerprints.json").read_bytes()) == pairs
    for group in summary["groups"]:
        values = [r["baseline_over_candidate"] for r in ratios if r["client"] == group["client"] and r["cadence"] == group["cadence"]
            and (r["config"].get("learner_optimizer") or "plain") == group["optimizer"]]
        assert len(values) == group["cases"] == 18
        assert group["min"] == min(values) and group["max"] == max(values)
        assert group["favorable"] == sum(v > 1 for v in values)
        assert group["geometric_mean"] == math.exp(statistics.mean(map(math.log, values)))
    assert summary["all_paired_fingerprints_equal"] is all(
        r["fingerprints"]["baseline"] == r["fingerprints"]["candidate"] for r in rows)
    validations = [json.loads(p.read_bytes()) for p in sorted((root / "validation").glob("*.json"))]
    assert len(validations) == 6 and all(v["status"] == "passed" and v["direct_learner_seeds"] is False for v in validations)
    assert sum(v["browser_intervals_revalidated"] for v in validations) == summary["browser_intervals_revalidated"] == 2160
    assert summary["maximum_saved_state_abs_error"] == max(
        error for v in validations for c in v["cases"] for error in c["max_abs_errors"].values())
    inventory = json.loads((root / "raw-inventory.json").read_bytes())
    records = {r["path"]: r for r in inventory["records"]}
    assert len(records) == len(inventory["records"])
    assert inventory["bytes"] == sum(r["bytes"] for r in inventory["records"])
    ci_failure = json.loads((root / "preflight/hosted-wasm-test-target.json").read_bytes())
    assert ci_failure["source"] == summary["source"] and ci_failure["conclusion"] == "failure"
    assert ci_failure["followup"] == "test-cfg-followup.json"
    assert ci_failure["full_local_log"] == records["ci-wasm-job.log"]
    assert records["verified-a/receipt.json"]["sha256"] == sha(root / "verification/receipt.json")
    assert records["verified-b/receipt.json"]["sha256"] == sha(root / "verification-final/receipt.json")
    assert records["measurement-a/receipt.json"]["sha256"] == sha(root / "measurement-receipt.json")
    for public, raw, receipt in [("verification", "verified-a", verified), ("verification-final", "verified-b", final)]:
        for name, expected_hash in receipt["products"].items():
            assert records[raw + "/" + name]["sha256"] == expected_hash
        for path in (root / public).iterdir():
            assert path.is_file() and records[raw + "/" + path.name]["sha256"] == sha(path)
    assert len(measured["results"]) == 18
    for name, expected_hash in measured["results"].items():
        assert records["measurement-a/" + name]["sha256"] == expected_hash
        if name.endswith("-validation.json"):
            assert sha(root / "validation" / name) == expected_hash
    for path in sorted((root / "validation").glob("*.json")):
        validation = json.loads(path.read_bytes())
        run = path.name.removesuffix("-validation.json")
        assert len(validation["cases"]) == 9
        for lane in ["baseline", "candidate"]:
            expected_product = verified if lane == "candidate" else baseline
            product = validation["native_products"][lane]
            assert product["file"]["sha256"] == expected_product["products"]["native-worker"]
            assert product["binding"]["valid"] is True
            assert product["binding"]["source_commit"] == expected_product["source"]["commit"]
            assert product["binding"]["source_tree"] == expected_product["source"]["tree"]
            assert validation["browser_assets"]["/" + lane + "/spiraltorch_wasm_bg.wasm"] == expected_product["products"]["benchmark/spiraltorch_wasm_bg.wasm"]
        harness = validation["browser_harness_source"]
        assert (harness["commit"], harness["tree"], harness["tracked_dirty"]) == (
            summary["source"]["commit"], summary["source"]["tree"], False)
        for client in ["native", "browser"]:
            selected = [r for r in rows if r["run"] == run and r["client"] == client]
            assert [r["config"] for r in selected] == [c["config"] for c in validation["cases"]]
            for row, checked in zip(selected, validation["cases"]):
                for cadence in ["immediate", "deferred"]:
                    recorded = checked[client][cadence]
                    for lane, stats in recorded["lanes"].items():
                        times = [s["times_ms"][lane] for s in row["samples"] if
                            s["cadence"] == cadence and not s["warmup"]]
                        assert stats == dict(median_ms=statistics.median(times), min_ms=min(times),
                            max_ms=max(times), retained=len(times))
    baseline_inventory = json.loads((root / "baseline-inventory.json").read_bytes())["records"]
    assert len(baseline_inventory) == len({r["path"] for r in baseline_inventory})
    assert {r["path"]: r["sha256"] for r in baseline_inventory} == dict(baseline["products"], **{"receipt.json": sha(root / "baseline-verification.json")})
    if args.baseline_root:
        checked_files(args.baseline_root, baseline_inventory)
    if args.raw_root:
        checked_files(args.raw_root, inventory["records"])
    print(json.dumps(dict(status="passed", published_files=len(seen), recipes=summary["recipes"],
        intervals=total, retained=retained, raw_files_verified=len(inventory["records"]) if args.raw_root else 0,
        baseline_files_verified=len(baseline_inventory) if args.baseline_root else 0,
        numerical_or_gpu_replay=False, manifest_sha256=sha(root / "manifest.json"))))


if __name__ == "__main__":
    main()

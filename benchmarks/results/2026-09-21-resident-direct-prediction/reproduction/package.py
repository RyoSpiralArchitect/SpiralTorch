"""Publish all interval timings and validation records, not raw tensor payloads."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path("/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-graph-forward-v1")
LOG = Path(__file__).parent
OUT = ROOT / "benchmarks/results/2026-09-21-resident-direct-prediction"
BASE = Path("/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-pointwise-cotangent-20260915/verified-b")
verified = json.loads((LOG / "verified-a/receipt.json").read_bytes())
measured = json.loads((LOG / "measurement-a/receipt.json").read_bytes())
final = json.loads((LOG / "verified-b/receipt.json").read_bytes())
assert verified["status"] == measured["status"] == "passed"
assert "active" not in verified and "active" not in measured
assert verified["source"] == measured["source"] == measured["worker_source"]
assert final["status"] == "passed" and "active" not in final
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == final["source"]["commit"]
changed = subprocess.check_output(["git", "diff", "--name-only", measured["source"]["commit"], final["source"]["commit"]], cwd=ROOT, text=True).splitlines()
assert changed == ["crates/st-backend-wgpu/src/resident_tensor.rs"]
old = subprocess.check_output(["git", "show", measured["source"]["commit"] + ":" + changed[0]], cwd=ROOT)
new = subprocess.check_output(["git", "show", final["source"]["commit"] + ":" + changed[0]], cwd=ROOT)
assert old.replace(b"    #[cfg(test)]\n    pub(crate) fn capture_into(", b'    #[cfg(all(test, not(target_arch = "wasm32")))]\n    pub(crate) fn capture_into(') == new
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
OUT.mkdir()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy(path, name):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.copyfile(path, target)


rows, checks, ratios = [], [], []
for path in sorted((LOG / "measurement-a").glob("*-validation.json")):
    validation = json.loads(path.read_bytes())
    assert validation["status"] == "passed" and len(validation["cases"]) == 9
    prefix = path.name.removesuffix("-validation.json")
    copy(path, "validation/" + path.name)
    for client in ("native", "browser"):
        raw = json.loads((path.parent / (prefix + "-" + client + ".json")).read_bytes())
        assert raw["status"] == "passed" and raw["direct_learner_seeds"] is False
        for case, checked in zip(raw["cases"], validation["cases"]):
            assert case["config"] == checked["config"]
            rows.append(dict(run=prefix, client=client, config=case["config"],
                samples=case["samples"], fingerprints=case["fingerprints"]))
            checks.append(dict(run=prefix, client=client, config=case["config"],
                paired_fingerprints_equal=case["fingerprints"]["baseline"] == case["fingerprints"]["candidate"]))
            for cadence in ("immediate", "deferred"):
                values = checked[client][cadence]
                ratios.append(dict(run=prefix, client=client, config=case["config"], cadence=cadence,
                    baseline_over_candidate=values["baseline_over_candidate"],
                    **({"torch_over_candidate":values["torch_over_candidate"]} if client == "native" else {})))
assert len(rows) == 108 and len(ratios) == 216
write("intervals.json", dict(schema="spiraltorch.direct_prediction_intervals.v1", cases=rows))
write("paired_fingerprints.json", checks)
groups = []
for optimizer in ("plain", "topos_ema", "clipped_topos_ema"):
    for client in ("native", "browser"):
        for cadence in ("immediate", "deferred"):
            selected = [r for r in ratios if r["client"] == client and r["cadence"] == cadence
                and (r["config"].get("learner_optimizer") or "plain") == optimizer]
            assert len(selected) == 18
            values = [r["baseline_over_candidate"] for r in selected]
            groups.append(dict(optimizer=optimizer, client=client, cadence=cadence, cases=len(selected),
                geometric_mean=math.exp(statistics.mean(map(math.log, values))),
                min=min(values), max=max(values), favorable=sum(v > 1 for v in values)))
validation_files = list((OUT / "validation").glob("*.json"))
maximum = max(v for path in validation_files for case in json.loads(path.read_bytes())["cases"]
              for v in case["max_abs_errors"].values())
write("summary.json", dict(schema="spiraltorch.direct_prediction_summary.v1", source=measured["source"], baseline_source=measured["baseline_source"],
    recipes=54, measured_intervals=5400, retained_intervals=4320, browser_intervals_revalidated=2160,
    maximum_saved_state_abs_error=maximum, all_paired_fingerprints_equal=all(c["paired_fingerprints_equal"] for c in checks),
    ratios=ratios, groups=groups,
    boundary="Diagnostic correlated per-case medians, not GPU kernel times or confidence intervals. "
    "Both runtimes use identical ordinary unfused quadratic/quartic seed evaluation. Torch lacks equivalent guards and rollback. "
    "No exact browser physical adapter attestation, CUDA, fastest-Torch, uninterrupted FT or learning-quality claim."))
for name in ("verify.py", "verify-final.py", "measure.py", "package.py"):
    copy(LOG / name, "reproduction/" + name)
copy(LOG / "verified-a/receipt.json", "verification/receipt.json")
for path in (LOG / "verified-a").glob("*.log"):
    copy(path, "verification/" + path.name)
for path in (LOG / "verified-a").glob("*clients.json"):
    copy(path, "verification/" + path.name)
copy(BASE / "receipt.json", "baseline-verification.json")
copy(LOG / "verified-b/receipt.json", "verification-final/receipt.json")
for path in (LOG / "verified-b").glob("*.log"):
    copy(path, "verification-final/" + path.name)
for path in (LOG / "verified-b").glob("*clients.json"):
    copy(path, "verification-final/" + path.name)
write("test-cfg-followup.json", dict(source=final["source"], measured_source=measured["source"],
    changed=changed, old_sha256=hashlib.sha256(old).hexdigest(), new_sha256=hashlib.sha256(new).hexdigest(),
    diff=subprocess.check_output(["git", "diff", measured["source"]["commit"], final["source"]["commit"], "--", changed[0]], cwd=ROOT, text=True),
    boundary="The only source change after timing narrows a test-only helper to native tests. Production paths were not changed; timings remain pinned to the measured commit, not relabeled as a newer build."))
copy(LOG / "preflight-clippy.log", "preflight/first-clippy.log")
copy(LOG / "preflight-clippy-b.log", "preflight/corrected-clippy.log")
ci_log = LOG / "ci-wasm-job.log"
assert "method `capture_into` is never used" in ci_log.read_text()
write("preflight/hosted-wasm-test-target.json", dict(
    run="https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/35524688117",
    job_id=106114691010, source=measured["source"], conclusion="failure",
    error="method `capture_into` is never used; -D dead-code implied by -D warnings; could not compile st-backend-wgpu (lib test)",
    cause="The measured-source local wasm Clippy check covered --lib, while hosted --all-targets also compiled lib tests.",
    followup="test-cfg-followup.json", full_local_log=dict(path="ci-wasm-job.log", bytes=ci_log.stat().st_size, sha256=sha(ci_log))))
copy(LOG / "measurement-a/receipt.json", "measurement-receipt.json")
inventory = []
for folder in ("verified-a", "verified-b", "measurement-a"):
    for path in sorted((LOG / folder).rglob("*")):
        if path.is_file():
            inventory.append(dict(path=str(path.relative_to(LOG)), bytes=path.stat().st_size, sha256=sha(path)))
inventory.append(dict(path=ci_log.name, bytes=ci_log.stat().st_size, sha256=sha(ci_log)))
write("raw-inventory.json", dict(schema="spiraltorch.local_raw_inventory.v1", records=inventory,
    bytes=sum(r["bytes"] for r in inventory),
    boundary="Raw tensor captures, runtime products and complete logs retained locally; this list proves bytes, not rerun numerics."))
write("baseline-inventory.json", dict(records=[dict(path=name, bytes=(BASE / name).stat().st_size, sha256=sha(BASE / name))
    for name in [*measured["baseline_products"], "receipt.json"]]))
print(json.dumps(dict(output=str(OUT), maximum_saved_state_abs_error=maximum,
    all_paired_fingerprints_equal=all(c["paired_fingerprints_equal"] for c in checks), groups=groups), indent=2))

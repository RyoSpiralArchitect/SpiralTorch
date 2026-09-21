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
OUT = ROOT / "benchmarks/results/2026-09-15-resident-pointwise-cotangent"
OUT.mkdir()
verified = json.loads((LOG / "verified-b/receipt.json").read_bytes())
measured = json.loads((LOG / "measurement-a/receipt.json").read_bytes())
assert verified["status"] == measured["status"] == "passed"
assert "active" not in verified and "active" not in measured
assert verified["source"] == measured["source"] == measured["worker_source"]
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == measured["source"]["commit"]
assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).strip()


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
        assert raw["status"] == "passed" and raw["direct_learner_seeds"] is True
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
write("intervals.json", dict(schema="spiraltorch.pointwise_cotangent_intervals.v1", cases=rows))
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
write("summary.json", dict(schema="spiraltorch.pointwise_cotangent_summary.v1", source=measured["source"],
    recipes=54, measured_intervals=5400, retained_intervals=4320, browser_intervals_revalidated=2160,
    maximum_saved_state_abs_error=maximum, all_paired_fingerprints_equal=all(c["paired_fingerprints_equal"] for c in checks),
    ratios=ratios, groups=groups,
    boundary="Diagnostic correlated per-case medians, not GPU kernel times or confidence intervals. "
    "Both routes use identical fused seed programs. Torch lacks equivalent guards and rollback. "
    "No exact browser physical adapter attestation, CUDA, fastest-Torch, uninterrupted FT or learning-quality claim."))
for name in ("verify.py", "measure.py", "package.py"):
    copy(LOG / name, "reproduction/" + name)
copy(LOG / "verified-b/receipt.json", "verification/receipt.json")
for path in (LOG / "verified-b").glob("*.log"):
    copy(path, "verification/" + path.name)
for path in (LOG / "verified-b").glob("*clients.json"):
    copy(path, "verification/" + path.name)
copy(LOG / "verified-a/receipt.json", "earlier-attempt/receipt.json")
copy(LOG / "verified-a/gpu-test_runtime_imports.log", "earlier-attempt/test-path-error.log")
copy(LOG / "measurement-a/receipt.json", "measurement-receipt.json")
inventory = []
for folder in ("verified-a", "verified-b", "measurement-a"):
    for path in sorted((LOG / folder).rglob("*")):
        if path.is_file():
            inventory.append(dict(path=str(path.relative_to(LOG)), bytes=path.stat().st_size, sha256=sha(path)))
write("raw-inventory.json", dict(schema="spiraltorch.local_raw_inventory.v1", records=inventory,
    bytes=sum(r["bytes"] for r in inventory),
    boundary="Raw tensor captures, runtime products and complete logs retained locally; this list proves bytes, not rerun numerics."))
print(json.dumps(dict(output=str(OUT), maximum_saved_state_abs_error=maximum,
    all_paired_fingerprints_equal=all(c["paired_fingerprints_equal"] for c in checks), groups=groups), indent=2))

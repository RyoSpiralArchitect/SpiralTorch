"""Collect bounded regression evidence; not numerical reexecution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

raw, public = map(Path, sys.argv[1:])
source = json.loads((public / "source.json").read_text())
stages = ["resident-native-tests", "nn-full-tests", "selfsup-tests",
          "resident-browser-build", "resident-browser-bindgen", "resident-browser-training"]
receipts = []
for stage in stages:
    receipt = json.loads((raw / "accepted" / stage / "receipt.json").read_text())
    if receipt.pop("source") != source or receipt["exit_code"] != 0 or not receipt["source_unchanged"]:
        raise ValueError("unaccepted training stage: " + stage)
    log = raw / "accepted" / stage / "stdout.log"
    totals = [dict(zip(("passed", "failed", "ignored", "measured", "filtered_out"), map(int, m)))
              for m in re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", log.read_text())]
    receipts.append({"stage": stage, **receipt, "test_totals": totals})
    shutil.copyfile(log, public / (stage + ".log"))

report_path = raw / "accepted/resident-browser.json"
report = json.loads(report_path.read_text())
if report["status"] != "passed" or report["page_errors"]:
    raise ValueError("browser regression failure")
expected_counts = {"pointwise_fusion": 6, "autograd": 6, "learning": 12,
                   "resident_loss": 6, "classification": 6, "microbatch": 6,
                   "gradient_clip": 6, "momentum": 6}
sections = {}
fields = ["seed", "input_shape", "shape", "policy", "fused", "label_smoothing",
          "ignore_index", "reduction", "observations_after_updates", "microbatches",
          "module_parameters_applied", "max_abs_error", "initial_loss", "final_loss",
          "accepted_updates", "resume", "gradient_clip", "topos_momentum"]
for name, count in expected_counts.items():
    section = report[name]
    if len(section["cases"]) != count:
        raise ValueError("incomplete browser section: " + name)
    cases = []
    for case in section["cases"]:
        item = {k: case[k] for k in fields if k in case}
        for key in ["steps", "windows", "captures", "replays"]:
            if key in case:
                item[key + "_count"] = len(case[key]) if isinstance(case[key], list) else case[key]
        cases.append(item)
    sections[name] = {"cases": cases, "guards": len(section.get("guards", [])),
                      "probes": len(section.get("probes", []))}
if any(c["steps_count"] != 64 for c in sections["classification"]["cases"]):
    raise ValueError("classification trajectory shortened")
result = {"role": "Additional clean-source native/browser regression gates, separate from timing intervals",
          "source_commit": source["commit"],
          "source_json_sha256": hashlib.sha256((public / "source.json").read_bytes()).hexdigest(),
          "validation": receipts,
          "browser": {"raw_path": "accepted/resident-browser.json",
                      "raw_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      **{k: report[k] for k in ["status", "adapter", "browser_version", "asset_sha256", "page_errors"]},
                      "base_cases": len(report["cases"]), "base_guards": len(report["guards"]),
                      "primitive_checks": len(report["primitive_checks"]), "sections": sections},
          "numerical_reexecution": False}
with (public / "training-validation.json").open("x") as output:
    json.dump(result, output, indent=2, allow_nan=False)
    output.write("\n")
for stage in ["resident-graph-current", "resident-graph-diagnostic", "resident-graph-callsite",
              "resident-graph-old-formula-control", "resident-graph-full-exp-step", "resident-graph-hybrid"]:
    shutil.copyfile(raw / ("control-" + stage) / "stdout.log", public / (stage + ".log"))
print(json.dumps({"additional_stages": len(receipts), "browser_sections": list(sections)}))

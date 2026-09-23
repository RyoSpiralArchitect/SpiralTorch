"""Publish compact results and verify hashes without publishing local binaries."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

SHAPES = {(2, 3), (8, 257), (32, 256), (64, 768), (128, 1025), (256, 256)}
ACCEPTED = {"accepted-" + name for name in (
    "native", "wasm-clippy", "native-clippy", "tensor-clippy", "tensor-wasm-clippy",
    "tensor-tests", "wasm-build", "bindgen", "browser", "native-bench", "torch", "fmt")}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_intervals(report, routes):
    require({(c["rows"], c["cols"]) for c in report["cases"]} == SHAPES and len(report["cases"]) == 6, "shape coverage")
    medians = []
    for case in report["cases"]:
        intervals = case["intervals"]
        require(len(intervals) == 18 * len(routes), "interval count")
        require({(v["iteration"], v["route"]) for v in intervals} == {(i, r) for i in range(18) for r in routes}, "interval identities")
        require(all(math.isfinite(v["ms"]) and v["ms"] > 0 for v in intervals), "timing values")
        errors = case["max_scaled_error"]
        if isinstance(errors, dict):
            require(set(errors) == set(routes), "numerical route coverage")
            errors = list(errors.values())
        require(len(errors) == len(routes), "numerical route count")
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in errors), "numerical gate")
        medians.append(dict(rows=case["rows"], cols=case["cols"], median_ms={
            str(route): statistics.median(v["ms"] for v in intervals if v["route"] == route) for route in routes}))
    return medians


def validate_centered_browser(browser):
    require(browser["schema"] == "spiraltorch.resident_layer_norm.browser.v2", "browser schema")
    require(browser["status"] == "passed" and not browser["page_errors"] and not browser["console_messages"], "browser status")
    require((browser["cases"], browser["masks_per_case"], browser["training_steps"], browser["intermediate_readbacks"]) == (7, 8, 400, 0), "browser coverage")
    require((browser["scale_nullspace_cases"], browser["epsilon_cancellation_cases"], browser["dynamic_range_variants"], browser["guard_checks"]) == (5, 2, 20, 4), "browser regression coverage")
    require("backend: BrowserWebGpu" in browser["adapter"], "browser runtime backend")
    require(browser["browser_adapter_probe"]["is_fallback_adapter"] is False, "browser fallback probe")
    require(math.isfinite(browser["first_loss"]) and browser["first_loss"] > 0, "initial browser loss")
    require(math.isfinite(browser["last_loss"]) and 0 <= browser["last_loss"] < browser["first_loss"] * 1e-4, "browser learning")


def collect(raw, public, centered=False):
    stages = {p.parent.name: read(p) for p in raw.glob("*/receipt.json")}
    accepted = ACCEPTED | {"accepted-decimal"} if centered else ACCEPTED
    require({n for n in stages if n.startswith("accepted-")} == accepted, "accepted stage coverage")
    source = stages["accepted-native"]["source"]
    for name in accepted:
        record = stages[name]
        require(record["exit_code"] == 0 and record["source_unchanged"], name + " failed")
        require(record["source"] == source and not source["status"], name + " differs from clean source")
    negatives = ([("scale-nullspace-probe", 101), ("epsilon-cancellation-probe", 101),
                  ("review-dynamic-range", 101), ("review-division-refined", 101)] if centered else
                 [("prototype-01-native", 101), ("prototype-06-denormal-probe", 101),
                  ("training-controls-torch", 1), ("training-controls-torch-diagnostic", 1)])
    for name, code in negatives:
        require(stages[name]["exit_code"] == code and stages[name]["source_unchanged"], "missing negative control " + name)
    if not centered:
        require("0 != -0.09223365" in (raw / "prototype-06-denormal-probe/stderr.log").read_text(), "subnormal regression")
    tests = lambda name: [int(v) for v in re.findall(r"test result: ok\. (\d+) passed; 0 failed", (raw / name / "stdout.log").read_text())]
    require(tests("accepted-native") == [202 if centered else 198, 31], "native tests")
    require(tests("accepted-tensor-tests") == [11, 4], "Tensor tests")
    native = read(raw / "accepted-native-bench/stdout.log")
    torch = read(raw / "accepted-torch/stdout.log")
    for report in (native, torch):
        require((report["epsilon"], report["parameter_gradient_scale"], report["warmup"], report["iterations"]) == (1e-5, 0.5, 3, 18), "benchmark protocol")
    medians = dict(native=validate_intervals(native, (0, 1, 2)), pytorch=validate_intervals(torch, ("cpu", "mps")))
    browser = read(raw / "accepted-browser.json")
    extra = []
    if centered:
        validate_centered_browser(browser)
        decimal = read(raw / "accepted-decimal/stdout.log")
        require(decimal["precision"] == 100 and len(decimal["dx_f32"]) == 4, "decimal oracle")
        extra.append(("decimal-oracle.json", decimal))
    else:
        require(browser["status"] == "passed" and not browser["page_errors"] and not browser["console_messages"], "browser status")
        require((browser["cases"], browser["masks_per_case"], browser["training_steps"], browser["intermediate_readbacks"]) == (7, 7, 400, 0), "browser coverage")
        require(math.isfinite(browser["last_loss"]) and browser["last_loss"] < browser["first_loss"] * 1e-4, "browser learning")
        masks = read(raw / "torch-mask-probe/stdout.log")
        require(masks["diagnostic_only"] and not masks["all_masks_valid"] and len(masks["cases"]) == 14, "mask negative control")
        forced = read(raw / "training-controls-torch-forced-input-vjp/stdout.log")
        require(forced["status"] == "passed" and forced["forced_unused_input_vjp"] and len(forced["cases"]) == 4, "forced-input control")
        extra.extend([("pytorch-mask-diagnostic.json", masks), ("pytorch-forced-input-control.json", forced)])
    public.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (public / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for name, value in [("native.json", native), ("pytorch.json", torch), ("browser.json", browser),
                        ("measured-source.json", source), ("medians.json", medians)] + extra:
        write(name, value)
    stage_records = {name: {k: v for k, v in value.items() if k != "source"} for name, value in stages.items()}
    for name, record in stage_records.items():
        record["source_commit"] = stages[name]["source"]["commit"]
        record["source_dirty"] = bool(stages[name]["source"]["status"])
    write("validation.json", dict(schema="spiraltorch.resident_layer_norm.evidence.v2" if centered else "spiraltorch.resident_layer_norm.evidence.v1", measured_commit=source["commit"],
          scope="Exploratory host-to-host primitive comparison, not performance admission; ordinary Tensor route unchanged",
          accepted_stages=sorted(accepted), rust_tests=248 if centered else 244, measured_intervals=540, stages=stage_records))
    for name in [n for n, _ in negatives] + (["accepted-native"] if centered else []):
        for stream in ("stdout.log", "stderr.log"):
            text = (raw / name / stream).read_text()
            (public / (name + "-" + stream)).write_text("\n".join(text.splitlines()[-30:]) + "\n")
    write("raw-manifest.json", [file_record(p, raw) for p in sorted(raw.rglob("*")) if p.is_file()])
    write("public-manifest.json", [file_record(p, public) for p in sorted(public.iterdir()) if p.is_file()])
    verify(raw, public)


def verify(raw, public):
    manifests = [(public, "public-manifest.json")]
    if raw is not None:
        manifests.append((raw, "raw-manifest.json"))
    for root, manifest in manifests:
        for expected in read(public / manifest):
            path = root / expected["path"]
            require(path.is_relative_to(root) and ".." not in path.relative_to(root).parts, "manifest path")
            require(file_record(path, root) == expected, "hash mismatch: " + expected["path"])
    medians = dict(native=validate_intervals(read(public / "native.json"), (0, 1, 2)),
                   pytorch=validate_intervals(read(public / "pytorch.json"), ("cpu", "mps")))
    require(medians == read(public / "medians.json"), "derived medians differ")
    if read(public / "validation.json")["schema"] == "spiraltorch.resident_layer_norm.evidence.v2":
        validate_centered_browser(read(public / "browser.json"))
    print(json.dumps(dict(status="verified", raw_verified=raw is not None,
                         raw_files=len(read(public / "raw-manifest.json")), public=str(public))))


if __name__ == "__main__":
    if sys.argv[1] == "verify-public":
        verify(None, Path(sys.argv[2]))
    elif sys.argv[1] in ("collect", "collect-centered"):
        mode, raw, public = sys.argv[1:]
        collect(Path(raw), Path(public), centered=mode == "collect-centered")
    elif sys.argv[1] == "verify":
        _, raw, public = sys.argv[1:]
        verify(Path(raw), Path(public))
    else:
        raise SystemExit("expected collect|collect-centered|verify RAW PUBLIC, or verify-public PUBLIC")

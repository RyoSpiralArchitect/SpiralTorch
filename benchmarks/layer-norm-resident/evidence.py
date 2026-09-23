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


def collect(raw, public):
    stages = {p.parent.name: read(p) for p in raw.glob("*/receipt.json")}
    require({n for n in stages if n.startswith("accepted-")} == ACCEPTED, "accepted stage coverage")
    source = stages["accepted-native"]["source"]
    for name in ACCEPTED:
        record = stages[name]
        require(record["exit_code"] == 0 and record["source_unchanged"], name + " failed")
        require(record["source"] == source and not source["status"], name + " differs from clean source")
    for name, code in [("prototype-01-native", 101), ("prototype-06-denormal-probe", 101),
                       ("training-controls-torch", 1), ("training-controls-torch-diagnostic", 1)]:
        require(stages[name]["exit_code"] == code and stages[name]["source_unchanged"], "missing negative control " + name)
    require("0 != -0.09223365" in (raw / "prototype-06-denormal-probe/stderr.log").read_text(), "subnormal regression")
    tests = lambda name: [int(v) for v in re.findall(r"test result: ok\. (\d+) passed; 0 failed", (raw / name / "stdout.log").read_text())]
    require(tests("accepted-native") == [198, 31], "native tests")
    require(tests("accepted-tensor-tests") == [11, 4], "Tensor tests")
    native = read(raw / "accepted-native-bench/stdout.log")
    torch = read(raw / "accepted-torch/stdout.log")
    for report in (native, torch):
        require((report["epsilon"], report["parameter_gradient_scale"], report["warmup"], report["iterations"]) == (1e-5, 0.5, 3, 18), "benchmark protocol")
    medians = dict(native=validate_intervals(native, (0, 1, 2)), pytorch=validate_intervals(torch, ("cpu", "mps")))
    browser = read(raw / "accepted-browser.json")
    require(browser["status"] == "passed" and not browser["page_errors"] and not browser["console_messages"], "browser status")
    require((browser["cases"], browser["masks_per_case"], browser["training_steps"], browser["intermediate_readbacks"]) == (7, 7, 400, 0), "browser coverage")
    require(math.isfinite(browser["last_loss"]) and browser["last_loss"] < browser["first_loss"] * 1e-4, "browser learning")
    masks = read(raw / "torch-mask-probe/stdout.log")
    require(masks["diagnostic_only"] and not masks["all_masks_valid"] and len(masks["cases"]) == 14, "mask negative control")
    forced = read(raw / "training-controls-torch-forced-input-vjp/stdout.log")
    require(forced["status"] == "passed" and forced["forced_unused_input_vjp"] and len(forced["cases"]) == 4, "forced-input control")
    public.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (public / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for name, value in [("native.json", native), ("pytorch.json", torch), ("browser.json", browser),
                        ("pytorch-mask-diagnostic.json", masks), ("pytorch-forced-input-control.json", forced),
                        ("measured-source.json", source), ("medians.json", medians)]:
        write(name, value)
    stage_records = {name: {k: v for k, v in value.items() if k != "source"} for name, value in stages.items()}
    for name, record in stage_records.items():
        record["source_commit"] = stages[name]["source"]["commit"]
        record["source_dirty"] = bool(stages[name]["source"]["status"])
    write("validation.json", dict(schema="spiraltorch.resident_layer_norm.evidence.v1", measured_commit=source["commit"],
          scope="Exploratory host-to-host primitive comparison, not performance admission; ordinary Tensor route unchanged",
          accepted_stages=sorted(ACCEPTED), rust_tests=244, measured_intervals=540, stages=stage_records))
    for name in ("prototype-01-native", "prototype-06-denormal-probe", "training-controls-torch-diagnostic"):
        for stream in ("stdout.log", "stderr.log"):
            text = (raw / name / stream).read_text()
            (public / (name + "-" + stream)).write_text("\n".join(text.splitlines()[-30:]) + "\n")
    write("raw-manifest.json", [file_record(p, raw) for p in sorted(raw.rglob("*")) if p.is_file()])
    write("public-manifest.json", [file_record(p, public) for p in sorted(public.iterdir()) if p.is_file()])
    verify(raw, public)


def verify(raw, public):
    for root, manifest in ((raw, "raw-manifest.json"), (public, "public-manifest.json")):
        for expected in read(public / manifest):
            path = root / expected["path"]
            require(path.is_relative_to(root) and ".." not in path.relative_to(root).parts, "manifest path")
            require(file_record(path, root) == expected, "hash mismatch: " + expected["path"])
    print(json.dumps(dict(status="verified", raw_files=len(read(public / "raw-manifest.json")), public=str(public))))


if __name__ == "__main__":
    mode, raw, public = sys.argv[1:]
    if mode == "collect":
        collect(Path(raw), Path(public))
    elif mode == "verify":
        verify(Path(raw), Path(public))
    else:
        raise SystemExit("expected collect|verify RAW PUBLIC")

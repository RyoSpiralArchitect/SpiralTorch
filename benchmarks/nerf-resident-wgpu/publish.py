"""Publish compact correctness evidence. Raw arrays and binaries remain local."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("nerf_compare", Path(__file__).with_name("compare.py"))
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
STAGES = [
    "format", "inventory", "browser-syntax", "admission", "native-clippy", "wasm-clippy",
    "backend-tests", "native", "freeze-native", "wasm", "freeze-wasm", "bindgen", "browser", "torch",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def publish(raw, output):
    output.mkdir(parents=True, exist_ok=False)
    source = read(raw / "final-native/receipt.json")["source"]
    assert source["status"] == ""
    validation = []
    for stage in STAGES:
        receipt = read(raw / f"final-{stage}/receipt.json")
        assert receipt.pop("source") == source
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
        validation.append({"stage": stage, **receipt})
    native = read(raw / "final-native/stdout.log")
    browser = read(raw / "browser-final.json")
    torch = read(raw / "final-torch/stdout.log")
    a, b = control.admit(native), control.admit(browser)
    assert torch["status"] == "passed" and len(torch["cases"]) == 36
    assert list(a) == [control.key(case) for case in torch["cases"]]
    assert browser["page_errors"] == [] and browser["console_messages"] == []
    assert browser["browser_adapter_probe"]["is_fallback_adapter"] is False
    assert browser["wasm_sha256"] == sha(raw / "wasm-final/spiraltorch_wasm_bg.wasm")
    for name in ["weights", "bias"]:
        assert control.f32_bytes(native[name]) == control.f32_bytes(browser[name])
    cases = []
    for record in torch["cases"]:
        key = control.key(record)
        x, y = a[key], b[key]
        inputs = control.f32_bytes([v for ray in x["ray_inputs"] for v in ray])
        assert inputs == control.f32_bytes([v for ray in y["ray_inputs"] for v in ray])
        assert record["input_sha256"] == hashlib.sha256(inputs).hexdigest()
        reference = record["torch_rgba"]
        assert len(reference) == key[0] * 4
        assert record["torch_sha256"] == hashlib.sha256(control.f32_bytes(reference)).hexdigest()
        for runtime, values in [("native", x["rgba"]), ("browser", y["rgba"])]:
            deltas = [abs(value - ref) for value, ref in zip(values, reference)]
            assert all(d <= 4e-7 + 4e-6 * abs(ref) for d, ref in zip(deltas, reference))
            assert max(deltas) == record["max_abs_errors"][runtime]
        cases.append({
            **{k: v for k, v in record.items() if k != "torch_rgba"},
            "native_sha256": hashlib.sha256(control.f32_bytes(x["rgba"])).hexdigest(),
            "browser_sha256": hashlib.sha256(control.f32_bytes(y["rgba"])).hexdigest(),
            "native_browser_max_abs": max(abs(u - v) for u, v in zip(x["rgba"], y["rgba"])),
        })
    result = {
        "measured_commit": source["commit"], "status": "passed", "cases": cases,
        "guards": {"native": native["guards"], "browser": browser["guards"]},
        "native_adapter": native["adapter"], "browser_adapter": browser["adapter"],
        "browser_version": browser["browser_version"],
        "browser_adapter_probe": browser["browser_adapter_probe"],
        "browser_assets": browser["asset_sha256"],
        "torch": {k: torch[k] for k in ["torch_version", "device", "intra_op_threads", "inter_op_threads"]},
        "parameter_sha256": hashlib.sha256(control.f32_bytes(native["weights"] + native["bias"])).hexdigest(),
        "max_abs_errors": {name: max(c["max_abs_errors"][name] for c in cases) for name in ["native", "browser"]},
        "boundary": "Correctness fixture only; affine NN field, f32 shader versus eager CPU f64 integration control. No performance, training, cross-vendor or f64-equivalence claim.",
    }
    write(output / "results.json", result)
    write(output / "source.json", source)
    write(output / "validation.json", validation)
    shutil.copyfile(raw / "final-backend-tests/stdout.log", output / "backend-tests.log")
    shutil.copyfile(raw / "final-admission/stderr.log", output / "admission-tests.log")
    failure = read(raw / "initial-tests/receipt.json")
    assert failure["exit_code"] != 0 and failure["source_unchanged"] is True
    shutil.copytree(raw / "initial-tests", output / "failed-long-prefix")
    write(output / "local-raw-manifest.json", {
        str(p): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(raw.rglob("*")) if p.is_file()
    })
    return result


if __name__ == "__main__":
    if sys.argv[1] == "--seal":
        root = Path(sys.argv[2])
        write(root / "manifest.json", {
            str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        })
        sys.exit(0)
    result = publish(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({"commit": result["measured_commit"], "cases": len(result["cases"]),
                      "max_abs_errors": result["max_abs_errors"]}))

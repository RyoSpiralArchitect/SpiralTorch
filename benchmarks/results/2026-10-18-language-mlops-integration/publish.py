"""Retain only bounded result records from the fixed local integration run.

This checks receipt consistency, not independent execution or model quality.
Model data, weights, wheels, and logs remain in private validation storage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


FILES = {
    "python-recovery.json": "python-recovery-reviewed/result.json",
    "python-selection.json": "python-selection-reviewed.json",
    "python-torch.json": "python-torch-reviewed.json",
    "browser.json": "browser/main.json",
    "browser-resume.json": "browser/resume.json",
    "browser-torch.json": "browser/torch-comparison.json",
    "browser-inputs.json": "browser/inputs.json",
    "browser-verification.json": "browser/verification.json",
    "node-nn.json": "browser/node-nn.json",
    "node-webgpu.json": "browser/node-webgpu.json",
}
FORBIDDEN = {"model_json", "request_json", "checkpoint_json", "report_json",
             "final_parameters", "train_documents", "validation_documents",
             "generated_bytes", "optimizer_state"}
REQUEST = "208975bab0905a1ee968d5940975259491163680ce7bd2c0132bce23a637e523"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(value):
    if isinstance(value, dict):
        require(not FORBIDDEN.intersection(value), "private payload in public receipt")
        for item in value.values():
            inspect(item)
    elif isinstance(value, list):
        for item in value:
            inspect(item)


def identity(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def publish(private, output, repo):
    values, raw = {}, {}
    for name, source in FILES.items():
        raw[name] = (private / source).read_bytes()
        require(len(raw[name]) <= 1024 * 1024, "unexpectedly large receipt")
        values[name] = json.loads(raw[name])
        inspect(values[name])
    recovery, selection = values["python-recovery.json"], values["python-selection.json"]
    require(recovery["request_sha256"] == selection["request_sha256"] == REQUEST, "wrong request")
    require(recovery["exact_report_equal"] and recovery["exact_checkpoint_equal"]
            and recovery["source_unchanged"] and recovery["interrupted_exit"] == -9, "recovery failed")
    require(selection["cases"] == 9 and selection["stops"] == [0, 4, 8, 12]
            and selection["exact_final_report_and_checkpoint_equal"]
            and selection["selected_artifacts_equal_source_checkpoints"], "selection failed")
    for name in ("python-torch.json", "browser-torch.json"):
        require(values[name]["request_sha256"] == REQUEST
                and values[name]["numerical_checks_passed"], "short numerical comparison failed")
    require(values["browser-verification.json"]["verified"]
            and values["browser.json"]["count"] == 101
            and values["browser-resume.json"]["exact_report"]
            and values["browser-resume.json"]["exact_checkpoint"], "browser verification failed")
    wheel = private / "reviewed-wheels/spiraltorch-0.4.31-cp38-abi3-macosx_11_0_arm64.whl"
    require(identity(wheel)["sha256"] == recovery["wheel_sha256"], "wrong qualified wheel")
    with zipfile.ZipFile(wheel) as archive:
        for name in ("__init__.pyi", "byte_corpus_runs.py"):
            require(archive.read("spiraltorch/" + name)
                    == (repo / "bindings/st-py/spiraltorch" / name).read_bytes(), "stale wheel " + name)
    production = ("crates", "bindings", "tools", "vendor", "Cargo.toml", "Cargo.lock", ".github")
    require(not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *production],
                                        cwd=repo), "uncommitted production source")
    source_paths = ["bindings/st-py/src/nn_resident/byte_study.rs",
                    "bindings/st-py/tests/test_nn_byte_study.py",
                    "bindings/st-py/spiraltorch/__init__.pyi",
                    "tools/qualify_byte_corpus_runs.py", "tools/byte_corpus_study.py"]
    source_paths += [str(p.relative_to(repo)) for p in (Path(__file__).resolve(),
                                                       Path(__file__).with_name("qualify_selection.py").resolve())]
    binaries = [wheel, private / "wasm-webgpu/spiraltorch_wasm_bg.wasm",
                private / "wasm-nn/spiraltorch_wasm_bg.wasm"]
    manifest = {
        "schema": "spiraltorch.language_mlops.integration_receipts.v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "production_source_clean": True,
        "source": {path: identity(repo / path) for path in source_paths},
        "binaries": {str(path.relative_to(private)): identity(path) for path in binaries},
        "logs": {path.name: identity(path) for path in sorted(private.glob("*.log"))},
        "results": {name: identity(private / source) for name, source in FILES.items()},
        "geometric_quality_benefit_established": False,
        "long_run_numerical_gate_superseded": False,
        "package_release_performed": False,
        "scope": "Local receipt consistency and source identity, not independent attestation",
    }
    output.mkdir(exist_ok=False)
    for name, data in raw.items():
        with (output / name).open("xb") as handle:
            handle.write(data)
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"published_receipts": len(raw), "source_commit": manifest["source_commit"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    publish(args.private.resolve(), args.output, args.repo.resolve())

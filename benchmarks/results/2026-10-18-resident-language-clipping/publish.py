"""Publish only synthetic clipping reports and hashes, never training state."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


SOURCES = (
    "crates/st-kernel-contracts/src/gradient_clip.rs",
    "crates/st-kernel-contracts/src/gradient_clip.wgsl",
    "crates/st-backend-wgpu/src/resident_training/parameters.rs",
    "crates/st-backend-wgpu/src/resident_training/parameters/clipping.rs",
    "crates/st-backend-wgpu/src/resident_training/parameters/clipping/tests.rs",
    "crates/st-backend-wgpu/src/resident_training/shaders/parameter_clip.wgsl",
    "crates/st-backend-wgpu/src/resident_training/graph/autograd/learner/clipping.rs",
    "crates/st-backend-wgpu/src/resident_training/graph/autograd/learner/clipping/tests.rs",
    "crates/st-backend-wgpu/src/shaders/graph_gradient_clip.wgsl",
    "crates/st-backend-wgpu/examples/resident_parameter_clip.rs",
    "crates/st-backend-wgpu/examples/support/resident_parameter_clip.rs",
    "crates/st-backend-wgpu/tests/parameter_clip_browser.html",
    "crates/st-nn/src/resident/byte_decoder/training.rs",
    "crates/st-nn/src/resident/byte_decoder/training/tests.rs",
    "crates/st-nn/src/resident/byte_decoder/training/tests/clipping.rs",
    "tools/check_resident_parameter_clip.py",
    "tools/test_resident_parameter_clip.py",
)


def identity(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    repo = output.parents[2]
    spec = importlib.util.spec_from_file_location("clip_check", repo / SOURCES[-2])
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    reports = {}
    for name, path in (("native", "native-reviewed.json"), ("browser", "browser-reviewed/main.json")):
        raw = (args.private / path).read_bytes()
        report = json.loads(raw)
        for field in ("strided", "foreign_and_stale", "frozen_guard_rejection", "adamw_recovery"):
            if report.get(field) is not True:
                raise ValueError("missing GPU control: " + field)
        if report.get("quality_claim") is not False:
            raise ValueError("unexpected quality claim")
        checked = checker.check(report)
        checked["input_sha256"] = identity(raw)["sha256"]
        reports[name + ".json"] = raw
        reports[name + "-torch.json"] = (json.dumps(checked, indent=2, allow_nan=False) + "\n").encode()
    manifest = dict(
        schema="spiraltorch.resident_parameter_clip.validation.v1",
        parent_commit="18a9ed5978b0fd1804556f81063266d6317072d9",
        source={name: identity((repo / name).read_bytes()) for name in SOURCES},
        reports={name: identity(raw) for name, raw in reports.items()},
        private_logs={p.name: identity(p.read_bytes()) for p in sorted(args.private.glob("*.log"))},
        browser_inputs=json.loads((args.private / "browser-reviewed/inputs.json").read_bytes()),
        retained_initial_attempts={
            "test_compile_error": identity((args.private / "shader-check.log").read_bytes()),
            "browser_signed_zero_serialization": identity((args.private / "browser/main.json").read_bytes()),
            "native_floor_counterexample": identity((args.private / "floor-before-fix.log").read_bytes()),
            "nonzero_sgd_subnormal_flush": identity((args.private / "floor-after-fix.log").read_bytes()),
        },
        quality_claim=False,
        throughput_claim=False,
        study_recipe_persistence=False,
        packaged_client_exposure=False,
    )
    reports["validation.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    if any((output / name).exists() for name in reports):
        raise FileExistsError("refusing to replace published evidence")
    for name, raw in reports.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    print(json.dumps(dict(published=list(reports))))


if __name__ == "__main__":
    main()

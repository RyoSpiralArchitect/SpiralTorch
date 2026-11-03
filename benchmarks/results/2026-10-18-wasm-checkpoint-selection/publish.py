"""Verify this bounded qualification before publishing results, never raw inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


FILES = {
    "synthetic-nn.json": "nn-reviewed-node.json",
    "synthetic-webgpu.json": "webgpu-reviewed-node.json",
    "retained-nn.json": "retained-nn-reviewed-node.json",
    "retained-webgpu.json": "retained-webgpu-reviewed-node.json",
    "browser.json": "browser/main.json",
}
SOURCES = [
    ".github/workflows/ci.yml", "Cargo.lock",
    "crates/st-nn/src/resident.rs", "crates/st-nn/src/resident/byte_study.rs",
    "crates/st-nn/src/resident/byte_study/selection.rs",
    "crates/st-nn/src/resident/byte_study/selection/tests.rs",
    "bindings/st-py/src/nn_resident/byte_study.rs",
    "bindings/st-wasm/src/byte_lm.rs", "bindings/st-wasm/types/spiraltorch-wasm.d.ts",
    "docs/byte_corpus_checkpoint_selection.md", "docs/language_mlops.md",
    "docs/wasm_byte_language_mlops.md",
    *["bindings/st-wasm/tests/" + name for name in (
        "byte_lm_fixture.py", "byte_lm_node.mjs", "byte_lm_boundaries.mjs",
        "byte_lm_selection.mjs", "byte_lm_selection_types.cjs", "byte_lm_selection_browser.html")],
]


def require(value, message):
    if not value:
        raise ValueError(message)


def byte_identity(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def identity(path):
    return byte_identity(path.read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe(value):
    if isinstance(value, dict):
        forbidden = {"model_json", "artifact_json", "request_json", "checkpoint_json",
                     "checkpoint_jsons", "train_documents", "validation_documents",
                     "generated_bytes", "optimizer_state", "parameters", "windows"}
        require(not forbidden.intersection(value), "private payload in receipt")
        for item in value.values():
            safe(item)
    elif isinstance(value, list):
        for item in value:
            safe(item)
    elif isinstance(value, str):
        require("/Users/" not in value and "/tmp/" not in value, "private path in receipt")


def verify(repo, private):
    receipts = {name: json.loads((private / path).read_bytes()) for name, path in FILES.items()}
    fixture = json.loads((private / "retained.json").read_bytes())
    selection = fixture["selection"]
    require(selection["source"] == "retained_learning", "retained models required")
    require(len(selection["checkpoint_jsons"]) == 4 and len(selection["expected_indices"]) == 9,
            "retained coverage")
    expected = selection["expected"]
    expected_names = list(selection["expected_indices"])
    for name, report in receipts.items():
        safe(report)
        actual = report["selection"]
        require(actual["count"] == len(actual["checks"]), "selection check count")
        if name.startswith("synthetic"):
            require(actual["source"] == "synthetic_contract" and actual["count"] == 26,
                    "synthetic check coverage")
        else:
            require(actual["source"] == "retained_learning" and actual["count"] == 31,
                    "retained check coverage")
            require([model["case_name"] for model in actual["models"]] == expected_names,
                    "retained case coverage")
            for model in actual["models"]:
                chosen = json.loads(expected[model["case_name"]]["report_json"])["selected"]
                require(model["exact_saved_artifact"] and model["exact_python_selection"],
                        "selection equality")
                require(all(model[key] == chosen[key] for key in
                            ("candidate_index", "completed_updates", "artifact_id")),
                        "selected identity")
        if name != "browser.json":
            feature = "webgpu" if "webgpu" in name else "nn"
            require(report["schema"] == "spiraltorch.packaged_byte_mlops.cpu.v2"
                    and report["features"] == feature and report["feature_matrix_passed"],
                    "package feature matrix")
            require(report["count"] == len(report["checks"]) == (30 if feature == "webgpu" else 20),
                    "transport coverage")
    browser = receipts["browser.json"]
    require(browser["passed"] and browser["adapter_requests"] == 1
            and browser["selection_adapter_requests"] == 0, "browser execution")
    require(browser["new_training_updates"] == 0 and not browser["quality_claim"]
            and not browser["physical_adapter_verified"], "scope differs")
    require([model["case_name"] for model in browser["models"]] == expected_names, "browser cases")
    for model, selected in zip(browser["models"], browser["selection"]["models"], strict=True):
        require(all(model[key] == value for key, value in selected.items()), "browser selection identity")
        require(model["exact_direct_inference"] and model["exact_direct_generation"], "browser equality")
        for key in ("logits_f32le_sha256", "generation_sha256"):
            require(re.fullmatch("[a-f0-9]{64}", model[key]), "invalid output identity")
    require(len({model["logits_f32le_sha256"] for model in browser["models"]}) == 9,
            "model outputs collapsed")
    manifest = json.loads((private / "browser/inputs.json").read_bytes())
    require(identity(private / "retained.json") == identity(private / "browser/input.json")
            == manifest["fixture"], "browser fixture identity")
    for path, value in manifest["original_source"].items():
        require(identity(repo / path) == value, "browser source changed")
    for path, value in manifest["source"].items():
        require(identity(private / "browser/site" / path) == value, "staged asset changed")
    for path, value in manifest["package"].items():
        require(identity(private / "webgpu-reviewed-web" / path) == value, "browser package changed")
    log = (private / "rust-study-tests.log").read_text()
    require("test result: ok. 71 passed; 0 failed" in log, "Rust study tests incomplete")
    require("Finished" in (private / "python-check.log").read_text(), "Python check incomplete")
    type_logs = [feature + "-reviewed-" + suffix + ".log"
                 for feature in ("nn", "webgpu") for suffix in ("types", "web-types")]
    for name in type_logs:
        require((private / name).read_text().count("TypeScript contract passed:") == 2,
                "declaration check incomplete")
    paths = SOURCES + sorted(str(path.relative_to(repo)) for path in
                             Path(__file__).resolve().parent.iterdir()
                             if path.suffix in (".py", ".md"))
    result = dict(
        schema="spiraltorch.wasm_checkpoint_selection.validation.v1",
        source={name: identity(repo / name) for name in paths},
        results={name: byte_identity(encoded(value)) for name, value in receipts.items()},
        source_receipts={name: identity(private / path) for name, path in FILES.items()},
        packages={feature: identity(private / (feature + "-reviewed-node") / "spiraltorch_wasm_bg.wasm")
                  for feature in ("nn", "webgpu")},
        browser_inputs=manifest, retained_input=identity(private / "retained.json"),
        retained_source=selection["expected_source"],
        logs={name: identity(private / name) for name in
              ["rust-study-tests.log", "python-check.log", *type_logs]},
        rust_cpu_tests=71, final_independent_rereview_complete=False,
        new_training_updates=0, quality_claim=False)
    safe(result)
    return receipts, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "private", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipts, manifest = verify(args.repo.resolve(), args.private.resolve())
    payloads = {name: encoded(value) for name, value in {**receipts, "manifest.json": manifest}.items()}
    args.output.mkdir(exist_ok=False)
    for name, raw in payloads.items():
        with (args.output / name).open("xb") as output:
            output.write(raw)
    print(json.dumps(dict(verified=True, retained_models=9, new_training_updates=0)))


if __name__ == "__main__":
    main()

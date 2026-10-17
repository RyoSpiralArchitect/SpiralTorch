"""Pinned Attention-kernel follow-up using the existing language comparator."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


PINS = {
    "request": "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
    "torch": "96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b",
    "before": "6bb2a87dcf87ef83ef209402eb713b8d8bc196ea6ceca560136341982b427317",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs(paths):
    raw = {name: path.read_bytes() for name, path in paths.items()}
    if {name: digest(value) for name, value in raw.items()} != PINS:
        raise ValueError("pinned input drift")
    return raw


def check_sources(sources):
    if any(digest(path.read_bytes()) != expected for path, expected in sources.items()):
        raise ValueError("comparison or kernel source drift")


def native_artifact(messages, repo):
    artifacts = [item for line in messages.splitlines() if line.strip()
                 for item in [json.loads(line)]
                 if item.get("reason") == "compiler-artifact"
                 and item.get("target", {}).get("name") == "resident_byte_learning"
                 and item.get("target", {}).get("kind") == ["example"]
                 and item.get("executable")]
    if len(artifacts) != 1:
        raise ValueError("native build must report exactly one executable")
    artifact = artifacts[0]
    binary = Path(artifact["executable"]).resolve()
    if not binary.is_relative_to(repo / "target") or artifact["profile"]["opt_level"] != "0":
        raise ValueError("native build must use the repository debug profile")
    return binary, artifact["profile"]


def build_native(repo, output, sources, helpers):
    check_sources(sources)
    compiler = subprocess.check_output(["rustc", "+1.99.0", "-Vv"], cwd=repo).decode()
    command = ["cargo", "+1.99.0", "build", "--locked", "--profile", "dev",
               "--target-dir", str(repo / "target"), "-p", "st-nn",
               "--no-default-features", "--features", "wgpu", "--example",
               "resident_byte_learning", "--message-format=json"]
    with helpers.private_writer(output / "build.jsonl") as stdout, helpers.private_writer(output / "build.log") as stderr:
        subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr, check=True)
    check_sources(sources)
    binary, profile = native_artifact((output / "build.jsonl").read_text(), repo)
    binary_hash = digest(binary.read_bytes())
    receipt = {"compiler": compiler, "command": command, "profile": profile,
               "executable": str(binary.relative_to(repo)), "binary_sha256": binary_hash,
               "sources": {str(path.relative_to(repo)): value for path, value in sources.items()}}
    helpers.write(output / "build-receipt.json", receipt)
    return binary, binary_hash


def verify(paths, raw, sources, binary, binary_hash, request):
    inputs(paths)
    if request.read_bytes() != raw["request"]:
        raise ValueError("executed request drift")
    check_sources(sources)
    if digest(binary.read_bytes()) != binary_hash:
        raise ValueError("executed binary drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "prior", "reference", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    repo, prior, reference, output = (
        getattr(args, name).resolve() for name in ("repo", "prior", "reference", "output")
    )
    paths = {"request": prior / "language/request.json", "before": prior / "language/native.json",
             "torch": reference / "w16-seed11/torch.json"}
    raw = inputs(paths)
    helper_path = repo / "benchmarks/results/2026-10-16-attention-adamw-numerics/language_probe.py"
    helper_bytes = helper_path.read_bytes()
    helpers = module(helper_path)
    source_paths = sorted({*repo.glob("tools/byte_corpus*.py"),
                           *repo.glob("benchmarks/results/2026-10-16-language-capacity-budget/*.py"),
                           *repo.glob("crates/st-kernel-contracts/src/adamw.*"), helper_path,
                           repo / "Cargo.toml", repo / "Cargo.lock",
                           repo / "crates/st-nn/examples/resident_byte_learning.rs",
                           repo / "crates/st-backend-wgpu/src/resident_tensor/shaders/attention_backward.wgsl",
                           Path(__file__).resolve(), Path(__file__).with_name("PLAN.md").resolve()})
    sources = {path: digest(path.read_bytes()) for path in source_paths}
    if sources[helper_path] != digest(helper_bytes):
        raise ValueError("runner helper changed while loading")
    output.mkdir(mode=0o700, exist_ok=False)
    for filename, data in [("request.json", raw["request"]), ("runner.py", Path(__file__).read_bytes())]:
        with helpers.private_writer(output / filename) as stream:
            stream.write(data)
    binary, binary_hash = build_native(repo, output, sources, helpers)
    build_receipt_hash = digest((output / "build-receipt.json").read_bytes())
    helpers.write(output / "started.json", {
        "pins": PINS, "binary_sha256": binary_hash,
        "build_receipt_sha256": build_receipt_hash,
        "sources": {str(path.relative_to(repo)): value for path, value in sources.items()},
        "comparison_baseline": "f27ca0aae native debug uninterrupted run; AdamW already repaired",
    })
    check = lambda: verify(paths, raw, sources, binary, binary_hash, output / "request.json")
    check()
    with helpers.private_writer(output / "native.json") as stdout, helpers.private_writer(output / "native.log") as stderr:
        subprocess.run([str(binary), str(output / "request.json"), "--checkpoint-out",
                        str(output / "checkpoint.json")], stdout=stdout, stderr=stderr, check=True)
    check()
    protocol = module(repo / "tools/byte_corpus_study.py")
    diagnostics = module(repo / "benchmarks/results/2026-10-16-language-capacity-budget/diagnose.py")
    expected = protocol.decode_json(raw["torch"])
    actual = (output / "native.json").read_bytes()
    result = {"schema": "spiraltorch.byte_corpus.attention_numerical_followup.v1", "pins": PINS,
              "binary_sha256": binary_hash, "actual_sha256": digest(actual),
              "build_receipt_sha256": build_receipt_hash,
              "checkpoint_sha256": digest((output / "checkpoint.json").read_bytes()),
              "before": helpers.summarize(protocol, diagnostics, raw["request"], expected,
                                          protocol.decode_json(raw["before"])),
              "after": helpers.summarize(protocol, diagnostics, raw["request"], expected,
                                         protocol.decode_json(actual)),
              "scope": "One predeclared numerical follow-up; no promotion or retroactive qualification"}
    check()
    helpers.write(output / "comparison.json", result)


if __name__ == "__main__":
    main()

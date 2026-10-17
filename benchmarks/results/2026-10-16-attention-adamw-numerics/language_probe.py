"""One predeclared language follow-up; preserve strict failures and private weights."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


PINS = {
    "request.json": "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
    "torch.json": "96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b",
    "native/report-1024.json": "6f4ca125c9d4ded06d8cf3fb06f518992efa66dc90e7f912c38ba62efb2edcf5",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def private_writer(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def write(path, value):
    with private_writer(path) as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def inputs(prior):
    raw = {name: (prior / name).read_bytes() for name in PINS}
    if {name: digest(value) for name, value in raw.items()} != PINS:
        raise ValueError("pinned prior input drift")
    return raw


def verify_drift(repo, prior, output, binary, binary_hash, raw, identities):
    if digest(binary.read_bytes()) != binary_hash or any(digest((repo / path).read_bytes()) != expected for path, expected in identities.items()):
        raise ValueError("executed build or comparison source drift")
    if (output / "request.json").read_bytes() != raw["request.json"]:
        raise ValueError("executed request drift")
    inputs(prior)


def summarize(protocol, diagnostics, raw, reference, actual):
    request = protocol.decode_json(raw)
    try:
        comparison = protocol.compare(raw, reference, actual)
        gate = {"passed": True, "comparison": comparison}
    except ValueError as error:
        # Keep raw parameter-bearing errors private, not in the public summary.
        gate = {"passed": False, "error_sha256": digest(str(error).encode()),
                "scope": "Strict protocol rejected; diagnostics are not qualification"}
    names = [case["name"] for case in request["cases"]]
    if any([case["name"] for case in report["cases"]] != names for report in (reference, actual)):
        raise ValueError("case coverage drift")
    cases = []
    for case, got, expected in zip(request["cases"], actual["cases"], reference["cases"]):
        descriptors = case["parameters"]
        if not (len(descriptors) == len(got["final_parameters"]) == len(expected["final_parameters"])):
            raise ValueError("parameter coverage drift")
        parameters = []
        for index, descriptor in enumerate(descriptors):
            measure = diagnostics.errors(got["final_parameters"][index], expected["final_parameters"][index], protocol)
            item = {"name": descriptor["name"], "values": measure}
            if descriptor["name"].endswith("qkv.bias"):
                item["qkv_failure_counts"] = diagnostics.qkv_parts(measure["failed_indices"], 16)
            for moment in ("first", "second"):
                item[moment] = diagnostics.errors(got["optimizer_state"]["slots"][index][moment],
                                                 expected["optimizer_state"]["slots"][index][moment], protocol)
            parameters.append(item)
        cases.append({"name": case["name"], "parameters": parameters,
                      "training": diagnostics.errors([v["ce"] for v in got["training"]],
                                                     [v["ce"] for v in expected["training"]], protocol),
                      "validation": diagnostics.errors([v for p in got["validation"] for v in p["batch_losses"]],
                                                       [v for p in expected["validation"] for v in p["batch_losses"]], protocol)})
    return {"strict_gate": gate, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "prior", "output", "binary"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    repo, prior, output, binary = (getattr(args, flag).resolve() for flag in ("repo", "prior", "output", "binary"))
    raw = inputs(prior)
    output.mkdir(mode=0o700, exist_ok=False)
    sources = sorted({*repo.glob("tools/byte_corpus*.py"),
                      *repo.glob("tools/resident_byte*.py"),
                      *repo.glob("crates/st-kernel-contracts/src/adamw.*"),
                      *repo.glob("benchmarks/results/2026-10-16-language-capacity-budget/*.py"),
                      Path(__file__).resolve()})
    identities = {str(path.relative_to(repo)): digest(path.read_bytes()) for path in sources}
    binary_hash = digest(binary.read_bytes())
    write(output / "started.json", {"pins": PINS, "binary_sha256": binary_hash, "sources": identities,
                                   "plan_sha256": digest(Path(__file__).with_name("PLAN.md").read_bytes())})
    # The child reads the exact validated bytes, not a mutable prior experiment path.
    with private_writer(output / "request.json") as stream:
        stream.write(raw["request.json"])
    with private_writer(output / "native.json") as stdout, private_writer(output / "native.log") as stderr:
        subprocess.run([str(binary), str(output / "request.json"), "--checkpoint-out",
                        str(output / "checkpoint.json")], stdout=stdout, stderr=stderr, check=True)
    verify_drift(repo, prior, output, binary, binary_hash, raw, identities)
    protocol = module(repo / "tools/byte_corpus_study.py")
    diagnostics = module(repo / "benchmarks/results/2026-10-16-language-capacity-budget/diagnose.py")
    reference = protocol.decode_json(raw["torch.json"])
    actual_raw = (output / "native.json").read_bytes()
    result = {"schema": "spiraltorch.byte_corpus.adamw_numerical_followup.v1", "pins": PINS,
              "binary_sha256": binary_hash, "actual_sha256": digest(actual_raw),
              "checkpoint_sha256": digest((output / "checkpoint.json").read_bytes()),
              "before": summarize(protocol, diagnostics, raw["request.json"], reference,
                                  protocol.decode_json(raw["native/report-1024.json"])),
              "after": summarize(protocol, diagnostics, raw["request.json"], reference, protocol.decode_json(actual_raw)),
              "scope": "One predeclared seed; no promotion or retroactive qualification"}
    verify_drift(repo, prior, output, binary, binary_hash, raw, identities)
    write(output / "comparison.json", result)


if __name__ == "__main__":
    main()

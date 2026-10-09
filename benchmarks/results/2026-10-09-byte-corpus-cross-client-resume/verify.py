"""Verify this frozen cross-client resume experiment; no model mathematics."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume = load("verify_byte_corpus_resume")
study = load("byte_corpus_study")
require, exact, f32 = resume.require, resume.exact, resume.f32


def endpoint(request, request_hash, checkpoint, report, cursor):
    require(checkpoint["schema"] == "spiraltorch.byte_corpus.checkpoint.v1", "checkpoint schema")
    require(exact(checkpoint["completed_updates"], cursor), "checkpoint cursor")
    require(checkpoint["request_sha256"] == report["request_sha256"] == request_hash, "request identity")
    require(report["engine"] == "spiraltorch", "engine")
    expected_schema = study.RESULT if cursor == 128 else "spiraltorch.byte_corpus.partial.v1"
    require(report["schema"] == expected_schema, "report schema")
    require(len(checkpoint["cases"]) == len(report["cases"]) == len(request["cases"]), "case count")
    for spec, saved, row in zip(request["cases"], checkpoint["cases"], report["cases"]):
        require(saved["name"] == spec["name"], "saved case name")
        require(all(exact(row[key], spec[key]) for key in ("name", "seed", "geometry")), "case identity")
        require(exact([p["revision"] for p in row["training"]], list(range(1, cursor + 1))), "revisions")
        stored = resume.stored_parameters(json.loads(saved["model_json"]), cursor)
        require(len(stored) == len(row["final_parameters"]) == len(spec["parameters"]), "tensor count")
        for initial, parameter, values in zip(spec["parameters"], stored, row["final_parameters"]):
            require(exact(parameter["shape"], initial["shape"]), "tensor shape")
            require(len(values) == len(initial["values"]), "tensor length")
            require(list(map(f32, parameter["values"])) == list(map(f32, values)), "saved tensor bits")
        require(len(saved["training"]) == len(row["training"]), "saved training length")
        for a, b in zip(saved["training"], row["training"]):
            require(exact(a["revision"], b["revision"]) and f32(a["ce"]) == f32(b["ce"]), "saved loss")
        require(len(saved["validation"]) == len(row["validation"]), "saved validation length")
        for a, b in zip(saved["validation"], row["validation"]):
            require(exact(a["revision"], b["revision"]), "saved validation revision")
            require(list(map(f32, a["batch_losses"])) == list(map(f32, b["batch_losses"])), "saved validation")


def verify(raw, reference, partial, start, result, end, source_backend, target_backend):
    request, request_hash = json.loads(raw), study.digest(raw)
    require(len(request["train_batches"]) == 128 and request["checkpoint_every"] == 64, "fixed recipe")
    require(source_backend != target_backend, "distinct clients")
    require(f"backend: {source_backend}" in partial["adapter"], "source adapter")
    require(f"backend: {target_backend}" in result["adapter"], "target adapter")
    endpoint(request, request_hash, start, partial, 37)
    endpoint(request, request_hash, end, result, 128)
    for prefix, full in zip(partial["cases"], result["cases"]):
        require(exact(prefix["training"], full["training"][:37]), "training prefix")
        require(exact(prefix["validation"], [p for p in full["validation"] if p["revision"] <= 37]),
                "validation prefix")
    comparison = study.compare(raw, reference, result)
    return {"source_adapter": partial["adapter"], "target_adapter": result["adapter"],
            "checkpoint_parameter_bits_match_reports": True, "retained_prefix_exact": True,
            "source_cursor": 37, "target_cursor": 128, "torch_comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "reference", "resume_directory", "handoff_directory", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    inventory = json.loads((ROOT / "benchmarks/results/2026-10-09-byte-corpus-resume/validation.json").read_bytes())
    hashes = {}

    def read(path, frozen=False):
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        hashes[path.name] = checksum
        if frozen:
            require(checksum == inventory["local_artifact_inventory"][path.name]["sha256"], "frozen artifact changed")
        return json.loads(raw)

    raw = args.request.read_bytes()
    hashes["request.json"] = study.digest(raw)
    reference = read(args.reference)
    directions = {}
    for source, target, suffix, source_backend, target_backend in (
        ("native", "browser", "reviewed", "Metal", "BrowserWebGpu"),
        ("browser", "native", "opaque", "BrowserWebGpu", "Metal"),
    ):
        direction = f"{source}-to-{target}"
        partial = read(args.resume_directory / f"{source}-partial-37-{suffix}.json", frozen=True)
        start = read(args.resume_directory / f"{source}-checkpoint-37-{suffix}.json", frozen=True)
        result = read(args.handoff_directory / f"{direction}.json")
        end = read(args.handoff_directory / f"{direction}-checkpoint-128.json")
        directions[direction] = verify(raw, reference, partial, start, result, end, source_backend, target_backend)
        target_suffix = "reviewed" if target == "native" else "opaque"
        baseline = read(args.resume_directory / f"{target}-full-{target_suffix}.json", frozen=True)
        # Descriptive only: cross-client bitwise equality is not an acceptance gate.
        directions[direction]["target_uninterrupted_report_exact"] = exact(result, baseline)
        directions[direction]["target_uninterrupted_parameter_max_abs_difference"] = max(
            abs(a - b) for row, base in zip(result["cases"], baseline["cases"])
            for values, original in zip(row["final_parameters"], base["final_parameters"])
            for a, b in zip(values, original))
        # Mutations stay in memory; saved artifacts are never edited by controls.
        for label in ("prefix", "weights", "cursor", "identity"):
            changed_result, changed_end = copy.deepcopy(result), copy.deepcopy(end)
            if label == "prefix":
                changed_result["cases"][0]["training"][0]["ce"] += 1e-10
            elif label == "weights":
                model = json.loads(changed_end["cases"][0]["model_json"])
                model["model"]["token"]["values"][0] += 1e-6
                changed_end["cases"][0]["model_json"] = json.dumps(model)
            elif label == "cursor":
                changed_end["completed_updates"] = True
            else:
                changed_end["request_sha256"] = "wrong"
            try:
                verify(raw, reference, partial, start, changed_result, changed_end, source_backend, target_backend)
            except ValueError:
                continue
            raise ValueError("negative control accepted: " + label)
    record = {"schema": "spiraltorch.byte_corpus.cross_client_resume.v1", "passed": True,
              "directions": directions, "input_sha256": hashes, "negative_controls_rejected": 8,
              "scope": "Same-host native Metal and actual browser WebGPU handoff, fixed recipe; numerical agreement, not cross-device bitwise equivalence, runtime attestation, speed or quality evidence"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Verified both handoff directions, six cases each, and eight negative controls")


if __name__ == "__main__":
    main()

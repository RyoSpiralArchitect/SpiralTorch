#!/usr/bin/env python3
"""Check retained v3 reports and separately executed first-update v4 controls."""
import argparse
import importlib.util
import math
from pathlib import Path
import struct


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study, resume = load("byte_corpus_study"), load("verify_byte_corpus_resume")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(values):
    require(all(type(x) in (float, int) and math.isfinite(x) for x in values), "invalid parameter scalar")
    return [struct.pack("<f", x) for x in values]


def validate_report(report, request, request_hash, cursor):
    version = study.request_version(request)
    total = len(request["train_batches"])
    kind = "result" if cursor == total else "partial"
    require(report["schema"] == f"spiraltorch.byte_corpus.{kind}.v{version}"
            and report["engine"] == "spiraltorch" and report["request_sha256"] == request_hash,
            "request-derived report identity")
    require(len(report["cases"]) == len(request["cases"]), "request-derived case count")
    if version == 4:
        require(resume.exact(report.get("bias_calibration"), request["bias_calibration"]), "request-derived calibration")
    revisions = [0] + [r for r in range(1, cursor + 1) if r % request["checkpoint_every"] == 0 or r == total]
    for row, spec in zip(report["cases"], request["cases"]):
        for key in ("name", "seed", "geometry", "geometry_update", "pair_metric"):
            require(key in row and resume.exact(row[key], spec.get(key)), "request-derived case metadata")
        if version == 4:
            require(row.get("bias_initialization") == spec["bias_initialization"], "request-derived initialization")
        require(resume.exact(row["parameter_tensors"], len(spec["parameters"]))
                and resume.exact(row["parameter_scalars"], sum(len(p["values"]) for p in spec["parameters"]))
                and resume.exact(row["trainable_parameter_scalars"], study.trainable_scalars(spec)), "request-derived parameter counts")
        require(resume.exact([p["revision"] for p in row["training"]], list(range(1, cursor + 1))), "request-derived complete training history")
        for point in row["training"]:
            bits([point["ce"]])
        require(resume.exact([p["revision"] for p in row["validation"]], revisions), "request-derived evaluation schedule")
        for point in row["validation"]:
            losses = point["batch_losses"]
            require(len(losses) == len(request["validation_batches"]), "request-derived evaluation coverage")
            bits(losses)
            require(resume.exact(point["target_bytes"], request["config"]["batch"] * request["config"]["steps"] * len(losses)), "request-derived target count")
            mean = math.fsum(losses) / len(losses)
            study.close([point["mean_ce"], point["bits_per_byte"]], [mean, mean / math.log(2)], "evaluation aggregation")
        require(len(row["final_parameters"]) == len(spec["parameters"]), "request-derived tensor count")
        for desc, values in zip(spec["parameters"], row["final_parameters"]):
            require(len(values) == len(desc["values"]), "request-derived tensor shape")
            packed = bits(values)
            if study.frozen(spec) and desc["name"].startswith("geometry."):
                require(packed == bits(desc["values"]), "request-derived frozen bits")


def verify(request_raw, source_raw, full, retained, first):
    request, source = map(study.decode_json, (request_raw, source_raw))
    require(study.request_version(request) == 4 and study.request_version(source) == 3, "control schema")
    require(request["bias_calibration"]["source_request_sha256"] == study.digest(source_raw), "source binding")
    validate_report(full, request, study.digest(request_raw), len(request["train_batches"]))
    validate_report(retained, source, study.digest(source_raw), len(source["train_batches"]))
    validate_report(first, request, study.digest(request_raw), 1)
    require(full["schema"] == "spiraltorch.byte_corpus.result.v4"
            and retained["schema"] == "spiraltorch.byte_corpus.result.v3"
            and retained["engine"] == "spiraltorch"
            and retained["request_sha256"] == study.digest(source_raw), "baseline identity")
    require(full["adapter"] == retained["adapter"] == first["adapter"], "runtime differs")
    for report in (full, first):
        require(report["engine"] == "spiraltorch" and report["request_sha256"] == study.digest(request_raw)
                and resume.exact(report.get("bias_calibration"), request["bias_calibration"]), "v4 report identity")
        require([c["name"] for c in report["cases"]] == [c["name"] for c in request["cases"]], "report order/coverage")
    require(len(retained["cases"]) == len(source["cases"]), "retained case coverage")
    old_names = []
    for old, new, spec in zip(retained["cases"], full["cases"], source["cases"]):
        copy = dict(new)
        require(copy.pop("bias_initialization") == study.ORIGINAL and copy["name"] == spec["name"], "inherited identity")
        require(resume.exact(copy, old), "inherited full case report changed")
        old_names.append(old["name"])
    for spec, result, prefix in zip(request["cases"], full["cases"], first["cases"]):
        require(all(resume.exact(prefix[k], result[k]) for k in ("name", "seed", "geometry", "geometry_update", "pair_metric", "bias_initialization", "parameter_tensors", "parameter_scalars", "trainable_parameter_scalars")), "first-step case metadata")
        require(resume.exact(prefix["training"], result["training"][:1])
                and [p["revision"] for p in prefix["training"]] == [1]
                and resume.exact(prefix["validation"], [p for p in result["validation"] if p["revision"] <= 1]), "first-step prefix or evaluation schedule")
        require(len(prefix["final_parameters"]) == len(spec["parameters"]), "first-step parameter count")
        for desc, value in zip(spec["parameters"], prefix["final_parameters"]):
            require(len(value) == len(desc["values"]), "first-step parameter shape")
            bits(value)
    pairs = []
    for seed in sorted({c["seed"] for c in request["cases"]}):
        for metric, initialization in [(study.POINCARE, study.ORIGINAL), (study.FLAT, study.ORIGINAL), (study.FLAT, study.MATCHED)]:
            indices = [i for i, c in enumerate(request["cases"]) if c["seed"] == seed
                       and c.get("pair_metric") == metric and c["bias_initialization"] == initialization]
            trained = next(i for i in indices if not study.frozen(request["cases"][i]))
            fixed = next(i for i in indices if study.frozen(request["cases"][i]))
            a, b = first["cases"][trained], first["cases"][fixed]
            require(resume.exact(a["validation"][0], b["validation"][0]) and resume.exact(a["training"], b["training"]), "same-initialization forward differs")
            moving = []
            for desc, learned, frozen_values in zip(request["cases"][trained]["parameters"], a["final_parameters"], b["final_parameters"]):
                original = bits(desc["values"])
                if desc["name"].startswith("geometry."):
                    require(bits(frozen_values) == original, "frozen initial bits changed")
                    if bits(learned) != original:
                        moving.append(desc["name"])
                else:
                    require(bits(learned) == bits(frozen_values), "first backbone update differs")
                    if desc["name"] in ("token_embedding", "position_embedding"):
                        require(bits(learned) != original, "embedding did not learn")
            require(bool(moving), "trained geometry did not change")
            pairs.append(dict(seed=seed, pair_metric=metric, bias_initialization=initialization,
                              first_backbone_update_bits_identical=True, both_embeddings_moved=True,
                              frozen_initial_bits_preserved=True, moving_geometry_families=moving))
    return dict(schema="spiraltorch.byte_corpus.calibrated_controls.v1", passed=True,
                request_sha256=study.digest(request_raw), source_request_sha256=study.digest(source_raw),
                inherited_full_reports_exact=old_names, pairs=pairs,
                scope="Same-runtime inheritance and separately executed first-update controls, not quality or speed evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "source", "full", "retained", "first", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    raw = {n: getattr(args, n).read_bytes() for n in ("request", "source", "full", "retained", "first")}
    result = verify(raw["request"], raw["source"], *(study.decode_json(raw[n]) for n in ("full", "retained", "first")))
    result["input_sha256"] = {n: study.digest(v) for n, v in raw.items()}
    study.write_new(args.output, result)
    print(f"Verified {len(result['inherited_full_reports_exact'])} inherited reports and {len(result['pairs'])} first-update pairs")


if __name__ == "__main__":
    main()

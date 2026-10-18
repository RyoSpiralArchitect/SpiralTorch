"""Verify paired probe receipts, then summarize full logits without a foreign ML runtime."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, mode):
    report = json.loads(path.read_text())
    if report["schema"] != "spiraltorch.native_gpt2_probe.v1" or report["mode"] != mode:
        raise ValueError("wrong probe schema or mode")
    if report["byteorder"] not in {"little", "big"}:
        raise ValueError("unknown float byte order")
    rows = {}
    for row in report["rows"]:
        key = row["case"], row["arm"]
        if key in rows:
            raise ValueError("duplicate case/arm")
        rows[key] = row
    expected = {(case, arm) for case in ("meaning", "models", "spiraltorch") for arm in ("base", "wave")}
    if set(rows) != expected:
        raise ValueError("missing or unexpected case/arm")
    return report, rows


def values(directory, row, byteorder):
    name = row["logits_file"]
    if Path(name).name != name or not name.endswith(".f32"):
        raise ValueError("logits must be a sibling f32 file")
    shape = row["shape"]
    if shape != [1, len(row["ids"]), 50257]:
        raise ValueError("unexpected probe logits shape")
    count = math.prod(shape)
    if count > 1_000_000:
        raise ValueError("logit probe budget exceeded")
    with (directory / name).open("rb") as stream:
        data = stream.read(count * 4 + 1)
    if len(data) != count * 4 or hashlib.sha256(data).hexdigest() != row["logits_sha256"]:
        raise ValueError("logit length or digest mismatch")
    result = array.array("f")
    if result.itemsize != 4:
        raise ValueError("comparison requires IEEE float32 array storage")
    result.frombytes(data)
    if byteorder != sys.byteorder:
        result.byteswap()
    if not all(math.isfinite(x) for x in result) or not math.isfinite(row["nll"]):
        raise ValueError("non-finite result")
    return result


def errors(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("empty or unmatched arrays")
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    return {"max_abs": max(abs(d) for d in delta), "rms": math.sqrt(math.fsum(d * d for d in delta) / len(delta))}


def compare(native_path, reference_path):
    native, nr = load(native_path, "native")
    reference, rr = load(reference_path, "reference")
    for key in ("source_sha256", "steps", "base_model_identity", "geometry_artifact_id", "scope"):
        if native[key] != reference[key]:
            raise ValueError("probe identity mismatch: " + key)
    if native["optional_ml_imports_forbidden"] is not True:
        raise ValueError("native import guard not declared")
    arrays, results = {}, []
    for key in sorted(nr):
        a, b = nr[key], rr[key]
        for field in ("ids", "prompt", "shape"):
            if a[field] != b[field]:
                raise ValueError("case input mismatch: " + field)
        for row in (a, b):
            if (row["generated_ids"][:len(row["ids"])] != row["ids"]
                    or len(row["generated_ids"]) != len(row["ids"]) + native["steps"]):
                raise ValueError("generation does not preserve the declared prefix/step count")
        nv = values(native_path.parent, a, native["byteorder"])
        rv = values(reference_path.parent, b, reference["byteorder"])
        arrays[key] = nv, rv
        results.append({
            "case": key[0], "arm": key[1], "prompt": a["prompt"], "ids": a["ids"], "shape": a["shape"],
            "logit_error": errors(nv, rv), "native_nll": a["nll"], "reference_nll": b["nll"],
            "nll_abs_error": abs(a["nll"] - b["nll"]), "greedy_ids_equal": a["generated_ids"] == b["generated_ids"],
            "native_generated_ids": a["generated_ids"], "reference_generated_ids": b["generated_ids"],
            "reference_generated_text": b["generated_text"],
            "native_logits_sha256": a["logits_sha256"], "reference_logits_sha256": b["logits_sha256"],
        })
    interventions = []
    for name in ("meaning", "models", "spiraltorch"):
        for rows in (nr, rr):
            for field in ("ids", "prompt", "shape"):
                if rows[name, "base"][field] != rows[name, "wave"][field]:
                    raise ValueError("intervention inputs differ between base and wave: " + field)
        nb, rb = arrays[name, "base"]
        nw, rw = arrays[name, "wave"]
        interventions.append({"case": name, "native_wave_logit_delta": errors(nw, nb),
                              "reference_wave_logit_delta": errors(rw, rb),
                              "intervention_delta_error": errors([x - y for x, y in zip(nw, nb)],
                                                                 [x - y for x, y in zip(rw, rb)])})
    return {
        "schema": "spiraltorch.native_gpt2_comparison.v1", "scope": native["scope"],
        "source_sha256": native["source_sha256"], "steps": native["steps"],
        "base_model_identity": native["base_model_identity"], "geometry_artifact_id": native["geometry_artifact_id"],
        "native_report_sha256": digest(native_path), "reference_report_sha256": digest(reference_path),
        "native_backend": native["backend"], "reference_backend": reference["backend"],
        "reference_versions": {key: reference[key] for key in ("torch", "transformers")},
        "rows": results, "interventions": interventions,
        "interpretation": "Measured errors, not an empirically selected pass threshold, efficacy study or timing benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.native_report, args.reference_report)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

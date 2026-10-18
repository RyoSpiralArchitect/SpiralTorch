"""Validate paired three-step probe inputs/hashes and measure gradient/update error."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(directory, receipt):
    name = receipt["file"]
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("expected a sibling JSON artifact")
    path = directory / name
    if path.stat().st_size > 16_777_216 or digest(path) != receipt["sha256"]:
        raise ValueError("artifact size or hash mismatch")
    return json.loads(path.read_text())


def parameters(state):
    result, structure = {}, {}
    for site in state["placements"]:
        path = site["path"]
        if path in result:
            raise ValueError("duplicate placement")
        result[path] = {}
        structure[path] = {key: value for key, value in site.items() if key != "tensors"}
        structure[path]["tensors"] = {}
        for tensor in site["tensors"]:
            name = tensor["name"]
            if name in result[path] or len(tensor["values"]) != math.prod(tensor["shape"]):
                raise ValueError("duplicate parameter or invalid value count")
            structure[path]["tensors"][name] = {k: v for k, v in tensor.items() if k != "values"}
            result[path][name] = tensor["values"]
    return result, structure


def error(a, b):
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("invalid numerical arrays")
    delta = [x - y for x, y in zip(a, b)]
    return {"max_abs": max(abs(x) for x in delta), "rms": math.sqrt(math.fsum(x * x for x in delta) / len(delta)),
            "native_l2": math.sqrt(math.fsum(x * x for x in a)), "reference_l2": math.sqrt(math.fsum(x * x for x in b))}


def compare(native_path, reference_path):
    native = json.loads(native_path.read_text())
    reference = json.loads(reference_path.read_text())
    if native["mode"] != "native" or reference["mode"] != "reference":
        raise ValueError("wrong probe modes")
    if native["schema"] != "spiraltorch.native_gpt2_learning_probe.v1":
        raise ValueError("wrong probe schema")
    for field in ("schema", "scope", "source_sha256", "learning_rate", "base_model_identity", "initial_geometry_artifact_id"):
        if native[field] != reference[field]:
            raise ValueError("unmatched probe identity: " + field)
    if (native["optional_ml_imports_forbidden"] is not True or native["artifact_continuation_exact"] is not True
            or native["base_probe_unchanged"] is not True or reference["base_parameter_versions_unchanged"] is not True):
        raise ValueError("probe guard or continuation failed")
    initial = native_path.parent / "initial.json"
    if (digest(initial) != digest(reference_path.parent / "initial.json")
            or "sha256:" + digest(initial) != native["initial_geometry_artifact_id"]):
        raise ValueError("different initial geometry")
    initial_state = json.loads(initial.read_text())
    initial_values, structure = parameters(initial_state)
    if len(native["rows"]) != 3 or len(reference["rows"]) != 3:
        raise ValueError("expected exactly three steps")
    rows = []
    for step, (a, b) in enumerate(zip(native["rows"], reference["rows"])):
        if a["step"] != step or b["step"] != step:
            raise ValueError("steps are not consecutive")
        for field in ("case", "text", "ids"):
            if a[field] != b[field]:
                raise ValueError("unmatched learning inputs: " + field)
        ng, rg = artifact(native_path.parent, a["gradient"]), artifact(reference_path.parent, b["gradient"])
        ns, rs = artifact(native_path.parent, a["state"]), artifact(reference_path.parent, b["state"])
        np, n_structure = parameters(ns)
        rp, r_structure = parameters(rs)
        if n_structure != structure or r_structure != structure:
            raise ValueError("geometry structure or update flags changed")
        for state in (ns, rs, initial_state):
            if state["base_model_identity"] != native["base_model_identity"] or state["policy"] != initial_state["policy"]:
                raise ValueError("state base identity or policy changed")
        if set(ng) != set(initial_values) or set(rg) != set(initial_values):
            raise ValueError("gradient placement set differs")
        differences = {}
        for path, params in initial_values.items():
            expected = {name for name, spec in structure[path]["tensors"].items() if spec["role"] == "parameter"}
            if set(ng[path]) != expected or set(rg[path]) != expected:
                raise ValueError("gradient parameter set differs")
            differences[path] = {}
            for name in params:
                update = error(np[path][name], rp[path][name])
                if name in expected:
                    if len(ng[path][name]) != len(params[name]) or len(rg[path][name]) != len(params[name]):
                        raise ValueError("gradient shape differs")
                    gradient = error(ng[path][name], rg[path][name])
                    differences[path][name] = {"gradient_error": gradient, "parameter_error_after_step": update}
                spec = structure[path]["tensors"][name]
                if (spec["role"] == "buffer" or not spec["requires_grad"]) and (np[path][name] != params[name] or rp[path][name] != params[name]):
                    raise ValueError("frozen parameter or buffer changed")
        error([a["pre_nll"], a["post_nll"]], [b["pre_nll"], b["post_nll"]])
        rows.append({"step": step, "case": a["case"], "text": a["text"], "ids": a["ids"],
                     "native_pre_nll": a["pre_nll"], "native_post_nll": a["post_nll"],
                     "reference_pre_nll": b["pre_nll"], "reference_post_nll": b["post_nll"],
                     "pre_nll_abs_error": abs(a["pre_nll"] - b["pre_nll"]), "post_nll_abs_error": abs(a["post_nll"] - b["post_nll"]),
                     "native_gradient": a["gradient"], "reference_gradient": b["gradient"],
                     "native_state": a["state"], "reference_state": b["state"], "parameters": differences})
    return {"schema": "spiraltorch.native_gpt2_learning_comparison.v1", "scope": native["scope"],
            "source_sha256": native["source_sha256"], "learning_rate": native["learning_rate"],
            "base_model_identity": native["base_model_identity"], "initial_geometry_artifact_id": native["initial_geometry_artifact_id"],
            "initial_geometry_sha256": digest(initial), "native_report_sha256": digest(native_path), "reference_report_sha256": digest(reference_path),
            "reference_versions": {key: reference[key] for key in ("torch", "transformers")}, "native_artifact_continuation_exact": True,
            "rows": rows, "interpretation": "Measured three-window gradient/update agreement, not held-out quality, full-model FT, timing or a selected pass threshold."}


def public_projection(comparison, private_sha256):
    """Keep all discrepancies/cases, not operand norms exposing scalar weights."""
    if comparison["schema"] != "spiraltorch.native_gpt2_learning_comparison.v1":
        raise ValueError("unsupported comparison schema")
    result = copy.deepcopy(comparison)
    removed = 0
    for row in result["rows"]:
        for parameters in row["parameters"].values():
            for parameter in parameters.values():
                gradient = parameter["gradient_error"]
                gradient["native_nonzero"] = gradient["native_l2"] > 0
                gradient["reference_nonzero"] = gradient["reference_l2"] > 0
                for metric in (gradient, parameter["parameter_error_after_step"]):
                    for field in ("native_l2", "reference_l2"):
                        del metric[field]
                        removed += 1
    result["publication"] = {
        "schema": "spiraltorch.native_gpt2_learning_public_projection.v1",
        "private_comparison_sha256": private_sha256,
        "operand_norm_fields_removed": removed,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    result = compare(args.native_report, args.reference_report)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.public_output is not None:
        public = public_projection(result, digest(args.output))
        with args.public_output.open("x") as stream:
            json.dump(public, stream, indent=2, allow_nan=False)
            stream.write("\n")

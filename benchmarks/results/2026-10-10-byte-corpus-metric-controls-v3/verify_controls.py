#!/usr/bin/env python3
"""Check v2 input lineage and v3 first-step controls, not numerical equivalence."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("resume", ROOT / "tools/verify_byte_corpus_resume.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
S = R.STUDY


def verify(request, old_request, baseline, first, request_hash):
    R.require(S.request_version(request) == 3 and S.request_version(old_request) == 2,
              "expected v3 and retained v2 requests")
    common = lambda value: {k: v for k, v in value.items() if k not in ("schema", "cases")}
    R.require(R.exact(common(request), common(old_request)), "data/schedule/rate changed")
    inherited = [{k: v for k, v in c.items() if k != "pair_metric"}
                 for c in request["cases"] if c.get("pair_metric") != S.FLAT]
    R.require(R.exact(inherited, old_request["cases"]), "inherited initializations changed")
    for report, kind in ((baseline, "result"), (first, "partial")):
        R.require(report["schema"] == f"spiraltorch.byte_corpus.{kind}.v3", "report schema")
        R.require(report["request_sha256"] == request_hash and report["engine"] == "spiraltorch",
                  "report identity")
        R.require(len(report["cases"]) == len(request["cases"]), "missing cases")
    R.require(R.exact(first["adapter"], baseline["adapter"]), "runtime differs")
    rows = {}
    for spec, full, prefix in zip(request["cases"], baseline["cases"], first["cases"]):
        for row in (full, prefix):
            for key in ("name", "seed", "geometry", "geometry_update", "pair_metric"):
                R.require(key in row and R.exact(row[key], spec.get(key)), "case identity/order")
            R.require(R.exact(row["parameter_tensors"], len(spec["parameters"]))
                      and R.exact(row["parameter_scalars"], sum(len(p["values"]) for p in spec["parameters"]))
                      and R.exact(row["trainable_parameter_scalars"], S.trainable_scalars(spec)), "parameter counts")
        R.require(len(full["training"]) == len(request["train_batches"])
                  and R.exact([p["revision"] for p in full["training"]], list(range(1, len(request["train_batches"]) + 1))),
                  "incomplete baseline")
        R.require(len(prefix["training"]) == 1 and R.exact(prefix["training"], full["training"][:1]),
                  "first training prefix differs")
        total = len(request["train_batches"])
        revisions = [n for n in range(total + 1) if n % request["checkpoint_every"] == 0 or n == total]
        R.require(R.exact([p["revision"] for p in full["validation"]], revisions), "baseline evaluation schedule")
        expected_evals = [p for p in full["validation"] if p["revision"] <= 1]
        R.require(R.exact(prefix["validation"], expected_evals), "first validation prefix differs")
        values = prefix["final_parameters"]
        R.require(len(values) == len(spec["parameters"]), "missing first-step tensors")
        weights = {}
        for p, after in zip(spec["parameters"], values):
            R.require(len(after) == len(p["values"]), "first-step tensor shape")
            before, after = list(map(R.f32, p["values"])), list(map(R.f32, after))
            if S.frozen(spec) and p["name"].startswith("geometry."):
                R.require(after == before, "frozen geometry bits changed")
            weights[p["name"]] = (before, after)
        rows[(spec["seed"], spec.get("pair_metric"), spec["geometry_update"])] = (prefix, weights)
    summaries = []
    for seed in sorted({c["seed"] for c in request["cases"]}):
        for metric in (S.POINCARE, S.FLAT):
            learned, lw = rows[(seed, metric, "train")]
            frozen, fw = rows[(seed, metric, "frozen")]
            R.require(R.exact(learned["training"], frozen["training"])
                      and R.exact(learned["validation"][0], frozen["validation"][0]),
                      "learned/frozen initial function differs")
            core = [name for name in lw if not name.startswith("geometry.")]
            R.require(all(lw[name] == fw[name] for name in core), "first backbone update differs")
            embeddings = [name for name in core if name in ("token_embedding", "position_embedding")]
            R.require(len(embeddings) == 2 and all(lw[name][0] != lw[name][1] for name in embeddings),
                      "embedding update missing")
            changed = [name for name, (before, after) in lw.items()
                       if name.startswith("geometry.") and before != after]
            R.require(changed, "learned geometry did not move at first step")
            summaries.append(dict(seed=seed, pair_metric=metric, initial_loss_equal=True,
                                  first_backbone_update_bits_equal=True, frozen_geometry_bits_unchanged=True,
                                  embeddings_changed=embeddings, learned_geometry_changed=changed))
    return dict(schema="spiraltorch.byte_corpus.metric_controls.v1", passed=True,
                existing_cases_identical=len(inherited), data_schedule_rate_identical=True,
                all_geometric_initial_bits_identical=True, pairs=summaries,
                scope="Input lineage and within-runtime first-step controls only; use the independent Torch comparator and separate resume verifier")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "old_request", "baseline", "first_step", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    raw = {name: getattr(args, name).read_bytes() for name in ("request", "old_request", "baseline", "first_step")}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    values = {name: S.decode_json(value) for name, value in raw.items()}
    result = verify(values["request"], values["old_request"], values["baseline"], values["first_step"], hashes["request"])
    result["input_sha256"] = hashes
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(result['pairs'])} metric/trainability pairs")


if __name__ == "__main__":
    main()

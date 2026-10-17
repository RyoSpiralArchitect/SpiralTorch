"""Synthetic CPU transport fixture, never evidence of training or model quality.

Reuses the portable Python contract fixtures. No package install, device, corpus
download, or private validation artifact is needed by the WASM CI job.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from test_byte_corpus_study import fixture as study_fixture, study
from test_verify_byte_corpus_resume import fixture as resume_fixture


def fixture():
    request = json.loads(study_fixture()[0])
    for case in request["cases"]:
        for parameter in case["parameters"]:
            parameter["values"] = [0.1] * len(parameter["values"])
    raw = study.encoded(request)
    checkpoint = resume_fixture()[5]
    checkpoint["request_sha256"] = hashlib.sha256(raw).hexdigest()
    checkpoint["completed_updates"] = 0
    for case in checkpoint["cases"]:
        model = json.loads(case["model_json"])
        model["attempted_revision"] = "0"
        case["model_json"] = study.encoded(model).decode()
        case["training"] = []
    limits = dict(max_mean_ce=6.0, max_mean_ce_regression=0.01,
                  max_batch_ce_regression=0.1, max_geometry_centered_rms=0.5,
                  max_router_mass_error=0.00001, min_router_normalized_entropy=0.1)
    gate = dict(schema="spiraltorch.byte_corpus.serving_gate.v2", steps=2,
                documents=[[1, 2, 3]], cohorts=[
                    dict(name=name, batches=[[[0, 0]]], limits=limits)
                    for name in ("synthetic-a", "synthetic-b")])
    return dict(request_json=raw.decode(), checkpoint_json=study.encoded(checkpoint).decode(),
                case_names=[case["name"] for case in request["cases"]],
                serving=dict(gate_json=study.encoded(gate).decode()))


if __name__ == "__main__":
    print(json.dumps(fixture(), allow_nan=False))

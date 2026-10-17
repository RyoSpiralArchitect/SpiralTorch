"""Synthetic CPU transport fixture, never evidence of training or model quality.

Reuses the portable Python contract fixtures. No package install, device, corpus
download, or private validation artifact is needed by the WASM CI job.
"""
import copy
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
    selection_request = copy.deepcopy(request)
    selection_request["train_batches"] *= 2
    selection_raw = study.encoded(selection_request)
    snapshots = []
    for cursor in (0, 1, 2, 4):
        saved = copy.deepcopy(checkpoint)
        saved["request_sha256"] = hashlib.sha256(selection_raw).hexdigest()
        saved["completed_updates"] = cursor
        for index, case in enumerate(saved["cases"]):
            model = json.loads(case["model_json"])
            model["attempted_revision"] = str(cursor)
            case["model_json"] = study.encoded(model).decode()
            case["training"] = [dict(revision=n, ce=2.0) for n in range(1, cursor + 1)]
            case["validation"] = [
                dict(revision=n, batch_losses=[(3.0, 1.0, 2.0)[n // 2] if index == 0 else 2.0])
                for n in (0, 2, 4) if n <= cursor]
        snapshots.append(study.encoded(saved).decode())
    selection = dict(source="synthetic_contract", request_json=selection_raw.decode(),
                     checkpoint_jsons=[snapshots[n] for n in (0, 2, 3)],
                     non_boundary_json=snapshots[1],
                     expected_indices={case["name"]: 1 if i == 0 else 0
                                       for i, case in enumerate(request["cases"])})
    return dict(request_json=raw.decode(), checkpoint_json=study.encoded(checkpoint).decode(),
                case_names=[case["name"] for case in request["cases"]],
                serving=dict(gate_json=study.encoded(gate).decode()), selection=selection)


if __name__ == "__main__":
    print(json.dumps(fixture(), allow_nan=False))

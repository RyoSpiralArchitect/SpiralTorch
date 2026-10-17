"""Exercise installed durable checkpoints through the Rust saved-model selector.

Uses only the previously pinned small AdamW request. Raw data and models remain
in a new private run; stdout contains decision records, not model payloads.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

import spiraltorch as st


REQUEST_SHA256 = "208975bab0905a1ee968d5940975259491163680ce7bd2c0132bce23a637e523"
STOPS = (0, 4, 8, 12)


def require(value, message):
    if not value:
        raise ValueError(message)


def qualify(request_path, output, control):
    raw = request_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == REQUEST_SHA256, "request differs from pinned fixture")
    request = json.loads(raw)
    with redirect_stdout(sys.stderr):
        st.run_byte_corpus_segments(request_path, output, STOPS)
    saved = [st.load_byte_corpus_segment(output, step) for step in STOPS]
    study = saved[0].study
    checkpoints = [item.checkpoint for item in saved]
    for name in ("report-12.json", "checkpoint-12.json"):
        require((output / name).read_bytes() == (control / name).read_bytes(),
                "retention boundaries changed final " + name)
    decisions = []
    for case in request["cases"]:
        artifact, decision_json = study.select_checkpoint(
            checkpoints, case["name"], policy="fixed_window_frozen.v1")
        decision = json.loads(decision_json)
        rows = decision["candidates"]
        require([row["completed_updates"] for row in rows] == list(STOPS),
                "selection did not consider the declared saved revisions")
        expected = min(range(len(rows)), key=lambda i: (rows[i]["mean_ce"], STOPS[i]))
        require(decision["selected"]["candidate_index"] == expected,
                "selection disagrees with its reported scores")
        direct = study.case_artifact(checkpoints[expected], case["name"],
                                     policy="fixed_window_frozen.v1")
        require(artifact.to_json() == direct.to_json(), "selector changed the selected model")
        require(decision["selected"]["artifact_id"] == artifact.artifact_id,
                "selection record differs from its artifact")
        study.validate_artifact(checkpoints[expected], artifact)
        reloaded = st.nn.ByteCorpusModelArtifact.from_json(
            artifact.to_json(), expected_id=artifact.artifact_id)
        require(reloaded.to_json() == artifact.to_json(), "opaque artifact import changed bytes")
        decisions.append(decision)
    require(len(decisions) == 9, "expected all nine cases")
    return {
        "schema": "spiraltorch.byte_corpus.saved_selection_qualification.v1",
        "request_sha256": REQUEST_SHA256,
        "stops": STOPS,
        "cases": len(decisions),
        "exact_final_report_and_checkpoint_equal": True,
        "selected_artifacts_equal_source_checkpoints": True,
        "artifact_import_exact": True,
        "decisions": decisions,
        "geometric_quality_benefit_established": False,
        "serving_promotion_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("control", type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.request, args.output, args.control), indent=2, allow_nan=False))

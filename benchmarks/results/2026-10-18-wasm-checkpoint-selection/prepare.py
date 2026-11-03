"""Prepare private WASM selection inputs from an already-qualified local run.

No model execution, training, corpus acquisition or serving promotion occurs.
Use the installed Python client to bind retained seals and produce exact
selection/artifact bytes for the independent WASM client to consume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import spiraltorch as st


REQUEST_SHA256 = "208975bab0905a1ee968d5940975259491163680ce7bd2c0132bce23a637e523"
STOPS = (0, 4, 8, 12)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare(run, previous, synthetic):
    raw = (run / "request.json").read_bytes()
    require(sha(raw) == REQUEST_SHA256, "request differs from qualified run")
    request = json.loads(raw)
    previous_raw = previous.read_bytes()
    prior = json.loads(previous_raw)
    require(prior["request_sha256"] == REQUEST_SHA256 and prior["cases"] == 9,
            "prior selection qualification differs")
    files = ["request.json"] + [
        f"{kind}-{step}.json" for step in STOPS
        for kind in ("checkpoint", "report", "segment")]
    hashes = {name: sha((run / name).read_bytes()) for name in files}
    saved = [st.load_byte_corpus_segment(run, step) for step in STOPS]
    study = saved[0].study
    checkpoints = [item.checkpoint for item in saved]
    expected, indices = {}, {}
    for case, prior_decision in zip(request["cases"], prior["decisions"], strict=True):
        artifact, report_json = study.select_checkpoint(
            checkpoints, case["name"], policy="fixed_window_frozen.v1")
        report = json.loads(report_json)
        require(report == prior_decision, "retained Python selection changed")
        index = report["selected"]["candidate_index"]
        direct = study.case_artifact(checkpoints[index], case["name"],
                                     policy="fixed_window_frozen.v1")
        require(artifact.to_json() == direct.to_json(), "selected model differs")
        indices[case["name"]] = index
        expected[case["name"]] = dict(
            report_json=report_json, artifact_json=artifact.to_json())
    steps = request["config"]["steps"]
    windows = [request["validation_documents"][document][offset:offset + steps + 1]
               for document, offset in request["validation_batches"][0]]
    require(len(windows) == request["config"]["batch"] and
            all(len(row) == steps + 1 for row in windows), "window shape")
    fixture = json.loads(synthetic.read_bytes())
    fixture["selection"] = dict(
        source="retained_learning", request_json=raw.decode("utf-8"),
        checkpoint_jsons=[item.checkpoint.to_json() for item in saved],
        expected_indices=indices, expected=expected,
        windows=[byte for row in windows for byte in row],
        expected_source=dict(request_sha256=REQUEST_SHA256,
                             prior_selection_sha256=sha(previous_raw),
                             retained_files_sha256=hashes))
    require(all(sha((run / name).read_bytes()) == value for name, value in hashes.items()),
            "retained source changed during preparation")
    require(previous.read_bytes() == previous_raw, "prior selection source changed")
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "previous", "synthetic", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    fixture = prepare(args.run, args.previous, args.synthetic)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(fixture, output, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(cases=len(fixture["selection"]["expected_indices"]),
                          snapshots=len(STOPS), new_training_updates=0,
                          output_sha256=sha(args.output.read_bytes()))))


if __name__ == "__main__":
    main()

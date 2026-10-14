"""Exercise existing study artifacts through a fresh pinned inference owner."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import spiraltorch as st

root, archive, output = map(Path, sys.argv[1:])
output.mkdir(mode=0o700, parents=True, exist_ok=False)
helpers = runpy.run_path(str(root / "bindings/st-py/tests/test_nn_byte_study.py"))
raw = (archive / "input/request.json").read_bytes()
request = json.loads(raw)
study = st.nn.ByteCorpusStudy.from_json(raw)
query = next(c for c in request["cases"] if c.get("mixture", {}).get("query_projection", {}).get("weight_update") == "train")
frozen = next(c for c in request["cases"] if c.get("mixture", {}).get("query_projection", {}).get("weight_update") == "frozen")
names = [request["cases"][0]["name"], query["name"], frozen["name"]]
result = {"scope": "Existing trained native/browser studies; no new updates, promotion or quality claim",
          "request_sha256": study.request_sha256, "cases": []}
batch = st.nn.ByteLmBatch.from_documents(
    [bytes(doc) for doc in request["validation_documents"]],
    [tuple(window) for window in request["validation_batches"][0]], request["config"]["steps"],
)
for client in ("native", "browser"):
    source = (archive / (client + "-checkpoint-128.json")).read_bytes()
    saved = study.checkpoint_from_json(source)
    before = saved.to_json()
    assert saved.completed_updates == 128
    history = json.loads(before)
    for name in names:
        artifact = study.case_artifact(saved, name, policy="fixed_window_frozen.v1")
        loaded = st.nn.ByteCorpusModelArtifact.from_json(artifact.to_json(), expected_id=artifact.artifact_id)
        study.validate_artifact(saved, loaded)
        selected = next(c for c in history["cases"] if c["name"] == name)
        expected_model = study.case_checkpoint(saved, name)
        assert loaded.checkpoint().to_json() == expected_model.to_json()
        model = loaded.restore_wgpu()
        assert model.tensor_device().adapter_info()["device_type"] != "Cpu"
        forward = model.forward(model.prepare_batch(batch))
        logits = forward.prediction_tensor().snapshot().read_values()
        loss = forward.next_byte_loss().loss_tensor().snapshot().read_values()[0]
        assert forward.artifact_id == artifact.artifact_id
        reference = expected_model.restore_wgpu()
        expected = reference.forward(reference.prepare_batch(batch))
        assert expected.artifact_id is None
        assert logits == expected.prediction_tensor().snapshot().read_values()
        assert loss == expected.next_byte_loss().loss_tensor().snapshot().read_values()[0]
        try:
            st.nn.ByteCorpusModelArtifact.from_json(artifact.to_json() + " ", expected_id=artifact.artifact_id)
        except ValueError:
            pass
        else:
            raise AssertionError("altered artifact accepted")
        assert model.forward(model.prepare_batch(batch)).prediction_tensor().snapshot().read_values() == logits
        fresh = subprocess.run(
            [sys.executable, "-I", "-B", "-c", helpers["FRESH_INFERENCE"]],
            input=json.dumps({"artifact_json": artifact.to_json(), "expected_id": artifact.artifact_id,
                              "documents": request["validation_documents"],
                              "windows": request["validation_batches"][0], "steps": request["config"]["steps"]}),
            text=True, capture_output=True, timeout=120,
        )
        if fresh.returncode:
            raise RuntimeError(fresh.stderr)
        assert json.loads(fresh.stdout) == {
            "artifact_id": artifact.artifact_id, "revision": 128, "logits": logits,
            "loss": loss, "model_json": expected_model.to_json(),
        }
        assert saved.to_json() == before
        original_loss = selected["validation"][-1]["batch_losses"][0]
        row = {"source_client": client, "case_name": name, "artifact_id": artifact.artifact_id,
               "policy": artifact.policy, "completed_updates": 128,
               "source_file_sha256": hashlib.sha256(source).hexdigest(),
               "study_checkpoint_sha256": artifact.study_checkpoint_sha256,
               "model_sha256": hashlib.sha256(expected_model.to_json().encode()).hexdigest(),
               "logits_f32le_sha256": hashlib.sha256(struct.pack("<" + "f" * len(logits), *logits)).hexdigest(),
               "logits_count": len(logits), "heldout_first_batch_ce": loss,
               "historical_first_batch_ce": original_loss,
               "historical_ce_abs_difference": abs(loss - original_loss),
               "historical_ce_f32_exact": struct.pack("<f", loss) == struct.pack("<f", original_loss),
               "fresh_process_exact": True, "bare_model_exact": True,
               "altered_import_rejected": True, "active_owner_unchanged": True,
               "source_unchanged": True}
        result["cases"].append(row)
        print(json.dumps(row), flush=True)
        if client == "browser" and name == query["name"]:
            fd = os.open(output / "browser-expected-artifact.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(artifact.to_json())
    assert (archive / (client + "-checkpoint-128.json")).read_bytes() == source
with (output / "qualification.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")

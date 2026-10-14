"""Private historical replay of fixed serving mechanics; zero training updates."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import spiraltorch as st


def private_write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid transition accepted")


def logits(forward):
    return forward.prediction_tensor().snapshot().read_values()


def logits_hash(values):
    return sha(struct.pack("<" + "f" * len(values), *values))


def main():
    archive, output = map(Path, sys.argv[1:])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    paths = [archive / "input/request.json"] + [archive / f"{client}-checkpoint-128.json"
                                                for client in ("native", "browser")]
    source_hashes = {str(path.relative_to(archive)): sha(path.read_bytes()) for path in paths}
    raw = paths[0].read_bytes()
    request = json.loads(raw)
    study = st.nn.ByteCorpusStudy.from_json(raw)
    ordinary = next(c["name"] for c in request["cases"] if not c["geometry"])
    queries = [next(c["name"] for c in request["cases"]
                    if c.get("mixture", {}).get("query_projection", {}).get("weight_update") == mode)
               for mode in ("train", "frozen")]
    # Frozen before measurements. Deliberately permissive mechanics thresholds,
    # not thresholds chosen to endorse or rank these historical models.
    policy = {"schema": "spiraltorch.byte_corpus.serving_gate.v1",
              "documents": request["validation_documents"],
              "batches": request["validation_batches"], "steps": request["config"]["steps"],
              "limits": {"max_mean_ce": 10., "max_mean_ce_regression": 2.,
                         "max_batch_ce_regression": 2., "max_geometry_centered_rms": 100.,
                         "max_router_mass_error": 1e-5, "min_router_normalized_entropy": 0.}}
    gate_json = json.dumps(policy, separators=(",", ":"))
    private_write(output / "serving-gate.json", gate_json)
    gate = st.nn.ByteCorpusServingGate.from_json(gate_json)
    private_write(output / "selection.json", json.dumps({"incumbent": ordinary, "candidates": queries,
                  "source_hashes": source_hashes, "gate_sha256": gate.request_sha256,
                  "limits": policy["limits"], "scope": "predeclared mechanics, not quality thresholds"}, indent=2))
    host = st.nn.ByteLmBatch.from_documents([bytes(d) for d in policy["documents"]],
                                           [tuple(w) for w in policy["batches"][0]], policy["steps"])
    result = {"schema": "spiraltorch.byte_corpus.serving_replay.v1", "source_hashes": source_hashes,
              "gate_sha256": gate.request_sha256, "new_training_updates": 0, "cases": [],
              "scope": "Historical model selection mechanics only; no new quality/held-outness claim"}
    for client, path in zip(("native", "browser"), paths[1:]):
        saved = study.checkpoint_from_json(path.read_bytes())
        assert saved.completed_updates == 128
        initial = study.case_artifact(saved, ordinary, policy="fixed_window_frozen.v1")
        study.validate_artifact(saved, initial)
        for name in queries:
            candidate = study.case_artifact(saved, name, policy="fixed_window_frozen.v1")
            study.validate_artifact(saved, candidate)
            slot = gate.bootstrap_wgpu(initial)
            assert slot.tensor_device().adapter_info()["device_type"] != "Cpu"
            batch = slot.prepare_batch(host)
            before = slot.forward(batch)
            before_values = logits(before)
            permit = slot.prepare_candidate(candidate)
            assessment = json.loads(permit.to_json())
            assert assessment["incumbent"]["logits_f32le_sha256"][0] == logits_hash(before_values)
            record = {"source_client": client, "case_name": name, "bootstrap": json.loads(slot.bootstrap_json()),
                      "assessment": assessment, "before_logits_f32le_sha256": logits_hash(before_values)}
            if permit.accepted:
                stale = slot.prepare_candidate(candidate)
                aba = slot.prepare_candidate(candidate)
                assert aba.accepted
                record["activation"] = json.loads(slot.activate(permit))
                reject(lambda: slot.activate(stale))
                after = slot.forward(batch)
                assert after.artifact_id == candidate.artifact_id
                record["after_logits_f32le_sha256"] = logits_hash(logits(after))
                assert record["after_logits_f32le_sha256"] == assessment["candidate"]["logits_f32le_sha256"][0]
                record["rollback"] = json.loads(slot.rollback(1))
                restored = slot.forward(batch)
                assert logits_hash(logits(restored)) == logits_hash(before_values)
                assert restored.artifact_id == initial.artifact_id
                assert before.artifact_id == initial.artifact_id and logits_hash(logits(before)) == logits_hash(before_values)
                assert slot.previous_artifact_id is None and slot.generation == 2
                reject(lambda: slot.activate(aba))
                assert slot.artifact_id == initial.artifact_id and slot.generation == 2
                assert slot.previous_artifact_id is None
                assert logits_hash(logits(slot.forward(batch))) == logits_hash(before_values)
                reject(lambda: slot.rollback(2))
                record.update(stale_rejected=True, aba_generation_rejected=True,
                              measured_forward_hashes_exact=True, rollback_exact=True, retained_forward_exact=True)
            else:
                reject(lambda: slot.activate(permit))
                assert slot.artifact_id == initial.artifact_id and slot.generation == 0
                record["rejection_preserved_owner"] = True
            if client == "browser" and name == queries[0]:
                windows = [b for d, start in policy["batches"][0]
                           for b in policy["documents"][d][start:start + policy["steps"] + 1]]
                private_write(output / "serving-input.json", json.dumps({"gate_json": gate_json,
                    "incumbent": {"id": initial.artifact_id, "json": initial.to_json()},
                    "candidate": {"id": candidate.artifact_id, "json": candidate.to_json()},
                    "windows": windows}))
            private_write(output / f"case-{len(result['cases'])}.json", json.dumps(record, indent=2))
            result["cases"].append(record)
            print(json.dumps({"source": client, "case": name, "accepted": permit.accepted,
                              "failed_checks": assessment["failed_checks"],
                              "mean_ce_delta": assessment["mean_ce_delta"]}), flush=True)
    assert source_hashes == {str(path.relative_to(archive)): sha(path.read_bytes()) for path in paths}
    result["sources_unchanged"] = True
    private_write(output / "qualification.json", json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Installed Rust serving mechanics; fixtures/thresholds are not quality claims."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import struct
import unittest

import spiraltorch as st

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_nn_byte_study.py")))
POLICY = "fixed_window_frozen.v1"


def gate_value(request=None):
    return {
        "schema": "spiraltorch.byte_corpus.serving_gate.v1",
        "documents": request["validation_documents"] if request else [[7, 8, 9, 10, 11]],
        "batches": request["validation_batches"] if request else [[[0, 0]], [[0, 2]]],
        "steps": request["config"]["steps"] if request else 2,
        "limits": dict(max_mean_ce=10.0, max_mean_ce_regression=1.0,
                       max_batch_ce_regression=1.0, max_geometry_centered_rms=100.0,
                       max_router_mass_error=1e-5, min_router_normalized_entropy=0.0),
    }


def gate(value=None):
    return st.nn.ByteCorpusServingGate.from_json(json.dumps(gate_value() if value is None else value))


def output_hash(values):
    return hashlib.sha256(struct.pack("<" + "f" * len(values), *values)).hexdigest()


def duplicated_block_fixture(artifact):
    # The paired study recipe intentionally supports one block. Build a separate
    # synthetic two-block inference fixture; it is NOT a source-attested study.
    envelope = json.loads(artifact.to_json())
    checkpoint = json.loads(envelope["model_json"])
    model = checkpoint["model"]
    model["blocks"].append(copy.deepcopy(model["blocks"][0]))
    geometry = model["geometry"]
    geometry["raw_gains"].append(copy.deepcopy(geometry["raw_gains"][0]))
    if "head_pair_metrics" in geometry:
        geometry["head_pair_metrics"].append(copy.deepcopy(geometry["head_pair_metrics"][0]))
    for key in ("raw_logits", "scales", "query_projections"):
        if key in geometry["mixture"]:
            geometry["mixture"][key].append(copy.deepcopy(geometry["mixture"][key][0]))
    envelope["model_json"] = json.dumps(checkpoint)
    raw = json.dumps(envelope)
    return st.nn.ByteCorpusModelArtifact.from_json(raw, expected_id="sha256:" + hashlib.sha256(raw.encode()).hexdigest())


class ServingPortable(unittest.TestCase):
    def test_exact_bytes_thresholds_and_window_admission(self):
        raw = json.dumps(gate_value()).encode()
        value = st.nn.ByteCorpusServingGate.from_json(raw)
        self.assertEqual(value.to_json().encode(), raw)
        self.assertEqual(value.request_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(value.batch_shape, (1, 2))
        self.assertEqual(value.batch_count, 2)
        self.assertNotEqual(value.request_sha256,
                            st.nn.ByteCorpusServingGate.from_json(raw + b" ").request_sha256)
        with self.assertRaises(AttributeError):
            value.request_sha256 = "forged"
        for payload in (None, {}, bytearray(raw), True):
            with self.assertRaises(TypeError):
                st.nn.ByteCorpusServingGate.from_json(payload)
        for payload in (b"{}", b"null", b"\xff"):
            with self.assertRaises(ValueError):
                st.nn.ByteCorpusServingGate.from_json(payload)
        for key, changed in (("steps", 0), ("steps", True), ("batches", []),
                             ("batches", [[[0, 4]]]), ("unknown", 1)):
            invalid = gate_value()
            invalid[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate(invalid)

    def test_cpu_validation_and_no_constructor_or_permit_import(self):
        raw = HELPERS["fixture"]()[0]
        study = st.nn.ByteCorpusStudy.from_json(raw)
        record = HELPERS["resume_fixture"]()[5]
        record["request_sha256"] = study.request_sha256
        saved = study.checkpoint_from_json(json.dumps(record))
        model = study.case_artifact(saved, record["cases"][0]["name"], policy=POLICY)
        value = gate()
        value.validate_artifact(model)
        wrong = gate_value()
        wrong["steps"] = 1
        with self.assertRaises(ValueError):
            gate(wrong).validate_artifact(model)
        for name in ("ByteCorpusServingGate", "ResidentByteCandidate", "ResidentByteServing"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                getattr(st.nn, name)()
        self.assertFalse(hasattr(st.nn.ResidentByteCandidate, "from_json"))
        if not st.wgpu_kernel_reports_available():
            with self.assertRaises(NotImplementedError):
                value.bootstrap_wgpu(model)


@unittest.skipUnless(os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1"
                     and st.wgpu_kernel_reports_available(), "explicit WGPU runtime opt-in")
class ServingRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = st.nn.ByteCorpusStudy.from_json(HELPERS["fixture"]()[0])
        cls.initial = cls.study.case_artifact(cls.study.advance_wgpu(0).checkpoint,
                                             "geometry", policy=POLICY)
        cls.trained = cls.study.case_artifact(cls.study.advance_wgpu().checkpoint,
                                             "geometry", policy=POLICY)

    def test_switch_reject_stale_foreign_and_rollback_preserves_retained_forwards(self):
        serving = gate().bootstrap_wgpu(self.initial)
        self.assertNotEqual(serving.tensor_device().adapter_info()["device_type"], "Cpu")
        host = st.nn.ByteLmBatch.from_windows([b"\x07\x08\x09"])
        batch = serving.prepare_batch(host)
        before = serving.forward(batch)
        before_values = before.prediction_tensor().snapshot().read_values()
        prepared = serving.prepare_candidate(self.trained)
        aba = serving.prepare_candidate(self.trained)
        self.assertTrue(aba.accepted)
        stale = serving.prepare_candidate(self.trained)
        foreign = serving.prepare_candidate(self.trained)
        other = gate().bootstrap_wgpu(self.initial)
        with self.assertRaises(ValueError):
            other.activate(foreign)
        self.assertTrue(foreign.consumed)
        self.assertEqual(other.generation, 0)
        self.assertTrue(prepared.accepted)
        receipt = json.loads(prepared.to_json())
        self.assertEqual(receipt["incumbent"]["artifact_id"], self.initial.artifact_id)
        self.assertEqual(receipt["candidate"]["artifact_id"], self.trained.artifact_id)
        self.assertGreater(receipt["candidate"]["geometry_centered_rms"][0][0], 0)
        self.assertEqual(receipt["incumbent"]["logits_f32le_sha256"][0], output_hash(before_values))
        with self.assertRaises(TypeError):
            serving.activate(prepared.to_json())
        self.assertFalse(prepared.consumed)
        self.assertEqual(json.loads(serving.activate(prepared))["to"], self.trained.artifact_id)
        self.assertTrue(prepared.consumed)
        with self.assertRaises(ValueError):
            serving.activate(prepared)
        with self.assertRaises(ValueError):
            serving.activate(stale)
        self.assertEqual(serving.generation, 1)
        after = serving.forward(batch)
        self.assertEqual(after.artifact_id, self.trained.artifact_id)
        after_hash = output_hash(after.prediction_tensor().snapshot().read_values())
        self.assertNotEqual(after_hash, output_hash(before_values))
        self.assertEqual(after_hash, receipt["candidate"]["logits_f32le_sha256"][0])
        for generation in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                serving.rollback(generation)
        with self.assertRaises(ValueError):
            serving.rollback(0)
        self.assertEqual(serving.generation, 1)
        self.assertEqual(json.loads(serving.rollback(1))["to"], self.initial.artifact_id)
        self.assertEqual(serving.generation, 2)
        self.assertIsNone(serving.previous_artifact_id)
        with self.assertRaises(ValueError):
            serving.activate(aba)
        self.assertTrue(aba.consumed)
        self.assertEqual(serving.generation, 2)
        self.assertEqual(serving.artifact_id, self.initial.artifact_id)
        self.assertIsNone(serving.previous_artifact_id)
        self.assertEqual(output_hash(serving.forward(batch).prediction_tensor().snapshot().read_values()), output_hash(before_values))
        del serving
        self.assertEqual(before.artifact_id, self.initial.artifact_id)
        self.assertEqual(output_hash(before.prediction_tensor().snapshot().read_values()), output_hash(before_values))
        self.assertEqual(after.artifact_id, self.trained.artifact_id)

    def test_rejected_and_nonfinite_candidates_never_change_active_owner(self):
        value = gate_value()
        value["limits"].update(max_mean_ce_regression=-100.0, max_batch_ce_regression=-100.0)
        serving = gate(value).bootstrap_wgpu(self.initial)
        candidate = serving.prepare_candidate(self.trained)
        self.assertFalse(candidate.accepted)
        self.assertIn("mean_ce_regression", json.loads(candidate.to_json())["failed_checks"])
        with self.assertRaises(ValueError):
            serving.activate(candidate)
        self.assertEqual(serving.artifact_id, self.initial.artifact_id)
        self.assertEqual(serving.generation, 0)
        # Pinned, syntactically finite model with overflowing forward arithmetic.
        # This is an injected failure, not a trained or source-attested artifact.
        envelope = json.loads(self.trained.to_json())
        model = json.loads(envelope["model_json"])
        model["model"]["token"]["values"] = [3e38] * len(model["model"]["token"]["values"])
        envelope["model_json"] = json.dumps(model)
        raw = json.dumps(envelope)
        invalid = st.nn.ByteCorpusModelArtifact.from_json(raw, expected_id="sha256:" + hashlib.sha256(raw.encode()).hexdigest())
        with self.assertRaises(ValueError):
            serving.prepare_candidate(invalid)
        self.assertEqual(serving.artifact_id, self.initial.artifact_id)
        self.assertEqual(serving.generation, 0)
        self.assertIsNone(serving.previous_artifact_id)

    def test_actual_shared_and_contextual_router_readbacks_and_entropy_rejection(self):
        source = json.loads(HELPERS["two_head_source"]())
        source["validation_batches"] = [[[0, 0]], [[0, 4]]]
        source = json.dumps(source)
        fisher = st.nn.ByteCorpusBiasCalibration.from_json(source, [0, 1], fisher_rao_control=True).prepare_wgpu()
        mixed = st.nn.ByteCorpusMixtureControls.from_json(fisher.request_json).prepare_wgpu()
        prepared = st.nn.ByteCorpusQueryMixtureControls.from_json(mixed.request_json).prepare()
        request = json.loads(prepared.request_json)
        study = prepared.study()
        shared = next(c["name"] for c in request["cases"]
                      if c.get("mixture", {}).get("router_update") == "train"
                      and "query_projection" not in c["mixture"])
        contextual = next(c["name"] for c in request["cases"]
                          if c.get("mixture", {}).get("query_projection", {}).get("weight_update") == "train")
        initial_state, trained_state = study.advance_wgpu(0).checkpoint, study.advance_wgpu().checkpoint
        for name in (shared, contextual):
            with self.subTest(name=name):
                initial = duplicated_block_fixture(study.case_artifact(initial_state, name, policy=POLICY))
                trained = duplicated_block_fixture(study.case_artifact(trained_state, name, policy=POLICY))
                with self.assertRaises(ValueError):
                    study.validate_artifact(trained_state, trained)
                serving = gate(gate_value(request)).bootstrap_wgpu(initial)
                candidate = serving.prepare_candidate(trained)
                measured = json.loads(candidate.to_json())["candidate"]
                independent = trained.restore_wgpu()
                expected_hashes, expected_losses, all_rows = [], [], [[], []]
                components = len(json.loads(trained.checkpoint().to_json())["model"]["geometry"]["mixture"]["metrics"])
                for windows in request["validation_batches"]:
                    host = st.nn.ByteLmBatch.from_documents([bytes(d) for d in request["validation_documents"]],
                                                           [tuple(w) for w in windows], request["config"]["steps"])
                    forward = independent.forward(independent.prepare_batch(host))
                    logits = forward.prediction_tensor().snapshot().read_values()
                    expected_hashes.append(hashlib.sha256(struct.pack("<" + "f" * len(logits), *logits)).hexdigest())
                    expected_losses.append(forward.next_byte_loss().loss_tensor().snapshot().read_values()[0])
                    for block in range(2):
                        weights = forward.geometry_mixture_weights_tensor(block).snapshot().read_values()
                        all_rows[block].extend(weights[i:i+components] for i in range(0, len(weights), components))
                self.assertEqual(measured["logits_f32le_sha256"], expected_hashes)
                self.assertEqual(measured["batch_ce"], expected_losses)
                self.assertEqual(len(measured["router"]), 2)
                self.assertEqual([len(heads) for heads in measured["geometry_centered_rms"]], [2, 2])
                for block, rows in enumerate(all_rows):
                    expected_rows = 2 * 2 * (request["config"]["steps"] if name == contextual else 1)
                    self.assertEqual(len(rows), expected_rows)
                    self.assertEqual(measured["router"][block]["rows"], expected_rows)
                    self.assertAlmostEqual(measured["router"][block]["max_mass_error"],
                                           max(abs(sum(row)-1) for row in rows), places=12)
                    self.assertAlmostEqual(measured["router"][block]["min_normalized_entropy"],
                                           min(-sum((p/sum(row))*math.log(p/sum(row)) for p in row if p>0)
                                               / math.log(components) for row in rows), places=12)

                # Deliberately collapse only block 1/head 0, without training.
                # This tests admission, not a trained failure or model quality.
                envelope = json.loads(initial.to_json())
                model = json.loads(envelope["model_json"])
                model["model"]["geometry"]["mixture"]["raw_logits"][1][:components] = [20.] + [-20.] * (components-1)
                envelope["model_json"] = json.dumps(model)
                raw = json.dumps(envelope)
                collapsed = st.nn.ByteCorpusModelArtifact.from_json(raw, expected_id="sha256:" + hashlib.sha256(raw.encode()).hexdigest())
                policy = gate_value(request)
                policy["limits"].update(max_mean_ce_regression=100., max_batch_ce_regression=100.,
                                        min_router_normalized_entropy=0.5)
                guarded = gate(policy).bootstrap_wgpu(initial)
                rejected = guarded.prepare_candidate(collapsed)
                self.assertFalse(rejected.accepted)
                report = json.loads(rejected.to_json())
                self.assertEqual(report["failed_checks"], ["router_entropy"])
                self.assertGreater(report["candidate"]["router"][0]["min_normalized_entropy"], 0.99)
                self.assertLess(report["candidate"]["router"][1]["min_normalized_entropy"], 0.01)
                with self.assertRaises(ValueError):
                    guarded.activate(rejected)
                self.assertEqual(guarded.artifact_id, initial.artifact_id)
                self.assertEqual(guarded.generation, 0)


if __name__ == "__main__":
    unittest.main()

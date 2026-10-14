"""Rust study clients; fabricated fixtures below test contracts, not quality."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
# Only independent study/fixture utilities, never the checkout's Python package.
sys.path.insert(0, str(ROOT / "tools"))
import byte_corpus_study as protocol
from test_byte_corpus_study import fixture, frozen_fixture, metric_fixture
from test_byte_corpus_calibration import fixture as calibrated_fixture
from test_byte_corpus_fisher import fixture as fisher_fixture
from test_byte_corpus_headwise import fixture as headwise_fixture
from test_byte_corpus_mixture import fixture as mixture_fixture
from test_byte_corpus_query import fixture as query_fixture
from test_verify_byte_corpus_resume import fixture as resume_fixture

NAMES = (
    "ByteCorpusStudy", "ByteCorpusStudyCheckpoint", "ByteCorpusStudySegment",
    "ByteCorpusPreparedBiasStudy", "ByteCorpusBiasCalibration",
    "ByteCorpusHeadwiseControls", "ByteCorpusMixtureControls", "ByteCorpusQueryMixtureControls",
    "ByteCorpusModelArtifact", "ResidentByteInference",
)

FRESH_INFERENCE = """
import json
import sys
import spiraltorch as st

payload = json.load(sys.stdin)
artifact = None
if "artifact_json" in payload:
    artifact = st.nn.ByteCorpusModelArtifact.from_json(
        payload["artifact_json"], expected_id=payload["expected_id"])
    checkpoint = artifact.checkpoint()
    model = artifact.restore_wgpu()
else:
    checkpoint = st.nn.ByteDecoderCheckpoint.from_json(payload["model_json"])
    model = checkpoint.restore_wgpu()
assert model.tensor_device().adapter_info()["device_type"] != "Cpu"
batch = st.nn.ByteLmBatch.from_documents(
    [bytes(doc) for doc in payload["documents"]],
    [tuple(window) for window in payload["windows"]], payload["steps"],
)
forward = model.forward(model.prepare_batch(batch))
assert forward.artifact_id == (artifact.artifact_id if artifact is not None else None)
print(json.dumps({
    "artifact_id": forward.artifact_id,
    "revision": model.attempted_updates,
    "logits": forward.prediction_tensor().snapshot().read_values(),
    "loss": forward.next_byte_loss().loss_tensor().snapshot().read_values()[0],
    "model_json": (checkpoint.to_json() if artifact is not None
                   else model.checkpoint_snapshot().read().to_json()),
}))
"""


def two_head_source():
    request = protocol.decode_json(metric_fixture()[0])
    request["config"].update(width=4, hidden=8, heads=2, steps=4)
    plain, geometry = protocol.parameters(request["config"], 11)
    for case in request["cases"]:
        case["parameters"] = copy.deepcopy(geometry if case["geometry"] else plain)
    request.update(train_documents=[list(b"geometry changes attention")],
                   validation_documents=[list(b"unseen document")],
                   train_batches=[[(0, 0)], [(0, 4)]])
    return protocol.encoded(request)


class StudyPortable(unittest.TestCase):
    def test_exports_and_stubs(self):
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        classes = {n.name: n for n in ast.parse(
            (ROOT / "bindings/st-py/spiraltorch/__init__.pyi").read_text()).body
            if isinstance(n, ast.ClassDef)}
        exposed = {n.target.id for n in classes["_NnModule"].body if isinstance(n, ast.AnnAssign)}
        for name in NAMES:
            self.assertIn(name, st.nn.__all__)
            self.assertIn(name, st._rs.nn.__all__)
            self.assertIs(namespace[name], getattr(st._rs.nn, name))
            self.assertIn("_Nn" + name, classes)
            self.assertIn(name, exposed)
            with self.assertRaises(TypeError):
                getattr(st.nn, name)()
        self.assertEqual(st.nn.BYTE_CORPUS_STUDY_MAX_BYTES, 64 * 1024 * 1024)
        self.assertIn("BYTE_CORPUS_STUDY_MAX_BYTES", namespace)
        self.assertIn("case_checkpoint", {n.name for n in classes["_NnByteCorpusStudy"].body
                                          if isinstance(n, ast.FunctionDef)})

    def test_all_versions_portable_and_exact_byte_identity(self):
        requests = [make()[0] for make in (fixture, frozen_fixture, metric_fixture,
                    calibrated_fixture, fisher_fixture, headwise_fixture, mixture_fixture)]
        requests.append(query_fixture()[1])
        for version, raw in enumerate(requests, 1):
            with self.subTest(version=version):
                study = st.nn.ByteCorpusStudy.from_json(raw)
                self.assertEqual(study.request_sha256, hashlib.sha256(raw).hexdigest())
                self.assertEqual(study.total_updates, 2)
                self.assertGreater(study.checkpoint_size_bound(), 0)
                study.validate_segment(0)
                study.validate_segment()
                self.assertEqual(st.nn.ByteCorpusStudy.from_json(raw.decode()).request_sha256,
                                 study.request_sha256)
                self.assertNotEqual(st.nn.ByteCorpusStudy.from_json(raw + b" ").request_sha256,
                                    study.request_sha256)

    def test_json_bounds_types_and_cpu_preflight(self):
        raw = fixture()[0]
        for value in (json.loads(raw), bytearray(raw), memoryview(raw), None):
            with self.assertRaises(TypeError):
                st.nn.ByteCorpusStudy.from_json(value)
        for value in (b"{}", b"null", b"\xff", b" " * (st.nn.BYTE_CORPUS_STUDY_MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                st.nn.ByteCorpusStudy.from_json(value)
        study = st.nn.ByteCorpusStudy.from_json(raw)
        with self.assertRaises(AttributeError):
            study.total_updates = 99
        for call in (study.validate_segment, study.advance_wgpu):
            with self.assertRaises(ValueError):
                call(3)
            for value in (True, 1.5, "1"):
                with self.assertRaises(TypeError):
                    call(value)
            with self.assertRaises(OverflowError):
                call(-1)
        with self.assertRaises(ValueError):
            study.checkpoint_from_json("{}")

    def test_checkpoint_transport_and_foreign_rejection_without_device(self):
        raw = fixture()[0]
        study = st.nn.ByteCorpusStudy.from_json(raw)
        # Synthetic, shape-valid step-one state. Not a measured training result.
        record = resume_fixture()[5]
        record["request_sha256"] = study.request_sha256
        saved = study.checkpoint_from_json(protocol.encoded(record))
        self.assertEqual(saved.completed_updates, 1)
        self.assertEqual(saved.request_sha256, study.request_sha256)
        exact = saved.to_json()
        self.assertEqual(study.checkpoint_from_json(exact).to_json(), exact)
        study.validate_segment(resume=saved)
        with self.assertRaises(ValueError):
            study.validate_segment(0, resume=saved)
        other = st.nn.ByteCorpusStudy.from_json(raw + b" ")
        with self.assertRaises(ValueError):
            other.checkpoint_from_json(exact)
        for call in (other.validate_segment, other.advance_wgpu):
            with self.assertRaises(ValueError):
                call(resume=saved)
        record["cases"][0]["training"] = []
        with self.assertRaises(ValueError):
            study.checkpoint_from_json(protocol.encoded(record))
        self.assertEqual(saved.to_json(), exact)

    def test_case_handoff_is_portable_typed_and_request_bound(self):
        raw = fixture()[0]
        study = st.nn.ByteCorpusStudy.from_json(raw)
        record = resume_fixture()[5]
        record["request_sha256"] = study.request_sha256
        saved = study.checkpoint_from_json(protocol.encoded(record))
        before = saved.to_json()
        for case in record["cases"]:
            model = study.case_checkpoint(saved, case["name"])
            self.assertIs(type(model), st.nn.ByteDecoderCheckpoint)
            self.assertEqual(model.attempted_revision, saved.completed_updates)
            self.assertEqual(model.to_json(),
                             st.nn.ByteDecoderCheckpoint.from_json(case["model_json"]).to_json())
            if not st.wgpu_kernel_reports_available():
                with self.assertRaises(NotImplementedError):
                    model.restore_wgpu()
        other = st.nn.ByteCorpusStudy.from_json(raw + b" ")
        with self.assertRaises(ValueError):
            other.case_checkpoint(saved, record["cases"][0]["name"])
        for name in ("", "unknown", record["cases"][0]["name"] + " "):
            with self.assertRaisesRegex(ValueError, "case name"):
                study.case_checkpoint(saved, name)
        for name in (None, True, 0):
            with self.assertRaises(TypeError):
                study.case_checkpoint(saved, name)
        with self.assertRaises(TypeError):
            study.case_checkpoint(record, record["cases"][0]["name"])
        self.assertEqual(saved.to_json(), before)

    def test_measured_preparation_preflight_and_selection_bounds(self):
        raw = two_head_source()
        for fisher in (False, True):
            recipe = st.nn.ByteCorpusBiasCalibration.from_json(raw, [0, 1], fisher_rao_control=fisher)
            if not st.wgpu_kernel_reports_available():
                with self.assertRaises(NotImplementedError):
                    recipe.prepare_wgpu()
        for indices in ([], [0, 0], [2], itertools.count()):
            with self.assertRaises(ValueError):
                st.nn.ByteCorpusBiasCalibration.from_json(raw, indices)
        for indices in ([False], [0.5], ["0"]):
            with self.assertRaises(TypeError):
                st.nn.ByteCorpusBiasCalibration.from_json(raw, indices)
        source = headwise_fixture()[3]
        for cls in (st.nn.ByteCorpusHeadwiseControls, st.nn.ByteCorpusMixtureControls):
            cls.from_json(source)
            with self.assertRaises(ValueError):
                cls.from_json(raw)
            if not st.wgpu_kernel_reports_available():
                with self.assertRaises(NotImplementedError):
                    cls.from_json(source).prepare_wgpu()

    def test_artifact_identity_policy_companions_and_cpu_restore(self):
        raw = fixture()[0]
        study = st.nn.ByteCorpusStudy.from_json(raw)
        record = resume_fixture()[5]
        record["request_sha256"] = study.request_sha256
        saved = study.checkpoint_from_json(protocol.encoded(record))
        name = record["cases"][0]["name"]
        policy = "fixed_window_frozen.v1"
        artifact = study.case_artifact(saved, name, policy=policy)
        payload = artifact.to_json()
        self.assertEqual(artifact.artifact_id, "sha256:" + hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(artifact.policy, policy)
        self.assertEqual(artifact.case_name, name)
        self.assertEqual(artifact.request_sha256, study.request_sha256)
        self.assertEqual(artifact.study_checkpoint_sha256,
                         hashlib.sha256(saved.to_json().encode()).hexdigest())
        for source in (payload, payload.encode()):
            loaded = st.nn.ByteCorpusModelArtifact.from_json(source, expected_id=artifact.artifact_id)
            study.validate_artifact(saved, loaded)
            self.assertEqual(loaded.to_json(), payload)
            self.assertEqual(loaded.checkpoint().to_json(), study.case_checkpoint(saved, name).to_json())
            if not st.wgpu_kernel_reports_available():
                with self.assertRaises(NotImplementedError):
                    loaded.restore_wgpu()
        with self.assertRaises(AttributeError):
            artifact.artifact_id = "sha256:" + "0" * 64
        with self.assertRaises(TypeError):
            study.case_artifact(saved, name)
        for invalid in (None, True, {}):
            with self.assertRaises(TypeError):
                study.case_artifact(saved, name, policy=invalid)
        for invalid in ("", "mutable", "fixed_window_frozen.v2"):
            with self.assertRaises(ValueError):
                study.case_artifact(saved, name, policy=invalid)
        for source in (payload + " ", payload.replace(name, name + "x"), b"\xff"):
            with self.assertRaises(ValueError):
                st.nn.ByteCorpusModelArtifact.from_json(source, expected_id=artifact.artifact_id)
        for source in (json.loads(payload), bytearray(payload.encode()), None):
            with self.assertRaises(TypeError):
                st.nn.ByteCorpusModelArtifact.from_json(source, expected_id=artifact.artifact_id)
        with self.assertRaises(TypeError):
            st.nn.ByteCorpusModelArtifact.from_json(payload)
        with self.assertRaises(ValueError):
            st.nn.ByteCorpusModelArtifact.from_json(b" " * (st.nn.BYTE_CORPUS_STUDY_MAX_BYTES + 1),
                                                   expected_id=artifact.artifact_id)
        with self.assertRaises(ValueError):
            st.nn.ByteCorpusStudy.from_json(raw + b" ").validate_artifact(saved, artifact)
        # Structurally valid unrelated history changes must break the companion binding.
        record["cases"][1]["training"][0]["ce"] += 0.25
        changed = study.checkpoint_from_json(protocol.encoded(record))
        with self.assertRaises(ValueError):
            study.validate_artifact(changed, artifact)
        self.assertEqual(artifact.to_json(), payload)

    def test_artifact_preserves_explicit_empty_source_case_name(self):
        request = protocol.decode_json(fixture()[0])
        request["cases"][0]["name"] = ""
        study = st.nn.ByteCorpusStudy.from_json(protocol.encoded(request))
        record = resume_fixture()[5]
        record["request_sha256"] = study.request_sha256
        record["cases"][0]["name"] = ""
        saved = study.checkpoint_from_json(protocol.encoded(record))
        artifact = study.case_artifact(saved, "", policy="fixed_window_frozen.v1")
        loaded = st.nn.ByteCorpusModelArtifact.from_json(artifact.to_json(), expected_id=artifact.artifact_id)
        self.assertEqual(loaded.case_name, "")
        study.validate_artifact(saved, loaded)

    def test_query_preparation_is_cpu_only_immutable_and_rust_owned(self):
        source = mixture_fixture()[0]
        recipe = st.nn.ByteCorpusQueryMixtureControls.from_json(source)
        prepared = recipe.prepare()
        report = protocol.decode_json(prepared.report_json)
        request = protocol.decode_json(prepared.request_json)
        self.assertTrue(report["no_fitting"])
        self.assertTrue(report["no_updates_consumed"])
        self.assertEqual(report["source_request_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(report["request_sha256"], prepared.study().request_sha256)
        self.assertEqual(request["schema"], protocol.REQUEST_V8)
        self.assertEqual(len(request["cases"]), 25)
        self.assertEqual(recipe.prepare().request_json, prepared.request_json)
        with self.assertRaises(AttributeError):
            prepared.request_json = "{}"
        with self.assertRaises(ValueError):
            st.nn.ByteCorpusQueryMixtureControls.from_json(fixture()[0])

    def test_cpu_only_execution_is_not_a_fallback(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("requires CPU-only artifact")
        study = st.nn.ByteCorpusStudy.from_json(fixture()[0])
        for call in (study.run_wgpu, study.advance_wgpu):
            with self.assertRaisesRegex(NotImplementedError, "wgpu"):
                call()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class StudyGpu(unittest.TestCase):
    def test_detached_calls_share_a_recipe_not_mutable_models(self):
        study = st.nn.ByteCorpusStudy.from_json(fixture()[0])
        expected = study.run_wgpu()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: study.run_wgpu(), range(2)))
        self.assertEqual(results, [expected, expected])

    def test_full_run_and_split_resume_are_identical(self):
        raw = frozen_fixture()[0]
        study = st.nn.ByteCorpusStudy.from_json(raw)
        full = study.run_wgpu()
        first = study.advance_wgpu(1)
        saved = first.checkpoint.to_json()
        same = st.nn.ByteCorpusStudy.from_json(raw)
        loaded = same.checkpoint_from_json(saved.encode())
        resumed = same.advance_wgpu(resume=loaded)
        self.assertEqual(resumed.report_json, full)
        self.assertEqual(resumed.checkpoint.completed_updates, 2)
        self.assertEqual(first.checkpoint.to_json(), saved)
        self.assertEqual(same.advance_wgpu(resume=resumed.checkpoint).report_json, resumed.report_json)
        with self.assertRaises(ValueError):
            study.advance_wgpu(0, resume=first.checkpoint)
        self.assertEqual(first.checkpoint.to_json(), saved)
        self.assertEqual(study.run_wgpu(), full)

    def test_preparation_chain_uses_one_shared_runner(self):
        source = two_head_source()
        flat = st.nn.ByteCorpusBiasCalibration.from_json(source, [0, 1]).prepare_wgpu()
        fisher = st.nn.ByteCorpusBiasCalibration.from_json(
            source, [0, 1], fisher_rao_control=True).prepare_wgpu()
        headwise = st.nn.ByteCorpusHeadwiseControls.from_json(fisher.request_json).prepare_wgpu()
        mixed = st.nn.ByteCorpusMixtureControls.from_json(fisher.request_json).prepare_wgpu()
        query = st.nn.ByteCorpusQueryMixtureControls.from_json(mixed.request_json).prepare()
        for version, prepared in ((4, flat), (5, fisher), (6, headwise), (7, mixed), (8, query)):
            with self.subTest(version=version):
                self.assertEqual(protocol.decode_json(prepared.request_json)["schema"],
                                 f"spiraltorch.byte_corpus.request.v{version}")
                study = prepared.study()
                first = study.advance_wgpu(1)
                resumed = study.advance_wgpu(resume=study.checkpoint_from_json(first.checkpoint.to_json()))
                self.assertEqual(resumed.report_json, study.run_wgpu())
                if version == 8:
                    self.check_fresh_case_inference(study, prepared.request_json, resumed.checkpoint)

    def check_fresh_case_inference(self, study, raw, saved):
        request = protocol.decode_json(raw)
        before = saved.to_json()
        record = json.loads(before)
        for case in record["cases"]:
            model = study.case_checkpoint(saved, case["name"])
            self.assertEqual(model.to_json(),
                             st.nn.ByteDecoderCheckpoint.from_json(case["model_json"]).to_json())
            artifact = study.case_artifact(saved, case["name"], policy="fixed_window_frozen.v1")
            loaded = st.nn.ByteCorpusModelArtifact.from_json(artifact.to_json(), expected_id=artifact.artifact_id)
            study.validate_artifact(saved, loaded)
            self.assertEqual(loaded.checkpoint().to_json(), model.to_json())
        # A learned contextual router, not just the plain attention arm.
        index = next(i for i, case in enumerate(request["cases"])
                     if case.get("mixture", {}).get("query_projection", {}).get("weight_update") == "train")
        case = record["cases"][index]
        checkpoint = study.case_checkpoint(saved, case["name"])
        model = checkpoint.restore_wgpu()
        self.assertNotEqual(model.tensor_device().adapter_info()["device_type"], "Cpu")
        batch = st.nn.ByteLmBatch.from_documents(
            [bytes(doc) for doc in request["validation_documents"]],
            [tuple(window) for window in request["validation_batches"][0]], request["config"]["steps"],
        )
        forward = model.forward(model.prepare_batch(batch))
        expected = {
            "artifact_id": None,
            "revision": saved.completed_updates,
            "logits": forward.prediction_tensor().snapshot().read_values(),
            "loss": forward.next_byte_loss().loss_tensor().snapshot().read_values()[0],
            "model_json": checkpoint.to_json(),
        }
        # Rust JSON uses the shortest f32 spelling; Python decodes it as f64.
        # Compare the underlying f32 bits, not those different decimal spellings.
        self.assertEqual(struct.pack("<f", expected["loss"]),
                         struct.pack("<f", case["validation"][-1]["batch_losses"][0]))
        child = subprocess.run(
            [sys.executable, "-I", "-B", "-c", FRESH_INFERENCE],
            input=json.dumps({"model_json": checkpoint.to_json(),
                              "documents": request["validation_documents"],
                              "windows": request["validation_batches"][0],
                              "steps": request["config"]["steps"]}),
            text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), expected)
        artifact = study.case_artifact(saved, case["name"], policy="fixed_window_frozen.v1")
        restricted = artifact.restore_wgpu()
        for method in ("sgd", "sgd_with_rates", "backward", "parameter_tensors", "checkpoint_snapshot"):
            self.assertFalse(hasattr(restricted, method), method)
        prepared = restricted.prepare_batch(batch)
        with self.assertRaises(TypeError):
            restricted.forward(prepared, pair_biases=[None])
        retained = restricted.forward(prepared)
        self.assertEqual(retained.artifact_id, artifact.artifact_id)
        self.assertEqual(retained.prediction_tensor().snapshot().read_values(), expected["logits"])
        # Failed candidate import leaves the active inference owner and retained forward intact.
        with self.assertRaises(ValueError):
            st.nn.ByteCorpusModelArtifact.from_json(artifact.to_json() + " ", expected_id=artifact.artifact_id)
        self.assertEqual(restricted.forward(prepared).prediction_tensor().snapshot().read_values(), expected["logits"])
        self.assertEqual(restricted.attempted_updates, expected["revision"])
        del restricted
        self.assertEqual(retained.artifact_id, artifact.artifact_id)
        self.assertEqual(retained.prediction_tensor().snapshot().read_values(), expected["logits"])
        fresh = subprocess.run(
            [sys.executable, "-I", "-B", "-c", FRESH_INFERENCE],
            input=json.dumps({"artifact_json": artifact.to_json(), "expected_id": artifact.artifact_id,
                              "documents": request["validation_documents"],
                              "windows": request["validation_batches"][0], "steps": request["config"]["steps"]}),
            text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(json.loads(fresh.stdout), dict(expected, artifact_id=artifact.artifact_id))
        self.assertEqual(saved.to_json(), before)


if __name__ == "__main__":
    unittest.main()

"""Rust study clients; fabricated fixtures below test contracts, not quality."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
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
)


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


if __name__ == "__main__":
    unittest.main()

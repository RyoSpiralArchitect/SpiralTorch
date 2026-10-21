"""Separate opt-in diagnostic tests, never a replacement for the original suite."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from threading import Barrier
import unittest

import spiraltorch as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_nn_byte_study as source


def portable_reference(cursor=1, suffix=b"", head_overflow=False):
    # Fabricated state exercises preflight only, not proof of an actual update.
    request = source.protocol.decode_json(source.fixture()[0])
    for case in request["cases"]:
        for parameter in case["parameters"]:
            parameter["values"] = [0.1] * len(parameter["values"])
        if head_overflow:
            for parameter in case["parameters"][-2:]:
                parameter["values"] = [float.fromhex("0x1.fffffep+127")] * len(parameter["values"])
    raw = source.protocol.encoded(request) + suffix
    study = st.nn.ByteCorpusStudy.from_json(raw)
    record = source.resume_fixture()[5]
    record["request_sha256"] = study.request_sha256
    record["completed_updates"] = cursor
    for case in record["cases"]:
        model = json.loads(case["model_json"])
        model["attempted_revision"] = str(cursor)
        case["model_json"] = json.dumps(model)
        case["training"] = [{"revision": i, "ce": 2.0} for i in range(1, cursor + 1)]
        if cursor == 2:
            case["validation"].append({"revision": 2, "batch_losses": [2.0]})
    return study, study.checkpoint_from_json(source.protocol.encoded(record))


class BoundaryPortable(unittest.TestCase):
    def test_export_is_opaque_and_stubbed(self):
        name = "ByteCorpusBoundaryObservation"
        self.assertIn(name, st.nn.__all__)
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        self.assertIs(namespace[name], getattr(st._rs.nn, name))
        with self.assertRaises(TypeError):
            getattr(st.nn, name)()
        stub = ast.parse((source.ROOT / "bindings/st-py/spiraltorch/__init__.pyi").read_text())
        classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
        self.assertIn("_NnByteCorpusBoundaryObservation", classes)
        self.assertIn("report_json", {node.name for node in classes["_NnByteCorpusBoundaryObservation"].body
                                     if isinstance(node, ast.FunctionDef)})
        self.assertIn(name, {node.target.id for node in classes["_NnModule"].body
                             if isinstance(node, ast.AnnAssign)})
        self.assertIn("run_wgpu_with_retained_boundary", {
            node.name for node in classes["_NnByteCorpusStudy"].body
            if isinstance(node, ast.FunctionDef)
        })

    def test_bad_case_cursor_and_foreign_reference_reject_before_execution(self):
        study, first = portable_reference()
        with self.assertRaises(ValueError):
            study.run_wgpu_with_retained_boundary(first, "missing")
        for cursor in (0, 2):
            _, wrong = portable_reference(cursor)
            with self.assertRaises(ValueError):
                study.run_wgpu_with_retained_boundary(wrong, "plain")
        _, foreign = portable_reference(suffix=b" ")
        with self.assertRaises(ValueError):
            study.run_wgpu_with_retained_boundary(foreign, "plain")
        with self.assertRaises(TypeError):
            study.run_wgpu_with_retained_boundary(first, True)

    def test_cpu_artifact_rejects_valid_diagnostic_without_fallback(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("CPU-only artifact required")
        study, first = portable_reference()
        with self.assertRaisesRegex(NotImplementedError, "wgpu"):
            study.run_wgpu_with_retained_boundary(first, "plain")


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class BoundaryGpu(unittest.TestCase):
    def test_failed_study_outcome_survives_unavailable_boundary_reads(self):
        study, first = portable_reference(head_overflow=True)
        observed = study.run_wgpu_with_retained_boundary(first, "plain")
        original = observed.run_outcome_json
        outcome = json.loads(original)
        self.assertEqual(outcome["status"], "error")
        self.assertFalse(outcome["all_cases_completed"])
        self.assertFalse(outcome["boundary_captured"])
        self.assertFalse(outcome["release_ready"])
        self.assertTrue(outcome["error"])
        self.assertIsNone(observed.report_json)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "boundary unavailable"):
                observed.read_boundary()
            self.assertEqual(observed.run_outcome_json, original)
            self.assertIsNone(observed.report_json)

    def test_same_process_outcome_then_one_shot_boundary_for_sgd_and_adamw(self):
        design = source.protocol.decode_json(source.frozen_fixture()[0])
        for adamw in (False, True):
            with self.subTest(adamw=adamw):
                request = source.optimizer_request(copy.deepcopy(design)) if adamw else design
                study = st.nn.ByteCorpusStudy.from_json(source.protocol.encoded(request))
                first = study.advance_wgpu(1)
                saved = first.checkpoint.to_json()
                observed = study.run_wgpu_with_retained_boundary(first.checkpoint, "frozen")
                raw_outcome = observed.run_outcome_json
                outcome = json.loads(raw_outcome)
                self.assertEqual(outcome["status"], "completed")
                self.assertTrue(outcome["all_cases_completed"])
                self.assertTrue(outcome["boundary_captured"])
                self.assertFalse(outcome["release_ready"])
                self.assertEqual(outcome["request_sha256"], study.request_sha256)
                self.assertEqual(outcome["case"], "frozen")
                raw_report = observed.report_json
                self.assertIsInstance(raw_report, str)
                self.assertEqual(raw_report, study.run_wgpu())
                self.assertEqual(outcome["report"], json.loads(raw_report))
                reference = study.case_checkpoint(first.checkpoint, "frozen").to_json()
                self.assertEqual(first.checkpoint.to_json(), saved)
                del study
                ready = Barrier(2, timeout=10)

                def consume():
                    ready.wait()
                    try:
                        return "read", observed.read_boundary().to_json()
                    except RuntimeError as error:
                        return "error", str(error)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: consume(), range(2)))
                self.assertEqual([value for kind, value in results if kind == "read"], [reference])
                errors = [value for kind, value in results if kind == "error"]
                self.assertEqual(len(errors), 1)
                self.assertIn("already consumed", errors[0])
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    observed.read_boundary()
                self.assertEqual(observed.run_outcome_json, raw_outcome)
                self.assertEqual(observed.report_json, raw_report)
                with self.assertRaises(AttributeError):
                    observed.run_outcome_json = "{}"
                with self.assertRaises(AttributeError):
                    observed.report_json = "{}"


if __name__ == "__main__":
    unittest.main()

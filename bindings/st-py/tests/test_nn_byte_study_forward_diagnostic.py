"""Opt-in forward evidence, not ordinary replay or release qualification."""
import ast
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
from test_nn_byte_study_boundary import portable_reference


class ForwardDiagnosticPortable(unittest.TestCase):
    def test_exports_are_opaque_and_stubbed(self):
        name = "ByteCorpusForwardDiagnosticObservation"
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        self.assertIs(namespace[name], getattr(st._rs.nn, name))
        self.assertIn(name, st.nn.__all__)
        with self.assertRaises(TypeError):
            getattr(st.nn, name)()
        constants = {
            "BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES": 64 * 1024 * 1024,
            "BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_GRAPHS": 256,
            "BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_JSON_BYTES": 256 * 1024 * 1024,
        }
        stub = ast.parse((source.ROOT / "bindings/st-py/spiraltorch/__init__.pyi").read_text())
        classes = {n.name: n for n in stub.body if isinstance(n, ast.ClassDef)}
        exports = {n.target.id for n in classes["_NnModule"].body if isinstance(n, ast.AnnAssign)}
        for key, value in constants.items():
            self.assertEqual(namespace[key], value)
            self.assertIn(key, st.nn.__all__)
            self.assertIn(key, exports)
        self.assertIn(name, exports)
        methods = {n.name for n in classes["_NnByteCorpusStudy"].body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"validate_forward_diagnostic", "run_wgpu_with_forward_diagnostic"} <= methods)
        methods = {n.name for n in classes["_NnByteCorpusForwardDiagnosticObservation"].body
                   if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"run_outcome_json", "report_json", "read_diagnostic_json"} <= methods)

    def test_cpu_preflight_accepts_fresh_and_remaining_update_selection(self):
        study, first = portable_reference()
        study.validate_forward_diagnostic("plain", 1)
        study.validate_forward_diagnostic("plain", 2)
        study.validate_forward_diagnostic("plain", 2, resume=first)
        study.validate_forward_diagnostic("plain", 2, max_bytes=1)

    def test_invalid_selection_rejects_before_execution(self):
        study, first = portable_reference()
        _, foreign = portable_reference(suffix=b" ")
        for call in (study.validate_forward_diagnostic, study.run_wgpu_with_forward_diagnostic):
            for name, update, kwargs in [
                ("missing", 2, {}),
                ("plain", 0, {}),
                ("plain", 3, {}),
                ("plain", 1, {"resume": first}),
                ("plain", 2, {"resume": foreign}),
                ("plain", 2, {"max_bytes": 0}),
                ("plain", 2, {"max_bytes": 64 * 1024 * 1024 + 1}),
            ]:
                with self.subTest(method=call.__name__, case=name, update=update, kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        call(name, update, **kwargs)

    def test_indices_do_not_coerce_bool_float_string_or_index_objects(self):
        class Coerced:
            def __index__(self):
                return 2

        study, _ = portable_reference()
        for call in (study.validate_forward_diagnostic, study.run_wgpu_with_forward_diagnostic):
            for value in (True, False, 2.0, "2", Coerced()):
                with self.subTest(method=call.__name__, value=type(value).__name__):
                    with self.assertRaises(TypeError):
                        call("plain", value)
                    with self.assertRaises(TypeError):
                        call("plain", 2, max_bytes=value)
            for value in (-1, 1 << 256):
                with self.assertRaises(OverflowError):
                    call("plain", value)
                with self.assertRaises(OverflowError):
                    call("plain", 2, max_bytes=value)
            with self.assertRaises(TypeError):
                call(True, 2)

    def test_cpu_artifact_refuses_execution_without_a_fallback(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("CPU-only artifact required")
        study, _ = portable_reference()
        with self.assertRaisesRegex(NotImplementedError, "wgpu"):
            study.run_wgpu_with_forward_diagnostic("plain", 2)


@unittest.skipUnless(
    os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1"
    and os.environ.get("SPIRALTORCH_RUN_FORWARD_GRAPH_DIAGNOSTIC_TESTS") == "1",
    "separately authorized real-GPU forward diagnostic opt-in",
)
class ForwardDiagnosticGpu(unittest.TestCase):
    def test_one_shot_read_is_thread_safe_and_keeps_original_outcome(self):
        study, _ = portable_reference()
        observation = study.run_wgpu_with_forward_diagnostic("plain", 2)
        original = observation.run_outcome_json
        report = observation.report_json
        outcome = json.loads(original)
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["capture_status"], "copies_submitted")
        self.assertIsNone(outcome["resume_checkpoint_json_sha256"])
        self.assertFalse(outcome["release_ready"])
        self.assertEqual(json.loads(report), outcome["report"])
        del study
        ready = Barrier(2, timeout=10)

        def consume():
            ready.wait()
            try:
                return "read", observation.read_diagnostic_json()
            except RuntimeError as error:
                return "error", str(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: consume(), range(2)))
        reads = [value for kind, value in results if kind == "read"]
        errors = [value for kind, value in results if kind == "error"]
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("already consumed", errors[0])
        diagnostic = json.loads(reads[0])
        self.assertEqual(diagnostic["value_encoding"], "f32_raw_u32_bits")
        self.assertEqual(diagnostic["counter_encoding"], "u64_decimal_strings")
        self.assertEqual(diagnostic["parameter_revision"], "1")
        self.assertFalse(diagnostic["release_ready"])
        self.assertEqual(len(diagnostic["graphs"]), 5)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            observation.read_diagnostic_json()
        self.assertEqual(observation.run_outcome_json, original)
        self.assertEqual(observation.report_json, report)
        with self.assertRaises(AttributeError):
            observation.run_outcome_json = "{}"

    def test_small_capture_budget_does_not_replace_learning_outcome(self):
        study, _ = portable_reference()
        observation = study.run_wgpu_with_forward_diagnostic("plain", 2, max_bytes=1)
        original = observation.run_outcome_json
        outcome = json.loads(original)
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["capture_status"], "error")
        self.assertTrue(outcome["diagnostic_error"])
        self.assertFalse(outcome["release_ready"])
        with self.assertRaisesRegex(RuntimeError, "diagnostic unavailable"):
            observation.read_diagnostic_json()
        self.assertEqual(observation.run_outcome_json, original)


if __name__ == "__main__":
    unittest.main()

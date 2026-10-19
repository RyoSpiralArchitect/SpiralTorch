#!/usr/bin/env python3
"""Installed-native smoke validation, including dependency-free execution."""

import importlib.util
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import spiraltorch as st


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/smoke_learning_stack.py"
SPEC = importlib.util.spec_from_file_location("release_learning_smoke", SCRIPT)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class LearningSmokeTests(unittest.TestCase):
    def test_tolerance_helper_rejects_nonfinite_and_missing_values(self):
        for actual, expected in (([float("nan")], [0]), ([0], [float("inf")]),
                                 ([1], [0]), ([], [0]), ([0], [])):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(AssertionError):
                SMOKE.close(actual, expected)
        SMOKE.close([1.0], [1.0], atol=0, rtol=0)

    def test_finite_difference_helper_has_independent_known_derivative(self):
        values = [0.25, -0.5, 2.0]
        derivative = SMOKE.finite_difference(lambda x: sum(value ** 2 for value in x), values)
        SMOKE.close(derivative, [2 * value for value in values], atol=1e-10, rtol=0)
        self.assertEqual(values, [0.25, -0.5, 2.0])

    def test_missing_native_owner_is_failure_not_skip(self):
        with mock.patch.object(st, "ToposResonatorKernel", object()):
            with self.assertRaises(AssertionError):
                SMOKE.main()

    def test_non_extension_owner_is_rejected(self):
        native = mock.Mock(__file__="not_a_native_extension.py")
        with mock.patch.object(SMOKE.importlib, "import_module", return_value=native):
            with self.assertRaises(AssertionError):
                SMOKE.main()

    def test_native_smoke_without_optional_ml_packages(self):
        package_root = str(Path(st.__file__).resolve().parent.parent)
        code = (
            "import sys, runpy; "
            f"sys.path.insert(0, {package_root!r}); "
            "sys.modules.update({name: None for name in "
            "['torch', 'numpy', 'transformers', 'pytest', 'peft', 'accelerate', 'safetensors']}); "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code],
                                text=True, capture_output=True, timeout=60,
                                cwd=SCRIPT.parent)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["topos"]["updates"], 24)
        self.assertLess(report["topos"]["final_loss"], report["topos"]["initial_loss"])
        self.assertEqual(report["resident_plan"]["gpu_execution"], "not attempted")
        self.assertEqual(report["byte_decoder"], {
            "modes": 8, "checkpoint_roundtrip": "exact", "gpu_execution": "not attempted"})
        self.assertEqual(report["byte_corpus"], {
            "native_exports": 13, "request_identity": "exact bytes",
            "segment_validation": "CPU-only", "gpu_execution": "not attempted"})
        self.assertEqual(report["byte_clipping_api"], {
            "installed_stub_signatures": 2,
            "native_methods": 2 if st.wgpu_kernel_reports_available() else 0,
            "gpu_execution": "not attempted"})
        self.assertIn("not optimizer resume", report["topos"]["handoff"])
        self.assertEqual(set(report), {"scope", "sequential", "topos", "wave_gate",
                                       "elliptic", "fractional_history", "resident_plan",
                                       "byte_decoder", "byte_corpus", "byte_clipping_api"})

    def test_clipped_gate_reads_installed_stub_not_checkout(self):
        package = Path(st.__file__).resolve().parent
        with mock.patch.object(Path, "read_text", autospec=True,
                               side_effect=FileNotFoundError("missing installed stub")) as read:
            with self.assertRaises(FileNotFoundError):
                SMOKE.byte_clipped_api()
        self.assertEqual(read.call_args.args[0], package / "__init__.pyi")

    def test_clipped_gate_rejects_missing_or_changed_installed_signature(self):
        source = (Path(st.__file__).resolve().parent / "__init__.pyi").read_text(encoding="utf-8")
        for mutation in ("missing", "optional_norm", "positional_norm", "wrong_return"):
            tree = ast.parse(source)
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == "_NnResidentByteDecoder")
            method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                          and node.name == "sgd_with_rates_clipped")
            if mutation == "missing":
                owner.body.remove(method)
            elif mutation == "optional_norm":
                method.args.kw_defaults[0] = ast.Constant(value=1.0)
            elif mutation == "positional_norm":
                method.args.args += method.args.kwonlyargs
                method.args.kwonlyargs = []
                method.args.kw_defaults = []
            else:
                method.returns = ast.Name(id="Tensor")
            with self.subTest(mutation=mutation), mock.patch.object(SMOKE.ast, "parse", return_value=tree), \
                    self.assertRaises((AssertionError, KeyError)):
                SMOKE.byte_clipped_api()

    def test_clipped_gate_rejects_missing_native_method_and_python_substitute(self):
        native = SMOKE.importlib.import_module("spiraltorch.spiraltorch")
        with mock.patch.object(st.nn, "ResidentByteDecoder", object()), self.assertRaises(AssertionError):
            SMOKE.byte_clipped_api()
        substitute = SimpleNamespace()
        with mock.patch.object(st.nn, "ResidentByteDecoder", substitute), \
                mock.patch.object(native.nn, "ResidentByteDecoder", substitute), \
                mock.patch.object(st, "wgpu_kernel_reports_available", return_value=True), \
                self.assertRaises(AttributeError):
            SMOKE.byte_clipped_api()

    def test_foreign_import_guard_rejects_roots_and_submodules(self):
        guard = SMOKE.NativeOnly()
        for name in SMOKE.FOREIGN:
            for module in (name, name + ".internal"):
                with self.subTest(module=module), self.assertRaises(AssertionError):
                    guard.find_spec(module)
        self.assertIsNone(guard.find_spec("spiraltorch"))

    def test_missing_study_owner_is_failure_not_skip(self):
        for name in ("ByteCorpusStudy", "ByteCorpusModelArtifact", "ResidentByteInference",
                     "ByteCorpusServingGate", "ResidentByteCandidate", "ResidentByteServing"):
            with self.subTest(name=name), mock.patch.object(st.nn, name, object()), \
                    self.assertRaises(AssertionError):
                SMOKE.byte_corpus_portable()

    def test_study_star_import_omission_is_failure_not_skip(self):
        native = SMOKE.importlib.import_module("spiraltorch.spiraltorch")
        for label, module in (("facade", st.nn), ("native", native.nn)):
            for missing in ("ByteCorpusStudySegment", "ByteCorpusModelArtifact", "ResidentByteInference",
                            "ByteCorpusServingGate", "ResidentByteCandidate", "ResidentByteServing"):
                exports = [name for name in module.__all__ if name != missing]
                with self.subTest(surface=label, missing=missing), \
                        mock.patch.object(module, "__all__", exports), self.assertRaises(AssertionError):
                    SMOKE.byte_corpus_portable()

    def test_all_wheel_paths_execute_smoke_before_artifact_upload(self):
        for workflow in ("wheels.yml", "release_wheels.yml"):
            with self.subTest(workflow=workflow):
                text = (ROOT / ".github/workflows" / workflow).read_text()
                command = "python -I ../../tools/smoke_learning_stack.py"
                self.assertIn(command, text)
                self.assertLess(text.index(command), text.index("uses: actions/upload-artifact@"))
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("python -I tools/smoke_learning_stack.py", ci)
        self.assertIn("python -I tests/test_release_learning_smoke.py", ci)
        self.assertIn("python -I -B bindings/st-py/tests/test_nn_byte_decoder.py -v", ci)
        self.assertEqual(ci.count("python -I -B bindings/st-py/tests/test_nn_byte_serving.py -v"), 2)

    def test_homogeneous_substitute_does_not_pass_headwise_smoke(self):
        constructor = st.nn.ByteDecoderGeometryPlan

        class HomogeneousSubstitute:
            def __init__(self, *args):
                self.inner = constructor(*args)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def with_head_pair_metrics(self, grid):
                return self.inner.with_head_pair_metrics([[row[0]] * len(row) for row in grid])

        with mock.patch.object(st.nn, "ByteDecoderGeometryPlan", HomogeneousSubstitute), \
                self.assertRaisesRegex(AssertionError, "checkpoint schema"):
            SMOKE.byte_decoder_portable()


if __name__ == "__main__":
    unittest.main()

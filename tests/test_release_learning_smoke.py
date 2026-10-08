#!/usr/bin/env python3
"""Installed-native smoke validation, including dependency-free execution."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

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
            "['torch', 'numpy', 'transformers', 'pytest']}); "
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
        self.assertIn("not optimizer resume", report["topos"]["handoff"])
        self.assertEqual(set(report), {"scope", "sequential", "topos", "wave_gate",
                                       "elliptic", "fractional_history", "resident_plan"})

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


if __name__ == "__main__":
    unittest.main()

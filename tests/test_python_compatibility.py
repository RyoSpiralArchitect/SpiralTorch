#!/usr/bin/env python3
"""Source import boundaries; legacy simulation is not an older interpreter."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "bindings/st-py/spiraltorch"


class PythonCompatibilityTests(unittest.TestCase):
    def test_minimum_interpreter_gate_reuses_installed_linux_wheel(self):
        for workflow in ("ci.yml", "wheels.yml", "release_wheels.yml"):
            with self.subTest(workflow=workflow):
                text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
                gate = text.split("- name: Select minimum supported CPython (Linux)\n", 1)[1]
                gate = gate.split("\n      - name: Upload", 1)[0]
                self.assertEqual(gate.count("if: runner.os == 'Linux'"), 2)
                self.assertIn('python-version: "3.8"', gate)
                self.assertIn("assert sys.version_info[:2] == (3, 8)", gate)
                self.assertIn("--no-deps --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl", gate)
                self.assertIn("python -I tools/smoke_learning_stack.py", gate)
                self.assertNotIn("maturin", gate)
                self.assertNotIn("cargo ", gate)
                self.assertNotIn("continue-on-error", gate)
                self.assertNotIn("|| true", gate)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("python -I tests/test_python_compatibility.py", ci)

    def test_metrics_import_and_fields_across_dataclass_contracts(self):
        for version in ((3, 8), (3, 9), sys.version_info[:2]):
            with self.subTest(version=version):
                code = textwrap.dedent("""
                    import dataclasses
                    import importlib
                    import json
                    import sys
                    import types

                    package = types.ModuleType('compat_probe')
                    package.__path__ = [sys.argv[1]]
                    sys.modules[package.__name__] = package
                    version = tuple(json.loads(sys.argv[2]))
                    original = dataclasses.dataclass
                    if version < (3, 10):
                        def legacy(cls=None, *, init=True, repr=True, eq=True,
                                   order=False, unsafe_hash=False, frozen=False):
                            return original(cls, init=init, repr=repr, eq=eq,
                                            order=order, unsafe_hash=unsafe_hash,
                                            frozen=frozen)
                        dataclasses.dataclass = legacy
                        sys.version_info = version + (0, 'final', 0)
                    module = importlib.import_module('compat_probe.zspace_inference')
                    metrics = module.ZMetrics(0.2, 0.3, 0.4, gradient=[0.5],
                                              gradient_basis='test.control')
                    expected = dict(speed=0.2, memory=0.3, stability=0.4,
                                    gradient=[0.5], drs=0.0, telemetry=None,
                                    gradient_basis='test.control')
                    assert dataclasses.asdict(metrics) == expected
                    changed = dataclasses.replace(metrics, speed=0.6)
                    assert changed.speed == 0.6 and metrics.speed == 0.2
                    assert changed.gradient_basis == metrics.gradient_basis
                    slotted = hasattr(module.ZMetrics, '__slots__')
                    assert slotted == (version >= (3, 10))
                    print(json.dumps({'slotted': slotted, 'fields': len(expected)}))
                """)
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-B", "-c", code,
                     str(PACKAGE), json.dumps(version)],
                    text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stdout)["fields"], 7)


if __name__ == "__main__":
    unittest.main()

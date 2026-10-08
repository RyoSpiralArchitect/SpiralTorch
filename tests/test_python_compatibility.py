#!/usr/bin/env python3
"""Source import boundaries; legacy simulation is not an older interpreter."""

import ast
import collections.abc
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import typing
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "bindings/st-py/spiraltorch"


class PythonCompatibilityTests(unittest.TestCase):
    def test_callback_aliases_do_not_require_runtime_pep585_or_pep604(self):
        # Evaluate only the source aliases with pre-3.9 standard-library types.
        # This is a focused regression, not a Python 3.8 interpreter substitute.
        class LegacyMeta(type):
            def __getitem__(cls, item):
                raise TypeError("legacy runtime type is not subscriptable")

            def __or__(cls, other):
                raise TypeError("legacy runtime type does not support unions")

        class LegacyType(metaclass=LegacyMeta):
            pass

        cases = {
            "hf_generation.py": ("CheckpointGenerationRunner",),
            "hf_ft_status.py": ("HandoffRunner", "HandoffPackageRunner"),
            "hf_adapter_executor.py": (
                "CommandRunner", "ProcessStarted", "ProcessProgress",
                "ProcessStopRequested", "StopRequestLoader",
            ),
        }
        for filename, names in cases.items():
            with self.subTest(module=filename):
                path = PACKAGE / filename
                tree = ast.parse(path.read_text(encoding="utf-8"))
                namespace = {"subprocess": types.SimpleNamespace(CompletedProcess=LegacyType)}
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and node.level == 0:
                        if node.module == "typing":
                            for imported in node.names:
                                namespace[imported.asname or imported.name] = getattr(typing, imported.name)
                        elif node.module == "collections.abc":
                            for imported in node.names:
                                namespace[imported.asname or imported.name] = LegacyType
                found = set()
                for node in tree.body:
                    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                        continue
                    target = node.targets[0]
                    if not isinstance(target, ast.Name) or target.id not in names:
                        continue
                    found.add(target.id)
                    with self.subTest(alias=target.id):
                        self.assertFalse(any(isinstance(part, ast.BitOr) for part in ast.walk(node.value)))
                        value = eval(compile(ast.Expression(node.value), str(path), "eval"), namespace)
                        self.assertIs(typing.get_origin(value), collections.abc.Callable)
                self.assertEqual(found, set(names))

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

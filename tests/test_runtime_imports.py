from __future__ import annotations

import ast
import contextlib
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMPORTS_PATH = (
    REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "runtime_imports.py"
)
TOP_LEVEL_STUB_PATH = (
    REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "__init__.pyi"
)
SPIRALK_STUB_PATH = (
    REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "spiralk.pyi"
)
PY_TYPED_PATH = REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "py.typed"
PYPROJECT_PATH = REPO_ROOT / "bindings" / "st-py" / "pyproject.toml"


def load_runtime_imports():
    spec = importlib.util.spec_from_file_location(
        "spiraltorch_runtime_imports_test",
        RUNTIME_IMPORTS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def top_level_stub_public_names() -> set[str]:
    stub_mod = ast.parse(TOP_LEVEL_STUB_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in stub_mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def source_runtime_public_names() -> list[str]:
    script = """
import json
import spiraltorch as st
print("SPIRALTORCH_PUBLIC_ALL=" + json.dumps(list(st.__all__)))
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT / "bindings" / "st-py")
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"source spiraltorch import failed with code {completed.returncode}:\n"
            + completed.stdout
            + completed.stderr
        )
    for line in completed.stdout.splitlines():
        if line.startswith("SPIRALTORCH_PUBLIC_ALL="):
            return json.loads(line.partition("=")[2])
    raise AssertionError(
        "source spiraltorch import did not emit public surface marker:\n"
        + completed.stdout
        + completed.stderr
    )


class RuntimeImportsTest(unittest.TestCase):
    def test_pyproject_exposes_runtime_preflight_script(self) -> None:
        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'spiral-runtime-preflight = "spiraltorch.runtime_imports:main"',
            pyproject,
        )

    def test_pep561_marker_and_top_level_stubs_are_shipped(self) -> None:
        self.assertTrue(PY_TYPED_PATH.is_file())
        self.assertTrue(TOP_LEVEL_STUB_PATH.is_file())
        self.assertTrue(SPIRALK_STUB_PATH.is_file())

    def test_top_level_stub_covers_runtime_public_all(self) -> None:
        stub_names = top_level_stub_public_names()
        missing = sorted(
            name
            for name in source_runtime_public_names()
            if not name.startswith("_") and name not in stub_names
        )
        self.assertEqual(missing, [])

    def test_top_level_stub_exposes_runtime_import_helpers(self) -> None:
        stub = TOP_LEVEL_STUB_PATH.read_text(encoding="utf-8")

        for helper in [
            "RUNTIME_IMPORT_INSTALL_HINTS",
            "runtime_import_install_hint",
            "runtime_import_names_from_source",
            "runtime_import_probe_fields",
            "runtime_import_required_gate_fields",
            "runtime_imports_from_source",
            "required_runtime_imports_from_source",
        ]:
            with self.subTest(helper=helper):
                self.assertIn(helper, stub)

    def test_top_level_stub_exposes_shared_python_surface_helpers(self) -> None:
        stub = TOP_LEVEL_STUB_PATH.read_text(encoding="utf-8")

        for helper in [
            "AuditEvent",
            "AuditLog",
            "CpuSimdPackedRhs",
            "EllipticTelemetry",
            "EllipticWarp",
            "InferenceResult",
            "InferenceRuntime",
            "SafetyVerdict",
            "SafetyViolation",
            "capture",
            "cpu_simd_prepack_rhs",
            "describe_wgpu_softmax_variants",
            "ensure_zmetrics",
            "export",
            "hg",
            "hpo",
            "inference",
            "optim",
            "rg",
            "share",
            "spiralk",
            "trainer_events_to_atlas_route",
            "trainer_step_event_to_atlas_frame",
            "z",
            "z_metrics",
        ]:
            with self.subTest(helper=helper):
                self.assertIn(helper, stub)

    def test_ft_presets_extend_without_changing_hf_runtime(self) -> None:
        module = load_runtime_imports()

        self.assertEqual(
            module.TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS["hf-runtime"],
            ["transformers", "torch", "tokenizers"],
        )
        self.assertEqual(
            module.runtime_import_preset_modules(["hf-finetune"]),
            [
                "hf-finetune="
                "transformers|torch|tokenizers|datasets|accelerate|safetensors"
            ],
        )
        self.assertEqual(
            module.runtime_import_preset_modules(["hf-peft"]),
            ["hf-peft=transformers|torch|tokenizers|accelerate|peft|safetensors"],
        )

    def test_failed_ft_runtime_imports_emit_install_hints(self) -> None:
        module = load_runtime_imports()
        available = {"transformers", "torch", "tokenizers"}

        def fake_import(name: str):
            if name not in available:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            imported = types.ModuleType(name)
            imported.__version__ = f"{name}-test"
            return imported

        source = {
            "runtime_import_presets": ["hf-peft"],
            "required_runtime_import_presets": ["hf-peft"],
        }
        with mock.patch.object(module.importlib, "import_module", side_effect=fake_import):
            fields = module.runtime_import_probe_fields(source)

        self.assertEqual(fields["runtime_import_presets"], "hf-peft")
        self.assertEqual(
            fields["runtime_imports_requested"],
            "transformers,torch,tokenizers,accelerate,peft,safetensors",
        )
        self.assertEqual(
            fields["runtime_imports_failed"],
            "accelerate,peft,safetensors",
        )
        self.assertEqual(
            fields["runtime_import_failed_install_hints"],
            (
                "accelerate=pip install accelerate,"
                "peft=pip install peft,"
                "safetensors=pip install safetensors"
            ),
        )
        self.assertFalse(fields["required_runtime_import_presets_passed"])
        self.assertEqual(
            module.runtime_import_requirement_failures(fields),
            ["runtime_import_preset_unsatisfied:hf-peft"],
        )

    def test_unknown_runtime_import_has_no_install_hint(self) -> None:
        module = load_runtime_imports()

        self.assertIsNone(
            module.runtime_import_install_hint("spiraltorch_unknown_plugin")
        )
        self.assertEqual(
            module.runtime_import_install_hints_label(
                ["torch", "spiraltorch_unknown_plugin"]
            ),
            "torch=pip install torch",
        )

    def test_preflight_report_requires_requested_preset_when_require_all(self) -> None:
        module = load_runtime_imports()

        def fake_import(name: str):
            if name == "math":
                imported = types.ModuleType(name)
                imported.__version__ = "math-test"
                return imported
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        with mock.patch.object(
            module.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            report = module.runtime_import_preflight_report(
                runtime_imports=["math"],
                runtime_import_presets=["hf-finetune"],
                require_all=True,
            )

        self.assertEqual(report["runtime_imports_imported"], "math")
        self.assertEqual(
            report["required_runtime_import_presets"],
            "hf-finetune",
        )
        self.assertFalse(report["runtime_import_preflight_passed"])
        self.assertEqual(
            report["runtime_import_preflight_failures"],
            "runtime_import_preset_unsatisfied:hf-finetune",
        )
        self.assertIn(
            "runtime_import_preflight_failures",
            "\n".join(module.runtime_import_preflight_summary_lines(report)),
        )

    def test_cli_json_returns_nonzero_for_required_missing_preset(self) -> None:
        module = load_runtime_imports()

        def fake_import(name: str):
            if name in {"transformers", "torch", "tokenizers"}:
                return types.ModuleType(name)
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        stdout = io.StringIO()
        with mock.patch.object(
            module.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            with contextlib.redirect_stdout(stdout):
                exit_code = module.main(
                    ["--preset", "hf-peft", "--require", "--json"]
                )

        self.assertEqual(exit_code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["runtime_import_preflight_passed"])
        self.assertEqual(
            payload["runtime_imports_failed"],
            "accelerate,peft,safetensors",
        )
        self.assertEqual(
            payload["runtime_import_failed_install_hints"],
            (
                "accelerate=pip install accelerate,"
                "peft=pip install peft,"
                "safetensors=pip install safetensors"
            ),
        )

    def test_cli_lists_presets(self) -> None:
        module = load_runtime_imports()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = module.main(["--list-presets"])

        self.assertEqual(exit_code, 0)
        self.assertIn("hf-finetune=", stdout.getvalue())

    def test_cli_writes_json_out_in_quiet_mode(self) -> None:
        module = load_runtime_imports()
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "nested" / "runtime-preflight.json"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--import",
                        "math",
                        "--require",
                        "--json-out",
                        str(output_path),
                        "--quiet",
                    ]
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(payload["runtime_import_preflight_passed"])
        self.assertEqual(payload["runtime_imports_requested"], "math")
        self.assertEqual(payload["required_runtime_imports"], "math")


if __name__ == "__main__":
    unittest.main()

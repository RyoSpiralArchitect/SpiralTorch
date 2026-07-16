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


def pyproject_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts or "=" not in line:
            continue
        name, value = line.split("=", 1)
        scripts[name.strip()] = value.strip().strip('"')
    return scripts


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

    def test_pyproject_console_scripts_resolve_to_callables(self) -> None:
        scripts = pyproject_scripts()
        self.assertIn("spiral-hf-profile", scripts)
        self.assertIn("spiral-zspace-inference-distortion-probe", scripts)
        self.assertIn("spiral-zspace-inference-distortion-sweep", scripts)
        script = """
import importlib
import json
import sys

scripts = json.loads(sys.argv[1])
failures = []
for name, target in scripts.items():
    module_name, _, attr_path = target.partition(":")
    try:
        value = importlib.import_module(module_name)
        for attr in attr_path.split(".") if attr_path else []:
            value = getattr(value, attr)
        if not callable(value):
            failures.append([name, target, "not callable"])
    except Exception as exc:
        failures.append([name, target, f"{exc.__class__.__name__}: {exc}"])
print("SPIRALTORCH_SCRIPT_ENTRYPOINTS=" + json.dumps(failures, sort_keys=True))
"""
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "bindings" / "st-py")
        completed = subprocess.run(
            [sys.executable, "-P", "-c", script, json.dumps(scripts, sort_keys=True)],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )
        marker = "SPIRALTORCH_SCRIPT_ENTRYPOINTS="
        failures = None
        for line in completed.stdout.splitlines():
            if line.startswith(marker):
                failures = json.loads(line[len(marker) :])
                break
        self.assertIsNotNone(
            failures,
            completed.stdout + completed.stderr,
        )
        self.assertEqual(failures, [])

    def test_pyproject_exposes_hf_ft_extras(self) -> None:
        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

        self.assertIn("[project.optional-dependencies]", pyproject)
        for extra in [
            "hf-runtime",
            "hf-finetune",
            "hf-peft",
            "hf-peft-finetune",
            "hf-gpt2-ft",
            "hf-trl-sft",
        ]:
            with self.subTest(extra=extra):
                self.assertIn(f"{extra} = [", pyproject)
        for dependency in [
            '"transformers>=4.40"',
            '"torch>=2.2"',
            '"tokenizers>=0.15"',
            '"datasets>=2.18"',
            '"accelerate>=0.27"',
            '"safetensors>=0.4"',
            '"pyarrow>=15"',
            '"evaluate>=0.4"',
            '"peft>=0.10"',
            '"trl>=0.12"',
            '"trackio>=0.2"',
        ]:
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, pyproject)

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
            "runtime_device_backends_from_source",
            "runtime_device_report_fields",
            "runtime_device_requirement_failures",
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
        self.assertEqual(
            module.runtime_import_preset_modules(["hf-gpt2-ft"]),
            [
                "hf-gpt2-ft="
                "transformers|torch|tokenizers|datasets|accelerate|safetensors|"
                "pyarrow|tqdm|evaluate|peft"
            ],
        )
        self.assertEqual(
            module.runtime_import_preset_modules(["hf-trl-sft"]),
            [
                "hf-trl-sft="
                "transformers|torch|tokenizers|datasets|accelerate|safetensors|"
                "pyarrow|tqdm|peft|trl|trackio"
            ],
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
                ["torch", "pyarrow", "evaluate", "trl", "spiraltorch_unknown_plugin"]
            ),
            (
                "torch=pip install torch,"
                "pyarrow=pip install pyarrow,"
                "evaluate=pip install evaluate,"
                "trl=pip install trl"
            ),
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

    def test_runtime_device_report_fields_gate_ready_backends(self) -> None:
        module = load_runtime_imports()

        def fake_describe(backends, *, continue_on_error=True):
            self.assertTrue(continue_on_error)
            rows = []
            for backend in module.csv_values(backends):
                if backend == "mps":
                    rows.append(
                        {
                            "backend": "mps",
                            "requested_backend": "mps",
                            "runtime_ready": False,
                            "runtime_status": "error",
                            "error": "mps placeholder",
                        }
                    )
                else:
                    rows.append(
                        {
                            "backend": backend,
                            "requested_backend": backend,
                            "runtime_ready": backend == "wgpu",
                            "runtime_status": (
                                "kernel_wired" if backend == "wgpu" else "cpu"
                            ),
                        }
                    )
            return {"reports": rows}

        fields = module.runtime_device_report_fields(
            {
                "runtime_device_backends": ["wgpu", "cpu"],
                "required_runtime_device_backends": ["cpu"],
                "required_runtime_device_ready_backends": ["wgpu", "mps"],
            },
            describe_runtime_devices=fake_describe,
        )

        self.assertTrue(fields["runtime_device_report_requested"])
        self.assertEqual(
            fields["runtime_device_route_contract_version"],
            "spiraltorch.runtime_device_route.v3",
        )
        self.assertEqual(
            fields["runtime_device_route_semantic_owner"],
            "st-core::backend::runtime_route",
        )
        self.assertEqual(fields["runtime_device_route_semantic_backend"], "rust")
        self.assertEqual(fields["runtime_device_report_backends"], "wgpu,cpu,mps")
        self.assertEqual(
            fields["runtime_device_report_available_backends"],
            "wgpu,cpu",
        )
        self.assertEqual(fields["runtime_device_report_ready_backends"], "wgpu")
        self.assertEqual(fields["runtime_device_report_error_backends"], "mps")
        self.assertEqual(
            fields["runtime_device_report_statuses"],
            "wgpu=kernel_wired,cpu=cpu,mps=error",
        )
        self.assertTrue(fields["required_runtime_device_backends_passed"])
        self.assertFalse(fields["required_runtime_device_ready_backends_passed"])
        self.assertEqual(
            module.runtime_device_requirement_failures(fields),
            ["runtime_device_not_ready:mps"],
        )

    def test_runtime_device_report_keeps_ready_surrogate_diagnostic_available(
        self,
    ) -> None:
        module = load_runtime_imports()

        def fake_describe(backends, *, continue_on_error=True):
            self.assertTrue(continue_on_error)
            return {
                "reports": [
                    {
                        "backend": "mps",
                        "requested_backend": "mps",
                        "effective_backend": "wgpu",
                        "runtime_ready": True,
                        "requested_backend_runtime_ready": False,
                        "effective_backend_runtime_ready": True,
                        "runtime_status": "kernel_wired",
                        "requested_backend_runtime_status": "placeholder",
                        "effective_backend_runtime_status": "kernel_wired",
                        "error": "native MPS kernels are not wired",
                    }
                ]
            }

        fields = module.runtime_device_report_fields(
            {
                "runtime_device_backends": ["mps"],
                "required_runtime_device_backends": ["mps"],
                "required_runtime_device_ready_backends": ["mps"],
            },
            describe_runtime_devices=fake_describe,
        )

        self.assertEqual(fields["runtime_device_report_available_backends"], "mps")
        self.assertEqual(fields["runtime_device_report_ready_backends"], "mps")
        self.assertEqual(fields["runtime_device_report_error_backends"], "none")
        self.assertEqual(fields["runtime_device_route_fallback_backends"], "mps")
        self.assertEqual(fields["runtime_device_native_not_ready_backends"], "mps")
        self.assertTrue(fields["required_runtime_device_backends_passed"])
        self.assertTrue(fields["required_runtime_device_ready_backends_passed"])
        self.assertEqual(module.runtime_device_requirement_failures(fields), [])

    def test_runtime_device_report_preserves_unknown_readiness(self) -> None:
        module = load_runtime_imports()

        def fake_describe(backends, *, continue_on_error=True):
            self.assertTrue(continue_on_error)
            return {
                "reports": [
                    {
                        "backend": "cpu",
                        "requested_backend": "cpu",
                        "runtime_status": "cpu",
                    }
                ]
            }

        fields = module.runtime_device_report_fields(
            {
                "runtime_device_backends": ["cpu"],
                "required_runtime_device_ready_backends": ["cpu"],
            },
            describe_runtime_devices=fake_describe,
        )

        contract = json.loads(fields["runtime_device_route_contract_json"])
        self.assertEqual(fields["runtime_device_report_not_ready_backends"], "cpu")
        self.assertEqual(
            fields["runtime_device_native_readiness_unknown_backends"],
            "cpu",
        )
        self.assertEqual(
            fields["runtime_device_route_readiness_unknown_backends"],
            "cpu",
        )
        self.assertEqual(fields["runtime_device_route_not_ready_backends"], "none")
        self.assertEqual(
            fields["required_runtime_device_ready_backends_unknown"],
            "cpu",
        )
        self.assertEqual(contract["routes"][0]["native_readiness"], "unknown")
        self.assertEqual(contract["routes"][0]["route_readiness"], "unknown")
        self.assertFalse(fields["required_runtime_device_ready_backends_passed"])
        self.assertEqual(
            module.runtime_device_requirement_failures(fields),
            ["runtime_device_readiness_unknown:cpu"],
        )

    def test_preflight_report_includes_runtime_device_failures(self) -> None:
        module = load_runtime_imports()

        def fake_describe(backends, *, continue_on_error=True):
            return {
                "reports": [
                    {
                        "backend": backend,
                        "requested_backend": backend,
                        "runtime_ready": False,
                        "runtime_status": "feature_disabled",
                    }
                    for backend in module.csv_values(backends)
                ]
            }

        report = module.runtime_import_preflight_report(
            runtime_imports=["math"],
            required_runtime_imports=["math"],
            runtime_device_backends=["wgpu"],
            required_runtime_device_ready_backends=["wgpu"],
            describe_runtime_devices=fake_describe,
        )

        self.assertFalse(report["runtime_import_preflight_passed"])
        self.assertEqual(
            report["runtime_import_preflight_failures"],
            "runtime_device_not_ready:wgpu",
        )
        self.assertEqual(
            report["runtime_device_report_statuses"],
            "wgpu=feature_disabled",
        )
        self.assertIn(
            "runtime_device_reports backends=wgpu",
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
        self.assertIn("hf-gpt2-ft=", stdout.getvalue())
        self.assertIn("hf-trl-sft=", stdout.getvalue())

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

    def test_cli_runtime_device_ready_gate_uses_lazy_describer(self) -> None:
        module = load_runtime_imports()

        def fake_describe(backends, *, continue_on_error=True):
            return {
                "reports": [
                    {
                        "backend": backend,
                        "requested_backend": backend,
                        "runtime_ready": backend == "cpu",
                        "runtime_status": (
                            "cpu" if backend == "cpu" else "feature_disabled"
                        ),
                    }
                    for backend in module.csv_values(backends)
                ]
            }

        stdout = io.StringIO()
        with mock.patch.object(
            module,
            "_default_describe_runtime_devices",
            return_value=fake_describe,
        ):
            with contextlib.redirect_stdout(stdout):
                exit_code = module.main(
                    [
                        "--runtime-device-backend",
                        "wgpu",
                        "--require-runtime-device-ready-backend",
                        "wgpu",
                    ]
                )

        self.assertEqual(exit_code, 1)
        output = stdout.getvalue()
        self.assertIn("runtime_device_reports backends=wgpu", output)
        self.assertIn(
            "runtime_import_preflight_failures runtime_device_not_ready:wgpu",
            output,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import types
import unittest
from unittest import mock

import spiraltorch as st
import spiraltorch.runtime_imports as runtime_imports
from spiraltorch import hf_ft


class HuggingFaceFineTuneBridgeTest(unittest.TestCase):
    def test_rust_dependency_report_names_required_surfaces(self) -> None:
        report = hf_ft.hf_gpt2_finetune_rust_dependency_report()

        self.assertEqual(
            report["row_type"],
            "hf_gpt2_finetune_rust_dependency_report",
        )
        self.assertIn("st-tensor", report["rust_surface_crates"])
        self.assertIn("st-nn", report["rust_surface_crates"])
        self.assertIn("st-backend-wgpu", report["rust_surface_crates"])
        self.assertIn("transformers", report["python_package_label"])
        self.assertIn("pyarrow", report["python_package_label"])

    def test_preflight_defaults_require_hf_gpt2_ft_and_report_backends(self) -> None:
        available = set(hf_ft.HF_GPT2_FT_REQUIRED_PYTHON_PACKAGES)

        def fake_import(name: str):
            if name not in available:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            module = types.ModuleType(name)
            module.__version__ = f"{name}-test"
            return module

        def fake_describe(backends, *, continue_on_error=True):
            self.assertTrue(continue_on_error)
            return {
                "reports": [
                    {
                        "backend": backend,
                        "requested_backend": backend,
                        "runtime_ready": True,
                        "runtime_status": "ready",
                    }
                    for backend in runtime_imports.csv_values(backends)
                ]
            }

        with mock.patch.object(
            runtime_imports.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            report = hf_ft.hf_gpt2_finetune_preflight_report(
                describe_runtime_devices=fake_describe,
            )

        self.assertTrue(report["runtime_import_preflight_passed"])
        self.assertEqual(report["required_runtime_import_presets"], "hf-gpt2-ft")
        self.assertEqual(report["runtime_device_report_backends"], "wgpu,cpu")
        self.assertEqual(report["runtime_device_report_ready_backends"], "wgpu,cpu")
        self.assertEqual(report["hf_model_name"], "gpt2")
        self.assertEqual(report["hf_dataset_name"], "wikitext")

    def test_preflight_can_report_missing_without_requiring_preset(self) -> None:
        def fake_import(name: str):
            if name in {"transformers", "torch", "tokenizers"}:
                return types.ModuleType(name)
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        with mock.patch.object(
            runtime_imports.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            report = hf_ft.hf_gpt2_finetune_preflight_report(
                runtime_device_backends=[],
                require_hf_gpt2_ft=False,
                describe_runtime_devices=lambda backends, **_: {"reports": []},
            )

        self.assertTrue(report["runtime_import_preflight_passed"])
        self.assertEqual(report["required_runtime_import_presets"], "none")
        self.assertIn("datasets", report["runtime_imports_failed"])
        self.assertIn("peft", report["runtime_import_failed_install_hints"])

    def test_empty_token_probe_is_honest_without_native_runtime(self) -> None:
        probe = hf_ft.hf_gpt2_finetune_zspace_probe([])

        self.assertEqual(probe["zspace_probe_status"], "missing_tokens")
        self.assertEqual(probe["zspace_probe_observed_token_count"], 0)

    def test_top_level_exports_hf_ft_helpers(self) -> None:
        self.assertIs(
            st.hf_gpt2_finetune_preflight_report,
            hf_ft.hf_gpt2_finetune_preflight_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_rust_dependency_report,
            hf_ft.hf_gpt2_finetune_rust_dependency_report,
        )
        self.assertIn("hf_ft", st.__all__)
        self.assertIn("hf_gpt2_finetune_preflight_report", st.__all__)


if __name__ == "__main__":
    unittest.main()

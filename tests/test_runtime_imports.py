from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMPORTS_PATH = (
    REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "runtime_imports.py"
)


def load_runtime_imports():
    spec = importlib.util.spec_from_file_location(
        "spiraltorch_runtime_imports_test",
        RUNTIME_IMPORTS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RuntimeImportsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

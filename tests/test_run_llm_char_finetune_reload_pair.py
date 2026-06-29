from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_llm_char_finetune_reload_pair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_llm_char_finetune_reload_pair", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunLlmCharFinetuneReloadPairTests(unittest.TestCase):
    def test_dry_run_writes_base_reload_and_compare_contract(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            reload_data = root / "reload.txt"
            run_root = (root / "runs").resolve()
            data.write_text("spiral torch base corpus", encoding="utf-8")
            reload_data.write_text("spiral torch reload corpus", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = mod.main(
                    [
                        str(data),
                        "--reload-data",
                        str(reload_data),
                        "--run-root",
                        str(run_root),
                        "--base-epochs",
                        "1",
                        "--reload-epochs",
                        "2",
                        "--batches",
                        "3",
                        "--batch",
                        "2",
                        "--steps",
                        "4",
                        "--embed-dim",
                        "5",
                        "--hidden",
                        "6",
                        "--lr",
                        "0.03",
                        "--reload-lr",
                        "0.01",
                        "--eval-samples",
                        "7",
                        "--val-split",
                        "0.2",
                        "--gen",
                        "8",
                        "--seed",
                        "11",
                        "--reload-seed",
                        "19",
                        "--backend",
                        "cpu",
                        "--curves",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((run_root / "reload_pair.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], mod.SCHEMA)
        self.assertTrue(manifest["dry_run"])
        self.assertFalse(manifest["failed"])
        self.assertEqual(manifest["base_seed"], 11)
        self.assertEqual(manifest["reload_seed"], 19)
        self.assertEqual(manifest["settings"]["reload_lr"], 0.01)
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(manifest["runs"][0]["name"], "base_scratch")
        self.assertEqual(manifest["runs"][1]["name"], "reload_finetune")
        self.assertEqual(manifest["runs"][0]["status"], "dry_run")
        self.assertEqual(manifest["runs"][1]["status"], "dry_run")
        base_command = manifest["runs"][0]["command"]
        reload_command = manifest["runs"][1]["command"]
        compare_command = manifest["compare_command"]
        self.assertIn("--run-dir", base_command)
        self.assertNotIn("--load-run", base_command)
        self.assertIn("--load-run", reload_command)
        self.assertIn(str(run_root / "base_scratch"), reload_command)
        self.assertIn(str(reload_data), reload_command)
        self.assertIn("--aggregate", compare_command)
        self.assertIn("--curves", compare_command)
        self.assertIn(str(run_root / "compare.json"), compare_command)
        self.assertIsNone(manifest["compare_path"])

    def test_defaults_reload_seed_and_data_to_base_inputs(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            run_root = root / "runs"
            data.write_text("spiral torch shared corpus", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = mod.main(
                    [
                        str(data),
                        "--run-root",
                        str(run_root),
                        "--dry-run",
                    ]
                )
            manifest = json.loads((run_root / "reload_pair.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["reload_seed"], manifest["base_seed"] + 1)
        self.assertEqual(manifest["reload_data_paths"], manifest["data_paths"])
        self.assertIn(str(data), manifest["runs"][1]["command"])

    def test_invalid_budget_returns_usage_error(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            data.write_text("spiral torch", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                code = mod.main(
                    [str(data), "--run-root", str(root / "runs"), "--batches", "0"]
                )

        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

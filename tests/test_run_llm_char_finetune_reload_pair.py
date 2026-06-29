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
    def test_backend_readiness_requires_accelerator_feature(self) -> None:
        mod = _load_module()

        self.assertTrue(mod.backend_readiness("cpu", {})["backend_ready"])

        wgpu_missing = mod.backend_readiness(
            "wgpu", {"wgpu": False, "wgpu-rt": False}
        )
        self.assertFalse(wgpu_missing["backend_ready"])
        self.assertEqual(wgpu_missing["backend_status"], "backend_unavailable")
        self.assertEqual(
            wgpu_missing["backend_required_any_features"], ["wgpu", "wgpu-rt"]
        )

        self.assertTrue(mod.backend_readiness("cuda", {"cuda": True})["backend_ready"])
        self.assertEqual(
            mod.backend_readiness("quantum", {})["backend_status"], "unknown_backend"
        )

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
                        "--restore-best-at-end",
                        "--curves",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((run_root / "reload_pair.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], mod.SCHEMA)
        self.assertTrue(manifest["dry_run"])
        self.assertFalse(manifest["failed"])
        self.assertFalse(manifest["preflight_blocked"])
        self.assertIn("preflight_path", manifest)
        self.assertIn("preflight", manifest)
        self.assertIn("child_preflight_path", manifest["preflight"])
        self.assertEqual(manifest["base_seed"], 11)
        self.assertEqual(manifest["reload_seed"], 19)
        self.assertEqual(manifest["settings"]["reload_lr"], 0.01)
        self.assertTrue(manifest["settings"]["restore_best_at_end"])
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
        self.assertIn("--restore-best-at-end", base_command)
        self.assertIn("--load-run", reload_command)
        self.assertIn("--restore-best-at-end", reload_command)
        self.assertIn(str(run_root / "base_scratch"), reload_command)
        self.assertIn(str(reload_data), reload_command)
        self.assertIn("--aggregate", compare_command)
        self.assertIn("--curves", compare_command)
        self.assertIn(str(run_root / "compare.json"), compare_command)
        self.assertIsNone(manifest["compare_path"])

    def test_preflight_only_writes_readiness_and_skips_runs(self) -> None:
        mod = _load_module()
        original = mod.run_finetune_preflight
        mod.run_finetune_preflight = lambda data_paths, *, run_root, backend: {
            "schema": mod.PREFLIGHT_SCHEMA,
            "backend": backend,
            "ready": True,
            "reason": "ready",
            "missing_nn_symbols": [],
            "child_preflight_path": str(run_root / "_preflight" / "preflight.json"),
        }
        try:
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
                            "--preflight-only",
                        ]
                    )
                manifest = json.loads(
                    (run_root / "reload_pair.json").read_text(encoding="utf-8")
                )
                preflight = json.loads(
                    (run_root / "preflight.json").read_text(encoding="utf-8")
                )
        finally:
            mod.run_finetune_preflight = original

        self.assertEqual(code, 0)
        self.assertTrue(manifest["preflight_only"])
        self.assertFalse(manifest["failed"])
        self.assertEqual(manifest["runs"], [])
        self.assertTrue(preflight["ready"])
        self.assertEqual(manifest["preflight"]["reason"], "ready")

    def test_preflight_only_failure_marks_manifest_failed(self) -> None:
        mod = _load_module()
        original = mod.run_finetune_preflight
        mod.run_finetune_preflight = lambda data_paths, *, run_root, backend: {
            "schema": mod.PREFLIGHT_SCHEMA,
            "backend": backend,
            "ready": False,
            "reason": "missing_nn_symbols",
            "missing_nn_symbols": ["Sequential"],
            "child_preflight_path": str(run_root / "_preflight" / "preflight.json"),
        }
        try:
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
                            "--preflight-only",
                        ]
                    )
                manifest = json.loads(
                    (run_root / "reload_pair.json").read_text(encoding="utf-8")
                )
        finally:
            mod.run_finetune_preflight = original

        self.assertEqual(code, 1)
        self.assertTrue(manifest["preflight_only"])
        self.assertTrue(manifest["failed"])
        self.assertFalse(manifest["preflight_blocked"])
        self.assertEqual(manifest["runs"], [])

    def test_preflight_failure_blocks_real_run_before_training(self) -> None:
        mod = _load_module()
        original = mod.run_finetune_preflight
        mod.run_finetune_preflight = lambda data_paths, *, run_root, backend: {
            "schema": mod.PREFLIGHT_SCHEMA,
            "backend": backend,
            "ready": False,
            "reason": "missing_nn_symbols",
            "missing_nn_symbols": ["Sequential"],
            "child_preflight_path": str(run_root / "_preflight" / "preflight.json"),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "data.txt"
                run_root = root / "runs"
                data.write_text("spiral torch shared corpus", encoding="utf-8")

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    code = mod.main([str(data), "--run-root", str(run_root)])
                manifest = json.loads(
                    (run_root / "reload_pair.json").read_text(encoding="utf-8")
                )
        finally:
            mod.run_finetune_preflight = original

        self.assertEqual(code, 1)
        self.assertTrue(manifest["failed"])
        self.assertTrue(manifest["preflight_blocked"])
        self.assertEqual(manifest["runs"], [])
        self.assertEqual(manifest["preflight"]["missing_nn_symbols"], ["Sequential"])

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

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
                        "--early-stop-patience",
                        "2",
                        "--restore-best-at-end",
                        "--rollback-on-validation-regression",
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
        self.assertEqual(manifest["eval_seed"], 11)
        self.assertEqual(manifest["settings"]["reload_lr"], 0.01)
        self.assertEqual(manifest["settings"]["eval_seed"], 11)
        self.assertEqual(manifest["settings"]["early_stop_patience"], 2)
        self.assertTrue(manifest["settings"]["restore_best_at_end"])
        self.assertTrue(manifest["settings"]["rollback_on_validation_regression"])
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
        self.assertIn("--eval-seed", base_command)
        self.assertIn("--early-stop-patience", base_command)
        self.assertIn("--restore-best-at-end", base_command)
        self.assertIn("--rollback-on-validation-regression", base_command)
        self.assertIn("--load-run", reload_command)
        self.assertIn("--eval-seed", reload_command)
        self.assertIn("--early-stop-patience", reload_command)
        self.assertIn("--restore-best-at-end", reload_command)
        self.assertIn("--rollback-on-validation-regression", reload_command)
        self.assertEqual(
            base_command[base_command.index("--eval-seed") + 1],
            reload_command[reload_command.index("--eval-seed") + 1],
        )
        self.assertIn(str(run_root / "base_scratch"), reload_command)
        self.assertIn(str(reload_data), reload_command)
        self.assertIn("--aggregate", compare_command)
        self.assertIn("--curves", compare_command)
        self.assertIn(str(run_root / "compare.json"), compare_command)
        self.assertIsNone(manifest["compare_path"])
        self.assertIsNone(manifest["outcome_path"])
        self.assertIsNone(manifest["outcome"])

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
        self.assertEqual(manifest["eval_seed"], manifest["base_seed"])
        self.assertEqual(manifest["reload_data_paths"], manifest["data_paths"])
        self.assertIn(str(data), manifest["runs"][1]["command"])

    def test_reload_pair_outcome_reports_best_improvement(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base_scratch"
            reload = root / "reload_finetune"
            base.mkdir()
            reload.mkdir()
            (base / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 4.0},
                        "final_validation": {"mean_nll": 3.5},
                        "training_final_validation": {"mean_nll": 3.6},
                        "best_validation_mean_nll": 3.4,
                        "best_validation_epoch": 1,
                        "validation_nll_delta": -0.5,
                        "training_final_nll_delta": -0.4,
                        "final_minus_best_validation_nll": 0.1,
                        "training_final_minus_best_validation_nll": 0.2,
                        "restore_best_at_end": True,
                        "restored_best_at_end": True,
                        "best_checkpoint_exists": True,
                        "rollback_on_validation_regression": True,
                        "validation_rollback_count": 1,
                        "validation_rollback_epochs": [0],
                        "epochs_completed": 2,
                    }
                ),
                encoding="utf-8",
            )
            (reload / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.45},
                        "final_validation": {"mean_nll": 3.3},
                        "training_final_validation": {"mean_nll": 3.25},
                        "best_validation_mean_nll": 3.2,
                        "best_validation_epoch": 2,
                        "validation_nll_delta": -0.15,
                        "training_final_nll_delta": -0.2,
                        "final_minus_best_validation_nll": 0.1,
                        "training_final_minus_best_validation_nll": 0.05,
                        "early_stopped_epoch": 3,
                        "restore_best_at_end": True,
                        "restored_best_at_end": True,
                        "best_checkpoint_exists": True,
                        "rollback_on_validation_regression": True,
                        "validation_rollback_count": 2,
                        "validation_rollback_epochs": [1, 3],
                        "epochs_completed": 4,
                    }
                ),
                encoding="utf-8",
            )

            outcome = mod.reload_pair_outcome(base, reload)

        self.assertEqual(outcome["schema"], mod.OUTCOME_SCHEMA)
        self.assertTrue(outcome["ready"])
        self.assertTrue(outcome["evaluation_comparable"])
        self.assertEqual(outcome["issues"], [])
        self.assertEqual(outcome["status"], "improved")
        self.assertTrue(outcome["reload_improved_best"])
        self.assertFalse(outcome["reload_regressed_best"])
        self.assertAlmostEqual(outcome["reload_best_minus_base_best_nll"], -0.2)
        self.assertAlmostEqual(outcome["reload_final_minus_base_final_nll"], -0.2)
        self.assertEqual(outcome["reload_training_status"], "improved")
        self.assertAlmostEqual(outcome["reload_training_final_minus_base_best_nll"], -0.15)
        self.assertAlmostEqual(outcome["reload_best_minus_reload_initial_nll"], -0.25)
        self.assertAlmostEqual(
            outcome["reload_training_final_minus_reload_initial_nll"],
            -0.2,
        )
        self.assertEqual(outcome["base"]["best_epoch"], 1)
        self.assertAlmostEqual(outcome["reload"]["training_final_nll"], 3.25)
        self.assertAlmostEqual(outcome["reload"]["training_final_nll_delta"], -0.2)
        self.assertAlmostEqual(outcome["reload"]["training_final_minus_best_nll"], 0.05)
        self.assertEqual(outcome["reload_validation_rollback_count"], 2)
        self.assertEqual(outcome["reload_validation_rollback_epochs"], [1, 3])
        self.assertEqual(outcome["reload"]["validation_rollback_count"], 2)
        self.assertEqual(outcome["reload"]["early_stopped_epoch"], 3)
        self.assertTrue(outcome["reload"]["restored_best_at_end"])

    def test_reload_pair_outcome_rejects_eval_seed_mismatch(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base_scratch"
            reload = root / "reload_finetune"
            base.mkdir()
            reload.mkdir()
            for run_dir, eval_seed, best_nll in (
                (base, 7, 3.5),
                (reload, 8, 3.4),
            ):
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "initial_validation": {"mean_nll": best_nll},
                            "final_validation": {"mean_nll": best_nll},
                            "best_validation_mean_nll": best_nll,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "run.json").write_text(
                    json.dumps({"seed": 1, "eval_seed": eval_seed}),
                    encoding="utf-8",
                )

            outcome = mod.reload_pair_outcome(base, reload)

        self.assertFalse(outcome["ready"])
        self.assertFalse(outcome["evaluation_comparable"])
        self.assertEqual(outcome["status"], "unknown")
        self.assertIn("eval_seed_mismatch", outcome["issues"])
        self.assertIn("eval_seed_mismatch", outcome["comparison_issues"])

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

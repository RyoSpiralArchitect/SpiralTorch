from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_llm_char_finetune_reload_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_llm_char_finetune_reload_sweep", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunLlmCharFinetuneReloadSweepTests(unittest.TestCase):
    def test_dry_run_writes_grid_commands_and_summary(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            run_root = root / "runs"
            data.write_text("spiral torch reload sweep corpus", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                code = mod.main(
                    [
                        str(data),
                        "--run-root",
                        str(run_root),
                        "--seed-values",
                        "3,5",
                        "--reload-lr-values",
                        "0.02,0.005",
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
                        "--eval-samples",
                        "7",
                        "--gen",
                        "8",
                        "--early-stop-patience",
                        "2",
                        "--restore-best-at-end",
                        "--rollback-on-validation-regression",
                        "--runtime-import-preset",
                        "hf-finetune",
                        "--require-runtime-imports",
                        "--dry-run",
                    ]
                )
            manifest = json.loads(
                (run_root / "reload_sweep.json").read_text(encoding="utf-8")
            )
            markdown = (run_root / "reload_sweep.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], mod.SCHEMA)
        self.assertTrue(manifest["dry_run"])
        self.assertFalse(manifest["failed"])
        self.assertEqual(len(manifest["cells"]), 4)
        self.assertEqual(manifest["summary"]["cells"], 4)
        self.assertEqual(manifest["summary"]["run_status_counts"], {"dry_run": 4})
        self.assertEqual(manifest["summary"]["training_status_counts"], {"dry_run": 4})
        self.assertEqual(manifest["summary"]["adoption_status_counts"], {"dry_run": 4})
        self.assertEqual(
            manifest["summary"]["runtime_preflight_status_counts"],
            {"dry_run": 4},
        )
        self.assertEqual(len(manifest["summary"]["reload_lr_groups"]), 2)
        self.assertEqual(
            [group["cells"] for group in manifest["summary"]["reload_lr_groups"]],
            [2, 2],
        )
        self.assertEqual(manifest["settings"]["eval_seed_offset"], 0)
        self.assertTrue(manifest["settings"]["rollback_on_validation_regression"])
        self.assertEqual(
            manifest["settings"]["runtime_import_presets"],
            ["hf-finetune"],
        )
        self.assertTrue(manifest["settings"]["require_runtime_imports"])
        self.assertTrue(manifest["settings"]["runtime_import_preflight_requested"])
        first = manifest["cells"][0]
        self.assertEqual(first["status"], "dry_run")
        self.assertEqual(first["eval_seed"], first["seed"])
        self.assertIn("--reload-lr", first["command"])
        self.assertIn("--eval-seed", first["command"])
        self.assertIn("--early-stop-patience", first["command"])
        self.assertIn("--restore-best-at-end", first["command"])
        self.assertIn("--rollback-on-validation-regression", first["command"])
        self.assertIn("--runtime-import-preset", first["command"])
        self.assertIn("hf-finetune", first["command"])
        self.assertIn("--require-runtime-imports", first["command"])
        self.assertIn("seed3_reloadlr0p02", first["name"])
        self.assertIn("# LLM Char Finetune Reload Sweep", markdown)
        self.assertIn("runtime_preflight_status_counts", markdown)
        self.assertIn("## Reload LR Groups", markdown)
        self.assertIn("| 0.02 | 2 |", markdown)
        self.assertIn(
            "| cell | status | training_status | adoption_status | run_status | seed | reload_seed | eval_seed |",
            markdown,
        )
        self.assertIn("seed3_reloadlr0p02", markdown)

    def test_sweep_summary_ranks_best_delta_and_counts_statuses(self) -> None:
        mod = _load_module()
        cells = [
            {
                "name": "regressed",
                "status": "ok",
                "reload_lr": 0.02,
                "outcome": {
                    "status": "regressed",
                    "reload_training_status": "regressed",
                    "reload_adoption_status": "rejected_regressed",
                    "reload_best_minus_base_best_nll": 0.1,
                    "reload_training_final_minus_base_best_nll": 0.15,
                    "reload_final_minus_base_final_nll": 0.2,
                    "reload_validation_rollback_count": 2,
                },
            },
            {
                "name": "improved",
                "status": "ok",
                "reload_lr": 0.005,
                "outcome": {
                    "status": "improved",
                    "reload_training_status": "improved",
                    "reload_adoption_status": "accepted_improved",
                    "reload_best_minus_base_best_nll": -0.2,
                    "reload_training_final_minus_base_best_nll": -0.3,
                    "reload_final_minus_base_final_nll": -0.1,
                    "reload_validation_rollback_count": 0,
                },
            },
            {
                "name": "unknown",
                "status": "missing_outcome",
                "reload_lr": 0.02,
                "outcome": None,
            },
            {
                "name": "protected",
                "status": "ok",
                "reload_lr": 0.02,
                "outcome": {
                    "status": "tied",
                    "reload_training_status": "regressed",
                    "reload_adoption_status": "protected_noop",
                    "reload_best_minus_base_best_nll": 0.0,
                    "reload_training_final_minus_base_best_nll": 0.05,
                    "reload_final_minus_base_final_nll": 0.0,
                    "reload_validation_rollback_count": 2,
                },
            },
        ]

        summary = mod.sweep_summary(cells)

        self.assertEqual(summary["cells"], 4)
        self.assertEqual(summary["status_counts"]["improved"], 1)
        self.assertEqual(summary["status_counts"]["regressed"], 1)
        self.assertEqual(summary["status_counts"]["tied"], 1)
        self.assertEqual(summary["status_counts"]["missing_outcome"], 1)
        self.assertEqual(summary["training_status_counts"]["improved"], 1)
        self.assertEqual(summary["training_status_counts"]["regressed"], 2)
        self.assertEqual(summary["training_status_counts"]["missing_outcome"], 1)
        self.assertEqual(summary["adoption_status_counts"]["accepted_improved"], 1)
        self.assertEqual(summary["adoption_status_counts"]["rejected_regressed"], 1)
        self.assertEqual(summary["adoption_status_counts"]["protected_noop"], 1)
        self.assertEqual(summary["adoption_status_counts"]["missing_outcome"], 1)
        self.assertEqual(summary["run_status_counts"]["ok"], 3)
        self.assertEqual(
            summary["runtime_preflight_status_counts"],
            {"missing_outcome": 1, "ok": 3},
        )
        self.assertEqual(summary["protected_noop_cells"], 1)
        self.assertEqual(summary["accepted_improved_cells"], 1)
        self.assertEqual(summary["best_cell"], "improved")
        self.assertEqual(summary["best_training_cell"], "improved")
        self.assertAlmostEqual(
            summary["best_reload_best_minus_base_best_nll"],
            -0.2,
        )
        self.assertAlmostEqual(
            summary["best_reload_training_final_minus_base_best_nll"],
            -0.3,
        )
        self.assertAlmostEqual(
            summary["reload_training_final_minus_base_best_nll_stats"]["mean"],
            -0.033333333333333326,
        )
        self.assertAlmostEqual(
            summary["reload_validation_rollback_count_stats"]["mean"],
            1.3333333333333333,
        )
        groups = summary["reload_lr_groups"]
        self.assertEqual([group["reload_lr_label"] for group in groups], ["0.005", "0.02"])
        self.assertEqual(groups[0]["best_cell"], "improved")
        self.assertEqual(groups[0]["training_status_counts"], {"improved": 1})
        self.assertEqual(groups[0]["adoption_status_counts"], {"accepted_improved": 1})
        self.assertAlmostEqual(
            groups[0]["reload_training_final_minus_base_best_nll_stats"]["mean"],
            -0.3,
        )
        self.assertAlmostEqual(
            groups[0]["reload_validation_rollback_count_stats"]["mean"],
            0.0,
        )
        self.assertEqual(groups[1]["cells"], 3)
        self.assertEqual(
            groups[1]["training_status_counts"],
            {"missing_outcome": 1, "regressed": 2},
        )
        self.assertEqual(
            groups[1]["adoption_status_counts"],
            {"missing_outcome": 1, "protected_noop": 1, "rejected_regressed": 1},
        )

    def test_invalid_reload_lr_values_return_usage_error(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            data.write_text("spiral torch", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                code = mod.main(
                    [
                        str(data),
                        "--run-root",
                        str(root / "runs"),
                        "--reload-lr-values",
                        "0.01,0",
                        "--dry-run",
                    ]
                )

        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

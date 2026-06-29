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
        first = manifest["cells"][0]
        self.assertEqual(first["status"], "dry_run")
        self.assertIn("--reload-lr", first["command"])
        self.assertIn("--early-stop-patience", first["command"])
        self.assertIn("--restore-best-at-end", first["command"])
        self.assertIn("seed3_reloadlr0p02", first["name"])
        self.assertIn("# LLM Char Finetune Reload Sweep", markdown)
        self.assertIn("seed3_reloadlr0p02", markdown)

    def test_sweep_summary_ranks_best_delta_and_counts_statuses(self) -> None:
        mod = _load_module()
        cells = [
            {
                "name": "regressed",
                "status": "ok",
                "outcome": {
                    "status": "regressed",
                    "reload_best_minus_base_best_nll": 0.1,
                    "reload_final_minus_base_final_nll": 0.2,
                },
            },
            {
                "name": "improved",
                "status": "ok",
                "outcome": {
                    "status": "improved",
                    "reload_best_minus_base_best_nll": -0.2,
                    "reload_final_minus_base_final_nll": -0.1,
                },
            },
            {
                "name": "unknown",
                "status": "missing_outcome",
                "outcome": None,
            },
        ]

        summary = mod.sweep_summary(cells)

        self.assertEqual(summary["cells"], 3)
        self.assertEqual(summary["status_counts"]["improved"], 1)
        self.assertEqual(summary["status_counts"]["regressed"], 1)
        self.assertEqual(summary["status_counts"]["missing_outcome"], 1)
        self.assertEqual(summary["run_status_counts"]["ok"], 2)
        self.assertEqual(summary["best_cell"], "improved")
        self.assertAlmostEqual(
            summary["best_reload_best_minus_base_best_nll"],
            -0.2,
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

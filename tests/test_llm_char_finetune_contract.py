from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "models" / "python" / "llm_char_finetune.py"
MODEL_PYTHON = ROOT / "models" / "python"


def _load_module():
    if str(MODEL_PYTHON) not in sys.path:
        sys.path.insert(0, str(MODEL_PYTHON))
    spec = importlib.util.spec_from_file_location("llm_char_finetune", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LlmCharFinetuneContractTests(unittest.TestCase):
    def test_training_contract_records_ft_backend_and_reload_scope(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_meta = {
                "format": mod.FORMAT_V2,
                "steps": 16,
                "embed_dim": 8,
                "vocab_size": 32,
                "symbols_count": 32,
                "mode": "embedding(8)",
                "curvature": -1.0,
                "temperature": 0.75,
                "backend": "cpu",
                "epochs": 3,
                "batches_per_epoch": 5,
                "batch": 2,
                "lr": 0.01,
                "events_path": str(root / "events.jsonl"),
                "desire": True,
                "softlogic": {"enabled": False},
                "weights_loaded_from": str(root / "base.json"),
            }

            contract = mod._build_training_contract(
                run_meta,
                weights_path=root / "weights.json",
                metrics_path=root / "metrics.jsonl",
                samples_dir=root / "samples",
                save_weights=root / "exported.json",
            )

        self.assertEqual(contract["schema"], mod.TRAINING_CONTRACT_SCHEMA)
        self.assertEqual(contract["learning_mode"], "finetune")
        self.assertEqual(contract["input"]["representation"], "tokenizerless_char")
        self.assertEqual(contract["input"]["format"], mod.FORMAT_V2)
        self.assertIn("embedding", contract["parameter_policy"]["trainable"])
        self.assertEqual(contract["parameter_policy"]["frozen"], [])
        self.assertEqual(contract["backend"]["requested"], "cpu")
        self.assertEqual(contract["backend"]["status"], "available")
        self.assertTrue(contract["reload"]["reload_safe"])
        self.assertTrue(contract["reload"]["metadata_required"])
        self.assertTrue(contract["geometry"]["zspace_softmax"])
        self.assertTrue(contract["geometry"]["hypergrad_attached"])
        self.assertTrue(contract["controls"]["desire"])

    def test_completion_summary_reports_checkpoint_metrics_and_sample(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            weights_path = run_dir / "weights.json"
            meta_path = run_dir / "weights.meta.json"
            metrics_path = run_dir / "metrics.jsonl"
            sample_path = run_dir / "samples" / "epoch_000.txt"
            save_weights = root / "exported.json"
            save_meta = root / "exported.meta.json"

            weights_path.write_text("{}", encoding="utf-8")
            meta_path.write_text("{}", encoding="utf-8")
            save_weights.write_text("{}", encoding="utf-8")
            save_meta.write_text("{}", encoding="utf-8")
            sample_path.parent.mkdir(parents=True)
            sample_path.write_text("hello", encoding="utf-8")
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"epoch": 0, "average_loss": 4.0}),
                        "not-json",
                        json.dumps({"epoch": 1, "average_loss": 3.25}),
                    ]
                ),
                encoding="utf-8",
            )
            run_meta = {
                "schema": mod.RUN_SCHEMA,
                "arch": "llm_char_finetune",
                "weights_loaded_from": None,
                "training_contract": {"schema": mod.TRAINING_CONTRACT_SCHEMA},
            }

            mod._write_completion_summary(
                run_dir,
                run_meta,
                weights_path=weights_path,
                metrics_path=metrics_path,
                final_sample_path=sample_path,
                save_weights=save_weights,
            )

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["schema"], mod.SUMMARY_SCHEMA)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["checkpoint"]["weights_exists"])
        self.assertTrue(summary["checkpoint"]["meta_exists"])
        self.assertTrue(summary["checkpoint"]["save_weights_exists"])
        self.assertTrue(summary["checkpoint"]["save_meta_exists"])
        self.assertEqual(summary["metrics"]["epoch_count"], 2)
        self.assertEqual(summary["metrics"]["loss_count"], 2)
        self.assertAlmostEqual(summary["metrics"]["first_average_loss"], 4.0)
        self.assertAlmostEqual(summary["metrics"]["final_average_loss"], 3.25)
        self.assertAlmostEqual(
            summary["metrics"]["first_minus_final_average_loss"],
            0.75,
        )
        self.assertTrue(summary["sample"]["exists"])


if __name__ == "__main__":
    unittest.main()

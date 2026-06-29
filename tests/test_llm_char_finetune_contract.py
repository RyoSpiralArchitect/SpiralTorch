from __future__ import annotations

import importlib.util
import json
import math
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
                "eval_samples": 17,
                "lr": 0.01,
                "validation_start_fraction_requested": 0.9,
                "validation_start_fraction_actual": 0.875,
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
        self.assertEqual(contract["validation"]["eval_samples"], 17)
        self.assertAlmostEqual(
            contract["validation"]["validation_start_fraction_requested"],
            0.9,
        )
        self.assertAlmostEqual(
            contract["validation"]["validation_start_fraction_actual"],
            0.875,
        )

    def test_validation_summary_payload_matches_compare_contract(self) -> None:
        mod = _load_module()
        samples = [([0, 1], 2), ([1, 2], 0)]
        rows = [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]
        unigram_rows = [[1.0 / 3.0] * 3, [1.0 / 3.0] * 3]
        bigram_rows = [[0.2, 0.2, 0.6], [0.5, 0.25, 0.25]]

        final_validation = mod._validation_from_probability_rows(
            rows,
            samples,
            vocab_size=3,
            unigram_rows=unigram_rows,
            bigram_rows=bigram_rows,
        )
        unigram_validation = mod._validation_from_probability_rows(
            unigram_rows,
            samples,
            vocab_size=3,
        )
        bigram_validation = mod._validation_from_probability_rows(
            bigram_rows,
            samples,
            vocab_size=3,
        )
        initial_validation = {
            "mean_nll": final_validation["mean_nll"] + 0.5,
            "accuracy": 0.0,
        }

        payload = mod._validation_summary_payload(
            initial_validation=initial_validation,
            final_validation=final_validation,
            unigram_validation=unigram_validation,
            bigram_validation=bigram_validation,
            best_validation=final_validation,
            best_epoch=3,
        )

        expected_nll = (-math.log(0.7) - math.log(0.6)) / 2.0
        self.assertEqual(final_validation["windows"], 2)
        self.assertAlmostEqual(final_validation["mean_nll"], expected_nll)
        self.assertAlmostEqual(final_validation["accuracy"], 1.0)
        self.assertAlmostEqual(final_validation["mean_target_probability"], 0.65)
        self.assertIn("mean_target_logprob_lift", final_validation)
        self.assertIn("mean_target_logprob_lift_vs_bigram", final_validation)
        self.assertAlmostEqual(payload["validation_nll_delta"], -0.5)
        self.assertAlmostEqual(
            payload["final_vs_unigram_nll_delta"],
            final_validation["mean_nll"] - unigram_validation["mean_nll"],
        )
        self.assertAlmostEqual(
            payload["final_vs_bigram_nll_delta"],
            final_validation["mean_nll"] - bigram_validation["mean_nll"],
        )
        self.assertAlmostEqual(payload["final_minus_best_validation_nll"], 0.0)
        self.assertEqual(payload["best_validation_epoch"], 3)
        self.assertAlmostEqual(
            payload["best_validation_mean_nll"],
            final_validation["mean_nll"],
        )

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
            validation_payload = {
                "initial_validation": {"mean_nll": 3.0, "accuracy": 0.1},
                "final_validation": {"mean_nll": 2.5, "accuracy": 0.3},
                "validation_nll_delta": -0.5,
                "best_validation_mean_nll": 2.4,
                "final_minus_best_validation_nll": 0.1,
            }

            mod._write_completion_summary(
                run_dir,
                run_meta,
                weights_path=weights_path,
                metrics_path=metrics_path,
                final_sample_path=sample_path,
                save_weights=save_weights,
                validation=validation_payload,
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
        self.assertAlmostEqual(summary["final_validation"]["mean_nll"], 2.5)
        self.assertAlmostEqual(summary["validation_nll_delta"], -0.5)
        self.assertAlmostEqual(summary["final_minus_best_validation_nll"], 0.1)


if __name__ == "__main__":
    unittest.main()

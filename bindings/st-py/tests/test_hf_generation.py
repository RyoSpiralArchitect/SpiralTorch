from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import spiraltorch as st
from spiraltorch import hf_ft
from spiraltorch.hf_generation import ZSpaceRepressionLogitsProcessor

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional test env
    torch = None


SWEEP_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_zspace_generation_control_sweep.py"
)


def load_generation_control_sweep_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_zspace_generation_control_sweep_test",
        SWEEP_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not installed")
class ZSpaceRepressionLogitsProcessorTests(unittest.TestCase):
    def test_repression_can_change_greedy_top_token(self) -> None:
        processor = ZSpaceRepressionLogitsProcessor(
            top_k=3,
            curvature=-1.0,
            temperature=1.0,
            entropy_target=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
            repression_window=4,
            repression_strength=2.0,
            last_token_repression=1.0,
            use_native_zspace=False,
        )
        input_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
        scores = torch.tensor([[4.0, 3.5, 1.0]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()
        aggregate_only = processor.report(limit=0)

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 1)
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["reported_rows"], 1)
        self.assertEqual(report["top_token_changed_count"], 1)
        self.assertEqual(report["reported_top_token_changed_count"], 1)
        self.assertEqual(aggregate_only["calls"], 1)
        self.assertEqual(aggregate_only["reported_rows"], 0)
        self.assertEqual(aggregate_only["rows"], [])
        self.assertEqual(aggregate_only["top_token_changed_count"], 1)
        self.assertEqual(report["rows"][0]["backend"], "math_zspace_softmax")
        self.assertEqual(report["backend"], "math_zspace_softmax")
        self.assertGreater(report["rows"][0]["max_repression"], 0.0)

    def test_softmax_only_records_entropy_without_reordering_greedy(self) -> None:
        processor = ZSpaceRepressionLogitsProcessor(
            top_k=3,
            curvature=-1.0,
            temperature=1.0,
            entropy_target=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
            repression_window=4,
            repression_strength=0.0,
            last_token_repression=0.0,
            use_native_zspace=False,
        )
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        scores = torch.tensor([[4.0, 3.5, 1.0]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 0)
        self.assertEqual(report["top_token_changed_count"], 0)
        self.assertIsNotNone(report["entropy_min"])
        self.assertIsNotNone(report["temperature_max"])

    def test_generation_report_embeds_zspace_control_payload(self) -> None:
        control = {
            "row_type": "zspace_repression_generation_control",
            "status": "ok",
            "calls": 2,
        }

        report = hf_ft.hf_gpt2_finetune_generation_report(
            stage="after_train",
            prompt="SpiralTorch is",
            generated_text="SpiralTorch is a runtime.",
            generated_continuation_text=" a runtime.",
            generation_method="model.generate+zspace_repression_softmax",
            generation_control=control,
        )

        self.assertEqual(report["generation_control"], control)


class ZSpaceGenerationExportTests(unittest.TestCase):
    def test_top_level_exports_generation_processor(self) -> None:
        self.assertIn("ZSpaceRepressionLogitsProcessor", st.__all__)
        self.assertIn("build_zspace_repression_logits_processor", st.__all__)
        self.assertIn("build_zspace_softmax_logits_processor", st.__all__)
        self.assertIs(st.ZSpaceRepressionLogitsProcessor, ZSpaceRepressionLogitsProcessor)


class ZSpaceGenerationControlSweepExampleTests(unittest.TestCase):
    def test_dry_run_builds_control_grid_without_loading_model(self) -> None:
        module = load_generation_control_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "control-sweep.json"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--prompt",
                    "SpiralTorch is",
                    "--out",
                    str(out_path),
                    "--zspace-entropy-target-values",
                    "none,3.0",
                    "--repression-strength-values",
                    "0.0,1.25",
                    "--last-token-repression-values",
                    "0.0",
                    "--report-limit",
                    "2",
                ]
            )
            runs = module.build_control_runs(args)
            report = module.run_sweep(args)

        self.assertEqual(len(runs), 5)
        self.assertEqual(runs[0]["kind"], "baseline")
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["run_count"], 5)
        self.assertEqual(report["summary"]["completed_run_count"], 0)
        self.assertTrue(any(str(row["name"]).startswith("zt3") for row in runs))

    def test_repetition_report_scores_repeated_ngrams(self) -> None:
        module = load_generation_control_sweep_example()
        loop = module.text_repetition_report("a b c a b c a b c")
        clean = module.text_repetition_report("a b c d e f")

        self.assertGreater(loop["loop_score"], clean["loop_score"])
        self.assertEqual(loop["max_ngram_repetition"], 3)


if __name__ == "__main__":
    unittest.main()

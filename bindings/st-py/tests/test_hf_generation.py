from __future__ import annotations

import unittest

import spiraltorch as st
from spiraltorch import hf_ft
from spiraltorch.hf_generation import ZSpaceRepressionLogitsProcessor

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional test env
    torch = None


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

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 1)
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["top_token_changed_count"], 1)
        self.assertEqual(report["rows"][0]["backend"], "math_zspace_softmax")
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


if __name__ == "__main__":
    unittest.main()

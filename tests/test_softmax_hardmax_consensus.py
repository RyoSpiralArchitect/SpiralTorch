import math
import unittest

from spiraltorch import (
    _GOLDEN_RATIO,
    _GOLDEN_RATIO_BIAS,
    _GOLDEN_RATIO_CONJUGATE,
    _SPIRAL_RAMANUJAN_DELTA,
    _SPIRAL_RAMANUJAN_RATIO,
    _SPIRAL_LEECH_SCALE,
    _spiral_softmax_hardmax_consensus_python,
)


class SpiralConsensusTests(unittest.TestCase):
    def test_invalid_lengths_return_defaults(self) -> None:
        fused, metrics = _spiral_softmax_hardmax_consensus_python([1.0], [1.0, 0.0], rows=1, cols=2)

        self.assertEqual(fused, [0.0, 0.0])
        self.assertEqual(
            metrics,
            {
                "phi": _GOLDEN_RATIO,
                "phi_conjugate": _GOLDEN_RATIO_CONJUGATE,
                "phi_bias": _GOLDEN_RATIO_BIAS,
                "ramanujan_ratio": _SPIRAL_RAMANUJAN_RATIO,
                "ramanujan_delta": _SPIRAL_RAMANUJAN_DELTA,
                "average_enrichment": 0.0,
                "mean_entropy": 0.0,
                "mean_hardmass": 0.0,
                "spiral_coherence": 0.0,
            },
        )

    def test_singleton_entropy_and_enrichment(self) -> None:
        fused, metrics = _spiral_softmax_hardmax_consensus_python([1.0], [1.0], rows=1, cols=1)

        enrichment = _SPIRAL_LEECH_SCALE * _GOLDEN_RATIO
        scale = 1.0 + enrichment
        expected_value = (_GOLDEN_RATIO_CONJUGATE + _GOLDEN_RATIO_BIAS) * scale

        self.assertEqual(len(fused), 1)
        self.assertTrue(math.isclose(fused[0], expected_value, rel_tol=1e-12))
        self.assertTrue(math.isclose(metrics["average_enrichment"], enrichment, rel_tol=1e-12))
        self.assertEqual(metrics["mean_entropy"], 0.0)
        self.assertEqual(metrics["mean_hardmass"], 1.0)

        enrichment_norm = enrichment / (1.0 + abs(enrichment))
        expected_coherence = (0.0 + 1.0 + enrichment_norm) / 3.0
        self.assertTrue(math.isclose(metrics["spiral_coherence"], expected_coherence, rel_tol=1e-12))

    def test_non_finite_values_are_sanitised(self) -> None:
        fused, metrics = _spiral_softmax_hardmax_consensus_python(
            [0.5, math.nan, -0.1, 0.6], [1.0, math.nan, -3.0, 0.0], rows=2, cols=2
        )

        self.assertEqual(len(fused), 4)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in fused))
        self.assertTrue(math.isfinite(metrics["average_enrichment"]))
        self.assertTrue(math.isfinite(metrics["spiral_coherence"]))


if __name__ == "__main__":
    unittest.main()

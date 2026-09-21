"""Exercise the real Rust self-supervision and fractal APIs, not Python substitutes."""
import math
import struct
import unittest

import spiraltorch as st


class SourceCrosscutTests(unittest.TestCase):
    def test_info_nce_matches_independent_objective(self):
        anchors = [[0.0, 0.0, 0.0], [0.5, -0.25, 1.0], [1.0, 0.0, -0.5]]
        positives = [[1.0, 0.0, 0.0], [0.5, 1.0, -0.5], [-0.25, 0.5, 0.75]]
        for normalize in [False, True]:
            result = st.selfsup.info_nce(anchors, positives, temperature=0.5, normalize=normalize)
            norm = lambda row: max(math.sqrt(sum(x * x for x in row)), 2 ** -23)
            expected = [[sum(a * b for a, b in zip(anchor, positive)) /
                         (norm(anchor) * norm(positive) if normalize else 1) / 0.5
                         for positive in positives] for anchor in anchors]
            loss = 0.0
            for i, row in enumerate(expected):
                maximum = max(row)
                loss += maximum - row[i] + math.log(sum(math.exp(v - maximum) for v in row))
                for actual, reference in zip(result["logits"][i], row):
                    self.assertAlmostEqual(actual, reference, delta=1e-5)
            self.assertAlmostEqual(result["loss"], loss / len(anchors), delta=1e-5)
            self.assertEqual(result["labels"], [0, 1, 2])
            self.assertEqual(result["batch"], 3)

    def test_info_nce_rejects_empty_or_ragged_batches(self):
        for anchors, positives in [([], []), ([[]], [[]]), ([[1.0], []], [[1.0], [2.0]])]:
            with self.assertRaises(ValueError):
                st.selfsup.info_nce(anchors, positives)

    def test_fractal_weave_preserves_base_and_float32_addition(self):
        generator = st.frac.FractalFieldGenerator(4, iterations=16)
        for length in [1, 17, 257]:
            base = st.frac.MellinLogGrid(-1.5, 0.03125, [complex(i * 0.125, -i * 0.25) for i in range(length)])
            before = base.samples
            branch = generator.branching_field(base.log_start, base.log_step, length)
            woven = generator.weave_with_grid(base)
            self.assertEqual(base.samples, before)
            self.assertEqual(len(woven), length)
            self.assertEqual(woven.log_start, base.log_start)
            self.assertEqual(woven.log_step, base.log_step)
            for actual, original, addition in zip(woven.samples, before, branch):
                self.assertEqual(struct.pack("f", actual.real), struct.pack("f", original.real + addition.real))
                self.assertEqual(struct.pack("f", actual.imag), struct.pack("f", original.imag + addition.imag))


if __name__ == "__main__":
    unittest.main()

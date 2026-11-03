"""Independent endpoint checks of the one-coordinate SGD inference, without a device."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_hosted_trio import exact, inverse_negative_sgd, word


class InverseSGD(unittest.TestCase):
    def test_endpoints_and_their_immediate_neighbors_forward_round_correctly(self):
        w, r, bad, g = 0x3bcd8de5, 0x3d4ccccd, 0x3bcd8e6a, 0xb5ab005d
        for split in (False, True):
            inverse = inverse_negative_sgd(w, r, bad, g, split=split)
            lo, hi = [int(inverse[k], 16) for k in ("nearest_to_zero_hex", "farthest_from_zero_hex")]
            def update(negative_word):
                p = exact(r) * exact(negative_word)
                if split:
                    p = exact(word(float(p)))
                return word(float(exact(w) - p))
            self.assertEqual([update(v) for v in (lo-1, lo, hi, hi+1)], [bad-1, bad, bad, bad+1])
            self.assertEqual(update(g), 0x3bcd8e6e)
            self.assertEqual(inverse["minimum_f32_steps_from_measured"], g - hi)
            self.assertGreater(inverse["minimum_relative_change"], 0.0241)
            observed = inverse_negative_sgd(w, r, update(g), g, split=split)
            self.assertEqual(observed["minimum_relative_change"], 0)

    def test_does_not_generalize_this_inverse_to_another_sign_or_optimizer(self):
        with self.assertRaises(ValueError):
            inverse_negative_sgd(0x3bcd8de5, 0x3d4ccccd, 0x3bcd8e6a, 0x35ab005d, split=False)

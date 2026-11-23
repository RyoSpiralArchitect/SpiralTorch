import math
from pathlib import Path
import runpy
import unittest

split = runpy.run_path(str(Path(__file__).with_name("block_readback.py")))["split"]


class DerivativeReadback(unittest.TestCase):
    def test_direction_and_distance_need_not_pick_same_component(self):
        r = split(dict(affine=[2, 0], saturation=[1, 1], projection=[0.01, 0], both=[0.1, 0.1]))
        self.assertTrue(r["projection_closer"])
        self.assertTrue(r["saturation_more_aligned"])
        self.assertAlmostEqual(r["cosine_to_full"]["projection"], 1/math.sqrt(2))
        self.assertAlmostEqual(r["cosine_to_full"]["saturation"], 1.)
        self.assertAlmostEqual(r["ratios"]["both"], math.sqrt(.02)/2)

    def test_tie_is_not_a_win(self):
        r = split(dict(affine=[1], saturation=[.5], projection=[.5], both=[.5]))
        self.assertFalse(r["projection_closer"])
        self.assertIs(r["saturation_more_aligned"], False)

    def test_do_not_impute_zero_norms(self):
        r = split(dict(affine=[0], saturation=[1], projection=[0], both=[0]))
        self.assertTrue(all(x is None for x in r["ratios"].values()))
        self.assertIsNone(r["saturation_more_aligned"])

    def test_mismatched_width(self):
        with self.assertRaises(ValueError):
            split(dict(affine=[1], saturation=[1, 2], projection=[1], both=[1]))


if __name__ == "__main__":
    unittest.main()

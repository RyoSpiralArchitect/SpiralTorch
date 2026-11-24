import math
from pathlib import Path
import runpy
import unittest

reader = runpy.run_path(str(Path(__file__).with_name("block_readback.py")))
split, scalar_residual = reader["split"], reader["scalar_residual"]


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

    def test_nonnegative_scalar_bound_and_degenerate_directions(self):
        self.assertAlmostEqual(scalar_residual([11, 0], [3, 4]), 4/5)
        self.assertEqual(scalar_residual([1, -2], [3, -6]), 0.)
        self.assertEqual(scalar_residual([-1, 0], [1, 0]), 1.)
        self.assertEqual(scalar_residual([0, 0], [1, 0]), 1.)
        self.assertIsNone(scalar_residual([1, 0], [0, 0]))
        self.assertIsNone(scalar_residual([0], [0]))

    def test_small_residual_is_not_lost_and_old_output_is_unchanged(self):
        arms = dict(affine=[1, 0], saturation=[.5, 1e-9], projection=[0, 1], both=[1, 0])
        old = split(arms)
        with_bounds = split(arms, scalar_bound=True)
        bounds = with_bounds.pop("scalar_residual")
        self.assertEqual(old, with_bounds)
        self.assertAlmostEqual(bounds["saturation"], 2e-9, delta=1e-22)
        self.assertEqual(bounds["both"], 0.)
        self.assertEqual(bounds["projection"], 1.)


if __name__ == "__main__":
    unittest.main()

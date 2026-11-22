"""Independent analytic and finite-difference controls; no ML runtime."""
import math
from pathlib import Path
import runpy
import unittest

M = runpy.run_path(str(Path(__file__).with_name("training_anchors.py")))


class AnchorSplit(unittest.TestCase):
    def test_known_probabilities_survive_f32_rounding(self):
        full, choice = -math.log(.25), -math.log(.25/.375)
        r = M["coverage"](M["f32"](full), choice)
        lo, hi = r["pair_mass_rounding"]
        self.assertLessEqual(lo, .375)
        self.assertGreaterEqual(hi, .375)
        self.assertAlmostEqual(r["gold_given_pair"], 2/3)
        self.assertLess(hi-lo, 1e-6)
        self.assertAlmostEqual(r["gold_mass"], r["pair_mass"]*r["gold_given_pair"])

    def test_large_rounded_loss_does_not_identify_pair_mass(self):
        r = M["coverage"](2.**30, 2.**30)
        self.assertEqual(r["coverage_nll_rounding"], [0.,64.])
        self.assertEqual(r["pair_mass_rounding"], [math.exp(-64),1.])
        self.assertEqual(M["coverage"](0.,0.)["coverage_nll_rounding"], [0.,2.**-150])
        for full, choice in [(0., 1.), (-1., 0.), (float("nan"), 0.), (1., float("inf"))]:
            with self.assertRaises(ValueError):
                M["coverage"](full, choice)

    def test_directional_loss_split_by_finite_differences(self):
        # Three logits [y, -x, 5x]; first two form the declared pair, first is gold.
        def losses(x, y):
            gold, rival, other = map(math.exp, (y, -x, 5*x))
            total, inside = gold+rival+other, gold+rival
            return dict(full_nll=-math.log(gold/total), choice_nll=-math.log(gold/inside),
                        coverage_nll_estimate=-math.log(inside/total))
        gf, gc, eps = [4/3, -2/3], [-.5, -.5], 1e-5
        for dx, dy in [(1.,0.), (0.,1.), (-.5,.9), (0.,0.)]:
            slopes = M["slopes"](gf, gc, [dx,dy])
            left, right = losses(-eps*dx,-eps*dy), losses(eps*dx,eps*dy)
            for key, value in slopes.items():
                self.assertAlmostEqual(value, (right[key]-left[key])/(2*eps), delta=1e-9)
        split = M["slopes"](gf, gc, [1.,0.])
        self.assertLess(split["choice_nll"], 0.)
        self.assertGreater(split["full_nll"], 0.)

    def test_zero_coverage_gradient_and_malformed_vectors(self):
        r = M["slopes"]([2.,-3.], [2.,-3.], [.75,1.])
        self.assertEqual(r["coverage_nll_estimate"], 0.)
        self.assertEqual(r["choice_nll"], r["full_nll"])
        for full, choice, move in [([],[],[]), ([1.,2.],[1.],[1.]), ([float("nan")],[1.],[0.])]:
            with self.assertRaises(ValueError):
                M["slopes"](full,choice,move)

    def test_independent_margin_pullback_uses_negative_rival_probability(self):
        self.assertEqual(M["independent_choice"]([4.,-8.],.25,[-1.,2.],.001), ([-1.,2.], 0.))
        for g, p, grad in [([4.,-8.],.25,[1.,-2.]), ([4.],.25,[-1.,2.]), ([4.],0.,[-1.]), ([4.],.25,[0.])]:
            with self.assertRaises(ValueError):
                M["independent_choice"](g,p,grad,.001)

    def test_reject_different_owner_prefix_label_position_and_objective(self):
        case = dict(key="a",gold=2,rival=3,prefix=[4,5])
        row = dict(key="a",gold=2,rival=None,position=2,objective="full",declared=None,loss=1.,
                   step=dict(origin_id="same"))
        M["check_join"](case,row,"full","same")
        for edit in [dict(key="b"),dict(gold=3),dict(position=1),dict(objective="rival"),
                     dict(rival=1),dict(step=dict(origin_id="warm"))]:
            with self.assertRaises(ValueError):
                M["check_join"](case,{**row,**edit},"full","same")


if __name__=="__main__":
    unittest.main()

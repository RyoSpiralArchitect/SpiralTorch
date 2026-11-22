import math
from pathlib import Path
import runpy
import unittest


M = runpy.run_path(str(Path(__file__).with_name("metric.py")))
p, recover, alignment = (M[k] for k in ("packed", "recover", "alignment"))


class LocalMetricTests(unittest.TestCase):
    def test_affine_inverse_preserves_nontrivial_gate_and_bias_bytes(self):
        x, dy = (2., -.5, 1.5), (-3., 8., .375)
        bias = tuple(y*.25 for y in dy)
        g = p(tuple(a*b for a,b in zip(x,bias))+bias)
        self.assertEqual(recover(p(x),g,.25),p(dy))
        for site, grad, strength in [(p(x),g,.5),(p(x),g[:-4],.25),(p((2.,1.,1.5)),g,.25)]:
            with self.assertRaises(ValueError): recover(site,grad,strength)

    def test_even_a_round_tripping_ambiguous_underflow_is_not_a_witness(self):
        for bias in [0., -0., 2.**-149, 2.**-126, float("inf"), float("nan")]:
            with self.assertRaises(ValueError): recover(p([1.]),p([bias,bias]),.25)

    def test_exact_cancellation_signed_and_undefined_slopes(self):
        a = alignment(p([1.,1.]),p([1.,-1.]))
        self.assertEqual((a["gate_dot"],a["bias_dot"],a["cosine"],a["full_along_negative_choice"]), (1.,-1.,0.,"flat"))
        a = alignment(p([2.,0.]),p([-4.,0.]))
        self.assertEqual((a["full_slope_at_1_256"],a["cosine"],a["full_along_negative_choice"]), (2./256,-1.,"up"))
        self.assertIsNone(alignment(p([1.,2.]),p([0.,0.]))["full_slope_at_1_256"])
        self.assertEqual(alignment(p([0.,0.]),p([0.,1.]))["full_along_negative_choice"],"flat")

    def test_positive_rescaling_cannot_flip_sign_but_anisotropic_map_can(self):
        # diag(1, 2) is its own adjoint; weighting axis two can flip the Gram sign.
        a, b = ([2.,1.], [-1.,1.])
        for scale in [1., 7.]:
            self.assertEqual(alignment(p([scale*x for x in a]),p([scale*x for x in b]))["full_along_negative_choice"],"up")
        self.assertEqual(alignment(p([2.,2.]),p([-1.,2.]))["full_along_negative_choice"],"down")

    def test_independent_margin_requires_matched_nonzero_reference(self):
        self.assertAlmostEqual(M["relative"]([2.,3.],[2.,2.]),1/math.sqrt(8))
        for a,b in [([1.],[0.]),([1.,1.],[1.])]:
            with self.assertRaises(ValueError): M["relative"](a,b)


if __name__ == "__main__":
    unittest.main()

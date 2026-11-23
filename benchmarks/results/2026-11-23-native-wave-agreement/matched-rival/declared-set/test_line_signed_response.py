import math
from pathlib import Path
import runpy
import unittest

probe = runpy.run_path(str(Path(__file__).with_name("line_signed_response.py")))
kl = probe["kl"]


class SignedResponse(unittest.TestCase):
    def test_zero_and_logit_shift(self):
        self.assertEqual(kl([0., 1.], [2., 3.]), 0.)
        self.assertEqual(kl([3.], [4.]), 0.)

    def test_uniform_binary_both_signs(self):
        for d in (-.25, .25):
            self.assertAlmostEqual(kl([0., 0.], [d, -d]), math.log(math.cosh(d)), places=15)

    def test_order_matters(self):
        z = math.log(3.)
        self.assertAlmostEqual(kl([0., z], [0., 0.]), .75*math.log(1.5)+.25*math.log(.5))
        self.assertNotAlmostEqual(kl([0., z], [0., 0.]), kl([0., 0.], [0., z]))

    def test_reject_malformed_row(self):
        for bad in ([0., 1.], [], [math.nan]):
            with self.assertRaises(ValueError):
                kl([0.], bad)


if __name__ == "__main__":
    unittest.main()

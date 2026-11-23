"""A finite sign reversal need not mean the origin VJP is wrong."""
from pathlib import Path
import runpy
import unittest

s=runpy.run_path(str(Path(__file__).with_name("secant.py")))


class Secants(unittest.TestCase):
    def test_curved_end_and_consistent_origin(self):
        eps=1/256
        f=lambda t:-t+2*t*t
        odd=(f(eps)-f(-eps))/2
        r=s["bracket"]([-eps,eps],[odd,-odd],[1,-1])
        self.assertTrue(r["H_consistent"] and r["rows_consistent"])
        self.assertLess(r["observed_H"],0)
        self.assertGreater(f(1)-f(0),0)

    def test_absent_and_reversed_signals_are_not_confirmations(self):
        invisible=s["bracket"]([1e-8,-1e-8],[-1e-8,1e-8],[1,-1])
        self.assertIsNone(invisible["H_consistent"])
        self.assertIsNone(invisible["rows_consistent"])
        reversed=s["bracket"]([.01,-.02],[-.01,.02],[1,-1])
        self.assertFalse(reversed["H_consistent"] or reversed["rows_consistent"])

    def test_float_endpoints_and_signed_zero_equality(self):
        self.assertEqual(s["interpolate"]([1.,-2.],[1.125,-3.],256),[1.125,-3.])
        self.assertTrue(s["same_logits"]({"a":[0.]*11},{"a":[0.]*11}))
        self.assertFalse(s["same_logits"]({"a":[-0.]+[0.]*10},{"a":[0.]*11}))


if __name__=="__main__":
    unittest.main()

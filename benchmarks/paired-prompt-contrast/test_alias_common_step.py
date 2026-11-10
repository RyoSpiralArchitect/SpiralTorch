import math
import unittest
from alias_common_step import triangle
from alias_common_step_audit import qualifies


class CommonStepTests(unittest.TestCase):
    def test_unit_weights_and_raw_weights_are_different_objects(self):
        r=triangle([[1,0,0],[0,2,0],[0,0,4]])
        self.assertAlmostEqual(r["margin"],1/math.sqrt(3))
        for x,w in zip(r["raw_weights"],[4/7,2/7,1/7]):self.assertAlmostEqual(x,w)
        self.assertTrue(all(x<0 for x in r["slopes"]))

    def test_strict_infeasible_or_unmeasurably_small_is_not_success(self):
        with self.assertRaises(ValueError):triangle([[1,0],[-1,0],[0,1]])
        p=dict(absolute_margin=5e-6,linear_rel=.15)
        self.assertTrue(qualifies(-.0001,-.000101,p))
        self.assertFalse(qualifies(-.0001,+.000101,p))
        self.assertFalse(qualifies(-1e-8,-1e-8,p))
        self.assertFalse(qualifies(-.0001,-.0002,p))

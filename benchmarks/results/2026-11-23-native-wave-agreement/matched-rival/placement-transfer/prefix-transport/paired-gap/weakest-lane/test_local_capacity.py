from pathlib import Path
import math
import runpy
import unittest

capacity=runpy.run_path(str(Path(__file__).with_name("local_capacity.py")))["common_direction"]


class SimultaneousMargins(unittest.TestCase):
    def test_aligned_opposed_and_zero(self):
        for b,ratio in (([3.,0.],1.),([-3.,0.],0.),([0.,0.],None)):
            r=capacity([1.,0.],b,[1.,1.],.1)
            if ratio is None:self.assertIsNone(r["ratio"])
            else:self.assertAlmostEqual(r["ratio"],ratio)
        self.assertAlmostEqual(capacity([1.,0.],[1.,0.],[1.,1.],.1)["common_gain"],.1*math.sqrt(2))

    def test_diagonal_scaling_and_order_exchange(self):
        a,b,q=[2.,0.],[0.,1.],[4.,1.]
        r=capacity(a,b,q,.25)
        self.assertAlmostEqual(r["common_gain"],.25)
        self.assertAlmostEqual(r["ratio"],1/math.sqrt(2))
        rev=capacity(b,a,q,.25)
        self.assertAlmostEqual(r["common_gain"],rev["common_gain"])
        self.assertAlmostEqual(r["lambda_first"],1-rev["lambda_first"])

    def test_unequal_opposed_intersection(self):
        r=capacity([1.,2.],[-2.,-4.],[3.,9.],.25)
        self.assertAlmostEqual(r["ratio"],0.,places=14)
        self.assertLess(r["physical_l2"],16.)


if __name__=="__main__":unittest.main()

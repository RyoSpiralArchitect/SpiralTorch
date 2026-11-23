"""Check the exploratory linear bound with explicit signed, nonorthogonal columns."""
import itertools
import math
from pathlib import Path
import runpy
import unittest

bound=runpy.run_path(str(Path(__file__).with_name("line_sensitivity_summary.py")))["column_bound"]


class LinearBoundTests(unittest.TestCase):
    def test_reverse_triangle_and_residual_hold_for_signed_combinations(self):
        # For each prompt: three parameter columns in a two-dimensional output space.
        maps=[[(.1,0),(.1,.2),(-.3,.1)],[(2.,-1.),(-.4,.5),(.1,-.3)],[(.3,.1),(-.1,.3),(.1,.1)]]
        mags=[[math.hypot(*m[j]) for m in maps] for j in range(3)]
        r,cap=2.,1.5
        b=bound(mags,0,1,r,cap)
        self.assertLess(b["dominant_coordinate_bound_in_probe_units"],r)
        for x in itertools.product((-1.,-.5,0.,.5,1.),repeat=3):
            lengths=[sum(sum(xj*v[k] for xj,v in zip(x,m))**2 for k in range(2)) for m in maps]
            if not max(lengths):
                continue
            scale=min(r/math.sqrt(sum(v*v for v in x)),math.sqrt(cap/max(lengths)))
            self.assertLessEqual(abs(x[0])*scale,b["dominant_coordinate_bound_in_probe_units"]+1e-12)
            for v,upper in zip(lengths,b["upper_per_anchor_over_T"]):
                self.assertLessEqual(scale*scale*v,upper+1e-12)

    def test_one_concentrated_coordinate_is_tight_without_residual_axes(self):
        b=bound([[0.,2.,.2]],0,1,32.,2.)
        self.assertAlmostEqual(b["upper_mean_over_T"],(2+.02)/3)
        self.assertAlmostEqual(b["dominant_coordinate_bound_in_probe_units"],math.sqrt(2)/2)


if __name__=="__main__":
    unittest.main()

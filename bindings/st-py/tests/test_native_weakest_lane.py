"""The Python client transports the native signed-lane choice; no Python loss or optimizer."""
import math
from pathlib import Path
import runpy
import unittest

Fixture = runpy.run_path(str(Path(__file__).with_name("test_native_paired_token_gap.py")))["PairedGap"]


class WeakestLane(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.f = Fixture()

    def test_both_lane_choices_match_single_lane_unit_derivative(self):
        f=self.f
        model=f.model()
        selected=set()
        for a,b in ((7,8),(8,7)):
            rows=[model.last_mlp_prefix_margin_signal(p,a,b) for p in (f.first,f.second)]
            signed=[row["contrast"]["target_minus_alternative"]*s for row,s in zip(rows,(1,-1))]
            self.assertNotEqual(*signed)
            weak=signed.index(min(signed))
            selected.add(weak)
            g=model.geometry_paired_token_gap_gradients(f.first,f.second,[a,b],"weakest_lane")
            r=g.paired_token_gap_report()
            self.assertEqual(r,model.paired_token_gap(f.first,f.second,[a,b],"weakest_lane"))
            self.assertEqual(r["weakest_lane"],dict(signed_margin=min(signed),active=[weak==0,weak==1]))
            self.assertAlmostEqual(g.loss,math.log1p(math.exp(-min(signed))),places=6)
            derivative=(-1 if weak==0 else 1)/(1+math.exp(min(signed)))
            self.assertEqual(r["row_derivatives"][1-weak],0.)
            for name,vec in g.parameter_gradients()["transformer.h.1.mlp"].items():
                for value,unit in zip(vec,rows[weak][name+"_gradient"]):
                    self.assertAlmostEqual(value,derivative*unit,delta=2e-7+3e-4*abs(derivative*unit))
            self.assertNotIn("weakest_lane",model.paired_token_gap(f.first,f.second,[a,b],"binary"))
        self.assertEqual(selected,{0,1})

    def test_preflight_does_not_accept_anchor_options(self):
        class Poison:
            def __len__(self): return 3
            def __getitem__(self,k): raise AssertionError("copied an unadmitted prefix")
        with self.assertRaises((TypeError,ValueError)):
            self.f.model().geometry_paired_token_gap_gradients(Poison(),Poison(),[7,8],"weakest_lane",stiffness=1.)


if __name__=="__main__":
    unittest.main()

"""Model-free checks for the independent source oracle and paired transfer metric."""
import math
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

s=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))

class Controls(unittest.TestCase):
    def test_constant_and_causal_changes(self):
        src,mu=s.feature([2.,3.,5.]*4,"change")
        self.assertTrue(all(x==0. for x in src+mu))
        raw=[2.,4.,3.,6.,8.,5.,8.,10.,9.]
        src,mu=s.feature(raw,"change")
        exp=[0.,0.,0.,-4./3.,4./3.,2./3.,2./3.,2./3.,4./3.]
        self.assertEqual(src,exp)
        self.assertEqual(mu,[s.d.f32(-2/3),2.,2.])
        self.assertNotEqual(mu,s.feature(raw,"level")[1])

    def test_template_and_single_marker_pairs(self):
        for kind in s.PARENT:
            train,transfer=[s.cases(kind,k) for k in (False,True)]
            self.assertFalse({tuple(r["prompt"]) for r in train}&{tuple(r["prompt"]) for r in transfer})
            for r in train+transfer:
                self.assertEqual(r["prompt"][r["prompt"].index(9)+1],0)
            for teacher in s.x.TEACHERS:
                self.assertEqual(len(s.x.pairs(train,teacher))*2,len(train))

    def test_gain_cannot_be_replaced_by_ratio(self):
        a=s.cap([1.,0.,0.],[-1.,0.,0.],[1.]*3,.001)
        b=s.cap([1.,0.,0.],[0.,1.,0.],[1.]*3,.001)
        self.assertEqual(a["common_gain"],0.)
        self.assertTrue(math.isclose(b["ratio"],1/math.sqrt(2)))

if __name__=="__main__":
    unittest.main()

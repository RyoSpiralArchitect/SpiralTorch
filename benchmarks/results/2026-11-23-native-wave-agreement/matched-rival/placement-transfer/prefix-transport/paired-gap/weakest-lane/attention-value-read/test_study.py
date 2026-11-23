"""No model calls: heads, lagged routing and fixed position controls."""
from pathlib import Path
import math
import runpy
import unittest

s=runpy.run_path(str(Path(__file__).with_name("study.py")))

class Equations(unittest.TestCase):
    def test_two_heads_do_not_share_probabilities_or_values(self):
        r=dict(shape=[3,2],heads=2,key_lag=0,scale=1.,query=[1.,-1.],
               keys=[0.,0.,1.,1.,2.,2.],values=[1.,4.,2.,5.,3.,6.])
        ps,mu=s["mix"](r)
        expected=[math.exp(i)/sum(math.exp(j) for j in range(3)) for i in range(3)]
        for p,e in zip(ps[0],expected):self.assertAlmostEqual(p,e)
        for p,e in zip(ps[1],reversed(expected)):self.assertAlmostEqual(p,e)
        self.assertAlmostEqual(mu[0],sum(p*(i+1) for i,p in enumerate(expected)))
        r["key_lag"]=1
        lagged,_=s["mix"](r)
        self.assertEqual(lagged[0][0],lagged[0][1]);self.assertNotEqual(lagged,ps)

    def test_slot_two_is_a_predeclared_positional_control(self):
        for kind in s["PARENT"]:
            for test in (False,True):
                for row in s["cases"](kind,test):
                    p=row["prompt"]
                    self.assertEqual(p[p.index(9)-1],0)
                    self.assertEqual(p[2],row["head"]+1 if row["template"]=="prehead" else 0)

if __name__=="__main__":unittest.main()

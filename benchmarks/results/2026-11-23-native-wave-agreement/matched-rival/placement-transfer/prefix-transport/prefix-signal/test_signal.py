import copy
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

v=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
u=v.u


def record(rows,up):
    w,n=3,len(rows)
    features=[x/n for r in rows for x in r]
    g=[sum(features[j::w])*up[j] for j in range(w)]
    return {"trace":dict(shape=[n,w],rotation="identity",site_input=[x for r in rows for x in r],source_features=features,
                         gate_gradient=g,bias_gradient=up)}


class SignalTests(unittest.TestCase):
    def test_separates_feature_and_differential_readout(self):
        x=[[0,0,0],[4,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,0]]
        y=copy.deepcopy(x);y[1][0]=-2
        f=v.routes(record(x,[2,0,0]),record(y,[2,0,0]),"head")
        self.assertEqual((f["feature_norm"],f["readout_norm"],f["input_contrast"]),(2.,0.,6.))
        r=v.routes(record(x,[2,0,0]),record(x,[-2,0,0]),"head")
        self.assertEqual(r["feature_norm"],0)
        self.assertAlmostEqual(r["readout_norm"],10/3)
        both=v.routes(record(x,[3,0,0]),record(y,[-1,0,0]),"head")
        self.assertLess(both["residual"],1e-12)
        y[0][2]=1
        with self.assertRaises(ValueError):
            v.routes(record(x,[2,0,0]),record(y,[2,0,0]),"head")

    def test_cancellation_and_undefined_coupling(self):
        x=[[0,0,0] for _ in range(6)]
        y=copy.deepcopy(x);y[1][0]=6;y[2][0]=-6
        r=v.routes(record(x,[1,0,0]),record(y,[1,0,0]),"head")
        self.assertIsNone(r["feature_coupling"])
        self.assertEqual(r["feature_kept"],0.)
        self.assertEqual(r["direct_feature_norm"],1.)

    def test_oracle_is_wrong_only_signed_and_fixed_length(self):
        tr={"contrast":{"target_minus_alternative":-1.},"gate_gradient":[3.,4.,0.]}
        self.assertIsNone(u.oracle_values(tr,1))
        self.assertEqual(u.oracle_values(tr,0),[u.d.f32(.15),u.d.f32(.2),0.])
        tr["contrast"]["target_minus_alternative"]=1.
        self.assertEqual(u.oracle_values(tr,1),[u.d.f32(-.15),u.d.f32(-.2),0.])
        tr["gate_gradient"]=[0.]*3
        self.assertIsNone(u.oracle_values(tr,1))


if __name__=="__main__":
    unittest.main()

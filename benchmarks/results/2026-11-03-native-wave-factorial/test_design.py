import copy
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as d  # noqa: E402


class Design(unittest.TestCase):
    def test_signed_factor_attribution_and_zero_reference(self):
        e = dict(affine=4.,saturation=2.,projection=1.,both=0.5)
        r = d.decompose(e)
        self.assertEqual(r["saturation_average"],.3125)
        self.assertEqual(r["projection_average"],.5625)
        self.assertEqual(r["saturation_average"] + r["projection_average"],r["total_reduction"])
        self.assertEqual(r["interaction"],.375)
        self.assertIsNone(d.decompose(dict(e,affine=0.)))
        self.assertLess(d.decompose(dict(e,saturation=10.))["saturation_average"],0.)

    def test_factual_check_and_shape_are_done_before_a_study(self):
        f = runpy.run_path(str(d.m.ROOT/"bindings/st-py/tests/test_native_wave_factorial.py"))
        import spiraltorch as st
        base = st.NativeCausalLm.from_gpt2(*f["fixture"]())
        for source,kind in (("affine","affine"),("wave_fixed","both")):
            model = f["attach"](base,kind)
            ids,mask = f["IDS"],f["MASK"]
            old = model.adaptation_loss_gradients(ids,[1,len(ids)],labels=mask,mode="joint",observe_geometry=True)
            new = model.wave_factorial_observation_json(ids,[1,len(ids)],d.CFG,labels=mask)
            meta=dict(order="test",arm=source,index=0,key="test")
            prior=dict(meta,observation=json.loads(old.observation_json()))
            row=dict(meta,observation=json.loads(new))
            with patch.object(d.m,"SITE",f["SITE"]):
                d.validate(row,prior)
                bad=copy.deepcopy(row)
                bad["observation"]["loss"]+=.001
                with self.assertRaisesRegex(ValueError,"factual CE"):
                    d.validate(bad,prior)
        self.assertEqual(len(d.anchors()),576)


if __name__=="__main__":
    unittest.main()

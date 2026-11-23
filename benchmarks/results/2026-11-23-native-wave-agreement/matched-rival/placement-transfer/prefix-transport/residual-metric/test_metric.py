"""No native model: analytic oracles and decision gate tests."""
import copy
import math
from pathlib import Path
import runpy
import unittest
from types import SimpleNamespace

z=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
v=z.v


class MetricTests(unittest.TestCase):
    def test_positive_turn_and_signed_odd_tail(self):
        # Final mean is (1, 2, -1): R(2,-1,4) + (1,2,-6), divided by 2.
        rec=dict(shape=[2,3],rotation="quarter_turn",strength=.25,site_input=[2.,-1.,4.,1.,2.,-6.],
                 source_features=[.5,1.,2.,.5,1.,-3.])
        self.assertEqual(v.moments([rec]),[1/16,4/16,1/16])
        baseline=z.expected_outputs([rec],[0.]*3)
        end=z.expected_outputs([rec],[1.,2.,3.])
        self.assertEqual(end,[[1.25,3.,-6.75]])
        self.assertAlmostEqual(v.rms_rows(baseline,end),math.sqrt((.25**2+1+.75**2)/3))

    def test_validate_rejects_same_length_wrong_direction(self):
        q=[1.,1.,1.]
        target=v.TARGET
        amp=v.d.f32(target*math.sqrt(3))
        values=[0.,0.,0.,-amp,0.,0.]
        obs=amp/math.sqrt(3)
        step=dict(schema="spiraltorch.prefix_residual_step.v1",metric_id="m",pre_geometry_id="a",
                  post_geometry_id="b",rule="uniform",members=2,total_target_tokens=2,target_rms=target,
                  realized_rms=obs,requested_parameter_l2=target*math.sqrt(3),realized_parameter_l2=amp,
                  member_loss_dots=[-amp,-amp],mean_input_loss=1.)
        sample=dict(cell="x",i=0,id="b",values=values,report=step,members=[dict(loss=1.,v=[0.,0.,0.,1.,0.,0.])]*2)
        v.validate_step(sample,[0.]*6,"a","uniform",q,"m")
        bad=copy.deepcopy(sample)
        bad["values"][3],bad["values"][4]=bad["values"][4],bad["values"][3]
        with self.assertRaises(ValueError):
            v.validate_step(bad,[0.]*6,"a","uniform",q,"m")


if __name__=="__main__":
    unittest.main()

import math
from pathlib import Path
import runpy
import unittest

S = runpy.run_path(str(Path(__file__).with_name("sensitivity.py")))
D = S["D"]


def observation(m):
    loss = D["binary_loss"](m)
    report = dict(mean_target_minus_declared_rival=m, declared_ce=loss, full_ce=loss+1,
                  mix=1.,target_tokens=1,contrasted_tokens=1,rival_ids=1)
    return loss,report


class SensitivityTests(unittest.TestCase):
    def test_recovers_linear_margin_from_numerical_loss_gradient(self):
        for m in (-6., 0., 6.):
            loss,report = observation(m)
            # Margin m + 2*x - 3*y: use centered differences of CE, not its derivative formula.
            eps=1e-5
            g = [(D["binary_loss"](m+eps*a)-D["binary_loss"](m-eps*a))/(2*eps) for a in (2,-3)]
            norm=math.hypot(*g)
            dx=[-x*.01/norm for x in g]
            record = dict(gradient_l2=norm,gradient_dot_step=sum(a*b for a,b in zip(g,dx)),realized_l2=.01)
            v = S["local"](loss,report,record)
            self.assertAlmostEqual(v["margin_gradient_l2"],math.sqrt(13),places=6)
            self.assertAlmostEqual(v["first_order_own_margin_delta"],2*dx[0]-3*dx[1],places=8)

    def test_refuses_saturated_zero_full_mix_and_wrong_update_direction(self):
        step=dict(gradient_l2=1.,gradient_dot_step=-1.,realized_l2=1.)
        loss,report=observation(1000.)
        with self.assertRaisesRegex(ValueError,"undefined"):
            S["local"](loss,report,step)
        loss,report=observation(0.)
        with self.assertRaises(ValueError):
            S["local"](loss,dict(report,mix=.5),step)
        with self.assertRaises(ValueError):
            S["local"](loss,report,dict(step,gradient_dot_step=1.))


if __name__=="__main__":
    unittest.main()

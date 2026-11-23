import copy
import itertools
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

v = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
p, d = v.p, v.d


class OwnerTests(unittest.TestCase):
    def test_zero_and_mask_are_part_of_the_artifact(self):
        host = {"base":"fixture"}
        r = p.artifact("fixture","identity","gate",[0.,0.,0.,-.125,0.,.25])
        with self.assertRaises(ValueError):
            p.state(d.encoded(r),host,"identity","bias")
        r["placements"][0]["tensors"][0]["requires_grad"] = True
        with self.assertRaises(ValueError):
            p.state(d.encoded(r),host,"identity","gate")
        with self.assertRaises(ValueError):
            p.coords([0.,-0.,0.,0.,0.,0.],"gate")

    def test_ray_uses_active_coordinates_only(self):
        target = 2*d.D
        members = [{"loss":.75,"v":[1e8,-1e8,1e8,4.,0.,0.]} for _ in range(2)]
        ray = dict(origin_id="a",result_id="b",requested_l2=target,trainable_values=3,
                   method="cohort_mean_gradient_ray_f64.v1",gradient_l2=4.,realized_l2=target,
                   relative_vector_error=0.,gradient_dot_step=-4*target)
        report = dict(step=ray,members=2,total_target_tokens=2,readout_accumulator="f32",
                      lora_artifact_id=None,base_model_identity="fixture",mean_input_loss=.75,member_loss_dots=[-4*target]*2)
        rec = dict(id="b",report=report,members=members,values=[0.,0.,0.,-target,0.,0.])
        self.assertEqual(p.check_motion(rec,[0.]*6,"a",{"base":"fixture"},"gate"),0.)
        for change in ("sign","fixed","all"):
            bad = copy.deepcopy(rec)
            if change == "sign":
                bad["values"][3] *= -1
            elif change == "fixed":
                bad["values"][0] = 1e-12
            else:
                bad["report"]["step"]["trainable_values"] = 6
            with self.assertRaises(ValueError):
                p.check_motion(bad,[0.]*6,"a",{"base":"fixture"},"gate")

    def test_role_contrasts_and_conflict(self):
        rows, logits = [], {}
        for i,j in itertools.product(range(2),repeat=2):
            h,t = 1-2*i,1-2*j
            key = str(i)+str(j)
            rows.append(dict(frame="a",key=key,head=i,attractor=j))
            logits[key] = [0.]*7+[2+3*h-5*t+7*h*t,0.,0.,0.]
        pos = v.position_stats(rows,logits)
        self.assertEqual([pos[k] for k in ("offset","head","attractor","interaction")],[2,3,-5,7])
        self.assertEqual(pos["conflict_wins"],2)
        with self.assertRaises(ValueError):
            v.position_stats(rows[:-1],logits)

    def test_thresholds_are_strict_per_control_and_order(self):
        row = dict(h=.002,S=.002,positive_pairs=3,wins=9,wins_before=8,
                   owner_gap_h=.002,owner_gap_S=.002,owner_wins=9,
                   position_gap_h=.002,position_gap_S=.002,position_wins=8)
        self.assertEqual(p.qualify([row],"owner"),dict(probability=True,decision=False))
        self.assertTrue(p.qualify([row],"position")["decision"])
        self.assertFalse(p.qualify([row,dict(row,position_gap_h=.001)],"position")["probability"])


if __name__ == "__main__":
    unittest.main()

import copy
import math
from pathlib import Path
import runpy
import unittest

a = runpy.run_path(str(Path(__file__).with_name("audit.py")))
x, check = a["x"], a["check"]


class Pairs(unittest.TestCase):
    def test_matched_splits_and_teachers(self):
        all_ids = set()
        lexical = []
        for kind in ("formal","rehearsal"):
            for test,n in ((False,64),(True,32)):
                rows = x.rows(kind,test)
                self.assertEqual(len(rows),n if kind=="formal" else 8)
                ids = {tuple(r["prompt"]) for r in rows}
                self.assertEqual(len(ids),len(rows))
                self.assertFalse(ids & all_ids)
                all_ids |= ids
                if kind=="formal":
                    lexical.append({(r["frame"][0],r["frame"][2]) for r in rows})
                for t in x.TEACHERS:
                    pairs=x.pairs(rows,t,17)
                    self.assertEqual(pairs,x.pairs(rows[::-1],t,17))
                    self.assertNotEqual(pairs,x.pairs(rows,t,31))
                    self.assertEqual(len(pairs)*2,len(rows))
                    for u,v in pairs:
                        self.assertEqual((u[t],v[t]),(0,1))
                        self.assertEqual([i for i,(m,n) in enumerate(zip(u["prompt"],v["prompt"])) if m!=n],
                                         [3 if u["template"].startswith("head") and t=="head" else
                                          2 if t=="head" else 6])
        self.assertFalse(lexical[0]&lexical[1])
        rows=x.rows("formal")
        with self.assertRaises(ValueError):
            x.pairs(rows+[rows[0]],"head")
        with self.assertRaises(ValueError):
            x.pairs(rows[1:],"head")

    def test_direction_and_center_have_different_evidence(self):
        rows=x.rows("rehearsal")
        z=lambda gap: [0.]*7+[gap,0.,0.,0.]
        origin={r["key"]:z(.5) for r in rows}
        offset={r["key"]:z(1.) for r in rows}
        ordered={r["key"]:z(.5+.125*(1-2*r["head"])) for r in rows}
        s=a["metrics"](rows,origin,offset,"head")
        self.assertEqual((s["rank"]["correct"],s["abs_center_drift"],s["rank"]["ties"]),(0,.5,4))
        s=a["metrics"](rows,origin,ordered,"head")
        self.assertEqual((s["rank"]["up"],s["abs_center_drift"],s["mean_half_gap"]),(4,0.,.125))
        self.assertEqual(s["raw"]["up"]+s["raw"]["down"],0)

    def test_loss_uses_two_opposite_logits_and_keeps_origin_center(self):
        z=[[.75,-.5],[-.125,.875]]
        for arm in x.ARMS:
            c=check.reference(z,arm,.125)
            for row in (0,1):
                za,zb=copy.deepcopy(z),copy.deepcopy(z)
                za[row][0]+=1e-5;zb[row][0]-=1e-5
                numeric=(check.reference(za,arm,.125)["loss"]-check.reference(zb,arm,.125)["loss"])/2e-5
                self.assertAlmostEqual(c["row_derivatives"][row],numeric,places=9)
        self.assertEqual(check.reference(z,"rank",.125)["loss"],check.reference(z,"anchored_rank",.125)["loss"])
        self.assertGreater(check.reference(z,"anchored_rank",1.)["center_penalty"],0)

    def test_frozen_gates_require_every_order_template_and_teacher(self):
        sp=dict(bases=[11],orders=[3,5])
        cases=x.rows("rehearsal",True)
        def make(gap):
            return {r["key"]:[0.]*7+[gap(r),0.,0.,0.] for r in cases}
        bank={x.name(11):make(lambda r:0.)}
        for _,arm,t,o in x.cells(sp,11):
            offset={"anchored_rank":.1,"rank":.2,"binary":0.}[arm]
            bank[x.name(11,arm,t,o)]=make(lambda r: offset + (1-2*r[t])*(0. if arm=="binary" else 1.))
        gates=lambda:a["analyze"](sp,cases,bank)["base_gates"]["11"]
        self.assertTrue(all(gates().values()))
        bank[x.name(11,"anchored_rank","attractor",5)]=bank[x.name(11,"rank","attractor",5)]
        self.assertEqual(gates(),dict(ordering=True,ordering_and_raw=True,center_control=False))
        bank[x.name(11,"binary","head",5)]=bank[x.name(11,"anchored_rank","head",5)]
        self.assertFalse(gates()["ordering"])


if __name__=="__main__":
    unittest.main()

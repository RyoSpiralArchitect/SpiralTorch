import copy
from pathlib import Path
import runpy
import unittest

m=runpy.run_path(str(Path(__file__).with_name("metrics.py")))
s=m["s"]; x=s.x


class WeakerStudy(unittest.TestCase):
    def test_sets_and_number_pairing(self):
        seen=set()
        for kind in s.PREV:
            for test in (False,True):
                rows=s.cases(kind,test)
                ids={tuple(r["prompt"]) for r in rows}
                self.assertEqual(len(ids),len(rows))
                self.assertTrue(all(seq[-2:]==(0,10) and seq.count(0)==1 for seq in ids))
                self.assertFalse(ids&seen)
                self.assertFalse(ids&{tuple(r["prompt"]) for r in x.rows(kind,test)})
                seen.update(ids)
                for teacher in x.TEACHERS:
                    self.assertEqual(x.pairs(rows,teacher,31),x.pairs(rows[::-1],teacher,31))
                    for a,b in x.pairs(rows,teacher,31):
                        self.assertEqual((a[teacher],b[teacher]),(0,1))
                        self.assertEqual(sum(i!=j for i,j in zip(a["prompt"],b["prompt"])),1)
        self.assertEqual(len(seen),112)

    def test_seen_order_is_not_a_pair_win(self):
        rows=s.cases("rehearsal",True)
        def z(f):
            return {r["key"]:[0.]*7+[f(r),0.,0.,0.] for r in rows}
        old=z(lambda r: 0.)
        tie=z(lambda r: 1-r["head"])
        tilted=z(lambda r: 2-r["head"])
        both=z(lambda r: 1-2*r["head"])
        for bank,n,ties,w in ((tie,0,4,0.),(tilted,0,0,-1.),(both,4,0,1.)):
            q=m["outcomes"](rows,old,bank,"head")
            self.assertEqual((q["rank"]["correct"],q["both"],q["weak_ties"],q["mean_min"]),(4,n,ties,w))
            self.assertEqual(len(q["gained_both"]),n)

    def test_every_order_and_template_must_pass(self):
        rows=s.cases("rehearsal",True);sp=dict(bases=[99],orders=[3,5])
        def z(t,size):
            return {r["key"]:[0.]*7+[(1-2*r[t])*size,0.,0.,0.] for r in rows}
        bank={x.name(99):z("head",0.)}
        for n,arm,t,o in s.trajectories(sp,99):
            bank[n]=z(t,1. if arm=="weakest_lane" else 0.)
        gates=lambda:m["analyze"](sp,rows,bank)["base_gates"]["99"]
        self.assertTrue(all(gates()["head"].values()))
        for r in rows:
            if r["template"]=="prehead":
                bank[x.name(99,"binary","head",5)][r["key"]]=z("head",.25)[r["key"]]
        self.assertEqual(gates()["head"],dict(continuous=True,both_and_margin=False,quality=False))
        self.assertTrue(all(gates()["attractor"].values()))

    def test_auditor_checks_the_inactive_lane_and_tie(self):
        pair=x.pairs(s.cases("rehearsal",False),"head")[0]
        for v,act in (([[-1.,0.],[0.,1.]],[True,False]),([[1.,0.],[0.,-1.]],[False,True]),([[1.,0.],[0.,1.]],[True,True])):
            c=s.ck.reference(v,"binary",0.)
            import math
            margins=[v[0][0]-v[0][1],v[1][1]-v[1][0]]
            weak=min(margins);loss=math.log1p(math.exp(-weak));ws=1/(1+math.exp(weak))
            c.update(schema="spiraltorch.paired_token_gap.v1",objective=dict(kind="weakest_lane"),tokens=[7,8],logits=v,
                     weakest_lane=dict(active=act,signed_margin=weak),loss=loss,
                     row_derivatives=[-ws*act[0]/sum(act),ws*act[1]/sum(act)])
            r=dict(keys=[a["key"] for a in pair],objective=c,g=dict(geometry="id",base="base",tokens=2,shape=[2,9],
                loss=s.d.f32(loss),params=dict(bias=[0.]*3,gate=[0.]*3)))
            s.validate_gradient(r,pair,"weakest_lane",None,dict(base="base"),"id",False)
            bad=copy.deepcopy(r);bad["objective"]["row_derivatives"][1]+=.01
            with self.assertRaises(ValueError):
                s.validate_gradient(bad,pair,"weakest_lane",None,dict(base="base"),"id",False)


if __name__=="__main__":
    unittest.main()

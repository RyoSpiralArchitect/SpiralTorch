import math
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

a=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("analyze.py"))))
k,d=a.k,a.d


class ConditionalUpdate(unittest.TestCase):
    def test_pairing_preserves_exact_stream_and_both_teacher_balances(self):
        rows=d.design(True)["train"]
        perms=[]
        for seed in (17,31,73):
            pair=k.schedule(rows,seed,"pair")
            flat=[r for p in pair for r in p]
            self.assertEqual(k.schedule(rows,seed,"single"),[[r] for r in flat])
            self.assertCountEqual(flat,rows)
            for x,y in pair:
                self.assertEqual(x["frame"],y["frame"])
                self.assertTrue(all(x[t]+y[t]==1 for t in d.TEACHERS))
            perms.append(tuple(r["key"] for r in flat))
        self.assertEqual(len(set(perms)),3)
        sp=k.plan(True)
        cells=list(k.specs(sp))
        self.assertEqual(len(cells),144)
        self.assertEqual(sum(64//k.PACK[c[3]] for c in cells),6912)
        self.assertEqual(len(cells)*64,9216)
        self.assertEqual(48*(6+len(cells)),7200)
        self.assertEqual(len(set(sp["weights"].values())),3)

    def test_two_labels_identify_margin_jacobian_despite_shared_component(self):
        z=[0.]*11
        z[7],z[8]=2.,1.
        p=math.exp(-k.nll([2.,1.],0))
        q=1.-p
        slope=[.1,-.2,.3,.45,-.25,.01]
        pure=[k.scaled(slope,-q),k.scaled(slope,p)]
        common=[.65,.1,0.,-.02,.3,-.1]
        mixed=[[x+c for x,c in zip(v,common)] for v in pure]
        vals={}
        for scope,gs in (("mixed",mixed),("pure",pure)):
            vals[scope]=[]
            for i,g in enumerate(gs):
                ce,binary,loss=k.costs(z,i,k.MIX[scope])
                vals[scope].append(dict(gold=i,mix=k.MIX[scope],ce=ce,binary=binary,loss=loss,v=list(map(d.f32,g))))
        J,result=k.check_bank(vals,z,True)
        self.assertLess(k.norm(k.sub(J,slope)),1e-7)
        self.assertAlmostEqual(result["pure_cos"],-1.)
        self.assertGreater(result["mixed_cos"],0.)
        vals["pure"][0]["v"][0]+=.125
        with self.assertRaises(ValueError):
            k.check_bank(vals,z,True)

    def test_unnormalized_pair_mean_then_normalize_is_not_mean_of_rays(self):
        x,y={"key":"x","head":0},{"key":"y","head":1}
        def g(row):
            return {"pure":[{"v":v} for v in row]}
        bank={"x":g(([2.,0.],[-1.,0.])),"y":g(([0.,1.],[0.,-.5]))}
        p=a.frozen_ray([[x,y]],bank,"pure","head",1)
        one=a.frozen_ray([[x],[y]],bank,"pure","head",1)
        self.assertAlmostEqual(k.norm(p),2*d.D)
        self.assertAlmostEqual(k.norm(one),math.sqrt(2)*d.D)
        self.assertNotEqual(p,one)
        bank["y"]["pure"][1]={"v":[-2.,0.]}
        self.assertIsNone(a.frozen_ray([[x,y]],bank,"pure","head",1))

    def test_choice_global_choice_and_teacher_specificity_are_distinct(self):
        rows=d.design(True)["test"]
        def snapshot(teacher):
            z={}
            for r in rows:
                v=[0.]*11
                v[10]=6.
                m=0. if teacher is None else .25*(1-2*r[teacher])
                v[7]=m/2; v[8]=-m/2
                z[r["key"]]=v
            return a.metrics(rows,z)
        before,head,other=map(snapshot,(None,"head","attractor"))
        e=a.effect(before,head,other)
        self.assertEqual((e["wins_before"],e["ties_before"],e["wins"],e["global_wins"],e["positive_pairs"]),(0,16,16,0,4))
        self.assertAlmostEqual(e["S"],.25)
        row=dict(pure=e,mixed=dict(e,wins=8),gap_h=.2,gap_S=.2)
        self.assertTrue(a.qualify([row]*3)["decision"])
        self.assertFalse(a.qualify([row,dict(row,gap_h=.001),row])["probability"])
        self.assertFalse(a.qualify([dict(row,mixed=e)]*3)["decision"])

    def test_train_only_scaling_and_no_position_leak_into_bag(self):
        rows=d.design(True)["test"]
        x=next(r for r in rows if r["head"]==0 and r["attractor"]==1)
        y=next(r for r in rows if r["frame"]==x["frame"] and r["head"]==1 and r["attractor"]==0)
        self.assertEqual(*a.probe.features("bag",[x,y],None))
        self.assertNotEqual(*a.probe.features("position",[x,y],None))
        mu,std=a.probe.standardize([[1.,2.],[3.,2.]])
        self.assertEqual(mu,[2.,2.])
        self.assertEqual(std,[1.,.0001])
        self.assertEqual(a.probe.apply([[5.,2.]],mu,std),[[3.,0.,1.]])


if __name__=="__main__":
    unittest.main()

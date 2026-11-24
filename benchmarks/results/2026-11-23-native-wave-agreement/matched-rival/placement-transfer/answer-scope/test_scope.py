import json
import math
from pathlib import Path
import runpy
import struct
import types
import unittest

v=types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
s=v.s


class ScopeTest(unittest.TestCase):
    def test_seeds_are_independent_and_all_layout_bytes_covered(self):
        template=s.fixtures_cached()[2]
        n,=struct.unpack_from("<Q",template)
        variants=[s.generate(template,k) for k in (101,211,307,409)]
        self.assertEqual(variants[0],s.generate(template,101))
        self.assertEqual(len(set(variants+[template])),5)
        headers=json.loads(template[8:8+n])
        for raw in variants:
            self.assertEqual(template[:8+n],raw[:8+n])
            for name,t in headers.items():
                start,end=t["data_offsets"]
                old=template[8+n+start:8+n+end]
                data=raw[8+n+start:8+n+end]
                self.assertNotEqual(old,data)
                vals=struct.unpack("<%df"%(len(data)//4),data)
                self.assertTrue(all(map(math.isfinite,vals)))
                if name.endswith(".weight") and len(t["shape"])==1:
                    self.assertTrue(all(.89<x<1.11 for x in vals))

    def test_stop_only_changes_terminated_loss_and_not_content_transfer(self):
        rows=s.d.design(True)["test"]
        def costs(r, swapped):
            stops=[2.,2.]
            if swapped:
                stops[r["head"]]=1.
            tokens=[[3.,stops[0]],[3.,stops[1]]]
            return dict(token_nlls=tokens,line_nlls=[sum(p) for p in tokens])
        x={r["key"]:costs(r,False) for r in rows}
        y={r["key"]:costs(r,True) for r in rows}
        for r in rows:
            k=r["key"]
            self.assertEqual(s.loss(x[k],r["head"],"answer"),s.loss(y[k],r["head"],"answer"))
            self.assertGreater(s.loss(x[k],r["head"],"through_stop"),s.loss(y[k],r["head"],"through_stop"))
        before,after=(v.components(rows,m) for m in (x,y))
        effect=v.effects(before,after,before)
        self.assertEqual((effect["h"],effect["S"],effect["switches"],effect["ties"]),(0,0,0,16))
        self.assertEqual(effect["delta_stop"],1.)
        self.assertEqual(effect["delta_line"],1.)

    def test_two_scopes_use_the_same_binary_mix_but_distinct_gold_length(self):
        costs=dict(token_nlls=[[1.,4.],[2.,3.]],line_nlls=[5.,5.])
        for gold in (0,1):
            self.assertAlmostEqual(s.loss(costs,gold,"through_stop"),.5*(2.5+math.log(2)))
            other=1-gold
            margin=costs["token_nlls"][gold][0]-costs["token_nlls"][other][0]
            self.assertAlmostEqual(s.loss(costs,gold,"answer"),.5*(1+gold+math.log1p(math.exp(margin))))

    def test_probability_without_decision_is_not_a_switch(self):
        base=dict(h=.01,S=.01,positive_pairs=4,before_wins=8,wins=8)
        r=v.compare(base,dict(base,h=0,S=0))
        self.assertEqual(v.gates([r]*3),dict(probability=True,decision=False))
        improve=v.compare(dict(base,wins=9),dict(base,h=0,S=0))
        self.assertTrue(v.gates([improve]*3)["decision"])
        miss=dict(improve,gap_h=.001)
        self.assertFalse(v.gates([improve,miss,improve])["probability"])
        self.assertFalse(v.wave_gate([improve]*3,[improve]*3))
        with self.assertRaises(ValueError):
            v.gates([])
        with self.assertRaises(ValueError):
            v.wave_gate([improve],[])

    def test_switch_when_gold_margin_crosses_zero(self):
        rows=s.d.design(True)["test"]
        def costs(row,mode):
            margin=.06 if mode is None else .2*(1-2*row[mode])
            return dict(token_nlls=[[2.-margin/2,2.],[2.+margin/2,2.]])
        before,head,att=(v.components(rows,{r["key"]:costs(r,m) for r in rows}) for m in (None,"head","attractor"))
        e=v.effects(before,head,att)
        self.assertAlmostEqual(e["h"],.2)
        self.assertAlmostEqual(e["S"],.2)
        self.assertEqual((e["before_wins"],e["wins"],e["switches"],e["positive_pairs"]),(8,16,8,4))


if __name__=="__main__":
    unittest.main()

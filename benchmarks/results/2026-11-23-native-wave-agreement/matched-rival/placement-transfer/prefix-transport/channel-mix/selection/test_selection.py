"""Small independent checks; no decoder, pretrained data or transfer scores."""
from pathlib import Path
import os
import runpy
import struct
import subprocess
import unittest

r=runpy.run_path(str(Path(__file__).with_name("run.py")))
a=r["a"];u=a.u
def frame(g):
    return ((5.,0.),(5.+g,0.),(5.-g,0.),(5.,0.))

class RouteSelection(unittest.TestCase):
    def test_design_is_crossed_and_disjoint(self):
        train=u.fresh("rehearsal")
        rows=u.fresh("formal")
        self.assertEqual((len(train),len(rows)),(8,32))
        self.assertFalse({tuple(r["prompt"]) for r in train}&{tuple(r["prompt"]) for r in rows})
        for h,t in zip(rows[::2],rows[1::2]):
            self.assertNotEqual(h["prompt"],t["prompt"])
            self.assertEqual(sorted(h["prompt"]),sorted(t["prompt"]))
            self.assertEqual((h["frame"],h["head"],h["attractor"]),(t["frame"],t["head"],t["attractor"]))
        self.assertEqual(len(u.quartets(rows[::2])),4)
        for bad in (rows[::2][:-1],rows[::2]+[rows[0]],[]):
            with self.assertRaises(ValueError):u.quartets(bad)

    def test_worst_before_mean_and_role(self):
        groups=[[frame(v) for v in xs] for xs in ((1.,11.,2.,-1.),(1.,-1.,3.,-2.))]
        self.assertEqual(a.reference(groups,"head"),dict(frames=2,selected=1,threshold=5.,worst=[-2.,1.,-3.],
                                                          mean=[4.,1.5,-2.5]))
        self.assertEqual(a.reference(groups,"attractor")["selected"],2)
        self.assertEqual(a.reference([[frame(0.)]*4],"head")["selected"],0)

    def test_separates_raw_decision_centered_decision_and_rank(self):
        rows=u.fresh("rehearsal")[::2]
        def logits(v):return [0.]*7+[v,0.,0.,0.]
        z0={r["key"]:logits(1.) for r in rows}
        z={r["key"]:logits(3. if r["head"]==0 else 1.) for r in rows}
        h,c,st=a.role_stats(rows,z0,z,"head",2.)
        self.assertEqual((h,c),(1.,1.))
        self.assertEqual((st["raw_after"],st["raw_up"],st["raw_down"]),(2,0,0))
        self.assertEqual((st["centered_before"],st["centered_after"],st["rank_up"],st["rank_after"]),(2,4,2,2))
        other=a.role_stats(rows,z0,z,"attractor",2.)[2]
        self.assertEqual((other["rank_up"],other["rank_ties"]),(0,2))

    def test_real_rust_binary_and_rejections(self):
        exe=os.environ.get("ST_ROUTE_SELECTOR")
        if not exe:self.skipTest("set ST_ROUTE_SELECTOR to the compiled Rust driver")
        groups=[[frame(v) for v in xs] for xs in ((1.,11.,2.,-1.),(1.,-1.,3.,-2.))]
        for t in a.LABELS:
            self.assertEqual(r["native_selector"](Path(exe))(groups,t),a.reference(groups,t))
        record=struct.pack("<I32f",1,*([0.]*32))
        for data in (b"",record[:-1],record+b"x",struct.pack("<I",4097),struct.pack("<I",0),
                     record[:-4]+struct.pack("<f",float("nan"))):
            check=subprocess.run([exe,"first"],input=data,capture_output=True)
            self.assertNotEqual(check.returncode,0);self.assertFalse(check.stdout)

if __name__=="__main__":unittest.main()

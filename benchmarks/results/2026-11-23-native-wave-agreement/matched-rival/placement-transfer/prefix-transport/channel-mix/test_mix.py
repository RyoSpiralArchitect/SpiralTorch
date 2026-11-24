import copy
import math
from pathlib import Path
import runpy
import unittest
from types import SimpleNamespace

v=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
c=v.c


class Controls(unittest.TestCase):
    def test_rotation_is_dense_orthogonal_common_mode_preserving_and_invertible(self):
        for i in range(3):
            for j in range(3):
                self.assertEqual(sum(c.MATRIX[i][k]*c.MATRIX[j][k] for k in range(3)),361*(i==j))
            self.assertEqual(sum(c.MATRIX[i]),19)
            self.assertNotIn(0,c.MATRIX[i])
        x=[.12,-.8,1.3]
        f=c.mix3(x,c.MAPS[1]); inv=c.mix3(x,c.MAPS[2])
        self.assertNotEqual(f,inv)
        self.assertAlmostEqual(sum(x),sum(f))
        self.assertAlmostEqual(math.hypot(*f),math.hypot(*x))
        for a,b in zip(c.mix3(f,c.MAPS[2]),x):self.assertAlmostEqual(a,b)

    def test_fresh_templates_are_matched_by_multiset_but_not_by_placement(self):
        new=c.fresh("formal")
        self.assertEqual(len(new),len({tuple(r["prompt"]) for r in new}))
        self.assertEqual(len(new),32)
        self.assertTrue({tuple(r["prompt"]) for r in new}.isdisjoint(tuple(r["prompt"]) for r in c.fresh("rehearsal")))
        for a,b in zip(new[::2],new[1::2]):
            self.assertEqual(sorted(a["prompt"]),sorted(b["prompt"]))
            self.assertNotEqual(a["prompt"],b["prompt"])
            self.assertEqual((a["head"],a["attractor"],a["frame"]),(b["head"],b["attractor"],b["frame"]))

    def test_gate_requires_both_controls_every_order_and_template(self):
        r=dict(h=.1,S=.2,vs_identity_h=.01,vs_inverse_h=.02,vs_identity_S=.01,vs_inverse_S=.01,
               wins=9,wins_before=8,vs_identity_wins=1,vs_inverse_wins=1,positive_pairs=3)
        self.assertEqual(v.qualifies([r]*6,3),dict(margin=True,decision=True))
        self.assertFalse(v.qualifies([r]*5+[dict(r,vs_inverse_S=.001)],3)["margin"])
        self.assertEqual(v.qualifies([r,dict(r,vs_identity_wins=0)],3),dict(margin=True,decision=False))

    def test_origin_adjoint_is_transpose_not_the_forward_matrix(self):
        baseline=dict(v=[.2,-.4,.7,0,0,0],gold=0,mix=1.,loss=.5,ce=.8,binary=.5)
        rec=dict(trace=dict(site_input=[1.,2.,3.],shape=[1,3]),ce=[baseline])
        g=c.mix3(baseline["v"][:3],c.MAPS[2])
        member=dict(baseline,v=g+[x*y for x,y in zip(g,[1,2,3])])
        c.check_origin(member,rec,c.MAPS[1])
        with self.assertRaises(ValueError):
            c.check_origin(member,rec,c.MAPS[2])


if __name__=="__main__":
    unittest.main()

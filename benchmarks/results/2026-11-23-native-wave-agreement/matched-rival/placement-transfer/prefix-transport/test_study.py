from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

v=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))
u=v.u


class PrefixDesign(unittest.TestCase):
    def test_all_arms_match_examples_steps_and_capacity(self):
        sp=u.plan(True,"a"*64);cells=list(u.cells(sp))
        self.assertEqual((len(cells),len(set(cells))),(54,54))
        self.assertEqual(sum(len(u.k.schedule(sp["train"],o,"pair"))*sp["epochs"] for _,_,_,o in cells),1728)
        self.assertEqual(len(set(sp["weights"].values())),3)
        for t in u.TURNS:
            self.assertEqual(u.coordinates(u.json.dumps(u.record("a",t,[0.]*6)),t,{"base":"a"}),[0.]*6)
            bad=u.record("a",t,[0.]*6);bad["placements"][0]["tensors"][0]["requires_grad"]=False
            with self.assertRaises(ValueError):
                u.coordinates(u.json.dumps(bad),t,{"base":"a"})

    def test_thresholds_are_strict_and_decision_needs_an_actual_win(self):
        row=dict(h=.002,S=.002,gap_h=.002,gap_S=.002,positive_pairs=3,wins=9,wins_before=8,identity_wins=8)
        self.assertEqual(v.gate([row]*3),dict(probability=True,decision=True))
        self.assertEqual(v.gate([row,row,dict(row,wins=8)]),dict(probability=True,decision=False))
        self.assertFalse(v.gate([row,row,dict(row,gap_S=.001)])["probability"])
        self.assertFalse(v.gate([dict(row,positive_pairs=2)])["decision"])

    def test_frozen_logits_compare_bytes_not_just_equal_numbers(self):
        self.assertNotEqual(u.bits([0.]*11),u.bits([-0.]*11))
        with self.assertRaises(ValueError): u.bits([float("nan")]*11)


if __name__=="__main__":
    unittest.main()

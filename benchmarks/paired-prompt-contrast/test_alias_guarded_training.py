import unittest
from pathlib import Path

from alias_guarded_training import ARMS, BUDGET, HORIZON, SPLIT, VECTOR, block, boundary, check_replay, geometry_path
from alias_guarded_training_audit import evaluate, streaks, summary


class GuardedTrainingTests(unittest.TestCase):
    def test_missing_harm_and_unresolved_progress_are_not_the_same_gate(self):
        groups = [[0],[1],list(range(2,8))]
        def scores(delta):
            return [dict(cost=1.+delta, single=1.+delta, lines=[1.]*5+[1.+delta]) for _ in range(8)]
        base = scores(0.)
        tiny = evaluate(base,scores(-1e-7),groups,[-1e-7]*9,5e-6,.15)
        self.assertFalse(tiny["harmful"] or tiny["qualifies"] or tiny["strict_finite"])
        after = scores(-.02)
        after[6]["lines"][5] = 1.01
        broken = evaluate(base,after,groups,[-.02]*9,5e-6,.15)
        self.assertTrue(broken["harmful"])
        self.assertFalse(broken["qualifies"])
        self.assertTrue(evaluate(base,scores(-.02),groups,[-.021]*9,5e-6,.15)["qualifies"])

    def test_no_claim_of_autonomy_from_an_earlier_streak(self):
        self.assertEqual(streaks([True]*20+[False,True]),dict(longest=20,trailing=1))
        yes = dict(qualifies=True,harmful=False,strict_finite=True)
        no = dict(qualifies=False,harmful=True,strict_finite=False)
        steps = [dict(step=i+1,repaired=i==31,candidate=no if i==31 else yes,
                      selected=yes,replayed=i==0) for i in range(32)]
        point = dict(nine=[1.]*9,terms=[[1.]*5]*6,absolute_idle=1.,other_idle_line=1.)
        v=summary(steps,{"0":point,"32":point},5e-6)["counts"]
        self.assertEqual((v["rescued"],v["prevented_harm"],v["last_repair"],v["safe_streak"]["longest"]),(1,1,32,31))
        self.assertFalse(v["last16_without_repair_and_qualified"])

    def test_cold_boundary_binds_path_cursor_owner_and_live_score(self):
        marker = dict(plan="p",stage=0,cursor=SPLIT,state="initial",arm=ARMS[1],owner="a",scores="s",
                      shadow=dict(pre="a",before="s"))
        boundary(marker,"p",0,"initial",ARMS[1])
        for edit in (dict(cursor=SPLIT-1),dict(arm=ARMS[0]),dict(owner="b"),dict(scores="x"),dict(plan="q")):
            with self.assertRaises(ValueError):
                boundary(marker|edit,"p",0,"initial",ARMS[1])

    def test_archive_control_mismatch_is_not_a_new_state_but_guard_may_diverge(self):
        p=dict(refs=dict(initial=[dict(pre="a",gradient="g",before="b",candidate="c",proposal="r",
                                      candidate_scores="sc",guarded="d",guard="proof",guarded_scores="sg")]))
        f=dict(pre="a",gradient="g",before="b",candidate="c",proposal="r",chosen="d",guard="proof")
        self.assertTrue(check_replay(p,"initial",ARMS[1],0,f,"sc","sg"))
        with self.assertRaises(ValueError):
            check_replay(p,"initial",ARMS[1],0,f|dict(guard="bad"),"sc","sg")
        self.assertFalse(check_replay(p,"initial",ARMS[1],0,f|dict(pre="b"),"sc","sg"))
        with self.assertRaises(ValueError):
            check_replay(p,"initial",ARMS[0],0,f|dict(pre="b"),"sc","sc")

    def test_gradient_transport_preserves_order_sign_and_work_budget(self):
        pack = block([[float(i)]+[-0.]*63 for i in range(24)])
        self.assertEqual([v[0] for v in VECTOR.iter_unpack(pack)],list(range(24)))
        self.assertEqual(pack[4:8],bytes.fromhex("00000080"))
        self.assertEqual(BUDGET["backwards"],(4*HORIZON+4)*24)
        self.assertEqual(BUDGET["forwards"],6*HORIZON*24)
        with self.assertRaises(ValueError):
            geometry_path(Path("."),"sha256:"+"../x"*16)

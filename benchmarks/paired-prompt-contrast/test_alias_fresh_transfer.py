import math
from pathlib import Path
import tempfile
import unittest

from alias_fresh_transfer import EFFECT, EPS, MENU, claim, dataset, reference
from alias_fresh_transfer_audit import check_scores, decisions, log_mass_nll, paired


class FreshTransferTests(unittest.TestCase):
    def test_crossed_prompt_deck_is_fresh_matched_and_label_blind(self):
        qs,key=dataset()
        by={q["id"]:q["prompt"] for q in qs}
        self.assertEqual(len(by),72)
        self.assertEqual(list(by),sorted(by))
        self.assertTrue(all(set(q)=={"id","prompt"} for q in qs))
        rows={(k["pool"],k["order"],k["block"],k["meaning"]):by[k["id"]] for k in key}
        train={(49,32+i,24+c,27+i%2,29+c) for i in range(6) for c in range(3)}
        self.assertFalse(train & {tuple(q) for q in by.values()})
        for i in range(6):
            for c in range(3):
                a=rows["familiar","original",i,c]; b=rows["novel","original",i,c]
                self.assertEqual(a[:1]+a[2:],b[:1]+b[2:])
                self.assertEqual((a[1],b[1],a[3]),(32+i,53+i,28-i%2))
                self.assertEqual(rows["familiar","swapped",i,c],[a[0],a[1],a[4],a[3],a[2]])
                self.assertEqual(rows["novel","swapped",i,c],[b[0],b[1],b[4],b[3],b[2]])

    def test_shared_lift_is_not_selectivity_and_specificity_is_not_containment(self):
        uniform=paired([[.12]*3]*6)
        stats=decisions(uniform,uniform,EPS,EFFECT)
        self.assertTrue(stats["broad_absolute"] and stats["broad_menu"])
        self.assertFalse(stats["specific"] or stats["contained"])
        leaky=paired([[.08,.08,.20]]*6)
        stats=decisions(leaky,leaky,EPS,EFFECT)
        self.assertTrue(stats["specific"])
        self.assertFalse(stats["contained"])
        selective=paired([[0.,-.01,.08]]*6)
        self.assertTrue(decisions(selective,selective,EPS,EFFECT)["contained"])
        # Positives lose probability too; suppressing everything is not a successful transfer.
        down=paired([[-.2,-.2,-.01]]*6)
        self.assertFalse(decisions(down,down,EPS,EFFECT)["specific"])

    def test_gains_must_agree_in_the_same_blocks_for_both_readouts(self):
        def result(ids):return paired([[0.,0.,.2 if i in ids else -.02] for i in range(6)])
        stats=decisions(result({0,1,2,3}),result({2,3,4,5}),EPS,EFFECT)
        self.assertEqual(stats["same_blocks_better"],2)
        self.assertFalse(stats["specific"])
        with self.assertRaises(ValueError):
            paired([[1.,2.,3.]]*5)

    def test_independent_terms_detect_a_wrong_stop_even_if_the_total_does_not_change(self):
        terms=[[1.,1.+i/8,1.] for i in range(6)]
        line=list(map(math.fsum,terms)); mass=log_mass_nll(line)
        cond=[x-mass for x in line]
        common=dict(stop_id=1,target_counts=[3]*6,prefix_nodes=8,decoder_lanes=6,winner_index=0,line_nlls=line,menu_set_nll=mass)
        new=dict(common,token_nlls=terms,conditional_nlls=cond,shared_content_tokens=1)
        old=dict(common,token_nlls=terms,menu_probabilities=[math.exp(-x) for x in cond])
        self.assertEqual(check_scores(new,old,2e-5),(0.,0.))
        bad=dict(new,token_nlls=[*terms[:5], [terms[5][0],terms[5][1]+.05,terms[5][2]-.05]])
        with self.assertRaises(ValueError):
            check_scores(bad,old,2e-5)
        with self.assertRaises(ValueError):
            check_scores(dict(new,gold_index=1),old,2e-5)

    def test_reference_has_one_dummy_partition_and_thirteen_distinct_prefix_targets(self):
        calls=[]
        class Fake:
            def line_subset_loss(self,p,menu,*,accepted_indices,stop_id):
                calls.append((tuple(accepted_indices),stop_id))
                return dict(accepted_indices=accepted_indices,stop_id=stop_id,target_counts=[3]*6,prefix_nodes=8,
                            decoder_lanes=6,winner_index=0,line_nlls=[3.]*6,menu_probabilities=[1/6]*6,menu_set_nll=3.-math.log(6))
            def next_token_score(self,p,t):
                calls.append((tuple(p),t))
                return dict(target_id=t,vocabulary=64,nll=1.)
        r=reference(Fake(),[49,32,24,28,29])
        self.assertEqual(calls[0],((0,3),1))
        self.assertEqual(len(set(calls[1:])),13)
        self.assertEqual(r["token_nlls"],[[1.]*3 for _ in MENU])

    def test_consumed_deck_cannot_be_rescored_to_a_different_output(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); a=d/"a"; b=d/"b"; a.mkdir(); b.mkdir()
            claim(d,"p","q",a)
            original=(d/"consumed.json").read_bytes()
            with self.assertRaises(FileExistsError):
                claim(d,"p","q",b)
            self.assertEqual((d/"consumed.json").read_bytes(),original)
            self.assertFalse((b/"attempt.json").exists())

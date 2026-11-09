import copy
from collections import Counter
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import entity_swap_study as study
import entity_swap_task as t
from entity_swap_readback import direct_step
from alias_interference_audit import f32
from shared_prefix_study import put


def observed(rows, gap):
    table={}
    for r in rows:
        x=gap(r["prompt"]);z=[-12.]*64;z[t.ON]=0.;z[t.IDLE]=x;table[r["id"]]=z
    return t.metrics(t.key(rows),table)


def symbolic(prompt):
    # The oracle is given a relational lookup rule, not trained by the decoder.
    value=prompt[2] if prompt[1]==prompt[7] else prompt[5]
    return 1. if value==26 else -1.


class EntitySwapTests(unittest.TestCase):
    def setUp(self):
        self.tr,self.te=(t.deck(k) for k in ("train","test"))

    def test_disjoint_unordered_pairs_exact_factorial_and_equal_exposure(self):
        t.design(self.tr,self.te)
        tr={tuple(sorted(x["entities"])) for x in self.tr};te={tuple(sorted(x["entities"])) for x in self.te}
        self.assertEqual((len(tr),len(te),len(tr&te)),(8,8,0))
        for rows in (self.tr,self.te):
            for edge in range(8):
                self.assertEqual({(r["order"],r["idle"],r["query"]) for r in rows if r["edge"]==edge},set(itertools.product(range(2),repeat=3)))
        train_counts=Counter(r["id"] for i in range(t.STEPS) for r in t.batch(self.tr,i))
        self.assertEqual(set(train_counts.values()),{16});self.assertEqual(len(train_counts),64)
        with self.assertRaises(ValueError): t.design(self.tr,self.tr)
        broken=copy.deepcopy(self.te);broken[0]["gold"]=0
        with self.assertRaises(ValueError): t.design(self.tr,broken)

    def test_rotated_negatives_change_only_the_distractor_not_label_marginals_or_slot(self):
        for step in range(4):
            rows=t.batch(self.tr,step);ns=t.negatives(rows)
            for targets in t.ANSWERS:
                ids=[i for i,r in enumerate(rows) if r["gold"]==targets]
                self.assertEqual(sorted(ns["matched"][i] for i in ids),sorted(ns["rotated"][i] for i in ids))
                for i in ids:
                    pos=rows[i]; a=rows[ns["matched"][i]];b=rows[ns["rotated"][i]]
                    self.assertNotEqual(pos["gold"],a["gold"]); self.assertEqual(a["gold"],b["gold"])
                    self.assertEqual(Counter(pos["prompt"]),Counter(a["prompt"]))
                    self.assertEqual([k for k in range(9) if pos["prompt"][k]!=a["prompt"][k]],[2,5])
                    self.assertEqual(a["prompt"][7],b["prompt"][7])
                    self.assertEqual(sum(x!=y for x,y in zip(a["prompt"],b["prompt"],strict=True)),1)
        self.assertTrue(all(set(r)=={"id","prompt"} for r in t.queries(self.te)))

    def test_relational_oracle_solves_all_views_but_shortcuts_do_not(self):
        r=observed(self.te,symbolic)
        self.assertEqual((r["interaction"],r["complete_edges"],r["exact_quartets"],r["correct_choices"]),(1.,8,16,64))
        self.assertEqual((r["assignment_pairs"],r["query_pairs"],r["rank_quartets"]),(32,32,16))
        for f in (lambda p:1.,lambda p:1. if p[2]==26 else -1.,lambda p:1. if p[5]==26 else -1.,
                  lambda p:1. if p[7]%2 else -1.,lambda p:sum(p)):
            v=observed(self.te,f)
            self.assertEqual((v["interaction"],v["correct_choices"],v["complete_edges"],v["rank_quartets"]),(0.,32,0,0))
        v=observed(self.te,lambda p:0.)
        self.assertEqual((v["correct_choices"],v["ties"],v["full_vocab_strict_top1"]),(0,64,0))

    def test_good_ranking_or_train_advantage_is_not_exact_binding(self):
        biased=observed(self.te,lambda p:10.+symbolic(p));ref=observed(self.te,lambda p:10.)
        self.assertEqual((biased["interaction"],biased["rank_quartets"],biased["correct_choices"],biased["complete_edges"]),(1.,16,32,0))
        scores=dict.fromkeys(("origin",*t.ARMS),ref);scores["safe_matched"]=biased
        gates=t.decisions(scores)["safe"];self.assertTrue(gates["advantage"]);self.assertFalse(gates["competence"])
        scores["safe_rotated"]=None;self.assertFalse(t.decisions(scores)["safe"]["advantage"])
        scores["safe_matched"]=None;self.assertFalse(t.decisions(scores)["safe"]["competence"])
        a=ref|dict(by_edge=[1.]*5+[0.]*3,by_order=[.1,.1])
        self.assertFalse(t.advantage(a,ref)["passed"])
        a=ref|dict(by_edge=[1.]*8,by_order=[1.,-1.]);self.assertFalse(t.advantage(a,ref)["passed"])

    def test_key_join_requires_all_cells_and_finite_logits(self):
        labels=t.key(self.te); logits={x["id"]:[0.]*64 for x in labels}
        for new in (labels[:-1],labels[:-1]+[labels[0]]):
            with self.assertRaises(ValueError):t.metrics(new,logits)
        logits[labels[0]["id"]][3]=float("nan")
        with self.assertRaises(ValueError):t.metrics(labels,logits)

    def test_direct_receipt_binds_actual_rounded_endpoint_and_budget(self):
        a=[0.]*64;z=[-f32(.002)]+[0.]*63;gs=[dict(v=[1.]+[0.]*63,loss=.5)]*16
        p=dict(radius=.002,rate=.125,base="b",lora="l")
        r=dict(pre_geometry_id="a",post_geometry_id="z",base_model_identity="b",lora_artifact_id="l",
               readout_accumulator="f32_wide_backward",members=16,total_target_tokens=16,trainable_values=64,learning_rate=.125,
               mean_gradient_l2=1.,mean_input_loss=.5,applied_l2=-z[0],member_loss_dots=[z[0]]*16,
               distance_match=dict(origin_id="a",result_id="z",requested_l2=.002,direction_l2=.125,realized_l2=-z[0],relative_vector_error=1e-7))
        direct_step(r,p,"a","z",a,z,gs,16)
        with self.assertRaises(ValueError):direct_step(r,p,"a","z",a,[z[0]/2]+z[1:],gs,16)
        self.assertEqual(study.LIMITS["backwards"],896*88)
        self.assertEqual(study.LIMITS["score_rows"],16*128)

    def test_claim_binds_one_output_and_worker_cannot_open_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/"result";pp="P";v=study.claim(root,"score",pp,out,["T","A"])
            put(root/"score-claim.json",v);study.check_claim(root,"score",pp,out,["T","A"])
            with self.assertRaises(FileExistsError):put(root/"score-claim.json",v)
            with self.assertRaises(ValueError):study.check_claim(root,"score",pp,root/"again",["T","A"])
            for name in ("blind","keys","fit"):
                (root/name).mkdir();put(root/name/"probe.json",{})
            code=("import sys, pathlib;sys.path.insert(0,sys.argv[1]);from entity_swap_study import fence;"
                  "p=pathlib.Path(sys.argv[2]);s=sys.argv[3];fence(p,s);"
                  "(p/('fit' if s=='train' else 'blind')/'probe.json').read_bytes();"
                  "\nfor n in ('keys','blind' if s=='train' else 'fit'):\n"
                  " try:(p/n/'probe.json').read_bytes()\n except PermissionError:pass\n else:raise SystemExit(3)")
            for s in ("train","score"):
                subprocess.run([sys.executable,"-I","-B","-S","-c",code,str(Path(__file__).resolve().parent),str(root),s],check=True)

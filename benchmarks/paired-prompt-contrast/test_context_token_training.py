import copy
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from alias_guarded_training import VECTOR
from alias_interference_audit import f32
import context_token_training as study
from context_token_training import ARMS, BUDGET, GROUPS, REFUSALS, STATES, STEPS, TARGET, check_seed, nll, pairs
from context_token_training_audit import audit_vectors, contrast, decisions, proposal_groups, validate_rows
from shared_prefix_study import put, sha


def roster():
    return [dict(meaning=c,prompt=[49,32+b,24+c,27+b%2,29+c]) for c in (1,2,0) for b in reversed(range(6))]


def score_rows(data, gain=lambda i,c: 0.):
    rows = []
    for i,e in enumerate(data):
        c = e["meaning"]
        terms = [2.,3.-gain(i,c),1.]
        lines = [5.]*5+[sum(terms)]
        conditional = [v-nll(lines) for v in lines]
        row = dict(slot=i,terms=terms,lines=lines,conditional=conditional,cost=nll([conditional[c],conditional[c+3]]))
        if c == 2:
            row["single"] = lines[5]
        rows.append(row)
    return rows


class ContextPreferenceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.data = roster()
        self.triples, self.pairs = pairs(self.data)
        self.groups = [[i for i,v in enumerate(self.data) if v["meaning"] == c] for c in range(3)]
        self.rows = score_rows(self.data)

    def test_rotation_preserves_both_marginals_and_parity_but_no_matched_edge(self):
        m,r = (self.pairs[k] for k in ("matched_pair","rotated_pair"))
        self.assertEqual(len(m),12)
        self.assertTrue(set(map(tuple,m)).isdisjoint(map(tuple,r)))
        for side in (0,1):
            self.assertEqual(Counter(p[side] for p in m),Counter(p[side] for p in r))
        for (i,j),(x,y) in zip(m,r,strict=True):
            a,b,z = (self.data[k] for k in (i,j,y))
            self.assertEqual((i,a["meaning"],b["meaning"]),(x,2,z["meaning"]))
            self.assertEqual(a["prompt"][1],b["prompt"][1])
            self.assertEqual((z["prompt"][1]-b["prompt"][1])%6,2)
            self.assertEqual(a["prompt"][3],z["prompt"][3])
        for bad in ([*self.data[:-1],self.data[0]], self.data[:-1], [dict(meaning=1,prompt=[]),*self.data[1:]]):
            with self.assertRaises(ValueError):
                pairs(bad)

    def test_a_uniform_boost_is_not_selectivity_and_a_relative_gain_is_not_containment(self):
        def effects(gains):
            rows = score_rows(self.data,lambda _,c:gains[c])
            return contrast(self.rows,rows,self.triples)["word"]
        uniform = effects([.2,.2,.2])
        self.assertFalse(uniform["selectivity"])
        self.assertAlmostEqual(uniform["gap"],0.)
        rising = effects([.1,.1,.2])
        self.assertTrue(rising["selectivity"])
        self.assertFalse(rising["contained"])
        self.assertTrue(effects([-.1,0.,.1])["contained"])
        self.assertFalse(effects([-.3,-.3,-.1])["contained"])

    def test_no_pair_specific_benefit_without_rotated_control_and_no_partial_endpoint_pass(self):
        changed = score_rows(self.data,lambda _,c:.3 if c == 2 else -.05)
        paths = {k:dict(complete=True,rows=changed if k in ("matched_pair","rotated_pair") else self.rows) for k in ARMS}
        result = decisions(self.rows,paths,self.triples)
        self.assertTrue(result["primary"] and result["stopped_line"] and result["contained"])
        self.assertFalse(result["entity_binding"])
        paths["positive_ce"]["complete"] = False
        self.assertFalse(decisions(self.rows,paths,self.triples)["primary"])
        paths["matched_pair"]["complete"] = False
        gate = decisions(self.rows,paths,self.triples)
        self.assertFalse(any(gate[k] for k in ("primary","entity_binding","stopped_line","contained")))
        self.assertTrue(all(v is None for v in gate["contrasts"].values()))

    def test_effect_must_be_distributed_over_entities_not_carried_by_one(self):
        selected = set(self.triples[0])
        rows = score_rows(self.data,lambda i,c: .7 if c == 2 and i in selected else 0.)
        d = contrast(self.rows,rows,self.triples)["word"]
        self.assertGreater(d["gap"],.01)
        self.assertFalse(d["selectivity"])

    def test_signed_vjp_and_complete_report_bind_the_pair_order(self):
        p = dict(groups=self.groups,pairs=self.pairs)
        vectors = [[0.]*64 for _ in range(24)]
        word = [[float(j == i) for j in range(64)] for i in range(18)]
        vectors.extend(word)
        reports = {name:[] for name in self.pairs}
        loss = [f32(r["cost"]) for r in self.rows]+[6.]*6+[3.]*18
        for name, edges in self.pairs.items():
            for i,j in edges:
                reports[name].append(dict(target_id=TARGET,vocabulary=64,preferred_nll=3.,other_nll=3.,margin=0.,
                                          gradient_weight=.5,loss=math.log(2.),log_probability_gap=0.))
                vectors.append([.5*(a-b) for a,b in zip(word[i],word[j],strict=True)])
                loss.append(f32(math.log(2.)))
        raw = b"".join(VECTOR.pack(*v) for v in vectors)
        event = dict(sha=sha(raw),old_gradient=sha(raw[:24*VECTOR.size]),loss=loss,reports=reports)
        gs,err = audit_vectors(raw,event,self.rows,p)
        self.assertEqual(err,0.)
        self.assertEqual(len(gs),GROUPS)
        with self.assertRaises(ValueError):
            audit_vectors(raw,event | dict(old_gradient="foreign"),self.rows,p)
        with self.assertRaises(ValueError):
            audit_vectors(raw,event,self.rows,p | dict(pairs=dict.fromkeys(self.pairs,self.pairs["rotated_pair"])))
        bad = copy.deepcopy(event)
        bad["reports"]["matched_pair"][2]["log_probability_gap"] = 1.
        with self.assertRaises(ValueError):
            audit_vectors(raw,bad,self.rows,p)
        with self.assertRaises(ValueError):
            check_seed([-.5*x for x in word[0]],word[0],word[1],.5)

    def test_all_arms_share_first_four_proposal_groups_and_all_nine_guard_groups(self):
        fake = list(range(GROUPS))
        p = dict(groups=self.groups,pairs=self.pairs)
        base,guard = proposal_groups(fake,p,"guard_only")
        for arm in ARMS:
            gs,g = proposal_groups(fake,p,arm)
            self.assertEqual((gs[:4],g),(base,guard))
            if arm == "positive_ce":
                self.assertEqual(Counter(gs[4]),Counter({24+i:2 for i in self.groups[2]}))
            elif arm != "guard_only":
                self.assertEqual(gs[4],fake[42:54] if arm == "matched_pair" else fake[54:66])
        self.assertEqual(BUDGET["max_backwards"],len(STATES)*len(ARMS)*STEPS*GROUPS)
        self.assertTrue(REFUSALS["guard"].isdisjoint(REFUSALS["proposal"]))

    def test_word_stop_and_menu_are_independent_checked_observations(self):
        validate_rows(self.rows,self.groups)
        for field, idx in (("lines",5),("terms",2),("conditional",4)):
            damaged = copy.deepcopy(self.rows)
            damaged[0][field][idx] += .1
            with self.assertRaises(ValueError):
                validate_rows(damaged,self.groups)
        damaged = copy.deepcopy(self.rows)
        del damaged[self.groups[2][0]]["single"]
        with self.assertRaises(ValueError):
            validate_rows(damaged,self.groups)


class WheelAdmissionTests(unittest.TestCase):
    def archive(self, root, stub=None):
        wheel = root/"spiraltorch-0.4.33-cp38-abi3-macosx_11_0_arm64.whl"
        metadata = "spiraltorch-0.4.33.dist-info/"
        source = study.REPO/"bindings/st-py/spiraltorch"
        with ZipFile(wheel,"w") as z:
            z.writestr("spiraltorch/__init__.py",(source/"__init__.py").read_bytes())
            z.writestr("spiraltorch/__init__.pyi",(source/"__init__.pyi").read_bytes() if stub is None else stub)
            z.writestr("spiraltorch/spiraltorch.abi3.so",b"not loaded in this test")
            z.writestr(metadata+"METADATA","Name: spiraltorch\nVersion: 0.4.33\n")
            z.writestr(metadata+"WHEEL","Tag: cp38-abi3-macosx_11_0_arm64\n")
            z.writestr(metadata+"RECORD","")
        return wheel,sha(wheel.read_bytes()),sha(b"not loaded in this test")

    def test_archive_digest_and_extension_are_not_independent_attestations(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            wheel,digest,ext = self.archive(root)
            inventory = study.candidate_inventory(wheel,digest,ext)
            self.assertEqual(inventory["native"]["sha256"],ext)
            for w,e in (("f"*64,ext),(digest,"f"*64)):
                with self.assertRaisesRegex(ValueError,"pinned pair"):
                    study.freeze(root/"unused",root/"unused","unused",e,w,wheel,root/"output")
                self.assertFalse((root/"output").exists())
            wheel,stale,ext = self.archive(root,stub=b"old facade")
            with self.assertRaisesRegex(ValueError,"type stub"):
                study.candidate_inventory(wheel,stale,ext)

    def test_failed_install_check_never_launches_a_worker_or_writes_an_attempt(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            wheel,digest,ext = self.archive(root)
            inv = study.candidate_inventory(wheel,digest,ext)
            inv_pin = put(root/"wheel-inventory.json",inv)
            p = dict(kind=study.KIND,code=study.sources(),max_budget=study.BUDGET,wall=study.WALL,retries=0,
                     wheel=digest,native=ext,inventory=inv_pin)
            pp = put(root/"plan.json",p)
            with patch.object(study,"installed_identity",side_effect=ValueError("foreign installation")), \
                 patch.object(study.subprocess,"run",side_effect=AssertionError("worker launched")):
                with self.assertRaisesRegex(ValueError,"foreign installation"):
                    study.once(root,pp,root/"physical")
            self.assertFalse((root/"physical").exists())
            with self.assertRaisesRegex(ValueError,"inventory differs"):
                study.frozen_inventory(root,p | dict(native="f"*64))

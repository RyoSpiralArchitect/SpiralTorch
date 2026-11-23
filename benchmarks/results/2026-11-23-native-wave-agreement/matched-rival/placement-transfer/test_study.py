"""Study design, decompositions, negative controls and qualification boundaries."""
from collections import Counter
import math
from pathlib import Path
import runpy
import types
import unittest

a = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("analyze.py"))))
d = a.d


def synthetic(margin, coverage, stop_margin=0.):
    choice = [math.log1p(math.exp(-margin)), math.log1p(math.exp(margin))]
    totals = [x+coverage for x in choice]
    stops = [1.-stop_margin/2, 1.+stop_margin/2]
    return dict(stop_id=0, target_counts=[2,2], line_nlls=totals, conditional_nlls=choice, menu_set_nll=coverage,
                token_nlls=[[x-z,z] for x,z in zip(totals,stops)], winner_index=int(margin<0))


class Study(unittest.TestCase):
    def test_frames_split_and_balance(self):
        spec = d.design(True)
        seen = set()
        for s,n in (("train",32),("test",16)):
            rows = spec[s]
            self.assertEqual(len(rows),n)
            self.assertEqual(len({r["key"] for r in rows}),n)
            self.assertEqual(sum(r["head"] != r["attractor"] for r in rows), n//2)
            for role,ix in (("head",0),("attractor",3)):
                self.assertEqual(Counter((r["prompt"][ix],r[role]) for r in rows),
                                 {(noun,bit):n//8 for noun in range(3,7) for bit in (0,1)})
            frames = Counter(r["frame"] for r in rows)
            self.assertEqual(set(frames.values()),{4})
            self.assertTrue(seen.isdisjoint(frames))
            seen.update(frames)
        for r in d.design(False)["train"]+d.design(False)["test"]:
            self.assertNotIn(r["frame"],seen)

    def test_work_order_and_label_only_intervention(self):
        spec = d.design(True)
        self.assertEqual(len(d.cells(spec))*len(spec["train"])*spec["epochs"],4992)
        self.assertEqual((len(d.cells(spec))+len(spec["arms"]))*48,4368)
        perms = set()
        for seed in spec["orders"]:
            import json
            first,second = (d.session_plan(spec, "wave@"+d.SITES[0], r, seed) for r in d.TEACHERS)
            self.assertEqual(first[1], second[1])
            perms.add(tuple(r["key"] for r in first[1]))
            h,att = json.loads(first[0]), json.loads(second[0])
            diff = sum(x["objective"].pop("gold_index") != y["objective"].pop("gold_index")
                       for x,y in zip(h["examples"],att["examples"]))
            self.assertEqual(h,att)
            self.assertEqual(diff,16)
        self.assertEqual(len(perms),3)

    def test_coverage_and_stop_bias_are_not_role_learning(self):
        rows = d.design(True)["test"]
        cases = []
        for coverage,margin,stop in ((3,0.,0.),(2,.4,.4)):
            costs = {r["key"]:synthetic(margin,coverage,stop) for r in rows}
            for c in costs.values():
                a.verify_costs(c)
            cases.append(a.reports(rows,costs))
        out = a.contrast(cases[0],cases[1],cases[1])
        self.assertAlmostEqual(out["coverage_gain"],.5)
        self.assertAlmostEqual(out["gold_gain"],.5-math.log(math.cosh(.2))/2)
        self.assertLess(out["choice_gain"],0.)
        for k in ("B","S","delta_H","delta_H_stop","positive_pairs"):
            self.assertAlmostEqual(out[k],0.)
        self.assertFalse(a.passes([out]))

    def test_orthogonal_role_contrasts_have_known_sign_and_size(self):
        rows = d.design(True)["test"]
        def run(role):
            costs = {}
            for r in rows:
                m = 0 if not role else (1-2*r[role])*.4
                costs[r["key"]] = synthetic(m,3.,m*.25)
            return a.reports(rows,costs)
        result = a.contrast(run(None),run("head"),run("attractor"))
        for k in ("delta_H","B","S"):
            self.assertAlmostEqual(result[k],.4)
        self.assertAlmostEqual(result["delta_H_content"],.3)
        self.assertAlmostEqual(result["delta_H_stop"],.1)
        self.assertAlmostEqual(result["control_delta_H"],0)
        self.assertEqual(result["wins_before"],0)
        self.assertEqual(result["ties_before"],16)
        self.assertEqual(result["wins_after"],16)
        self.assertEqual(result["positive_pairs"],4)
        self.assertTrue(a.passes([result]*3))

    def test_all_orders_strictly_over_floor_and_paired(self):
        good = dict.fromkeys(a.GATES,.1)
        good["positive_pairs"] = 3
        self.assertTrue(a.passes([good]*3))
        for k in a.GATES:
            self.assertFalse(a.passes([good,dict(good,**{k:d.FLOOR}),good]))
        self.assertFalse(a.passes([dict(good,positive_pairs=2)]))
        self.assertFalse(a.wave_advantage([good],[good]))
        self.assertTrue(a.wave_advantage([good],[dict(good,delta_H=.01,S=.01)]))
        with self.assertRaises(ValueError):
            a.wave_advantage([good],[])

    def test_full_vocab_recheck_uses_candidate_specific_stop(self):
        logits = [[0.]*11,[1.]+[0.]*10,[-1.]+[0.]*10]
        target = [math.log(math.exp(z)+10)-z for z in (1.,-1.)]
        c = {"token_nlls":[[math.log(11),v] for v in target]}
        self.assertLess(a.verify_sentinel(logits,c),1e-12)
        with self.assertRaises(ValueError):
            a.verify_sentinel([logits[0],logits[2],logits[1]],c)
        with self.assertRaises(ValueError):
            a.verify_costs(dict(synthetic(.4,3.),winner_index=1))


if __name__ == "__main__":
    unittest.main()

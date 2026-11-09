from collections import Counter
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import answer_quartet_learning as a
import answer_loss_breakdown as breakdown


class AnswerQuartetLearning(unittest.TestCase):
    def test_answer_set_and_conditional_loss_are_separable(self):
        z = [-20.0, -18.0, 1.0, 4.0]
        support, choice = breakdown.loss_parts(z, 0, (0, 1))
        self.assertAlmostEqual(support + choice, a.nll(z, 0)[0])
        sup2, choice2 = breakdown.loss_parts([-15.0, -13.0, 1.0, 4.0], 0, (0, 1))
        self.assertLess(sup2, support)
        self.assertAlmostEqual(choice, choice2)
        with self.assertRaises(ValueError):
            breakdown.loss_parts(z, 3, (0, 1))

    def test_endpoint_readback_uses_the_recorded_control_for_each_mixture(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / "protocol"
            trained = base / "trained"
            scored = base / "scored"
            out = base / "readback"
            for folder in (root, trained, scored, out, base / "calibration"):
                folder.mkdir()
            p = dict(
                base="base",
                lora="lora",
                origins={state: state for state in a.old.STATES},
                folds={k: a.put(root / (k + ".json"), v) for k, v in a.decks().items()},
            )
            pin = a.put(root / "plan.json", p)
            cp = a.put(base / "calibration/complete.json", dict(plan=pin))
            records = {}
            ends = {}
            resume = None
            for state, order, policy in a.lanes():
                tag = ".".join((state, order, policy))
                owner = a.cx.owner_id(p, state, *a.SITE, [0] * 32)
                step = dict(owner=owner, right_bits=[0] * 32)
                ends[tag] = a.put(trained / (tag + ".json"), dict(steps=[step] * 64, owner=owner))
                if tag == "initial.ab.fit_scale":
                    resume = a.put(
                        trained / "resume.json", dict(initial=owner, steps=[step] * 32, owner=owner)
                    )
            tp = a.put(
                trained / "complete.json",
                dict(plan=pin, calibration=cp, ends=ends, updates=1536, resume=resume),
            )
            for tag in set(ends) | {s + ".hold" for s in a.old.STATES}:
                records[tag] = {
                    k: a.write(scored / (tag + "." + k), bytes(192 * 256)) for k in ("full", "cache")
                }
                records[tag]["max_error"] = 0.0
            sp = a.put(scored / "complete.json", dict(plan=pin, train=tp, rows=4992, records=records))
            result = a.audit(root, p, pin, out, cp, tp, sp)
            gates = a.obj(out / "summary.json", result, 2 << 20)["gates"]
            for state, order, arm, split in itertools.product(
                a.old.STATES, ("ab", "ba"), ("half", "fit_scale"), ("held", "transfer")
            ):
                control = "half_control" if arm == "half" else "fit_control"
                gate = gates[".".join((state, order, arm, split))]
                self.assertEqual(set(gate["vs"]), {".".join((state, order, k)) for k in ("answer", control)})
                self.assertFalse(gate["behavior_gate"] or gate["shared_gate"])

    def test_cross_pools_and_fold_boundaries(self):
        old = {tuple(sorted((q[1], q[4]))) for qs in a.cx.decks().values() for q in qs}
        fits = set()
        seen = set()
        for fold, qs in a.decks().items():
            edges = {tuple(sorted((q[1], q[4]))) for q in qs}
            self.assertEqual(len(edges), 4)
            self.assertFalse(edges & (seen | old))
            seen |= edges
            ids = {v for e in edges for v in e}
            if fold.startswith("train"):
                fits |= ids
            elif fold.startswith("held"):
                self.assertEqual(fits, ids)
            else:
                self.assertFalse(fits & ids)
            self.assertEqual(Counter(map(a.gold, qs)), Counter({2: 16, 16: 16}))
        self.assertEqual(len(seen), 24)

    def test_reflection_changes_only_binding_not_gold(self):
        qs = a.decks()["train0"]
        factual, control = a.batch(qs, False), a.batch(qs, True)
        self.assertEqual(factual["prompts"], control["prompts"])
        self.assertEqual(factual["gold"], control["gold"])
        self.assertNotEqual(factual["groups"], control["groups"])
        for i, (group, other) in enumerate(zip(factual["groups"], control["groups"], strict=True)):
            self.assertEqual([factual["gold"][k] for k in group], [16, 2, 2, 16])
            self.assertEqual(
                [control["gold"][k] for k in other], ([2, 16, 16, 2] if i % 2 else [16, 2, 2, 16])
            )

    def test_nll_and_reference_seed_are_full_vocabulary(self):
        v = [2.0, -1.0, 3.0, 4.0]
        loss, p = a.nll(v, 2)
        self.assertAlmostEqual(loss, -math.log(p[2]), places=14)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertNotAlmostEqual(a.nll(v[:3], 2)[0], loss)
        g = [0.1, -0.3, 0.6, -0.7]
        deriv = sum(p[k] * (x - g[2]) for k, x in enumerate(g))
        hi, lo = [a.nll([x + t * d for x, d in zip(v, g, strict=True)], 2)[0] for t in (1e-5, -1e-5)]
        self.assertAlmostEqual(deriv, (hi - lo) / 2e-5, places=9)
        self.assertEqual(a.nll([x + 32 for x in v], 2), (loss, p))

    def test_fixed_weights_and_order_budget(self):
        self.assertEqual(len(set(a.lanes())), 24)
        ce, inter = 3.0, 0.04
        mix = ce / (ce + inter)
        self.assertAlmostEqual((1 - mix) * ce, mix * inter)
        for p in ("fit_scale", "fit_control"):
            self.assertEqual(a.mixture(p, mix), mix)
        self.assertEqual(a.mixture("half", mix), a.mixture("half_control", mix))
        self.assertEqual(a.mixture("answer", mix), 0.0)

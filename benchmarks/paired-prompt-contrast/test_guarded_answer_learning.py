from collections import Counter
import itertools
import unittest

import guarded_answer_learning as g
import fit_logit_probe as probe


class GuardedAnswers(unittest.TestCase):
    def test_probe_does_not_confuse_interaction_with_global_or_within_row_bias(self):
        ys = [1, -1, -1, 1] * 2
        rows = [[float(t), -float(t), 0.0] for t in ys]
        shifted = [[x + i * 4 for x in row] for i, row in enumerate(rows)]
        for feature in (rows, shifted):
            m = probe.fit(feature, ys, 1e-4)
            self.assertEqual(
                probe.measure(probe.predictions(m, feature), ys)["robust"], dict(correct=8, ranked=2)
            )
            self.assertAlmostEqual(sum(m["weights"]), 0.0, places=12)
        no_signal = [[float(i % 2), -float(i % 2), 0.0] for i in range(8)]
        m = probe.fit(no_signal, ys, 1e-4)
        self.assertEqual(probe.measure(probe.predictions(m, no_signal), ys)["nominal"]["ranked"], 0)

    def test_remaining_graph_is_closed_without_reusing_a_pair(self):
        used = {
            tuple(sorted((q[1], q[4])))
            for ds in (g.a.decks(), g.cx.decks())
            for qs in ds.values()
            for q in qs
        }
        fit_ids = set()
        ds = g.decks()
        for f, rows in ds.items():
            pairs = {tuple(sorted((r[1], r[4]))) for r in rows}
            self.assertEqual(len(pairs), 4)
            self.assertFalse(pairs & used)
            used |= pairs
            ids = {x for p in pairs for x in p}
            if f.startswith("train"):
                fit_ids |= ids
            elif f.startswith("held"):
                self.assertTrue(ids <= fit_ids)
            else:
                self.assertFalse(ids & fit_ids)
            self.assertEqual(Counter(g.a.gold(q) for q in rows), Counter({16: 16, 2: 16}))
        self.assertEqual(len(used), 72)
        self.assertEqual(len(fit_ids), 12)

    def test_forward_constraints_are_conjunctive(self):
        def point(a, i, p):
            return dict(answer=a, interaction=i, protected_answer=p)

        base = point(2.0, 1.0, 0.5)
        anchor = point(1.5, 0.7, 0.49)
        self.assertTrue(g.accepted(base, point(1.625, 0.69, 0.5), 0.375, anchor, True))
        self.assertFalse(g.accepted(base, point(1.626, 0.5, 0.5), 0.375, anchor, True))
        self.assertFalse(g.accepted(base, point(1.6, 0.8, 0.5), 0.375, anchor, True))
        self.assertFalse(g.accepted(base, point(1.6, 0.5, 0.501), 0.375, anchor, True))
        self.assertTrue(g.accepted(base, point(1.6, 1.3, 0.5), 0.375, base, False))

    def test_gate_needs_each_reference_and_separates_transfer(self):
        scores = {}
        for state, order, arm in g.lanes():
            lane = ".".join((state, order, arm))
            scores[lane] = {}
            for fold in g.cx.FOLDS:
                improve = arm == "correct" and not fold.startswith("transfer")
                scores[lane][fold] = dict(
                    correct=16 + improve,
                    ranked=int(improve),
                    full_correct=2,
                    answer_nll=1.0,
                    edges=[[0.1 if improve else 0.0] * 4 for _ in range(4)],
                )
        gates = g.endpoint_gates(scores)
        for state, order in itertools.product(g.old.STATES, ("ab", "ba")):
            stem = ".".join((state, order, "correct"))
            for subset, want in (("held", True), ("transfer", False)):
                r = gates[stem + "." + subset]
                self.assertEqual(set(r["vs"]), {"answer", "anchor", "reflected"})
                self.assertEqual((r["shared"], r["behavior"]), (want, want))
        scores["initial.ab.reflected"]["held0"]["correct"] = 32
        self.assertFalse(g.endpoint_gates(scores)["initial.ab.correct.held"]["behavior"])

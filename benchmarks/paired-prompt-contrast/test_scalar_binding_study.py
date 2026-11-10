import math
import unittest
from collections import Counter

from scalar_binding_study import STEPS, factors, fresh, pairs, task, validate_splits
from scalar_binding_readback import decision, pair_means


class ScalarBindingTests(unittest.TestCase):
    def test_unused_combinations_have_the_same_entities_and_all_eight_views(self):
        train, test = task.deck("train"), fresh()
        validate_splits(train, test)
        self.assertEqual({x for r in train for x in r["entities"]}, {x for r in test for x in r["entities"]})
        self.assertEqual(Counter(r["edge"] for r in test), Counter({i: 8 for i in range(8)}))
        self.assertEqual(len({tuple(k) for r in test for k in [sorted(r["entities"])]}), 8)
        with self.assertRaises(ValueError):
            validate_splits(train, task.deck("test"))

    def test_fixed_sgd_schedule_is_balanced_by_pair_and_supervision(self):
        rows = task.deck("train")
        counts = Counter()
        for t in range(STEPS):
            group = pairs(rows, t)
            self.assertEqual(len(group), 4)
            for pos, neg in group:
                self.assertEqual((pos["gold"], neg["gold"]), (task.IDLE, task.ON))
                self.assertEqual(pos["entities"], neg["entities"])
                self.assertEqual(pos["query"], neg["query"])
                counts.update((pos["id"], neg["id"]))
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual(len(counts), 64)

    def test_regression_uses_a_difference_of_gaps_not_absolute_token_bias(self):
        for a, b in ((1.0, 4.0), (1000.0, 0.0), (0.0, 1000.0)):
            data = [[a, b, 0.4, -0.3], [-b, -a, 0.1, 0.6]]
            loss, g = pair_means(data)

            def displaced(s):
                return pair_means([[x + s * dx, y + s * dy, dx, dy] for x, y, dx, dy in data])[0]

            self.assertTrue(math.isfinite(loss))
            self.assertAlmostEqual(g, (displaced(0.0001) - displaced(-0.0001)) / 0.0002, places=7)
            self.assertEqual(pair_means([[x + 9, y + 9, dx, dy] for x, y, dx, dy in data]), (loss, g))

    def test_direction_gate_and_decision_gate_are_distinct(self):
        ref = dict(by_edge=[0.0] * 8, by_order=[0.0, 0.0], correct_choices=32, complete_edges=0)
        better = ref | dict(by_edge=[0.009] * 8, by_order=[0.009] * 2)
        out = decision(dict(hold=ref, orthogonal=ref, aligned=better))
        self.assertTrue(out["resolved"])
        self.assertFalse(out["practical"] or out["more_correct"] or out["competence"])
        self.assertEqual(
            factors({"s": [[0.6, 0.8], [0.8, -0.6]]}, "s", "orthogonal"), ([0.6, 0.8], [0.6, 0.8])
        )

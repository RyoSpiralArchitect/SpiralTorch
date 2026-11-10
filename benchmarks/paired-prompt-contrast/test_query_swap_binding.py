import math
import unittest
from collections import Counter

from query_swap_binding import design, fit_request, task
from query_swap_readback import cells, factorial, group_means


class QuerySwapTests(unittest.TestCase):
    def setUp(self):
        self.rows = [r for r in task.deck("train") if r["edge"] % 2 == 0]
        self.d = design(self.rows)

    def test_every_query_control_preserves_both_facts(self):
        self.assertEqual(Counter(self.d["groups"]), Counter({k: 2 for k in range(8)}))
        for (p, a), (q, b) in zip(self.d["A"], self.d["Q"], strict=True):
            self.assertEqual(p, q)
            raw = self.rows[p]["prompt"]
            self.assertEqual(
                [i for i, (x, y) in enumerate(zip(raw, self.rows[b]["prompt"], strict=True)) if x != y], [7]
            )
            self.assertEqual(raw[7], self.rows[a]["prompt"][7])
        corrupted = [dict(r) for r in self.rows]
        corrupted[0]["prompt"] = corrupted[0]["prompt"][:]
        corrupted[0]["prompt"][1] = 0
        with self.assertRaises(ValueError):
            design(corrupted)

    def test_balanced_means_identical_but_identity_means_need_not_be(self):
        gs = [[float(i), (3 * i % 7) / 4.0] for i in range(32)]
        groups = [group_means(gs, h, self.d["groups"]) for h in (self.d["A"], self.d["Q"])]
        self.assertNotEqual(*groups)
        self.assertEqual(*(list(map(math.fsum, zip(*v, strict=True))) for v in groups))
        pairs, labels, rule = fit_request(self.d, "joint_id")
        self.assertEqual((len(pairs), rule), (32, "common_cosine"))
        self.assertEqual(Counter(labels), Counter({k: 4 for k in range(8)}))
        self.assertEqual(len({i for pair in pairs for i in pair}), 32)

    def test_four_margins_reject_query_or_assignment_only_shortcuts(self):
        for a, b, c in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, -0.2, 0.3)):
            gaps = [
                2.0 + a * (-1) ** r["idle"] + b * (-1) ** r["query"] + c * (-1) ** (r["query"] + r["idle"])
                for r in self.rows
            ]
            for v in cells(self.d, gaps):
                d = factorial(v)
                self.assertAlmostEqual(d["shared"], 2 * c)
                self.assertAlmostEqual(d["assignment_only"], 2 * a)
                self.assertAlmostEqual(d["query_only"], 2 * b)
                self.assertEqual(d["worst"] > 0.0, c > max(abs(a), abs(b)))
        with self.assertRaises(ValueError):
            factorial([1.0, 0.0, 0.0, 0.0])

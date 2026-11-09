import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from completion_evidence import checked_native_nll, contrast, rebuild, retained_heads
from field_completions import CompletionMenu


class Readback(unittest.TestCase):
    def setUp(self):
        self.menu = CompletionMenu(6, 0, ((1,), (1, 2), (3, 2)))
        self.heads = {1: 1.1, 3: 1.0}
        self.obs = [((1,), 0, 2.), ((1,), 2, .2), ((1, 2), 0, .5),
                    ((3,), 2, .6), ((3, 2), 0, .4)]

    def test_shared_heads_and_distinct_stop(self):
        rows = rebuild(self.menu, self.heads, reversed(self.obs))
        self.assertEqual([x.content for x in rows], [(1, 2), (3, 2), (1,)])
        self.assertEqual(rows[0].parts, (1.1, .2, .5))
        self.assertAlmostEqual(rows[0].cost("content"), 1.3)
        self.assertAlmostEqual(rows[0].cost("mean_line"), .6)
        self.assertEqual(rows[0].cost("first"), 1.1)
        self.assertAlmostEqual(rows[0].cost("line"), 1.8)
        with self.assertRaises(ValueError):
            rows[0].cost("best")

    def test_contrast_keeps_the_same_wrong_line_for_every_component(self):
        rows = rebuild(self.menu, self.heads, self.obs)
        c = contrast(rows, (1,))
        self.assertEqual(c["rival"], (1, 2))
        self.assertEqual(c["head"], 0)
        self.assertAlmostEqual(c["body"], -.2)
        self.assertAlmostEqual(c["stop"], 1.5)
        self.assertAlmostEqual(c["total"], 1.3)
        self.assertAlmostEqual(contrast(rows, (1, 2))["total"], -.2)
        self.assertIsNone(contrast(rows, (4,)))
        self.assertIsNone(contrast(rows[:1], rows[0].content))

    def test_tie_is_not_a_win_for_both(self):
        menu = CompletionMenu(6, 0, ((1,), (3,)))
        rows = rebuild(menu, {1: .25, 3: .75}, [((3,), 0, .25), ((1,), 0, .75)])
        self.assertEqual(rows[0].content, (1,))
        self.assertEqual(contrast(rows, (3,))["total"], 0)

    def test_missing_duplicate_wrong_prefix_and_nonfinite_rejected(self):
        cases = [self.obs[:-1], self.obs+[self.obs[-1]], self.obs[:-1]+[self.obs[0]],
                 self.obs[:-1]+[((2, 3), 0, .4)], self.obs[:-1]+[((3, 2), False, .4)],
                 self.obs[:-1]+[((3, 2), 0, math.inf)], self.obs[:-1]+[((3, 2), 0, -1)]]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rebuild(self.menu, self.heads, bad)
        with self.assertRaises(ValueError):
            rebuild(self.menu, {True: 1., 3: 0.}, self.obs)

    def test_heads_are_full_vocab_and_keep_high_confidence_precision(self):
        raw = struct.pack("<6f", -60, 0, -60, -60, -60, -60)
        got = retained_heads(raw, self.menu)
        self.assertEqual(got[1], math.log1p(5*math.exp(-60)))
        self.assertGreater(got[1], 0.)
        self.assertAlmostEqual(got[3], 60.)
        tied = retained_heads(struct.pack("<6f", *([9]*6)), self.menu)
        self.assertAlmostEqual(tied[1], math.log(6))
        for row in (raw[:-1], struct.pack("<6f", 1, 2, 3, 4, 5, math.nan)):
            with self.assertRaises(ValueError):
                retained_heads(row, self.menu)

    def test_sparse_native_scores_have_bounds_not_an_exact_normalizer(self):
        s = dict(target_id=2, strongest_rival_id=0, greedy_id=0, target_rank=3, vocabulary=5,
                 target_logit=0., strongest_rival_logit=0., target_minus_best_rival=0., nll=math.log(5))
        self.assertEqual(checked_native_nll(s, 2, 5), math.log(5))
        for patch in [dict(nll=.9), dict(nll=2.), dict(target_id=True), dict(target_rank=1),
                      dict(greedy_id=1), dict(target_minus_best_rival=-.5), dict(strongest_rival_id=2)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                checked_native_nll({**s, **patch}, 2, 5)
        tail = dict(s, target_logit=-2000., target_minus_best_rival=-2000., nll=2000.2)
        self.assertEqual(checked_native_nll(tail, 2, 5), 2000.2)


if __name__ == "__main__":
    unittest.main()

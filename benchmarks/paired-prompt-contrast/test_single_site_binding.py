from collections import Counter
import itertools
import math
import unittest

import single_site_binding as s


class SingleSiteBinding(unittest.TestCase):
    def test_remaining_graph_is_pair_disjoint_and_exhaustive(self):
        prior = {tuple(sorted((r[1], r[4]))) for rows in s.pp.roster().values() for r in rows}
        ds = s.decks()
        self.assertEqual(set(ds), set(s.FOLDS))
        for rows in ds.values():
            pairs = {tuple(sorted((r[1], r[4]))) for r in rows}
            self.assertEqual(len(pairs), 4)
            self.assertFalse(pairs & prior)
            self.assertEqual(Counter(i for e in pairs for i in e), Counter(s.pp.IDS))
            prior.update(pairs)
        self.assertEqual(prior, set(itertools.combinations(sorted(s.pp.IDS), 2)))

    def test_orientation_and_reference_loss(self):
        gaps = [math.sin(i * 0.6) - 0.3 * i for i in range(32)]
        for view, obj in itertools.product(s.VIEWS, s.OBJECTIVES):
            groups = s.groups(view)
            self.assertEqual(
                Counter(i for group in groups for i in group), Counter({i: 2 for i in range(32)})
            )

            def value(xs):
                ms = []
                for group in groups:
                    a, b, c, d = (xs[i] for i in group)
                    margins = [a - c, d - b, a - b, d - c]
                    ms.extend(margins if obj == "pairwise" else [sum(margins) / 4])
                return math.fsum(math.log1p(math.exp(-x)) for x in ms) / len(ms)

            loss, g = s.reference_seed(gaps, obj, view)
            self.assertAlmostEqual(loss, value(gaps), places=14)
            for j in range(32):
                a, b = gaps.copy(), gaps.copy()
                a[j] += 1e-5
                b[j] -= 1e-5
                self.assertAlmostEqual(g[j], (value(a) - value(b)) / 2e-5, places=10)

    def test_interaction_ignores_query_and_fact_main_effects(self):
        gaps = [i * 0.125 for i in range(32)]
        nuisance = [x + (2 if i & 2 else -1) + (5 if i & 1 else 3) for i, x in enumerate(gaps)]
        for view in s.VIEWS:
            self.assertEqual(
                s.reference_seed(gaps, "interaction", view), s.reference_seed(nuisance, "interaction", view)
            )
            self.assertNotEqual(
                s.reference_seed(gaps, "pairwise", view), s.reference_seed(nuisance, "pairwise", view)
            )

    def test_sites_cross_factors_and_bits_round_trip(self):
        self.assertEqual(len(set(s.lanes())), 32)
        for u in s.pp.FACTORS:
            self.assertEqual(list(u), s.doubles(s.bits(u)))

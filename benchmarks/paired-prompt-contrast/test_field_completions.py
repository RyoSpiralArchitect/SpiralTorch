"""Synthetic causal, length, mask, and budget controls; never loads a model."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_completions import CompletionMenu, fit_menus, score_menu, supervised_content


class CompletedAnswers(unittest.TestCase):
    def test_project_only_exact_active_run(self):
        prefix = [5, 2, 5]
        doc = prefix + [3, 4, 0]
        mask = [-100]*3 + doc[3:]
        self.assertEqual(supervised_content(prefix, doc, mask, 10, 0), (3, 4))
        for p, d, m in (([5, 3, 5], doc, mask), (prefix, doc, [5]+mask[1:]),
                        (prefix, doc, mask[:-2]+[-100, 0]), (prefix, doc, mask[:-1]),
                        (prefix, doc[:-1], mask[:-1]), (prefix, prefix+[0], [-100]*3+[0]),
                        (prefix, prefix+[3, 0, 0], [-100]*3+[3, 0, 0]),
                        (prefix, doc, mask[:-1]+[False])):
            with self.subTest(p=p, d=d, m=m), self.assertRaises(ValueError):
                supervised_content(p, d, m, 10, 0)

    def test_field_fit_preserves_two_answers_with_same_head(self):
        train = [('shape', (1,)), ('shape', (1, 2)), ('material', (1,)), ('color', (7,))]
        menus = fit_menus(train + train, 10, 0)
        self.assertEqual(menus['shape'].choices, ((1,), (1, 2)))
        self.assertEqual(menus['material'].choices, ((1,),))
        self.assertEqual(menus['shape'].edges, (((1,), 0), ((1,), 2), ((1, 2), 0)))
        with self.assertRaises(ValueError):
            fit_menus(train[:-1], 10, 0)

    def test_stop_and_full_denominator_can_reverse_the_winner(self):
        m = CompletionMenu(7, 0, ((1,), (2, 3)))
        values = {((1,), 0): .03, ((2,), 3): .9, ((2, 3), 0): .8}
        trace = []
        def observe(p, target):
            trace.append((p, target))
            return -math.log(values[p, target])
        lines = score_menu(m, {1: -math.log(.55), 2: -math.log(.35)}, observe, calls=3)
        self.assertEqual(tuple(trace), m.edges)
        self.assertEqual([x.content for x in lines], [(2, 3), (1,)])
        self.assertAlmostEqual(math.exp(-lines[0].line_nll), .35*.9*.8)
        self.assertAlmostEqual(math.exp(-lines[1].content_nll), .55)

    def test_shared_edges_and_lexical_ties(self):
        m = CompletionMenu(7, 0, ((1, 2, 3), (1, 2, 4), (3, 4, 1)))
        trace = []
        def read(p, target):
            trace.append((p, target))
            return 1
        lines = score_menu(m, {3: 2, 1: 2}, read, calls=8)
        self.assertEqual(trace.count(((1,), 2)), 1)
        self.assertEqual(tuple(x.content for x in lines), m.choices)

    def test_normalization_is_not_joint_likelihood(self):
        m = CompletionMenu(7, 0, ((1,), (2, 3)))
        edges = {((1,), 0): .45, ((2,), 3): .44, ((2, 3), 0): .41}
        a, b = score_menu(m, {1: 1.35, 2: 1.20}, lambda p,t: edges[p,t], calls=3)
        self.assertEqual(a.content, (1,))
        self.assertGreater(a.line_nll/2, b.line_nll/3)

    def test_budget_bad_inputs_and_first_failure(self):
        m = CompletionMenu(7, 0, ((1,), (2, 3)))
        for roots, count in (({1: 1, 2: 2}, 2), ({1: 1}, 3), ({1: 2, 2: math.nan}, 3),
                             ({True: 1, 2: 1}, 3), ({1: -1, 2: 0}, 3)):
            with self.assertRaises(ValueError):
                score_menu(m, roots, lambda *a: self.fail("must not forward"), calls=count)
        trace = []
        def fail(p, t):
            trace.append((p, t))
            return math.inf
        with self.assertRaises(ValueError):
            score_menu(m, {1: 0, 2: 1}, fail, calls=3)
        self.assertEqual(trace, [m.edges[0]])
        for choices in ((), ((1,), (1,)), ((1, 0),), ((2,), (1,)), ((True,),), ((8,),), ((3,)*33,)):
            with self.assertRaises(ValueError):
                CompletionMenu(7, 0, choices)


if __name__ == "__main__":
    unittest.main()

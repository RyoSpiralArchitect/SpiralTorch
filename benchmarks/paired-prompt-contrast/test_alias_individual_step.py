import unittest
from alias_individual_step import FLOOR, witness
from alias_individual_step_audit import nine, passes


class IndividualStepTests(unittest.TestCase):
    def test_midpoint_repairs_the_worst_not_the_mean(self):
        gs = [[1., 0.]] * 8 + [[-1.,2.]]
        r = witness(gs, [-.125,0.])
        self.assertEqual(r["worst"], 8)
        self.assertAlmostEqual(r["mix"], sum(r["interval"])/2)
        self.assertGreater(r["margin"], FLOOR)
        self.assertLess(r["cosine"], 1.)
        kept = witness(gs, r["direction"])
        self.assertEqual((kept["mix"], kept["interval"]), (0., None))

    def test_unavailable_segment_is_not_a_full_space_infeasibility_claim(self):
        with self.assertRaises(ValueError):
            witness([[0.,1.,0.],[0.,-1.,1.]], [-1.,0.,0.])

    def test_every_prompt_counts_and_first_order_alone_does_not_pass(self):
        groups = [[0],[1],[2,3]]
        rows = [dict(cost=v, single=s, lines=[0.]*5+[s]) for v,s in ((1,0),(2,0),(3,9),(5,1))]
        self.assertEqual(nine(rows,groups), [1.,2.,4.,9,1])
        self.assertEqual(nine(rows,groups,True), nine(rows,groups))
        self.assertFalse(passes(-.1,.1,5e-6,.15))
        self.assertFalse(passes(-.1,-.02,5e-6,.15))
        self.assertFalse(passes(-1e-6,-1e-6,5e-6,.15))
        self.assertTrue(passes(-.01,-.0101,5e-6,.15))

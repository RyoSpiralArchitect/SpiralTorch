import math
import unittest

from primary_context_audit import direction, primary_cost
from primary_context_training import LIMITS, prior


class PrimaryContextTrainingTests(unittest.TestCase):
    def test_direct_and_limited_segments_do_not_share_a_mix(self):
        base=[0.,0.]; anchor=[-.125,-.125]; guard=[[1.,0.],[0.,1.]]
        direct,_,_,_,cap,lim,mix=direction(base,anchor,[[1.,2.],*guard],.1)
        self.assertEqual((cap,lim,mix),(1.,None,1.))
        self.assertAlmostEqual(direct[1],2*direct[0],places=7)
        turned,_,_,_,cap,lim,mix=direction(base,anchor,[[1.,-1.],*guard],.1)
        self.assertEqual(lim,1);self.assertEqual(mix,cap/2)
        self.assertLess(turned[0]-turned[1],0.)
        self.assertTrue(all(x<-.1*math.hypot(*turned) for x in turned))
        with self.assertRaisesRegex(ValueError,"infeasible anchor"):
            direction(base,[-1.,0.],[[1.,-1.],*guard],.1)

    def test_primary_loss_uses_signed_context_not_global_popularity(self):
        p=dict(matched_pair=[[0,1]],rotated_pair=[[0,2]])
        def rows(xs):return [dict(terms=[5.,x,1.]) for x in xs]
        first=rows([4.,5.,3.]);more=rows([3.,4.,2.])
        self.assertLess(primary_cost(more,"positive_ce",p),primary_cost(first,"positive_ce",p))
        for name in p:self.assertEqual(primary_cost(first,name,p),primary_cost(more,name,p))
        self.assertLess(primary_cost(first,"matched_pair",p),primary_cost(first,"rotated_pair",p))
        self.assertEqual(LIMITS["max_steers"],384)
        self.assertEqual(LIMITS["max_backwards"],prior.BUDGET["max_backwards"])

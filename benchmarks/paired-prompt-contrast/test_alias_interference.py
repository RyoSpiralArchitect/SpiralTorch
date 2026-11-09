import unittest
from alias_interference import DIRECTIONS
from alias_interference_audit import cosine,directional_gate,dot,mean_vec


class InterferenceTests(unittest.TestCase):
    def test_negative_cosine_does_not_mean_the_balanced_step_harms_either(self):
        a=[2.,0.]; b=[-.25,2.]
        self.assertLess(cosine(a,b),0.)
        v=mean_vec([a,b])
        self.assertGreater(dot(v,a),0.);self.assertGreater(dot(v,b),0.)
        self.assertIsNone(cosine([0.,0.],a))
        self.assertEqual(DIRECTIONS["non_idle"],(0,1))

    def test_direction_requires_both_measured_sign_and_local_agreement(self):
        plan=dict(margin=5e-6,linear_rel=.15)
        def match(p,m):return directional_gate(dict(predicted=p,measured=m),1,plan)
        self.assertTrue(match(.001,.00101))
        self.assertFalse(match(.001,-.001))
        self.assertFalse(match(.001,.002))
        self.assertFalse(match(1e-8,1e-8))
        self.assertFalse(match(-.001,.001))

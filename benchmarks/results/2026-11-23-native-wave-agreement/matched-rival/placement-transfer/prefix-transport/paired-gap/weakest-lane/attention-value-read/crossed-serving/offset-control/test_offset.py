import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze as a
import contract as k


class Controls(unittest.TestCase):
    def test_maximum_joint_not_individually_attainable(self):
        self.assertEqual(k.exhaustive([(-1.,-3.),(-4.,-6.),(1.,2.)]), (2,1,2.))
        self.assertEqual(k.exhaustive([(1.,-1.),(-4.,-6.)]), (2,1,0.))

    def test_exact_signs_including_ties_and_unrepresentable_interiors(self):
        self.assertTrue(a.exact_decisions([[1,1]],[[0,0]]))
        self.assertFalse(a.exact_decisions([[1,2]],[[0,0]]))
        self.assertFalse(a.exact_decisions([[0,0]],[[0,1]]))
        self.assertFalse(a.exact_decisions([[-1.,-math.nextafter(1.,math.inf)]],[[1,-1]]))
        self.assertTrue(a.exact_decisions([[2,0]],[[1,-1]]))
        self.assertTrue(a.exact_decisions([[2,0]],[[1,1]]))

    def test_correct_count_does_not_imply_same_winning_pairs(self):
        data=[("x",["a","b"],[1,-1],[-1,1]),("x",["c","d"],[0,2],[1,-1])]
        st=a.stats(data)
        self.assertEqual((st["both"],st["target_both"],st["shared"],st["missed"],st["other"],st["ties"]),(1,1,0,1,1,1))

    def test_verbs_have_the_same_contrast_orientation_in_both_lanes(self):
        rows=[[0.]*7+[3.,1.,0.,0.],[0.]*7+[2.,4.,0.,0.]]
        self.assertEqual(k.contrasts(rows),[2.,-2.])
        self.assertTrue(a.both(k.contrasts(rows)))
        self.assertFalse(a.both([0.,-2.]))


if __name__=="__main__":
    unittest.main()

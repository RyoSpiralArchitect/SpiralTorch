import itertools
from pathlib import Path
import runpy
import unittest

g = runpy.run_path(str(Path(__file__).with_name("gap_readback.py")))


class GapReadbackTests(unittest.TestCase):
    def test_common_shift_alone_recovers_one_failed_pair(self):
        before, after = (g["geometry"](*v) for v in ((5, 1), (3, -1)))
        self.assertEqual(g["change"](before, after), [0, 2, 2])

    def test_signed_identity_including_ties_and_reversed_lanes(self):
        for p, n in itertools.product((-2., -.5, 0., 1.), repeat=2):
            sep, offset, weak = g["geometry"](p, n)
            self.assertEqual(weak, min(p, -n))
            self.assertEqual(weak, sep-offset)
            self.assertEqual(int(weak>0), int(p>0 and n<0))

    def test_no_cases_not_a_zero_effect(self):
        self.assertIsNone(g["stats"]([])["weaker"])
        v = g["stats"]([[2, -1, 1], [-1, 3, 2]])
        self.assertEqual((v["separation_larger"],v["recentering_larger"],v["weaker"]), (1,1,1.5))


if __name__ == "__main__":
    unittest.main()

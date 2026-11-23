"""Small dependency-free checks on crossing, signed pairs, physical scale and ties."""
import itertools
import math
from pathlib import Path
import runpy
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_design as c
a = runpy.run_path(str(HERE / "audit.py"))


class CrossTests(unittest.TestCase):
    def test_delivered_not_parameter_length(self):
        tr = [{"site_input":[2., -1., 0.], "source":[1., 4., .5]}]
        w = [0., 0., 0., 4., 2., 8.]
        outs = c.delivery(w, tr)
        self.assertEqual(outs, [[3., 1., 1.]])
        self.assertAlmostEqual(c.rms(outs, tr), math.sqrt(2.))
        big = [{"site_input":[2.**26]*3, "source":[1.]*3}]
        self.assertEqual(c.delivery([0.]*3 + [1.]*3, big), [big[0]["site_input"]])
        with self.assertRaisesRegex(ValueError, "singular"):
            c.rms(c.delivery(w, big), big)
        with self.assertRaisesRegex(ValueError, "bias/gate"):
            c.delivery([1.]*6, tr)

    def test_raw_and_equal_energy_not_equal_vectors(self):
        orig = [{"site_input":[1., 1., 1.], "source":[1., 0., 0.]}]
        other = [{"site_input":[1., 1., 1.], "source":[0., 3., 0.]}]
        w = [0.]*3 + [4.]*3
        target = c.rms(c.delivery(w, orig), orig)
        before = c.rms(c.delivery(w, other), other)
        p = [0.]*3 + [c.d.f32(k*target/before) for k in w[3:]]
        matched = c.delivery(p, other)
        self.assertNotEqual(matched, c.delivery(w, orig))
        self.assertEqual(c.rms(matched, other), target)

    def test_four_sufficiency_patterns_and_nulls(self):
        by = {"0001":"both_needed", "0011":"weights_only_suffices",
              "0101":"read_only_suffices", "0111":"either_suffices"}
        for v in itertools.product((0,1), repeat=4):
            expected = "".join(map(str, v))
            self.assertEqual(a["pattern"](v), (expected, by.get(expected)))

    def test_verb_polarity_ties_and_interaction(self):
        def z(gap):
            return [0.]*7 + [gap]+[0.]*3
        self.assertEqual(a["paired"](z(3), z(0)), dict(both=0, correct=1, ties=1, weaker=0))
        def hit(w):
            return a["paired"](z(w), z(-w))
        group = [{"outcomes":{k:hit(v) for k,v in zip(a["CELLS"], [-1.,2.,-3.,4.])}}]
        row = a["summarize"](group)
        self.assertEqual(row["transitions"]["read_only_suffices"], 1)
        self.assertEqual(sum(row["histogram"].values()), 1)
        self.assertEqual(row["margin_effects"], dict(read_at_A=3., read_at_U=7.,
                         weights_at_A=-2., weights_at_U=2., interaction=4.))


if __name__ == "__main__":
    unittest.main()

import math
from pathlib import Path
import runpy
import unittest


M = runpy.run_path(str(Path(__file__).resolve().with_name("transfer.py")))
D = M["DISTANCE"]


def groups(names):
    return [dict(key="%s.%d%d" % (name,p,c),pair=name,plural=bool(p),conflict=bool(c),
                 gold=p,rival=1-p) for name in names for p in (0,1) for c in (0,1)]


class Transfer(unittest.TestCase):
    def test_receiver_is_not_normalized_and_indices_are_not_swapped(self):
        a = [[1,0],[1,2],[0,2],[3,4]]
        gram = M["reference"](a)
        x = M["pair"](1,0,2,gram,M["norms"](gram,4))
        self.assertEqual([x["t"+i] for i in ("11","14","41","44")], [D,2*D,3*D,4*D])
        self.assertEqual((x["donor"], x["receiver"], x["delta"]), (D,2*D,3*D))
        u = [0.,1.]
        score = lambda step: math.fsum(x*y for x,y in zip((3.,4.),step))
        self.assertEqual(score([D*x for x in u])-score([0.,0.]), x["t44"])

    def test_verb_polarity_cancels_a_pure_gain(self):
        cases = groups(("a","b","c"))
        a = [[1.-2*c["plural"],0.] for c in cases]
        rows, result = M["analyze"](cases, M["reference"](a+[[3*x,3*y] for x,y in a]))
        self.assertEqual(len(rows),96)
        self.assertFalse(any(r["target"].split(".")[0] == r["source"].split(".")[0] for r in rows))
        self.assertEqual((result["same"]["n"],result["opposite"]["n"]), (48,48))
        self.assertEqual(result["same"]["delta"],2*D)
        self.assertEqual(result["opposite"]["delta"],-2*D)
        self.assertEqual((result["all"]["delta"],result["all"]["donor"]), (0.,0.))
        self.assertFalse(result["consistent"])

    def test_a_shared_transfer_component_survives_balance(self):
        cases = groups(("a","b"))
        a = [[1.-2*c["plural"],1.] for c in cases]
        _, result = M["analyze"](cases, M["reference"](a+[[2*x,2*y] for x,y in a]))
        self.assertTrue(result["consistent"])
        self.assertAlmostEqual(result["all"]["receiver"],D/math.sqrt(2))
        self.assertAlmostEqual(result["all"]["donor"],0)
        self.assertAlmostEqual(result["all"]["cosine_change"],0)

    def test_single_group_missing_supervision_and_bad_matrix(self):
        cases = groups(("a",))
        rows, result = M["analyze"](cases, M["reference"]([[1.]]*8))
        self.assertEqual(rows, [])
        self.assertIsNone(result["consistent"])
        with self.assertRaisesRegex(ValueError,"incomplete lexical"):
            M["checked_cases"](cases[:-1])
        broken = [dict(c) for c in cases]
        broken[2]["rival"] = 17
        with self.assertRaisesRegex(ValueError,"supervision"):
            M["checked_cases"](broken)
        for bad in ([[0.]],[[float("nan")]],[[1.,.2],[.3,1.]],[[1.,.3],[.3,1.]]):
            exact = [[1.]] if len(bad) == 1 else [[1.,0.],[0.,1.]]
            with self.assertRaises(ValueError):
                M["check_gram"](bad,exact)


if __name__ == "__main__":
    unittest.main()

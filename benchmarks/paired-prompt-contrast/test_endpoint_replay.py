import hashlib
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endpoint_replay import Case, first_token_metrics, measure


def pack(*values):
    return struct.pack(f"<{len(values)}f", *values)


class EndpointTests(unittest.TestCase):
    def test_all_token_ties_and_correct_winner(self):
        for gold, rank, winner, wrong, margin in [(0, 1, 0, 1, 0), (1, 2, 0, 0, 0), (2, 3, 0, 0, -2)]:
            m = first_token_metrics(pack(2, 2, 0), gold, 3)
            self.assertEqual((m["rank"], m["greedy"], m["strongest_wrong"], m["margin"]), (rank, winner, wrong, margin))
            self.assertAlmostEqual(m["ce"], math.log(2 + math.exp(-2)) - margin)
        self.assertEqual(first_token_metrics(pack(-1000, 1000), 1, 2)["ce"], 0.)
        self.assertEqual(first_token_metrics(pack(0, -60), 0, 2)["ce"], math.log1p(math.exp(-60)))

    def test_wrong_shape_invalid_target_and_nonfinite_logits(self):
        for raw, gold, size in [(pack(0), 0, 1), (pack(0, 1), -1, 2), (pack(0, 1), True, 2),
                                (pack(0), 0, 2), (pack(0, float("nan")), 0, 2), (pack(0, float("inf")), 1, 2)]:
            with self.subTest(raw=raw, gold=gold, size=size), self.assertRaises(ValueError):
                first_token_metrics(raw, gold, size)

    def setUp(self):
        self.cases = [Case(str(i), (i,), i) for i in range(3)]
        self.arms = ["ce", "step", "double"]
        self.raw = pack(0., -1., 1.)
        self.pins = {(a, c): hashlib.sha256(self.raw).hexdigest() for a in self.arms[:2] for c in self.cases}
        self.starts, self.writes = [], []

    def run_grid(self, **changes):
        args = dict(cases=self.cases, arms=self.arms, references=self.pins, controls=self.arms[:2],
                    vocab_size=3, calls=9, observe=lambda a,c:self.raw,
                    on_start=lambda a,c:self.starts.append((a,c)),
                    on_raw=lambda a,c,raw,h:self.writes.append((a,c,raw,h)))
        return measure(**(args | changes))

    def test_rotation_and_complete_observations(self):
        result = self.run_grid()
        self.assertEqual([a for a, _ in self.starts], ["ce","step","double","step","double","ce","double","ce","step"])
        self.assertEqual(set(result), set(self.starts))
        self.assertEqual(len(self.writes), 9)
        self.assertTrue(all(m["greedy"] == 2 for m in result.values()))

    def test_replay_checks_bytes_even_if_argmax_and_margin_are_unchanged(self):
        bad = pack(-0., -1., 1.)
        with self.assertRaisesRegex(ValueError, "ce, 0"):
            self.run_grid(observe=lambda a,c:bad)
        self.assertEqual(self.starts, [("ce", self.cases[0])])
        self.assertEqual(self.writes[0][2:], (bad, hashlib.sha256(bad).hexdigest()))

    def test_bad_new_readout_and_observer_errors_do_not_advance(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            self.run_grid(controls=[], references={}, observe=lambda a,c:b"x")
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(self.writes[0][2], b"x")
        def broken(a, c):
            raise OSError("readout failed")
        with self.assertRaisesRegex(OSError, "readout failed"):
            self.run_grid(observe=broken)
        self.assertEqual((len(self.starts), len(self.writes)), (2, 1))

    def test_entire_roster_admitted_before_first_call(self):
        # True == 1 in Python: a malformed stored identity must not pass set equality.
        stale = {("ce", Case("1", (True,), 1)) if k == ("ce", self.cases[1]) else k: v for k, v in self.pins.items()}
        for changes in [dict(calls=8), dict(arms=["ce"]*3), dict(controls=["unknown"]),
                        dict(controls=["ce"]*2), dict(references={}), dict(references=stale),
                        dict(references={**self.pins, ("ce", self.cases[0]):"X"*64}),
                        dict(cases=[*self.cases[:2], self.cases[0]]),
                        dict(cases=[*self.cases[:2], Case("2", (True,), 2)]),
                        dict(cases=[*self.cases[:2], Case("2", (1,), 2)]),
                        dict(cases=[*self.cases[:2], Case("2", (2,), 1)])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_grid(**changes)
            self.assertFalse(self.starts)


if __name__ == "__main__":
    unittest.main()

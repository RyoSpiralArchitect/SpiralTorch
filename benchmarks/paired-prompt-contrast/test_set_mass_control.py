import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endpoint_replay import first_token_metrics
from set_mass_control import MAX_VOCABULARY, match_mass, reference_support


def pack(*values):
    return struct.pack(f"<{len(values)}f", *values)


class MassControlTests(unittest.TestCase):
    def test_support_selection_ties_and_label_blind_counterpart(self):
        row = pack(5, 4, 4, 3, 0)
        self.assertEqual(reference_support(row, 2), (0, 1))
        self.assertEqual(reference_support(row, 2, include_target=4), (0, 4))
        self.assertEqual(reference_support(row, 2, include_target=1), (0, 1))
        self.assertEqual(reference_support(row, 1, include_target=4), (4,))

    def test_signed_offset_achieves_target_mass_and_preserves_two_conditional_distributions(self):
        source = pack(0, 1, -2, 3, -0.)
        for offset in (-2., 2.):
            ref = pack(offset, 1, -2+offset, 3, -0.)
            r = match_mass(ref, source, (2,0))
            self.assertEqual(r.offset, offset)
            self.assertEqual(r.raw, ref)
            self.assertEqual(r.realized_minus_reference_log_odds, 0.)
            self.assertEqual(r.max_within_margin_drift, 0.)
            self.assertEqual(r.raw[-4:], pack(-0.))
            want = (math.exp(offset)+math.exp(offset-2))/(math.exp(offset)+math.exp(offset-2)+math.e+math.exp(3)+1)
            self.assertAlmostEqual(r.realized_probability, want, places=14)
        ref = pack(3, 1, -1, -1)
        src = pack(0, 2, 0, 0)
        r = match_mass(ref, src, (0,2))
        vs = struct.unpack("<4f", r.raw)
        self.assertAlmostEqual(vs[0]-vs[2], 0.)
        self.assertAlmostEqual(vs[1]-vs[3], 2.)
        self.assertNotEqual(first_token_metrics(r.raw, 3, 4)["greedy"], first_token_metrics(src, 3, 4)["greedy"])

    def test_rounded_member_offsets_can_differ_even_when_the_ideal_offset_is_exact(self):
        source = pack(0, 2**24, 2**24)
        ref = pack(0, 2**24, 2**24-1)
        m = match_mass(ref, source, (0,1))
        self.assertEqual(m.offset, 1.)
        self.assertEqual(struct.unpack("<3f", m.raw), (1., 2**24, 2**24))
        self.assertEqual(m.max_within_margin_drift, 1.)
        self.assertEqual(m.realized_minus_reference_log_odds, -1.)
        self.assertLess(m.realized_probability, m.reference_probability)

    def test_gauge_invariance_saturated_probabilities_and_noop_preserve_bytes(self):
        src, ref = pack(-0., 1., -2., 4.), pack(8., 9., 6., 12.)
        self.assertIs(match_mass(ref, src, (0,2)).raw, src)
        self.assertIs(match_mass(src, src, (0,2)).raw, src)
        m = match_mass(pack(-1000., 0), pack(-1001., 0), (0,))
        self.assertEqual((m.source_probability, m.reference_probability), (0.,0.))
        self.assertEqual(m.offset, 1.)
        self.assertEqual(m.realized_minus_reference_log_odds, 0.)

    def test_invalid_inputs_are_not_silently_normalized_or_clipped(self):
        row = pack(1,0,2)
        for bad in ((), (0,1,2), (0,0), (True,), (3,), (-1,)):
            with self.assertRaises(ValueError): match_mass(row,row,bad)
        for bad in (True, 0, 3, 1.5):
            with self.assertRaises(ValueError): reference_support(row,bad)
        for bad in (True, -1, 3):
            with self.assertRaises(ValueError): reference_support(row,2,include_target=bad)
        for bad in (row[:-1], pack(0), pack(0, 1), pack(0, math.nan, 1), bytes(4*(MAX_VOCABULARY+1))):
            with self.assertRaises(ValueError): match_mass(bad,row,(0,))
        high = float.fromhex("0x1.fffffep+127")
        with self.assertRaisesRegex(ValueError, "representable"):
            match_mass(pack(high,-high), pack(high,high), (0,))


if __name__ == "__main__":
    unittest.main()

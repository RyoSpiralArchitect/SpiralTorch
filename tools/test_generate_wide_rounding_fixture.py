import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location(
    "wide_oracle", Path(__file__).with_name("generate_wide_rounding_fixture.py"))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class OracleTests(unittest.TestCase):
    def test_known_normal_midpoints_observe_remote_tail(self):
        for sign in (0, 0x80000000):
            for high in (0x3F000000, 0x3F000001):
                for tail in (0, 1, 0x80000001):
                    actual = oracle.rounded(sum(oracle.units(v ^ sign)
                                               for v in (high, 102 << 23, tail)), 1)
                    next_up = tail == 1 or (tail == 0 and high & 1)
                    self.assertEqual(actual, ((high + (1 << 23)) + int(next_up)) ^ sign)

    def test_underflow_normal_transition_and_overflow_boundaries(self):
        for sign in (0, 0x80000000):
            for scale, high, low, tail, expected in (
                (-149, 0x3F000000, 0, 0, 0),
                (-149, 0x3F000000, 1, 0, 1),
                (-149, 0x3F000000, 0x80000001, 0, 0),
                (-125, 0x3F000000, (102 << 23) | 0x80000000, 0, 0x800000),
                (-125, 0x3F000000, (102 << 23) | 0x80000000, 0x80000001, 0x7FFFFF),
                (129, 0x3F000000, (101 << 23) | 0x80000000, 0, 0x7F800000),
                (129, 0x3F000000, (101 << 23) | 0x80000000, 0x80000001, 0x7F7FFFFF),
            ):
                total = sum(oracle.units(v ^ sign) for v in (high, low, tail))
                self.assertEqual(oracle.rounded(total, scale), expected | sign)

    def test_exact_single_components_match_host_f32_scaling(self):
        for bits in (1, 0x7FFFFF, 0x800000, 0x3F000001, 0x3F7FFFFF, 0x7F7FFFFF):
            for sign in (0, 0x80000000):
                for scale in (-10, -1, 0, 1, 10):
                    value = struct.unpack("<f", struct.pack("<I", bits | sign))[0] * 2.**scale
                    try:
                        expected = struct.unpack("<I", struct.pack("<f", value))[0]
                    except OverflowError:
                        expected = 0x7F800000 | sign
                    self.assertEqual(oracle.rounded(oracle.units(bits | sign), scale), expected)

    def test_fixture_is_finite_bounded_unique_and_deterministic(self):
        rows = oracle.cases()
        self.assertEqual(rows, oracle.cases())
        self.assertEqual(len(rows), len({tuple(row) for row in rows}))
        self.assertTrue(1000 < len(rows) <= 8192)
        self.assertTrue(all(((x >> 23) & 255) != 255 for row in rows for x in row[:3]))


if __name__ == "__main__":
    unittest.main()

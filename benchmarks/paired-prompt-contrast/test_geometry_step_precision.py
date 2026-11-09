from fractions import Fraction
import hashlib
from itertools import product
import math
import struct
import unittest

from geometry_step_precision import MAX_COORDINATES, certify_f32_delta


def packed(*xs):
    return struct.pack("<" + str(len(xs)) + "f", *xs)


class GeometryStepPrecisionTests(unittest.TestCase):
    def test_identity_and_zero_radius_preserve_signed_zero_and_layout(self):
        a = {"bias": packed(-0., 1.), "gate": packed(-2.)}
        b = {"bias": packed(-0., 4.), "gate": packed(2.)}
        for radius, endpoint in ((5., b), (0., a)):
            r = certify_f32_delta(a, b, radius, tolerance=0.)
            self.assertEqual(r["status"], "witness_within_limits")
            self.assertEqual(r["direction_l2"], 5.)
            self.assertEqual(r["nearest_l2"], radius)
            self.assertEqual(r["min_relative_error_squared"], ["0", "1"])
            self.assertEqual(r["witness_coordinates_sha256"], hashlib.sha256(b"".join(endpoint.values())).hexdigest())

    def test_half_an_ulp_on_either_side_is_not_a_small_representable_step(self):
        h = 2.**-23
        for sign in (1., -1.):
            a = {"g": packed(sign)}
            r = certify_f32_delta(a, {"g": packed(sign*(1+h))}, h/2)
            self.assertEqual(r["status"], "vector_impossible")
            self.assertEqual(r["min_relative_error_squared"], ["1", "1"])
            self.assertFalse(r["vector_ok"])

    def test_length_matches_while_no_direction_reaches_the_bound(self):
        h = 2.**-23
        radius = math.sqrt(8)*h
        r = certify_f32_delta({"bias": packed(1.), "gate": packed(1.)},
                              {"bias": packed(1.+3*h), "gate": packed(1.+4*h)}, radius)
        self.assertTrue(r["length_ok"])
        self.assertEqual(r["status"], "vector_impossible")
        self.assertAlmostEqual(r["nearest_l2"], radius)
        self.assertGreater(r["min_relative_vector_error"], .1)
        self.assertAlmostEqual(math.hypot(*r["group_relative_l2"].values()), r["min_relative_vector_error"])
        self.assertEqual(r["witness_coordinates_sha256"], hashlib.sha256(packed(1.+2*h, 1.+2*h)).hexdigest())

    def test_independent_full_enumeration_on_small_subnormal_lattices(self):
        tiny = 2.**-149
        for origin, desired, radius in (((-2, 1), (1, -1), 4.8), ((-1, 3), (-4, 3), 2.5),
                                         ((2, -1), (-1, 5), 3.5), ((0, 0), (3, 4), .1)):
            a = [v*tiny for v in origin]
            b = [v*tiny for v in desired]
            rad = radius*tiny
            actual = certify_f32_delta({"x": packed(*a)}, {"x": packed(*b)}, rad)
            direction = [y-x for x, y in zip(a, b)]
            length = math.hypot(*direction)
            units = [d / length for d in direction]
            # Includes both ends bracketing each ideal; farther points cannot be closer.
            def exact(i, j):
                return sum(Fraction(((k*tiny-x)/rad)-u)**2 for k,x,u in zip((i,j), a, units))
            expected = min(exact(i, j) for i, j in product(range(-9, 10), repeat=2))
            n, d = map(int, actual["min_relative_error_squared"])
            self.assertEqual(Fraction(n, d), expected)

    def test_tiny_or_enormous_radius_needs_no_unsafe_f32_cast(self):
        a = {"x": packed(0.)}
        for radius in (float.fromhex("0x1.fffffffffffffp1023"), float.fromhex("0x0.0000000000001p-1022")):
            r = certify_f32_delta(a, {"x": packed(2.**-149)}, radius)
            self.assertEqual(r["status"], "vector_impossible")
            self.assertTrue(math.isfinite(r["min_relative_vector_error"]))

    def test_threshold_is_checked_before_rounding_a_report_number(self):
        h = 2.**-23
        a, b = {"x": packed(1.)}, {"x": packed(1.+2*h)}
        radius = 1.5*h
        error = certify_f32_delta(a,b,radius)["min_relative_vector_error"]
        below = certify_f32_delta(a,b,radius,tolerance=math.nextafter(error, 0.))
        equal = certify_f32_delta(a,b,radius,tolerance=error)
        self.assertFalse(below["vector_ok"])
        self.assertTrue(equal["vector_ok"])
        self.assertFalse(equal["length_ok"])
        self.assertEqual(equal["status"], "witness_misses_length")

    def test_rejects_wrong_layout_budget_nonfinite_or_undefined_direction(self):
        v = packed(1.)
        x = {"bias": v}
        for a,b,rad,tol in [({}, {}, 0., 0.), (x, {}, 1., 0.), (x, x, 1., 0.),
                            ({"b":v,"g":v},{"g":v,"b":v},0.,0.),
                            (x, {"bias":b"x"}, 0., 0.), ({"bias":b""},{"bias":b""},0.,0.),
                            ({"":v},{"":v},0.,0.), (x,{"bias":packed(float("nan"))},0.,0.),
                            ({"bias":packed(float("inf"))},x,0.,0.), (x,x,-1.,0.),
                            (x,x,True,0.), (x,x,0.,float("nan")), (x,x,0.,1.), (x,x,10**400,0.)]:
            with self.subTest(radius=rad,tol=tol), self.assertRaises(ValueError):
                certify_f32_delta(a,b,rad,tolerance=tol)
        block = {"x": v * (MAX_COORDINATES+1)}
        with self.assertRaises(ValueError):
            certify_f32_delta(block, block, 0.)

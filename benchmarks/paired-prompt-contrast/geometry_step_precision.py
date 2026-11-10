"""Model-free lower bound for a fixed-radius, fixed-direction geometry step in f32 storage."""
import hashlib
import math
import struct
from fractions import Fraction


MAX_COORDINATES = 16_384
MAX_RELATIVE_DELTA_ERROR = 1e-4
_U = struct.Struct("<I")
_F = struct.Struct("<f")
_SIGN = 1 << 31
_MASK = (1 << 32) - 1
# Finite f32 bit patterns, ordered from -MAX through the two zeroes to +MAX.
_FIRST, _LAST = 0x00800000, 0xff7fffff


def _float(key):
    bits = (key ^ _SIGN) if key & _SIGN else (~key & _MASK)
    return _F.unpack(_U.pack(bits))[0]


def _key(value):
    bits = _U.unpack(_F.pack(value))[0]
    return (~bits & _MASK) if bits & _SIGN else (bits ^ _SIGN)


def _nearest(origin, direction, radius, direction_l2):
    if radius == 0. or direction == 0.:
        return origin, 0.
    unit = direction / direction_l2

    def residual(key):
        return (_float(key) - origin) / radius - unit

    # Subtraction and division by a positive radius are monotone, including after
    # binary64 rounding. Only the two sides of the sign crossing can minimize |r|.
    lo, hi = _FIRST, _LAST
    while lo < hi:
        mid = (lo + hi) // 2
        if residual(mid) < 0.:
            lo = mid + 1
        else:
            hi = mid
    choices = [lo, max(_FIRST, lo - 1)]
    # Prefer the ordinary round-to-nearest proposal when tied (including plateaus).
    nominal = origin + direction * (radius / direction_l2)
    if math.isfinite(nominal) and abs(nominal) <= _float(_LAST):
        choices.insert(0, _key(nominal))
    best = min(choices, key=lambda k: abs(residual(k)))
    return _float(best), residual(best)


def _norm(coordinates):
    # The native displacement contract uses compensated sums in artifact order.
    high = correction = 0.
    for a, b in coordinates:
        term = (b - a) * (b - a)
        total = high + term
        correction += (high - total) + term if high >= term else (term - total) + high
        high = total
    return math.sqrt(high + correction)


def certify_f32_delta(origin, proposal, radius, *, tolerance=MAX_RELATIVE_DELTA_ERROR):
    """Check a rounded proposal without changing weights, guessing a radius, or running a model.

    Supply matching ordered dictionaries of named trainable f32le byte blocks in artifact
    order. The vector test sums the squares of binary64 component residuals *exactly*,
    then compares squared rationals. Frozen owners/layout must be checked by the caller.
    A witness is not a valid model artifact or a guarantee about either example's loss.
    """
    if (not isinstance(origin, dict) or not isinstance(proposal, dict) or not origin or
            len(origin) > MAX_COORDINATES or len(origin) != len(proposal) or list(origin) != list(proposal)):
        raise ValueError("expected the same nonempty ordered coordinate blocks")
    for val in (radius, tolerance):
        if type(val) not in (float, int):
            raise ValueError("radius and tolerance must be finite numbers")
        try:
            finite = math.isfinite(val)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("radius and tolerance must be finite numbers")
    if radius < 0. or not 0. <= tolerance < 1.:
        raise ValueError("invalid radius or relative tolerance")
    blocks = {}
    count = 0
    for key, a in origin.items():
        b = proposal[key]
        if (not isinstance(key, str) or not key or type(a) is not bytes or type(b) is not bytes
                or not a or len(a) % 4 or len(a) != len(b)):
            raise ValueError("invalid f32le coordinate block")
        count += len(a) // 4
        if count > MAX_COORDINATES:
            raise ValueError("too many coordinates for the independent readback")
        block = [(x, y) for (x,), (y,) in zip(_F.iter_unpack(a), _F.iter_unpack(b))]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in block):
            raise ValueError("nonfinite coordinate")
        blocks[key] = block

    radius, tolerance = float(radius), float(tolerance)
    direction_l2 = _norm(pair for block in blocks.values() for pair in block)
    if radius > 0. and direction_l2 == 0.:
        raise ValueError("positive radius has no direction")
    error_sq = length_sq = Fraction()
    groups = {}
    witness = hashlib.sha256()
    for key, block in blocks.items():
        group_sq = Fraction()
        for before, end in block:
            closest, r = _nearest(before, end - before, radius, direction_l2)
            witness.update(_F.pack(closest))
            group_sq += Fraction(r) ** 2
            length_sq += Fraction(closest - before) ** 2
        groups[key] = math.sqrt(float(group_sq))
        error_sq += group_sq

    tol, goal = Fraction(tolerance), Fraction(radius)
    vector_ok = error_sq <= tol**2
    length_ok = ((1-tol)*goal)**2 <= length_sq <= ((1+tol)*goal)**2
    status = ("vector_impossible" if not vector_ok else
              "witness_within_limits" if length_ok else "witness_misses_length")
    return dict(status=status, coordinates=count, requested_l2=radius, direction_l2=direction_l2,
                nearest_l2=math.sqrt(float(length_sq)), min_relative_vector_error=math.sqrt(float(error_sq)),
                min_relative_error_squared=[str(error_sq.numerator), str(error_sq.denominator)],
                group_relative_l2=groups, tolerance=tolerance, vector_ok=vector_ok, length_ok=length_ok,
                witness_coordinates_sha256=witness.hexdigest())

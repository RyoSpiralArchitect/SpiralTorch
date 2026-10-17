"""Exact integer oracle for retained Wide components, not a float64 surrogate."""
import json
import random


def units(bits):
    exponent = (bits >> 23) & 255
    if exponent == 255:
        raise ValueError("components must be finite")
    significand = (bits & 0x7FFFFF) | (0x800000 if exponent else 0)
    return (-1 if bits >> 31 else 1) * (significand << max(0, exponent - 1))


def rounded(total, scale):
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    if magnitude == 0:
        return 0
    leading = magnitude.bit_length() - 1 + scale - 149
    if leading > 127:
        return sign | 0x7F800000
    if leading < -150:
        return sign
    quantum = max(-149, leading - 23)
    shift = quantum - (scale - 149)
    if shift <= 0:
        high = magnitude << -shift
    else:
        high, low = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        if low > half or (low == half and high & 1):
            high += 1
    if high >= 1 << 24:
        high >>= 1
        quantum += 1
    if high < 1 << 23:
        return sign | high
    exponent = quantum + 23 + 127
    return sign | (0x7F800000 if exponent >= 255 else (exponent << 23) | (high & 0x7FFFFF))


def cases():
    values = set()
    for scale in (-2147483648, -151, -150, -149, -148, -127, -126, -125, -124,
                  0, 1, 2, 127, 128, 129, 130, 2147483647):
        for high in (0x3F000000, 0x3F000001, 0x3F000002, 0x3F7FFFFF):
            for low in (0, 101 << 23, 102 << 23):  # zero, 2^-26, 2^-25
                for low_sign in (0, 0x80000000):
                    for tail in (0, 57 << 23, 7 << 23, 1):  # 2^-70, 2^-120, min subnormal
                        for tail_sign in (0, 0x80000000):
                            for sign in (0, 0x80000000):
                                values.add((high ^ sign, low ^ low_sign ^ sign,
                                            tail ^ tail_sign ^ sign, scale))
    rng = random.Random(20261018)
    for _ in range(512):
        high = 0x3F000000 | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
        low = (rng.randrange(0, 102) << 23) | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
        # Keep the third component well below a low-component ulp.
        low_exponent = (low >> 23) & 255
        tail = ((rng.randrange(0, max(1, low_exponent - 25)) << 23)
                | rng.getrandbits(23) | (rng.getrandbits(1) << 31))
        values.add((high, low, tail, rng.randrange(-155, 135)))
    values.add((0, 0, 0, 0))
    values.add((0x80000000, 0, 0, 0))
    return [list(row) + [rounded(sum(units(x) for x in row[:3]), row[3])]
            for row in sorted(values)]


def main():
    print(json.dumps(dict(schema="spiraltorch.wide_rounding.fixture.v1", cases=cases()),
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()

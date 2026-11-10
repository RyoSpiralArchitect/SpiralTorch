"""Offline common-offset control on retained f32 logits. No model or learned policy."""
from array import array
from dataclasses import dataclass, field
import heapq
import math
import struct
import sys

MAX_VOCABULARY = 1_048_576


def _decode(raw):
    if not isinstance(raw, bytes) or len(raw) % 4 or not 8 <= len(raw) <= 4 * MAX_VOCABULARY:
        raise ValueError("expected one bounded little-endian f32 vocabulary row")
    values = array("f")
    if values.itemsize != 4:
        raise RuntimeError("platform does not use four-byte floats")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError("non-finite logit")
    return values


def reference_support(reference, size, *, include_target=None):
    """Take the reference top classes, optionally reserving one slot for the known answer."""
    row = _decode(reference)
    if type(size) is not int or not 0 < size < len(row):
        raise ValueError("support must leave at least one class outside")
    if include_target is not None and (type(include_target) is not int or not 0 <= include_target < len(row)):
        raise ValueError("invalid included target")
    ids = range(len(row))
    if include_target is None:
        highest = heapq.nsmallest(size, ids, key=lambda i: (-row[i], i))
    else:
        highest = [include_target] + heapq.nsmallest(
            size - 1, (i for i in ids if i != include_target), key=lambda i: (-row[i], i))
    return tuple(sorted(highest))


def _terms(row, members):
    inside = max(members, key=row.__getitem__)
    outside = max((i for i in range(len(row)) if i not in members), key=row.__getitem__)
    a, b = row[inside], row[outside]
    ai = math.log1p(math.fsum(math.exp(row[i] - a) for i in members if i != inside))
    bo = math.log1p(math.fsum(math.exp(v - b) for i, v in enumerate(row) if i != outside and i not in members))
    return a, ai, -b, -bo  # signed terms of log(Z_inside / Z_outside)


def _difference(to, before):
    # Do not subtract two materialized probabilities or two separately rounded,
    # possibly very large, log odds.
    return math.fsum((*to, *(-v for v in before)))


def _probability(terms):
    odds = math.fsum(terms)
    e = math.exp(-abs(odds))
    return 1 / (1 + e) if odds >= 0 else e / (1 + e)


@dataclass(frozen=True)
class MassMatch:
    offset: float
    reference_probability: float
    source_probability: float
    realized_probability: float
    realized_minus_reference_log_odds: float
    max_within_margin_drift: float
    raw: bytes = field(repr=False)


def match_mass(reference, source, members):
    """Match reference mass in real arithmetic and return the actually rounded f32 row.

    The support must already be chosen. Add the *same signed* offset to every
    member, leave all other bytes untouched, and never clip overflowing logits.
    A positive offset boosts the set. This is a reference-assisted diagnostic,
    not independent inference. Inspect the residual before claiming exact f32 matching.
    """
    r, s = _decode(reference), _decode(source)
    if len(r) != len(s):
        raise ValueError("vocabulary mismatch")
    ids = tuple(members)
    if not 0 < len(ids) < len(s) or any(type(i) is not int or not 0 <= i < len(s) for i in ids):
        raise ValueError("expected a nonempty proper subset of class IDs")
    members = frozenset(ids)
    if len(members) != len(ids):
        raise ValueError("duplicate class ID")
    desired, initial = _terms(r, members), _terms(s, members)
    offset = _difference(desired, initial)
    if offset == 0.0:
        return MassMatch(offset, _probability(desired), _probability(initial), _probability(initial), 0., 0., source)
    buf = bytearray(source)
    lo, hi = math.inf, -math.inf
    for i in ids:
        try:
            struct.pack_into("<f", buf, 4 * i, s[i] + offset)
        except OverflowError as exc:
            raise ValueError("shifted logit is not representable in f32") from exc
        value, = struct.unpack_from("<f", buf, 4 * i)
        if not math.isfinite(value):
            raise ValueError("shifted logit is not representable in f32")
        applied = value - s[i]
        lo, hi = min(lo, applied), max(hi, applied)
        s[i] = value
    realized = _terms(s, members)
    return MassMatch(offset, _probability(desired), _probability(initial), _probability(realized),
                     _difference(realized, desired), hi - lo, bytes(buf))

"""Compare saved endpoints on full first-token readouts. No model imports or calls on import."""
from array import array
from dataclasses import dataclass
import hashlib
import math
import re
import sys


@dataclass(frozen=True)
class Case:
    key: str
    prefix: tuple[int, ...]
    target: int


def _token(value, size):
    return type(value) is int and 0 <= value < size


def _valid_case(c, vocab_size):
    return (isinstance(c, Case) and isinstance(c.key, str) and bool(c.key) and
            type(c.prefix) is tuple and bool(c.prefix) and all(_token(i, vocab_size) for i in c.prefix)
            and _token(c.target, vocab_size))


def first_token_metrics(raw, target, vocab_size):
    """Summarize one little-endian f32 logit vector; lowest token ID wins exact ties."""
    if type(vocab_size) is not int or vocab_size < 2 or not _token(target, vocab_size):
        raise ValueError("invalid vocabulary or target")
    if not isinstance(raw, bytes) or len(raw) != 4 * vocab_size:
        raise ValueError("expected one complete little-endian f32 vocabulary")
    values = array("f")
    if values.itemsize != 4:
        raise RuntimeError("platform does not use four-byte floats")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError("non-finite logit")
    gold = values[target]
    wrong = max((i for i in range(vocab_size) if i != target), key=values.__getitem__)
    margin = gold - values[wrong]
    winner = target if margin > 0 or (margin == 0 and target < wrong) else wrong
    peak = values[winner]
    return dict(greedy=winner, strongest_wrong=wrong, margin=margin,
                rank=1 + sum(v > gold or (v == gold and i < target) for i, v in enumerate(values)),
                ce=(peak - gold) + math.log1p(math.fsum(math.exp(v - peak) for i, v in enumerate(values) if i != winner)))


def measure(cases, arms, references, observe, *, controls, vocab_size, calls, on_start, on_raw):
    """Observe once per slot, in rotating arm order. Pin models and authenticate references upstream.

    references[(arm, Case)] is the expected SHA-256 of that control's full raw logits.
    on_start runs before observe; on_raw receives even malformed bytes or a failed replay before
    the exception stops the experiment. No retry, fallback, model loading, or training is performed.
    """
    cases, arms, controls = tuple(cases), tuple(arms), tuple(controls)
    if not cases or not arms or type(calls) is not int or calls != len(cases) * len(arms):
        raise ValueError("wrong call budget")
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("invalid vocabulary")
    if (not all(isinstance(a, str) and a for a in (*arms, *controls)) or
            len(set(arms)) != len(arms) or len(set(controls)) != len(controls) or not set(controls) <= set(arms)):
        raise ValueError("invalid arm or control roster")
    if not all(_valid_case(c, vocab_size) for c in cases):
        raise ValueError("invalid case identity")
    if len({c.key for c in cases}) != len(cases):
        raise ValueError("duplicate case key")
    pins = dict(references)
    if (not all(isinstance(k, tuple) and len(k) == 2 and _valid_case(k[1], vocab_size) for k in pins) or
            set(pins) != {(a, c) for a in controls for c in cases} or
            not all(isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h) for h in pins.values())):
        raise ValueError("incomplete or stale control identities or hashes")
    measured = {}
    for i, case in enumerate(cases):
        start = i % len(arms)
        for arm in arms[start:] + arms[:start]:
            on_start(arm, case)
            raw = observe(arm, case)
            if not isinstance(raw, bytes):
                raise TypeError("observer must return immutable logit bytes")
            digest = hashlib.sha256(raw).hexdigest()
            on_raw(arm, case, raw, digest)
            if (arm, case) in pins and digest != pins[arm, case]:
                raise ValueError(f"control replay differs: {arm}, {case.key}")
            measured[arm, case] = first_token_metrics(raw, case.target, vocab_size)
    return measured

"""Matched 64-coordinate output and Q/K edits for the small, pinned GPT-format fixture."""

from collections import Counter
from itertools import product
import json
import math
import struct

import entity_swap_task as task
from shared_prefix_study import require

W = 32
IDS = (48, 53, 50, 55, 49, 54, 51, 52)
SITES = ("output", "query", "key")
ROUTES = ("output", "qk")
FACTORS = tuple(tuple((-1.0 if i & bit else 1.0) / math.sqrt(W) for i in range(W)) for bit in (1, 2))
ZERO = (0.0,) * (2 * W)
EPS = 5e-6


def roster():
    """Two disjoint edge colours form a cycle; opposite vertices give the unused matching."""
    require(
        len(set(IDS)) == 8 and set(IDS).isdisjoint((*task.ENTITIES, 2, 16, 23, 24, 26, 46, 47, 60)),
        "identity collision",
    )
    decks = {}
    seen = set()
    for fold, indices in {
        "even": [(i, i + 1) for i in range(0, 8, 2)],
        "odd": [(i, (i + 1) % 8) for i in range(1, 8, 2)],
        "check": [(i, i + 4) for i in range(4)],
    }.items():
        edges = [(IDS[i], IDS[j]) for i, j in indices]
        pairs = {tuple(sorted(e)) for e in edges}
        require(Counter(x for e in edges for x in e) == Counter(IDS) and not pairs & seen, "leaked partner")
        seen.update(pairs)
        decks[fold] = [task.render(e, o, a, q) for e in edges for o, a, q in product(range(2), repeat=3)]
    require(len({tuple(r) for deck in decks.values() for r in deck}) == 96, "reused prompt")
    return decks


def pairs(broken):
    """Both supervision conditions present the very same prompts at the same frequencies."""
    out = []
    for first in range(0, 32, 8):
        for order, q in product(range(2), repeat=2):
            p = first + 4 * order + 3 * q
            for n in (p ^ 2, p ^ 1):
                out.extend(((p, n), (p ^ int(broken), n ^ int(broken))))
    return out


def coefficients(gaps, broken):
    """Differentiate mean softplus(negative gap - positive gap) w.r.t. the 32 raw gaps."""
    require(len(gaps) == 32 and all(map(math.isfinite, gaps)), "invalid logit gaps")
    d = [0.0] * 32
    losses = []
    ps = pairs(broken)
    for i, j in ps:
        x = gaps[j] - gaps[i]
        exp = math.exp(-abs(x))
        losses.append(max(0.0, x) + math.log1p(exp))
        v = (1.0 if x >= 0 else exp) / ((1.0 + exp) * len(ps))
        d[i] -= v
        d[j] += v
    return math.fsum(losses) / len(ps), d


def coordinates(route):
    require(route in ROUTES, "unknown route")
    return zip(("output", "output") if route == "output" else ("query", "key"), FACTORS, strict=True)


def lift(route, theta):
    require(len(theta) == 2 * W and all(map(math.isfinite, theta)), "invalid coefficients")
    result = {}
    for k, (site, factor) in enumerate(coordinates(route)):
        block = result.setdefault(site, [0.0] * (W * W))
        for i, u in enumerate(factor):
            for j, v in enumerate(theta[k * W : (k + 1) * W]):
                block[i * W + j] += u * v
    return result


def pull(route, weights):
    result = []
    for site, factor in coordinates(route):
        matrix = weights[site]
        require(len(matrix) == W * W, "invalid dense projection pullback")
        result.extend(math.sumprod(factor, matrix[j::W]) for j in range(W))
    return result


class Archive:
    """Edit only admitted F32 block-zero matrices, never a bias, value, or source file."""

    def __init__(self, raw):
        require(8 < len(raw) <= 100000, "not the small fixture")
        front = 8 + struct.unpack_from("<Q", raw)[0]
        require(8 < front < len(raw), "invalid safetensors header size")
        header = json.loads(raw[8:front])
        spans = []
        for t in header.values():
            a, b = t["data_offsets"]
            require(
                t["dtype"] == "F32" and 0 <= a < b <= len(raw) - front and b - a == 4 * math.prod(t["shape"]),
                "invalid tensor range",
            )
            spans.append((a, b))
        spans.sort()
        require(
            spans[0][0] == 0
            and spans[-1][1] == len(raw) - front
            and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
            "overlapping or missing tensor bytes",
        )
        self.raw = raw
        self.layout = {}
        for site, kind, stride, offset in (
            ("output", "c_proj", W, 0),
            ("query", "c_attn", 3 * W, 0),
            ("key", "c_attn", 3 * W, W),
        ):
            t = header["h.0.attn." + kind + ".weight"]
            require(t["shape"] == [W, stride], "unexpected Conv1D packing")
            self.layout[site] = (front + t["data_offsets"][0], stride, offset)

    def edited(self, changes):
        out = bytearray(self.raw)
        for site, matrix in changes.items():
            start, stride, offset = self.layout[site]
            require(len(matrix) == W * W and all(map(math.isfinite, matrix)), "invalid weight edit")
            for i in range(W):
                for j, dv in enumerate(matrix[i * W : (i + 1) * W]):
                    if dv:
                        k = start + 4 * (i * stride + offset + j)
                        value = struct.unpack_from("<f", self.raw, k)[0] + dv
                        struct.pack_into("<f", out, k, value)
        return bytes(out)


def summary(logits):
    require(
        len(logits) == 32 and all(len(v) == 64 and all(map(math.isfinite, v)) for v in logits),
        "bad scored fold",
    )
    gaps = [v[task.IDLE] - v[task.ON] for v in logits]
    edges = []
    for f in range(0, 32, 8):
        # Order-averaged, positive minus negative. Neither view alone tests both assignments and queries.
        aa, ab, ba, bb = [(gaps[f + i] + gaps[f + i + 4]) / 2 for i in range(4)]
        edges.append([aa - ba, bb - ab, aa - ab, bb - ba])
    gold = [task.IDLE if i % 4 in (0, 3) else task.ON for i in range(32)]
    return dict(
        edges=edges,
        shared=sum(sum(e) / 4 for e in edges) / 4,
        correct=sum(
            (v[t] - v[task.ON if t == task.IDLE else task.IDLE]) > EPS
            for v, t in zip(logits, gold, strict=True)
        ),
        full_correct=sum(
            v[t] > max(x for j, x in enumerate(v) if j != t) for v, t in zip(logits, gold, strict=True)
        ),
        loss=coefficients(gaps, False)[0],
    )


def wins(a, b):
    return sum(
        all(x - y > EPS for x, y in zip(left, right, strict=True))
        for left, right in zip(a["edges"], b["edges"], strict=True)
    )

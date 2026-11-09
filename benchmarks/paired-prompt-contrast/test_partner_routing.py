from collections import Counter
import json
import math
import struct
import unittest

import partner_projection as p


class PartnerRouting(unittest.TestCase):
    def test_pairs_and_control(self):
        for rows in p.roster().values():
            factual, corrupted = p.pairs(False), p.pairs(True)
            self.assertEqual(len(factual), 64)
            self.assertEqual(
                Counter(i for ab in factual for i in ab), Counter(i for ab in corrupted for i in ab)
            )
            self.assertEqual(Counter(i for i, _ in corrupted), Counter(j for _, j in corrupted))
            for i, j in factual:
                x, y = rows[i], rows[j]
                self.assertIn([k for k, (a, b) in enumerate(zip(x, y, strict=True)) if a != b], ([2, 5], [7]))
                for row, expected in ((x, 26), (y, 24)):
                    self.assertEqual(row[2 if row[1] == row[7] else 5], expected)

    def test_fold_and_budget(self):
        decks = p.roster()
        sets = [set(tuple(sorted((r[1], r[4]))) for r in rows) for rows in decks.values()]
        self.assertTrue(all(len(x) == 4 for x in sets))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(
            Counter(i for edge in sets[0] | sets[1] for i in edge), Counter({i: 2 for i in p.IDS})
        )
        self.assertEqual(len({tuple(r) for deck in decks.values() for r in deck}), 96)

    def test_objective_derivative(self):
        gaps = [math.sin(i + 0.2) for i in range(32)]
        v = [math.cos(i * 0.37) for i in range(32)]

        def along(h):
            return [a + h * b for a, b in zip(gaps, v, strict=True)]

        for broken in (False, True):
            loss, g = p.coefficients(gaps, broken)
            hi, lo = [p.coefficients(along(h), broken)[0] for h in (1e-5, -1e-5)]
            self.assertAlmostEqual((hi - lo) / 2e-5, math.sumprod(g, v), places=9)
            if broken:
                self.assertEqual(loss, p.coefficients([-x for x in gaps], True)[0])
                self.assertTrue(all(x == 0 for x in p.coefficients([0] * 32, True)[1]))

    def test_adjoint_and_equal_physical_step(self):
        theta = [math.sin(i * 0.32) for i in range(64)]
        gs = {s: [math.sin(i * 0.73 + k) for i in range(1024)] for k, s in enumerate(p.SITES)}
        for route in p.ROUTES:
            blocks = p.lift(route, theta)
            self.assertAlmostEqual(
                math.hypot(*(x for b in blocks.values() for x in b)), math.hypot(*theta), places=13
            )
            self.assertAlmostEqual(
                math.sumprod(theta, p.pull(route, gs)),
                math.fsum(math.sumprod(b, gs[k]) for k, b in blocks.items()),
                places=12,
            )

    def test_archive_changes_only_admitted_bytes(self):
        w = p.W
        head = {
            "h.0.attn.c_attn.weight": dict(dtype="F32", shape=[w, 3 * w], data_offsets=[0, 12 * w * w]),
            "h.0.attn.c_proj.weight": dict(dtype="F32", shape=[w, w], data_offsets=[12 * w * w, 16 * w * w]),
        }
        hdr = json.dumps(head).encode()
        raw = struct.pack("<Q", len(hdr)) + hdr + bytes(16 * w * w)
        arch = p.Archive(raw)
        for site, stride, offset in (("query", 3 * w, 0), ("key", 3 * w, w), ("output", w, 3 * w * w)):
            m = [0.0] * (w * w)
            m[2 * w + 7] = 0.5
            expected = bytearray(raw)
            struct.pack_into("<f", expected, 8 + len(hdr) + 4 * (2 * stride + offset + 7), 0.5)
            self.assertEqual(arch.edited({site: m}), expected)
        self.assertEqual(arch.edited(p.lift("qk", p.ZERO)), raw)
        with self.assertRaises(KeyError):
            arch.edited({"value": [0.0] * (w * w)})
        with self.assertRaises(ValueError):
            arch.edited({"query": [float("nan")] * (w * w)})

    def test_score_requires_both_axes(self):
        logits = [[0.0] * 64 for _ in range(32)]
        zero = p.summary(logits)
        for i, row in enumerate(logits):
            row[p.task.IDLE] = 2 if i % 2 == 0 else -2
        bias = p.summary(logits)
        self.assertEqual((bias["correct"], p.wins(bias, zero), bias["shared"]), (16, 0, 0))
        for i, row in enumerate(logits):
            row[p.task.IDLE] = 2 if i % 4 in (0, 3) else -2
        paired = p.summary(logits)
        self.assertEqual((paired["correct"], p.wins(paired, zero), paired["shared"]), (32, 4, 4))

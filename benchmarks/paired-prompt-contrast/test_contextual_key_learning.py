from collections import Counter
import json
import struct
import unittest

import contextual_key_learning as d
import key_depth_offset_readback as calibration


class ContextualKeyLearning(unittest.TestCase):
    def test_offset_removes_a_constant_not_a_ranking_error(self):
        ys = [True, False, False, True]
        gs = [-1.0, -5.0, -5.0, -1.0]
        fitted = calibration.offset(gs, ys)
        self.assertAlmostEqual(fitted, 3.0)
        self.assertEqual(calibration.correct(gs, ys, fitted), 4)
        self.assertEqual(calibration.ceiling([-3.0] * 4, ys), 2)
        self.assertEqual(calibration.ceiling([1.0, 2.0, -1.0, 0.0], ys), 3)
        with self.assertRaises(ValueError):
            calibration.offset(gs, [True] * 4)

    def test_no_familiar_pair_or_transfer_identity_leaks(self):
        ds = d.decks()
        old = {
            tuple(sorted((q[1], q[4])))
            for deck in list(d.last.decks().values()) + list(d.pp.roster().values())
            for q in deck
        }
        seen = set()
        cohorts = {}
        for fold, qs in ds.items():
            pairs = {tuple(sorted((q[1], q[4]))) for q in qs}
            self.assertEqual(len(pairs), 4)
            self.assertFalse(pairs & (old | seen))
            seen |= pairs
            cohorts[fold] = set(x for pair in pairs for x in pair)
            self.assertEqual(len(cohorts[fold]), 8)
        self.assertEqual(cohorts["train0"], cohorts["held1"])
        self.assertFalse(cohorts["train1"] & cohorts["transfer0"])
        self.assertEqual(len(seen), 24)

    def test_two_real_queries_share_each_factual_prefix(self):
        for deck in d.decks().values():
            ps, paths = d.paths(deck)
            self.assertEqual(len(ps), 16)
            self.assertEqual(Counter(k["prefix"] for k in paths), Counter({i: 2 for i in range(16)}))
            self.assertEqual([ps[k["prefix"]] + k["suffix"] for k in paths], deck)
            for q, p in zip(deck, d.probe_cases(deck), strict=True):
                self.assertEqual(p["recipient"] + p["suffix"], q)
                self.assertEqual(
                    [i for i, (a, b) in enumerate(zip(p["recipient"], p["donor"], strict=True)) if a != b],
                    [1],
                )
                self.assertEqual(p["position"], 2)
                self.assertEqual(p["donor"], [60, q[4], q[2]])

    def test_independent_hf_layout_changes_only_one_site_in_one_layer(self):
        w = d.pp.W
        head = {}
        cursor = 0
        for layer in (0, 1):
            for key, cols in (("c_proj", w), ("c_attn", 3 * w)):
                end = cursor + 4 * w * cols
                head[f"h.{layer}.attn.{key}.weight"] = dict(
                    dtype="F32", shape=[w, cols], data_offsets=[cursor, end]
                )
                cursor = end
        hdr = json.dumps(head).encode()
        raw = struct.pack("<Q", len(hdr)) + hdr + bytes(cursor)
        arch = d.LayerArchive(raw)
        for layer in (0, 1):
            for site, col in (("query", 0), ("key", w)):
                m = [0.0] * (w * w)
                m[3 * w + 7] = 0.75
                off = head[f"h.{layer}.attn.c_attn.weight"]["data_offsets"][0]
                expect = bytearray(raw)
                struct.pack_into("<f", expect, 8 + len(hdr) + off + 4 * (3 * 3 * w + col + 7), 0.75)
                self.assertEqual(arch.edited({(layer, site): m}), expect)

    def test_cache_threshold_and_snapshot_identity(self):
        def pack(xs):
            return struct.pack("<64f", *xs)

        self.assertLess(d.agree(pack([1] + [0] * 63), pack([1] + [1e-6] * 63)), 5e-5)
        for x in (float("inf"), float("nan"), 3.0):
            with self.assertRaises(ValueError):
                d.agree(pack([x] + [0] * 63), pack([1] + [0] * 63))
        p = dict(base="base", lora="lora", origins={"initial": "g", "subset_rank": "h"})
        hs = {d.owner_id(p, *ln[:4], [0] * 32) for ln in d.lanes()}
        self.assertEqual(len(hs), 16)
        self.assertNotIn(d.owner_id(p, "initial", 0, "key", 1, [1] + [0] * 31), hs)

from collections import Counter
import copy
import itertools
import unittest

import causal_key_lag_transfer as study


class KeyLagTransfer(unittest.TestCase):
    def setUp(self):
        self.training = study.capacity.prev.decks()["train0"]
        self.decks = study.decks(self.training)

    def test_pairs_are_exhaustive_and_current_training_disjoint(self):
        rows = self.decks["original"]
        self.assertEqual([r["prompt"] for r in rows[:32]], self.training)
        fit = {tuple(r["pair"]) for r in rows[:32]}
        unseen = {tuple(r["pair"]) for r in rows[32:]}
        self.assertEqual((len(fit), len(unseen), len(rows)), (4, 12, 128))
        self.assertFalse(fit & unseen)
        self.assertFalse({r["id"] for r in rows[:32]} & {r["id"] for r in rows[32:]})
        self.assertEqual(Counter(r["gold"] for r in rows[32:]), {2: 48, 16: 48})
        self.assertEqual(len(set(study.KINDS)) * 3 * sum(len(x) for x in self.decks.values()), 8064)

    def test_distance_changes_only_entity_filler_positions(self):
        for filler in study.FILLERS:
            for i in range(128):
                cells = [self.decks[f"f{filler}.d{d}"][i] for d in (1, 2, 3)]
                self.assertEqual(len({tuple(sorted(r["prompt"])) for r in cells}), 1)
                self.assertEqual(len({r["gold"] for r in cells}), 1)
                for d, r in zip((1, 2, 3), cells, strict=True):
                    q = r["prompt"]
                    self.assertEqual(len(q), 13)
                    self.assertEqual(q[4], 26 if r["order"] == r["idle"] else 24)
                    self.assertEqual(q[9], 26 if 1 - r["order"] == r["idle"] else 24)
                    self.assertEqual(q[4 - d], r["pair"][r["order"]])
                    self.assertEqual(q[9 - d], r["pair"][1 - r["order"]])
                    self.assertEqual(q[-2:], [r["pair"][r["query"]], 23])
                self.assertEqual(
                    len({tuple(r["prompt"][j] for j in (0, 4, 5, 9, 10, 11, 12)) for r in cells}), 1
                )

    def test_measurement_is_full_vocab_not_just_binary(self):
        rows = self.decks["f58.d2"][32:]

        def raw(which):
            values = []
            for r in rows:
                z = [0.0] * 64
                if which == "target":
                    z[r["gold"]] = 3.0
                elif which == "outsider":
                    z[r["gold"]] = 3.0
                    z[63] = 4.0
                elif which == "constant":
                    z[16] = 3.0
                values.append(study.ROW.pack(*z))
            return b"".join(values)

        correct = study.measures(rows, raw("target"))
        outside = study.measures(rows, raw("outsider"))
        constant = study.measures(rows, raw("constant"))
        self.assertEqual(
            (correct["full_correct"], correct["binary_correct"], correct["ranked"]), (96, 96, 24)
        )
        self.assertEqual((outside["full_correct"], outside["binary_correct"], outside["ranked"]), (0, 96, 24))
        self.assertEqual(
            (constant["full_correct"], constant["binary_correct"], constant["ranked"]), (48, 48, 0)
        )
        self.assertTrue(study.competent(correct))
        self.assertFalse(study.competent(outside))

    def test_ties_and_incomplete_quartets_fail_or_do_not_count(self):
        rows = self.decks["original"][32:]
        tied = bytes(len(rows) * study.ROW.size)
        self.assertEqual(study.measures(rows, tied)["full_correct"], 0)
        self.assertEqual(study.measures(rows, tied)["ranked"], 0)
        for bad in (rows[:-1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                study.measures(bad, tied[: len(bad) * study.ROW.size])
        with self.assertRaises(ValueError):
            study.measures(rows, tied[:-1])
        wrong = copy.deepcopy(rows)
        wrong[0]["gold"] = 18 - wrong[0]["gold"]
        with self.assertRaises(ValueError):
            study.measures(wrong, tied)

    def test_prefix_sealing_uses_the_actual_prefix_length(self):
        rows = self.decks["f58.d1"]
        qs = [r["prompt"] for r in rows]
        # At seven tokens the second entity is not present; these are not the factual prefixes.
        actual = len({tuple(q[:-2]) for q in qs})
        old = len({tuple(q[:7]) for q in qs})
        self.assertNotEqual(actual, old)
        reply = dict(data=dict(prefixes=actual, rows=[dict(ordinary=[0] * 64, cached=[0] * 64) for _ in qs]))
        full, cached, error = study.seal(reply, qs)
        self.assertEqual((full, cached, error), (bytes(128 * 256), bytes(128 * 256), 0.0))
        reply["data"]["prefixes"] = old
        with self.assertRaises(ValueError):
            study.seal(reply, qs)

    def test_padded_matching_failure_does_not_prove_distance_dependence(self):
        def metric(correct, ranked, gap):
            return dict(rows=96, edges=12, full_correct=correct, ranked=ranked, by_edge=[gap] * 12)

        low = metric(48, 0, 0.0)
        high = metric(96, 24, 2.0)
        records = {
            f"{lag}.{kind}": {v: dict(untrained=copy.deepcopy(low)) for v, _, _ in study.variants()}
            for lag, kind in itertools.product((0, 1, 2), study.KINDS)
        }
        records["1.aligned"]["original"]["untrained"] = high
        result = study.decisions(records)
        self.assertTrue(result["primary"]["passed"])
        self.assertFalse(result["double_match_passed"])
        self.assertTrue(all(x["inconclusive_due_to_matching_failure"] for x in result["distance"].values()))
        for lag, filler in itertools.product((1, 2), study.FILLERS):
            records[f"{lag}.aligned"][f"f{filler}.d{lag}"]["untrained"] = high
        self.assertTrue(study.decisions(records)["double_match_passed"])
        records["2.aligned"]["original"]["untrained"] = high
        self.assertFalse(study.decisions(records)["primary"]["passed"])

    def test_invalid_source_layout_and_separation_rejected(self):
        bad = copy.deepcopy(self.training)
        bad[0][-1] = 22
        with self.assertRaises(ValueError):
            study.decks(bad)
        for gap, filler in ((0, 58), (4, 58), (2, None), (1, 57)):
            with self.assertRaises(ValueError):
                study.prompt((35, 48), 0, 0, 0, filler, gap)


if __name__ == "__main__":
    unittest.main()

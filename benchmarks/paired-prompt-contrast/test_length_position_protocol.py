import collections
import unittest

import length_position_protocol as p


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fit = p.questions([(35, 48), (38, 53), (34, 50), (37, 55)])

    def test_matched_all_update_work_and_labels(self):
        for step in range(256):
            a, b = (p.training(self.fit, view, step) for view in ("fixed", "varied"))
            self.assertEqual(a["gold"], b["gold"])
            self.assertEqual(len(a["prompts"]), 96)
            self.assertEqual(sum(map(len, a["prompts"])), 1056)
            for x, y in zip(a["prompts"], b["prompts"]):
                self.assertEqual(len(x), len(y))
                self.assertEqual(collections.Counter(x), collections.Counter(y))
                self.assertEqual(x[-2:], y[-2:])
            self.assertEqual(collections.Counter(a["gold"]), {2: 48, 16: 48})

    def test_varied_preserves_entity_cue_adjacency(self):
        for step in (0, 1, 2, 255):
            for q in p.training(self.fit, "varied", step)["prompts"]:
                for entity in set(sum(([q[1], q[4]] for q in self.fit), [])):
                    positions = [i for i, t in enumerate(q[:-2]) if t == entity]
                    for i in positions:
                        self.assertIn(q[i + 1], (24, 26))

    def test_eval_complete_disjoint_lengths_and_true_factual_prefix(self):
        cases = p.evaluation(self.fit)
        self.assertEqual(len(cases), 24)
        self.assertEqual(sum(len(v["prompts"]) for v in cases.values()), 1536)
        seen = set()
        for name, v in cases.items():
            self.assertEqual(len(v["prompts"]), 96 if "untrained" in name else 32)
            for q in v["prompts"]:
                self.assertIn(len(q), (10, 12))
                self.assertEqual(q[v["prefix_len"]:], q[-2:])
                self.assertNotIn(tuple(q), seen)
                seen.add(tuple(q))

    def test_bad_input_never_becomes_a_different_task(self):
        for view, step in (("best", 0), ("fixed", -1), ("varied", 256)):
            with self.assertRaises(ValueError):
                p.training(self.fit, view, step)
        for wrong in (self.fit[:-1], self.fit[:8] * 4):
            with self.assertRaises(ValueError):
                p.decks(wrong)


if __name__ == "__main__":
    unittest.main()

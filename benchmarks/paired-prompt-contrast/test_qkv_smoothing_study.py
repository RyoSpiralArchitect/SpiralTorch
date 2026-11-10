import copy
import unittest

import qkv_smoothing_study as s


class SmoothingStudyTests(unittest.TestCase):
    def test_only_explicit_coefficient_differs(self):
        old = [dict(prompts=[[1], [2, 3]], gold=[2, 4], microbatch=4, batch_kind="unpadded_sequences")]
        untouched = copy.deepcopy(old)
        self.assertEqual(s.batches(old, "ce"), old)
        new = s.batches(old, "smooth")
        self.assertEqual(new[0].pop("label_smoothing"), .1)
        self.assertEqual(new, old)
        self.assertEqual(old, untouched)
        with self.assertRaises(ValueError):
            s.batches(old, "best")

    def test_lower_nll_with_lost_accuracy_does_not_pass(self):
        cells = {f"lag{k}-{m}": {f"f58-n{n}-g{g}-untrained_pairs": dict(correct=80, count=96, nll=1.)
                  for n in (1, 3) for g in (0, 1, 2)} for k, m in s.POLICY["lanes"]}
        one = next(iter(cells["lag1-smooth"].values()))
        one.update(correct=79, nll=.5)
        self.assertFalse(s.summarize(cells)["efficacy"])
        one["correct"] = 80
        self.assertTrue(s.summarize(cells)["efficacy"])
        self.assertFalse(s.summarize(cells)["competence"])
        cells["lag1-smooth"].pop(next(iter(cells["lag1-smooth"])))
        with self.assertRaises(ValueError):
            s.summarize(cells)


if __name__ == "__main__":
    unittest.main()

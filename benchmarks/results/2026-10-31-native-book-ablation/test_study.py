"""Protocol controls without model weights, optional ML, or training."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    PLAN,
    cases,
    check_ids,
    full_windows,
    midpoint_indices,
    ordering,
    weighted,
)
from prepare import ROMAN, persuasion
from study import contrast


class Protocol(unittest.TestCase):
    def test_train_budget_and_permutations(self):
        self.assertEqual(PLAN["train_windows"] * 63, 32256)
        orders = [ordering(s, 512) for s in PLAN["seeds"]]
        self.assertEqual(len({tuple(x) for x in orders}), 3)
        for order in orders:
            self.assertEqual(sorted(order), list(range(512)))
        self.assertEqual(len(set(cases())), 9)
        self.assertEqual(set(a for _, a in cases()[::3]), set(PLAN["arms"]))

    def test_window_boundaries_and_midpoints(self):
        self.assertEqual(full_windows(list(range(7)), 3), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(midpoint_indices(8, 4), [1, 3, 5, 7])
        with self.assertRaises(ValueError):
            midpoint_indices(7, 8)
        with self.assertRaises(ValueError):
            check_ids({"train": [[0] * 64], "external": [[0] * 64]})

    def test_book_toc_footer_not_evaluated(self):
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK PERSUASION ***\n"
            + "\n".join(f" CHAPTER {r}." for r in ROMAN)
            + "\n"
            + "\n".join(f"CHAPTER {r}.\n\nbody-{i}" for i, r in enumerate(ROMAN))
            + "\nFinis\n*** END OF THE PROJECT GUTENBERG EBOOK PERSUASION ***\nlicense"
        )
        self.assertEqual(persuasion(text.encode()), [f"body-{i}" for i in range(24)])
        with self.assertRaises(ValueError):
            persuasion(text.replace("CHAPTER XII.\n\n", "CHAPTER IX.\n\n").encode())

    def test_paired_contrasts_and_denominators(self):
        def score(n, data_id="same"):
            return {
                "external": {
                    "nll": n,
                    "targets": 10,
                    "chapters": {"1": {"nll": n, "targets": 10, "data_id": data_id}},
                }
            }

        scores = {
            f"{s}-{a}": score(float(i))
            for s in PLAN["seeds"]
            for i, a in enumerate(PLAN["arms"])
        }
        self.assertEqual(contrast(scores, "external", "learned4", "affine")["mean"], 2)
        scores["101-affine"] = score(0, "other")
        with self.assertRaises(ValueError):
            contrast(scores, "external", "learned4", "affine")
        self.assertEqual(weighted([(1, 1), (2, 3)]), 1.75)
        with self.assertRaises(ValueError):
            weighted([(float("nan"), 1)])


if __name__ == "__main__":
    unittest.main()

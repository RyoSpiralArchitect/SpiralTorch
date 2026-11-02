import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from public_scores import check, read


class PublishedReceipts(unittest.TestCase):
    def test_hashes_round_trip_and_tampering_fails_closed(self):
        original, result = read(HERE / "section_scores.json"), read(HERE / "results.json")
        self.assertEqual(check(original, result), 270)
        edited = copy.deepcopy(original)
        edited["nll"]["austen"]["5"][3] += .001
        with self.assertRaises(ValueError):
            check(edited, result)
        reordered = copy.deepcopy(original)
        reordered["order"][3:5] = reversed(reordered["order"][3:5])
        with self.assertRaises(ValueError):
            check(reordered, result)
        altered = copy.deepcopy(result)
        altered["comparison"]["shelley"]["mean_free_minus_affine_by_cohort"]["release"] *= -1
        with self.assertRaises(ValueError):
            check(original, altered)


if __name__ == "__main__":
    unittest.main()

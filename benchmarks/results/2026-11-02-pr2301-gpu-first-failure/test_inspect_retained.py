import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_retained import f32_order, input_occurrences


class InspectionTests(unittest.TestCase):
    def test_count_input_slots_not_the_final_shifted_target(self):
        r = {
            "config": {"steps": 2},
            "train_documents": [[7, 1, 7, 8]],
            "train_batches": [[[0, 0], [0, 0]], [[0, 1]]],
        }
        self.assertEqual(input_occurrences(r, 7), [2, 1])
        self.assertEqual(input_occurrences(r, 8), [0, 0])
        r["train_batches"] = [[[0, 2]]]
        with self.assertRaises(ValueError):
            input_occurrences(r, 7)

    def test_ulp_distance_uses_float32_order_not_decimal_spelling(self):
        self.assertEqual(f32_order(1 + 2**-23) - f32_order(1.0), 1)
        self.assertEqual(f32_order(-1.0) - f32_order(-1.0 - 2**-23), 1)
        self.assertEqual(f32_order(-0.0), f32_order(0.0))

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_trajectories import comparison


class PairedTraceTests(unittest.TestCase):
    def test_one_temporary_divergence_is_not_hidden_by_terminal_equality(self):
        a = [{"post_lora_artifact_id": str(i), "loss": 2.0} for i in range(512)]
        b = [dict(r) for r in a]
        b[6]["post_lora_artifact_id"] = "other"
        b[9]["loss"] = 1.5
        v = comparison(a, b)
        for key, position in (("post_lora", 7), ("loss_f32_le", 10)):
            self.assertEqual(v[key]["different_updates"], 1)
            self.assertEqual(v[key]["first_different_update"], position)
            self.assertNotEqual(
                v[key]["affine_sequence_sha256"], v[key]["wave_sequence_sha256"]
            )
        self.assertIsNone(comparison(a, a)["post_lora"]["first_different_update"])
        with self.assertRaises(ValueError):
            comparison(a, b[:-1])


if __name__ == "__main__":
    unittest.main()

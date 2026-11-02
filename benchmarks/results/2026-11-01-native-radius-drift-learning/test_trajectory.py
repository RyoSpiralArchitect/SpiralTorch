import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory import compare


class Localization(unittest.TestCase):
    def test_null_and_unexpected_changes_remain_visible(self):
        rows = [
            json.dumps(
                {"update": u, "loss": 1.0, "post_geometry_artifact_id": "a"}
            ).encode()
            for u in (1, 2, 3, 4)
        ]
        self.assertIsNone(compare(rows, rows, [0, 0], [2, 4])["first_different_update"])
        changed = list(rows)
        changed[2] = json.dumps(
            {"update": 3, "loss": 1.0, "post_geometry_artifact_id": "b"}
        ).encode()
        out = compare(rows, changed, [0, 2], [2, 4])
        self.assertEqual(out["matched_prior_reports"], 2)
        self.assertEqual(out["first_different_fields"], ["post_geometry_artifact_id"])
        self.assertTrue(out["change_bracketed_by_projection"])
        changed[2] = json.dumps(
            {"update": 3, "loss": 2.0, "post_geometry_artifact_id": "a"}
        ).encode()
        out = compare(rows, changed, [1, 2], [2, 4])
        self.assertEqual(out["first_different_fields"], ["loss"])
        self.assertFalse(out["change_bracketed_by_projection"])
        with self.assertRaises(ValueError):
            compare(rows, changed[:-1], [0, 1], [2, 4])


if __name__ == "__main__":
    unittest.main()

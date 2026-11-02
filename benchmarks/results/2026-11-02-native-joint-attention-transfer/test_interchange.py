import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_interchange import contrasts


class InterchangeTests(unittest.TestCase):
    def test_cancellation_and_interaction_are_not_reported_as_independent_gains(self):
        self.assertEqual(
            contrasts({"AA": 5.0, "AW": 4.0, "WA": 7.0, "WW": 8.0}),
            {"total": 3.0, "lora": 3.0, "geometry": 0.0, "interaction": 2.0},
        )
        self.assertEqual(
            contrasts(dict.fromkeys(("AA", "AW", "WA", "WW"), 4.0)),
            dict.fromkeys(("total", "lora", "geometry", "interaction"), 0.0),
        )


if __name__ == "__main__":
    unittest.main()

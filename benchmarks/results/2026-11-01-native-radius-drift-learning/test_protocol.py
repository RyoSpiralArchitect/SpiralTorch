from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import KEYS, ROMAN, STOPS, chapters, criterion
from audit import audit
from fit import f32, radii


class Protocol(unittest.TestCase):
    @staticmethod
    def frame(s):
        return (
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            + s
            + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"
        )

    def test_ignore_contents_volume_markers_and_darwin_index(self):
        body = " Contents\n CHAPTER I.\n\n"
        for n in (18, 18, 19):
            body += "VOLUME I.\n" + "".join(
                "CHAPTER " + x + "\nAb  cd\n  ef\n" for x in ROMAN[:n]
            )
        self.assertEqual(chapters(self.frame(body), "emma"), ["Ab cd ef"] * 55)
        darwin = (
            "CHAPTER I. NOT A CHAPTER HEADING\n"
            + "".join("CHAPTER " + x + ".\nTest\n" for x in ROMAN[:14])
            + "INDEX.\nprivate back matter"
        )
        self.assertEqual(chapters(self.frame(darwin), "darwin"), ["Test"] * 14)
        with self.assertRaises(ValueError):
            chapters(
                self.frame(darwin.replace("CHAPTER V.\n", "CHAPTER IV.\n")), "darwin"
            )
        with self.assertRaises(ValueError):
            chapters(self.frame(darwin.replace("INDEX.\n", "")), "darwin")

    def test_joint_rule_requires_real_gain_retention_and_an_engaged_bound(self):
        self.assertTrue(criterion(4, 2, 3, 7, 6.5, 1)["joint_pass"])
        for args in (
            (4, 2, 3.1, 7, 6.5, 1),
            (2, 2, 1.9, 7, 6.5, 1),
            (4, 2, 3, 7, 7, 1),
            (4, 2, 3, 7, 6, 0),
        ):
            self.assertFalse(criterion(*args)["joint_pass"])
        with self.assertRaises(ValueError):
            criterion(4, 2, 3, float("nan"), 6, 1)

    def test_float_domain_comparison_uses_rust_f32_storage_not_json_decimal(self):
        record = {
            "placements": [
                {
                    "path": "a",
                    "tensors": [{"name": "log_radius", "values": [1.3862944]}],
                }
            ]
        }
        self.assertNotEqual(f32(1.3862944), 1.3862944)
        self.assertEqual(radii(record)["a"], 1.3862943649291992)

    def test_all_nine_cases_and_every_observed_boundary_are_required(self):
        self.assertEqual(
            KEYS,
            [
                "109-fixed",
                "109-free",
                "109-bounded",
                "113-free",
                "113-bounded",
                "113-fixed",
                "127-bounded",
                "127-fixed",
                "127-free",
            ],
        )
        self.assertEqual(STOPS, [32, 64, 96, 128, 160, 192, 224, 256])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                audit(root)
            (root / "first-error.json").touch()
            with self.assertRaisesRegex(ValueError, "first failure"):
                audit(root)


if __name__ == "__main__":
    unittest.main()

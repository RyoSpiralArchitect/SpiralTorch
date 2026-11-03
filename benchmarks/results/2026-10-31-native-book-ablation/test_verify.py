"""Negative controls for the post-fit audit (no native model or ML imports)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import checked_steps, verify_chapters


class Audit(unittest.TestCase):
    def steps(self):
        return [
            dict(
                update=i + 1,
                epoch=0,
                window_start=2 * i,
                window_end=2 * i + 2,
                microbatches=1,
                target_tokens=126,
                loss=1.0,
                pre_geometry_artifact_id=str(i),
                post_geometry_artifact_id=str(i + 1),
            )
            for i in range(256)
        ]

    def test_complete_history(self):
        checked_steps(self.steps(), 101, "affine", "0", "256")

    def test_incomplete_or_disconnected_history(self):
        for failure in ("short", "resume_gap", "loss", "targets"):
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                rows = self.steps()
                if failure == "short":
                    rows.pop()
                elif failure == "resume_gap":
                    rows[123]["pre_geometry_artifact_id"] = "wrong"
                elif failure == "loss":
                    rows[1]["loss"] = float("inf")
                else:
                    rows[0]["target_tokens"] = 125
                checked_steps(rows, 101, "affine", "0", "256")

    def test_all_declared_chapters(self):
        def split(start, end):
            rows = {
                str(i): dict(targets=504, nll=4.0, data_id=str(i))
                for i in range(start, end)
            }
            return dict(targets=504 * len(rows), nll=4.0, chapters=rows)

        scores = {
            k: dict(external=split(1, 25), development=split(50, 56))
            for k in ("base", "initial")
        }
        verify_chapters(scores)
        scores["initial"]["external"]["chapters"]["2"]["data_id"] = "other"
        with self.assertRaises(ValueError):
            verify_chapters(scores)
        scores["initial"]["external"]["chapters"].pop("2")
        with self.assertRaises(ValueError):
            verify_chapters(scores)


if __name__ == "__main__":
    unittest.main()

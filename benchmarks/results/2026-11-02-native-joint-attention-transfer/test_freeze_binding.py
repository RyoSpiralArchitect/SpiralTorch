import sys
from pathlib import Path
import unittest
import json
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_binding import frozen, match_rows, section_identity
from joint_interchange import returned


class Bindings(unittest.TestCase):
    def test_failed_secondary_worker_retains_its_only_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "result.json"
            cmd = [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).with_name("joint_interchange.py")),
                "--receipt",
                str(receipt),
                "--result-pin",
                "0" * 64,
            ]
            for k in ("out", "data", "old", "model", "wheel"):
                cmd.extend(["--" + k, str(Path(tmp) / ("absent-" + k))])
            self.assertNotEqual(
                subprocess.run(cmd, capture_output=True, timeout=15).returncode, 0
            )
            attempt = Path(tmp) / "result.json.attempt"
            first = (attempt / "claim.json").read_bytes()
            self.assertEqual(json.loads(first)["limits"]["per_worker_seconds"], 2400)
            self.assertTrue((attempt / "worker-start.json").is_file())
            self.assertTrue((attempt / "first-error.json").is_file())
            self.assertNotEqual(
                subprocess.run(cmd, capture_output=True, timeout=15).returncode, 0
            )
            self.assertEqual((attempt / "claim.json").read_bytes(), first)
            self.assertFalse(receipt.exists())

    def test_fixed_chapter_plan_is_shared_but_content_is_not(self):
        a = section_identity([[i] * 64 for i in range(12)])
        b = section_identity([[i + 1] * 64 for i in range(12)])
        self.assertEqual(a["plan_id"], b["plan_id"])
        self.assertNotEqual(a["data_id"], b["data_id"])
        with self.assertRaisesRegex(ValueError, "another chapter"):
            match_rows(a, [b] * 13, 13)
        self.assertEqual(frozen()["sections"]["mansfield"], [5, 13, 21, 29, 37, 45])

    def test_off_diagonal_swap_rejects_even_if_both_diagonals_still_match(self):
        i = {"data_id": "d", "plan_id": "p"}
        names = ["AA", "AW", "WA", "WW"]
        owners = {k: list(k) for k in names}
        rows = [
            {
                **i,
                "lora_artifact_id": k[0],
                "geometry_artifact_id": k[1],
                "nll": float(p),
            }
            for p, k in enumerate(names)
        ]
        self.assertTrue(returned(names, owners, i, rows))
        with self.assertRaisesRegex(ValueError, "returned crossover owner"):
            returned(names, owners, i, [rows[0], rows[2], rows[1], rows[3]])


if __name__ == "__main__":
    unittest.main()

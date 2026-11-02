"""File-only checks, including self-consistent but semantically invalid rewrites."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify as v


class Receipts(unittest.TestCase):
    def test_archive(self):
        report = v.verify(HERE / "receipts")
        self.assertEqual((report["workers"], report["policy_measurements"], report["model_executions_by_verifier"]), (32,96,0))

    def test_resealed_tampering_is_rejected(self):
        for mutation in ("token", "report", "duplicate", "timing", "summary"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in v.FILES:
                    shutil.copyfile(HERE / "receipts" / name, root / name)
                journal, summary = root / "runs.jsonl", root / "result.json"
                rows = [json.loads(line) for line in journal.read_text().splitlines()]
                r = json.loads(summary.read_text())
                if mutation == "token":
                    rows[3]["cases"]["sample32"]["histories"][0][0] ^= 1
                elif mutation == "report":
                    rows[1]["cases"]["plain"]["reports_sha256"] = "0"*64
                elif mutation == "duplicate":
                    rows[3] = rows[2]
                elif mutation == "timing":
                    rows[0]["cases"]["zspace32"]["ns"] += 100
                else:
                    r["ratios"]["plain"]["id_old_over_new"]["median"] = 0.0
                journal.write_text("".join(json.dumps(row)+"\n" for row in rows))
                r["runs_sha256"] = v.sha(journal)
                summary.write_text(json.dumps(r))
                with self.assertRaises(ValueError):
                    v.verify(root)


if __name__ == "__main__":
    unittest.main()

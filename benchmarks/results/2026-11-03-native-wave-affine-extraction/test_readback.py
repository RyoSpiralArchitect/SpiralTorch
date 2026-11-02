"""Adversarial offline checks: rehashing a modified journal must not make it valid."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as v  # noqa: E402


class Readback(unittest.TestCase):
    def test_intact_receipts_need_no_private_data_or_model(self):
        root = v.p.HERE / "receipts"
        self.assertEqual(v.verify(root)["arm_updates"], 864)
        self.assertEqual(v.describe(root)["lower_nll_in_every_order"], {"affine": 41, "wave_fixed": 5})
        self.assertFalse((root / "train.json").exists())
        self.assertFalse(any(root.rglob("*.ck")))

    def check_rehashed(self, filename, mutate, expected):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "receipts"
            shutil.copytree(v.p.HERE / "receipts", root)
            f = root / filename
            mutate(f)
            result = v.read(root / "result.json")
            result["receipt_sha256"][filename] = v.sha(f)
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, expected):
                v.verify(root)

    def test_changed_answer_fails_independent_reduction(self):
        def change(f):
            data = v.read(f)
            score = data["scores"]["answers"][0]["scores"][0]
            score["top1_correct"] = int(score["top1_correct"] == 0)
            f.write_text(json.dumps(data))
        self.check_rehashed("order-113/scores-096.json", change, "top-1 counts")

    def test_swapped_arm_fails_update_lineage(self):
        def change(f):
            lines = f.read_text().splitlines()
            row = json.loads(lines[15])
            rs = row["reports"]
            rs[0]["pre_geometry_artifact_id"] = rs[1]["pre_geometry_artifact_id"]
            lines[15] = json.dumps(row)
            f.write_text("\n".join(lines) + "\n")
        self.check_rehashed("order-109/steps.jsonl", change, "update lineage")

    def test_reordered_keys_fail_even_with_valid_step_count(self):
        def change(f):
            plan = v.read(f)
            plan["keys"].reverse()
            f.write_text(json.dumps(plan))
        self.check_rehashed("order-127/order.json", change, "training permutation")


if __name__ == "__main__":
    unittest.main()

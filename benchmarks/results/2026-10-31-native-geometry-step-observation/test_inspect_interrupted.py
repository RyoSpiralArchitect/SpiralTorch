import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import audit
from inspect_interrupted import (
    inspect_case,
    inventory,
    paired_training_bands,
    retained_rows,
)
from replay import ORIGINAL, PATHS, PLAN, cases


class InterruptedInspection(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ref = [{"update": u} for u in (1, 2, 3)]
        site = dict(
            elements=2 * 63 * 768,
            input_rms=2.0,
            residual_rms=1.0,
            relative_residual_rms=0.5,
            trainable_gradient_l2=1.0,
            pre_log_radius=None,
            log_radius_gradient=None,
        )
        self.lines = [
            json.dumps(
                dict(
                    schema="spiraltorch.native_lm.observed_training_step.v1",
                    base_model_identity=ORIGINAL["base_id"],
                    data_id="train",
                    step=r,
                    sites={p: site for p in PATHS},
                )
            ).encode()
            + b"\n"
            for r in self.ref
        ]

    def read_rows(self, content):
        log = self.root / "observations.jsonl"
        log.write_bytes(content)
        return retained_rows(log, self.ref, "train", "affine")

    def test_strict_prefix_does_not_certify_a_final_checkpoint(self):
        self.assertEqual(self.read_rows(b"".join(self.lines[:2])), 2)
        (self.root / "declaration.json").write_text("{}")
        with self.assertRaises(FileNotFoundError) as raised:
            audit(self.root, self.root)
        self.assertEqual(Path(raised.exception.filename), self.root / "status.json")

    def test_torn_skipped_repeated_or_extra_rows_never_pass(self):
        for content in (
            self.lines[0][:-1],
            self.lines[0] + self.lines[2],
            self.lines[0] * 2,
            b"".join(self.lines) + self.lines[-1],
        ):
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                self.read_rows(content)

    def test_partial_is_distinguished_from_complete_unstarted_and_mismatch(self):
        case = "101-affine"
        args = (self.root, self.root, case, self.ref, {"data_id": "train"}, {})
        row, _ = inspect_case(*args)
        self.assertEqual(row["status"], "unstarted")
        (self.root / (case + ".stderr")).touch()
        with self.assertRaisesRegex(ValueError, "worker output"):
            inspect_case(*args)
        folder = self.root / case
        folder.mkdir()
        log = folder / "observations.jsonl"
        log.write_bytes(self.lines[0])
        chapters = {
            str(i): dict(data_id=str(i), nll=2.0, targets=504) for i in range(50, 56)
        }
        score = dict(chapters=chapters, nll=2.0, targets=3024)
        old = self.root / "run/score"
        old.mkdir(parents=True)
        for name in ("initial", case):
            (old / (name + ".json")).write_text(json.dumps({"development": score}))
        (folder / "score-000.json").write_text(json.dumps(score))
        row, curve = inspect_case(*args)
        self.assertEqual(
            (row["status"], row["decoded_steps"], curve), ("partial", 1, {"0": 2.0})
        )
        self.assertNotIn("matched_checkpoint_id", row)
        (folder / "score-008.json").write_text(json.dumps(score))
        with self.assertRaisesRegex(ValueError, "beyond retained prefix"):
            inspect_case(*args)
        (folder / "score-008.json").unlink()
        (folder / "first-step-mismatch.json").touch()
        with self.assertRaisesRegex(ValueError, "unexpected case evidence"):
            inspect_case(*args)
        (folder / "first-step-mismatch.json").unlink()
        (folder / "result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "missing retained score"):
            inspect_case(*args)

    def test_original_training_comparison_requires_paired_batches(self):
        rows = [
            {
                "update": u,
                "epoch": 0,
                "window_start": 2 * u - 2,
                "window_end": 2 * u,
                "target_tokens": 126,
                "loss": 1.0,
            }
            for u in range(1, PLAN["updates_per_case"] + 1)
        ]
        steps = {
            f"{s}-{a}": [dict(r, loss=r["loss"] + (a == "learned4")) for r in rows]
            for s, a in cases()
        }
        expected = {case: {"data_id": case.split("-")[0]} for case in steps}
        table = paired_training_bands(steps, expected)
        self.assertEqual(table["101"]["9:32"]["learned4-affine"], 1.0)
        expected["103-affine"]["data_id"] = "different"
        with self.assertRaisesRegex(ValueError, "unpaired training data"):
            paired_training_bands(steps, expected)
        expected["103-affine"]["data_id"] = "103"
        steps["103-affine"][1]["window_start"] = 3
        with self.assertRaisesRegex(ValueError, "unpaired training windows"):
            paired_training_bands(steps, expected)

    def test_inventory_rejects_links_and_fingerprints_bytes(self):
        p = self.root / "test.txt"
        p.write_bytes(b"a")
        before = inventory(self.root)
        p.write_bytes(b"b")
        self.assertNotEqual(inventory(self.root), before)
        (self.root / "link").symlink_to(p)
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory(self.root)


if __name__ == "__main__":
    unittest.main()

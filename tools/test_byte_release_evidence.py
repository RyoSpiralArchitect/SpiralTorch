"""Offline evidence retention, without evaluating native payload factories."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_evidence as evidence


class Retention(unittest.TestCase):
    def test_disabled_retention_does_not_evaluate_factory(self):
        evidence.retain("input", lambda: self.fail("disabled factory ran"))

    def test_bytes_survive_unchanged_before_comparison_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "test"
            with self.assertRaises(AssertionError):
                with evidence.recording(root):
                    evidence.retain("report", lambda: {"raw.json": b'{ "value" : 0.00001 }\n'})
                    self.assertEqual((root / "0000-report/raw.json").read_bytes(), b'{ "value" : 0.00001 }\n')
                    raise AssertionError("synthetic numerical failure")
            saved = json.loads((root / "index.json").read_bytes())
            self.assertTrue(saved["events"][0]["complete"])
            self.assertFalse(saved["release_ready"])
            self.assertIsNone(evidence.ACTIVE.get())
            with self.assertRaises(FileExistsError), evidence.recording(root):
                pass

    def test_partial_failure_is_retained_and_scope_is_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "test"
            with self.assertRaises(ValueError), evidence.recording(root):
                evidence.retain("partial", lambda: {"good.bin": b"available", "../bad.bin": b"bad"})
            saved = json.loads((root / "index.json").read_bytes())
            self.assertFalse(saved["events"][0]["complete"])
            self.assertEqual((root / "0000-partial/good.bin").read_bytes(), b"available")
            self.assertIsNone(evidence.ACTIVE.get())

    def test_nested_scope_and_bounded_payloads_reject(self):
        with tempfile.TemporaryDirectory() as temporary, evidence.recording(Path(temporary) / "outer"):
            with self.assertRaises(ValueError), evidence.recording(Path(temporary) / "inner"):
                pass
            with patch.object(evidence, "MAX_TOTAL", 2), self.assertRaises(ValueError):
                evidence.retain("large", lambda: {"raw.bin": b"123"})
            with self.assertRaises(ValueError):
                evidence.retain("../label", lambda: {"raw.bin": b"1"})


if __name__ == "__main__":
    unittest.main()

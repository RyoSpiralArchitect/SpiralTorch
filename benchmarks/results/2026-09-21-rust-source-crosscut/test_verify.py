"""Tamper and condition-grid rejection checks against the published fixture."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("archive_verify", ROOT / "verify.py")
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns("__pycache__"))

    def mutate(self, update):
        path = self.root / "candidate-a.json"
        doc = json.loads(path.read_text())
        update(doc)
        path.write_text(json.dumps(doc))

    def test_real_archive(self):
        self.assertEqual(verification.verify(self.root)["candidate_numerical_passes"], 72)

    def test_modified_bytes_rejected(self):
        self.mutate(lambda doc: doc.update(label="tampered"))
        with self.assertRaises(ValueError):
            verification.verify(self.root)

    def test_duplicate_condition_rejected(self):
        self.mutate(lambda doc: doc["cases"].__setitem__(0, doc["cases"][1]))
        with self.assertRaises(AssertionError):
            verification.load_measure(self.root).report(self.root)

    def test_invalid_timing_rejected(self):
        self.mutate(lambda doc: doc["cases"][0]["measurement"]["elapsed_ns"].__setitem__(0, float("nan")))
        with self.assertRaises(AssertionError):
            verification.load_measure(self.root).report(self.root)


if __name__ == "__main__":
    unittest.main()

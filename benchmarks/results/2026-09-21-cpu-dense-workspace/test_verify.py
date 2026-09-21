import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("archive_verifier", ROOT / "verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"
        shutil.copytree(ROOT, self.root)

    def modify(self, name, change):
        path = self.root / name
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document))
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        data = path.read_bytes()
        manifest["files"][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest_path.write_text(json.dumps(manifest))

    def test_original(self):
        self.assertEqual(verifier.verify(self.root)["dense_bitwise_condition_runs"], 240)

    def test_modified_bytes(self):
        (self.root / "dense/serial-candidate-a.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Hash/size"):
            verifier.verify(self.root)

    def test_nan_timing(self):
        self.modify("dense/serial-candidate-a.json", lambda d: d["cases"][0]["elapsed_ns"].__setitem__(0, float("nan")))
        with self.assertRaisesRegex(ValueError, "timing"):
            verifier.verify(self.root)

    def test_duplicate_condition(self):
        self.modify("dense/serial-candidate-a.json", lambda d: d["cases"].__setitem__(1, d["cases"][0]))
        with self.assertRaisesRegex(ValueError, "conditions"):
            verifier.verify(self.root)

    def test_missing_verification(self):
        self.modify("verification/receipt.json", lambda d: d["steps"].pop())
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verifier.verify(self.root)

if __name__ == "__main__":
    unittest.main()

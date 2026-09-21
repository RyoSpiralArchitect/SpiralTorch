"""Tamper checks for the published-byte/aggregation verifier, not GPU tests."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PublicationVerificationTests(unittest.TestCase):
    def test_tampering_is_rejected(self):
        root = Path(__file__).resolve().parent
        changes = {
            "missing-group": ("summary.json", lambda d: d["groups"].pop()),
            "duplicate-group": ("summary.json", lambda d: d["groups"].__setitem__(0, d["groups"][1])),
            "recipe-count": ("summary.json", lambda d: d.__setitem__("recipes", 53)),
            "source": ("summary.json", lambda d: d["source"].__setitem__("commit", "0" * 40)),
            "pair": ("paired_fingerprints.json", lambda d: d[0].__setitem__("paired_fingerprints_equal", False)),
            "route": ("intervals.json", lambda d: d["cases"][0]["samples"][0]["pointwise_cotangent_routes"].__setitem__("candidate", "materialized")),
            "timing": ("intervals.json", lambda d: d["cases"][0]["samples"][2]["times_ms"].__setitem__("candidate", 0)),
            "raw-size": ("raw-inventory.json", lambda d: d.__setitem__("bytes", 0)),
        }
        with tempfile.TemporaryDirectory(prefix="pointwise-publication-test-") as tmp:
            target = Path(tmp) / "archive"
            shutil.copytree(root, target)
            for name, (filename, change) in changes.items():
                with self.subTest(name=name):
                    path = target / filename
                    original = path.read_bytes()
                    data = json.loads(original)
                    change(data)
                    path.write_text(json.dumps(data, indent=2) + "\n")
                    manifest = json.loads((root / "manifest.json").read_bytes())
                    for row in manifest["files"]:
                        if row["path"] == filename:
                            row.update(bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    # Rebind the changed bytes so this tests semantics, not only fixity.
                    (target / "manifest.json").write_text(json.dumps(manifest))
                    result = subprocess.run([sys.executable, "-S", target / "verify.py"],
                        capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("AssertionError", result.stderr)
                    path.write_bytes(original)
            shutil.copyfile(root / "manifest.json", target / "manifest.json")
            extra = target / "unmanifested.txt"
            extra.write_text("not part of the archive\n")
            result = subprocess.run([sys.executable, "-S", target / "verify.py"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("AssertionError", result.stderr)
        result = subprocess.run([sys.executable, "-S", root / "verify.py"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-O", "-S", root / "verify.py"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run without -O", result.stderr)


if __name__ == "__main__":
    unittest.main()

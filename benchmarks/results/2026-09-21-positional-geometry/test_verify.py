"""Deliberately corrupt compact evidence, including after rehashing the archive."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path(__file__).parent
spec = importlib.util.spec_from_file_location("positional_verifier", ROOT / "verify.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "evidence"
        shutil.copytree(ROOT, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def corrupt(self, name, change, rehash=True):
        path = self.root / name
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))
        if rehash:
            subprocess.run([sys.executable, "-B", "-I", str(self.root / "reproduction/manifest.py"),
                            str(self.root)], check=True)
        with self.assertRaises(ValueError):
            mod.verify(self.root)

    def test_valid(self):
        self.assertTrue(mod.verify(self.root)["ok"])

    def test_changed_bytes(self):
        self.corrupt("native/candidate-a.json", lambda r: r.update(warmups=4), rehash=False)

    def test_incomplete_grid(self):
        self.corrupt("native/candidate-a.json", lambda r: r["cases"].pop())

    def test_missing_negative_control(self):
        self.corrupt("wasm/baseline-a.json", lambda r: r["contracts"].update(rope_history_independent=True))

    def test_failed_gradient(self):
        self.corrupt("wasm/candidate-a.json", lambda r: r["contracts"]["fields"][0].update(gradient_equal=False))

    def test_invalid_timing(self):
        self.corrupt("native/candidate-a.json", lambda r: r["cases"][0]["elapsed_ns"].__setitem__(0, -1))

    def test_wrong_source(self):
        self.corrupt("candidate-native-build/receipt.json", lambda r: r["source"].update(commit="0"*40))

    def test_incomplete_validation(self):
        self.corrupt("verification/stages.json", lambda r: r.pop())

    def test_wrong_worker(self):
        self.corrupt("wasm/receipt.json", lambda r: r["sha256"].update(candidate_wasm="0"*64))

    def test_stale_summary(self):
        self.corrupt("native/comparison.json", lambda r: r["comparisons"][0].update(ratio=99))

    def test_lost_failure(self):
        self.corrupt("positive-density-preflight/receipt.json", lambda r: r.update(exit_code=0))

if __name__ == "__main__":
    unittest.main()

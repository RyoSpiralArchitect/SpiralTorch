import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("layout_archive", ROOT / "verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "archive"
        shutil.copytree(ROOT, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, name, mutate):
        path = self.root / name
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload) + "\n")
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest) + "\n")

    def rejects(self):
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            verifier.verify(self.root)

    def test_original(self):
        self.assertEqual(verifier.verify(self.root)["status"], "passed")

    def test_byte_corruption(self):
        with (self.root / "native/candidate-a.json").open("ab") as out:
            out.write(b" ")
        self.rejects()

    def test_nonfinite_timing_even_with_new_file_hash(self):
        self.change("native/candidate-a.json", lambda payload: payload["cases"][0]["elapsed_ns"].__setitem__(0, float("nan")))
        self.rejects()

    def test_duplicate_condition(self):
        self.change("native/candidate-a.json", lambda payload: payload["cases"].__setitem__(1, payload["cases"][0]))
        self.rejects()

    def test_wasm_output_mismatch(self):
        self.change("wasm/candidate-a.json", lambda payload: payload["cases"][0].__setitem__("output_sha256", "0" * 64))
        self.rejects()

    def test_missing_validation_stage(self):
        self.change("verification/receipt.json", lambda payload: payload["steps"].pop())
        self.rejects()

    def test_forged_source(self):
        self.change("verification/receipt.json", lambda payload: payload["source_hashes"].__setitem__("crates/st-tensor/src/pure.rs", "0" * 64))
        self.rejects()

    def test_extra_nested_manifest(self):
        (self.root / "native/manifest.json").write_text("{}\n")
        self.rejects()

if __name__ == "__main__":
    unittest.main()

"""Positive and tamper controls for the publication verifier."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gelu_verify", HERE / "verify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "archive"
        shutil.copytree(HERE, self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def change(self, relative, mutate):
        path = self.root / relative
        data = json.loads(path.read_bytes())
        mutate(data)
        path.write_text(json.dumps(data, indent=2) + "\n")
        manifest = json.loads((self.root / "manifest.json").read_bytes())
        for row in manifest["files"]:
            if row["path"] == relative:
                row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def rejected(self):
        with self.assertRaises(AssertionError):
            module.verify(self.root)

    def test_positive(self):
        self.assertEqual(module.verify(self.root)["status"], "passed")

    def test_raw_bytes(self):
        with (self.root / "provenance.json").open("ab") as stream:
            stream.write(b" ")
        self.rejected()

    def test_missing_case(self):
        self.change("gelu/candidate-a.json", lambda data: data["cases"].pop())
        self.rejected()

    def test_wrong_output_hash(self):
        self.change("gelu/candidate-a.json", lambda data: data["cases"][0].update(output_sha256="0" * 64))
        self.rejected()

    def test_missing_live_probe(self):
        self.change("verification-final/wgpu-layout/receipt.json", lambda data: data["command"].remove("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"))
        self.rejected()

    def test_wrong_runtime_source(self):
        self.change("verification-final/nn/receipt.json", lambda data: data["source"]["files"].update({module.RUNTIME[0]: "0" * 64}))
        self.rejected()

    def test_false_summary(self):
        self.change("summary.json", lambda data: data["gelu"]["groups"]["False"].update(geomean=100))
        self.rejected()

    def test_unlisted_file(self):
        (self.root / ".unlisted").write_bytes(b"unexpected")
        self.rejected()


if __name__ == "__main__":
    unittest.main()

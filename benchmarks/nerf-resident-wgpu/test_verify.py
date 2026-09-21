"""Archive regressions must not create bytecode inside the hashed evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("nerf_verify", Path(__file__).with_name("verify.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
ARCHIVE = Path(__file__).resolve().parents[1] / "results/2026-09-21-nerf-resident-wgpu"


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"
        shutil.copytree(ARCHIVE, self.root)

    def edit(self, name, update):
        path = self.root / name
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))
        manifest = self.root / "manifest.json"
        hashes = json.loads(manifest.read_text())
        hashes[name] = verifier.sha(path)
        manifest.write_text(json.dumps(hashes))

    def rejects(self):
        with self.assertRaises(AssertionError):
            verifier.verify(self.root)

    def test_archive(self):
        self.assertGreater(verifier.verify(self.root), 10)

    def test_extra_file_and_byte_mutation(self):
        (self.root / "unexpected.pyc").write_bytes(b"not evidence")
        self.rejects()
        (self.root / "unexpected.pyc").unlink()
        (self.root / "backend-tests.log").write_text("changed")
        self.rejects()

    def test_duplicate_condition(self):
        self.edit("results.json", lambda r: r["cases"].__setitem__(0, r["cases"][1]))
        self.rejects()

    def test_wrong_guard_name(self):
        def change(r):
            guards = r["guards"]["native"]
            guards["substitute"] = guards.pop("retained_version")
        self.edit("results.json", change)
        self.rejects()

    def test_inconsistent_aggregate(self):
        self.edit("results.json", lambda r: r["max_abs_errors"].__setitem__("browser", 0.))
        self.rejects()

    def test_missing_validation(self):
        self.edit("validation.json", lambda r: r.pop())
        self.rejects()

    def test_dirty_measured_source(self):
        self.edit("source.json", lambda r: r.__setitem__("status", " M source.rs"))
        self.rejects()

    def test_review_result_drift(self):
        self.edit("review-legacy/results.json",
                  lambda r: r["cases"][0].__setitem__("input_sha256", "0" * 64))
        self.rejects()

    def test_missing_review_validation(self):
        self.edit("review-legacy/validation.json", lambda r: r.pop())
        self.rejects()

    def test_dirty_review_source(self):
        self.edit("review-legacy/source.json",
                  lambda r: r.__setitem__("status", " M shader.wgsl"))
        self.rejects()

    def test_fabricated_review_failure(self):
        self.edit("review-legacy/negative-attempts.json",
                  lambda r: r[0]["receipt"].__setitem__("exit_code", 0))
        self.rejects()


if __name__ == "__main__":
    unittest.main()

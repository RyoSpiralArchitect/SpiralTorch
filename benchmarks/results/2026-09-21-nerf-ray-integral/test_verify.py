import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("archive_verify", Path(__file__).with_name("verify.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"
        shutil.copytree(Path(__file__).parent, self.root)

    def rehash(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"files": {str(p.relative_to(self.root)): module.sha(p)
            for p in self.root.rglob("*") if p.is_file() and p.name != "manifest.json"}}))

    def mutate(self, name, fn):
        path = self.root / name
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))
        self.rehash()
        with self.assertRaises((AssertionError, KeyError, IndexError)):
            module.verify(self.root)

    def test_untouched_archive(self):
        module.verify(self.root)

    def test_byte_tampering(self):
        path = self.root / "summary.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaises(AssertionError):
            module.verify(self.root)

    def test_missing_worker(self):
        self.mutate("results.json", lambda x: x["results"].pop())

    def test_duplicate_worker(self):
        self.mutate("results.json", lambda x: x["results"].__setitem__(1, x["results"][0]))

    def test_changed_aggregate(self):
        self.mutate("summary.json", lambda x: x["aggregate"]["native"].__setitem__("geomean", 12345))

    def test_missing_condition(self):
        self.mutate("results.json", lambda x: x["results"][0]["cases"].pop())

    def test_failed_stage(self):
        self.mutate("validation/stages.json", lambda x: x[0].__setitem__("exit_code", 1))

    def test_changed_source_identity(self):
        self.mutate("source.json", lambda x: x.__setitem__("commit", "0" * 40))

    def test_missing_preconditioning_condition(self):
        self.mutate("preflight.json", lambda x: x[0]["cases"].pop())

    def test_changed_error_summary(self):
        self.mutate("summary.json", lambda x: x.__setitem__("max_render_abs_vs_native", 0.1))

    def test_changed_input_fingerprint(self):
        self.mutate("results.json", lambda x: x["results"][0]["cases"][0].__setitem__("input_metadata_sha256", "0" * 64))


if __name__ == "__main__":
    unittest.main()

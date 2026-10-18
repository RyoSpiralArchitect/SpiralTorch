import json
from pathlib import Path
import runpy
import tempfile
import unittest

HERE = Path(__file__).parent
PREPARE = runpy.run_path(str(HERE / "prepare.py"))
PREFLIGHT = runpy.run_path(str(HERE / "preflight.py"))
FIXTURE = runpy.run_path(str(HERE / "test_prepare.py"))["fixture"]


class InputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        raw = FIXTURE()
        (self.root / "book.txt").write_bytes(raw)
        plan = json.loads((HERE / "plan.json").read_text())
        plan["source"]["raw_sha256"] = PREPARE["sha"](raw)
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps(plan))
        self.data = self.root / "chapters"
        self.manifest = PREPARE["prepare"](self.root / "book.txt", self.plan, self.data)
        self.pin = self.manifest["plan_id"]

    def call(self):
        return PREFLIGHT["inputs"](self.plan, self.pin, self.data)

    def test_validation_captures_text_without_opening_reserve(self):
        for path in (self.data / "reserve").glob("*.txt"):
            path.unlink()
        _, _, paths, texts, manifest_id = self.call()
        self.assertEqual([len(paths[k]) for k in ("train", "development")], [49, 6])
        self.assertNotIn("reserve", paths)
        self.assertNotIn("reserve", texts)
        self.assertEqual(manifest_id, PREFLIGHT["identity"]((self.data / "manifest.json").read_bytes()))
        captured = texts["train"][0]
        paths["train"][0].write_text("changed after capture")
        self.assertEqual(captured, texts["train"][0])
        with self.assertRaises(ValueError):
            self.call()

    def test_changed_plan_or_manifest_path_rejects(self):
        with self.assertRaises(ValueError):
            PREFLIGHT["inputs"](self.plan, "sha256:" + "0" * 64, self.data)
        self.manifest["documents"][0]["relative_path"] = "../outside.txt"
        (self.data / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self.call()

    def test_duplicate_or_boolean_chapters_reject(self):
        for bad in (True, 2):
            manifest = json.loads(json.dumps(self.manifest))
            manifest["documents"][0]["chapter"] = bad
            (self.data / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.call()


if __name__ == "__main__":
    unittest.main()

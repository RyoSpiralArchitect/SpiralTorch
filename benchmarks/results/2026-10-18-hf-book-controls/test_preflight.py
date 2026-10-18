import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

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
        self.manifest_pin = PREFLIGHT["identity"]((self.data / "manifest.json").read_bytes())

    def call(self):
        return PREFLIGHT["inputs"](self.plan, self.pin, self.data, self.manifest_pin)

    def replace_manifest(self, manifest):
        (self.data / "manifest.json").write_text(json.dumps(manifest))
        self.manifest_pin = PREFLIGHT["identity"]((self.data / "manifest.json").read_bytes())

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
            PREFLIGHT["inputs"](self.plan, "sha256:" + "0" * 64, self.data, self.manifest_pin)
        self.manifest["documents"][0]["relative_path"] = "../outside.txt"
        self.replace_manifest(self.manifest)
        with self.assertRaises(ValueError):
            self.call()

    def test_duplicate_or_boolean_chapters_reject(self):
        for bad in (True, 2):
            manifest = json.loads(json.dumps(self.manifest))
            manifest["documents"][0]["chapter"] = bad
            self.replace_manifest(manifest)
            with self.assertRaises(ValueError):
                self.call()

    def test_replaced_chapter_and_manifest_reject_without_new_pin(self):
        raw = (self.data / "train/chapter-01.txt").read_bytes()
        (self.data / "development/chapter-50.txt").write_bytes(raw)
        self.manifest["documents"][49].update(bytes=len(raw), sha256=PREPARE["sha"](raw))
        (self.data / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "separately pinned manifest"):
            self.call()

    def test_symlinked_split_rejects_before_opening_reserve(self):
        (self.data / "train").rename(self.data / "original-train")
        (self.data / "train").symlink_to(self.data / "reserve", target_is_directory=True)
        original_open = Path.open
        def no_reserve_open(path, *args, **kwargs):
            if (self.data / "reserve") in path.resolve().parents:
                self.fail("reserve was opened through a symbolic link")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", no_reserve_open):
            with self.assertRaisesRegex(ValueError, "canonical split"):
                self.call()

    def test_symlinked_manifest_rejects(self):
        path = self.data / "manifest.json"
        path.rename(self.data / "original-manifest.json")
        path.symlink_to(self.data / "original-manifest.json")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.call()

    def test_package_identity_detects_source_native_and_inventory_changes(self):
        package = self.root / "package"
        package.mkdir()
        (package / "core.py").write_bytes(b"first")
        (package / "native.so").write_bytes(b"native")
        fingerprint = PREFLIGHT["package_identity"]
        initial = fingerprint(package)
        self.assertEqual(set(initial), {"core.py", "native.so"})
        (package / "native.so").write_bytes(b"changed")
        self.assertNotEqual(fingerprint(package), initial)
        (package / "native.so").write_bytes(b"native")
        self.assertEqual(fingerprint(package), initial)
        (package / "new.py").write_bytes(b"added")
        self.assertNotEqual(fingerprint(package), initial)
        (package / "new.py").unlink()
        (package / "core.py").write_bytes(b"changed")
        self.assertNotEqual(fingerprint(package), initial)


if __name__ == "__main__":
    unittest.main()

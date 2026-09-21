"""Evidence mutation gates; generated fixtures are not measured results."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive
from test_protocol import source
from test_contract import fixture


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        (self.source_root / "source.rs").write_text("// fixture\n")
        identity = {"commit": "0" * 40, "status": "",
                    "files": {"source.rs": archive.sha(self.source_root / "source.rs")}}
        for stage in archive.STAGES:
            folder = self.raw / "accepted" / stage
            folder.mkdir(parents=True)
            receipt = {"source": identity, "exit_code": 0, "source_unchanged": True,
                       "command": ["unit-test-fixture"], "seconds": 0.01, "environment": {}}
            archive.write(folder / "receipt.json", receipt)
            (folder / "stdout.log").write_text("synthetic test\n")
            (folder / "stderr.log").write_text("")
        attempt = self.raw / "rejected-preflight"
        attempt.mkdir()
        archive.write(attempt / "receipt.json", {**receipt, "exit_code": 42})
        (attempt / "stderr.log").write_text("synthetic rejection")
        for r in range(3):
            for family in ("native", "browser", "torch"):
                report = fixture("torch") if family == "torch" else source()
                for root, stem in [
                    (self.raw / "accepted", f"round-{r}-{family}"),
                    (self.raw, f"screen-{family}-{r + 1}"),
                ]:
                    path = root / (stem + ".json" if family == "browser" else stem + "/stdout.log")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    archive.write(path, report)
        self.output = self.root / "published"

    def publish(self):
        archive.publish(self.raw, self.output)

    def seal(self):
        archive.write(self.output / "manifest.json", archive.files(self.output))

    def test_publish_verify_and_no_overwrite(self):
        self.publish()
        checked = archive.verify(self.output, self.raw, self.source_root)
        self.assertFalse(checked["numerical_reexecution"])
        earlier = archive.read(self.output / "exploration.json")
        self.assertEqual(earlier["attempts"][0]["failed_stderr"], "synthetic rejection")
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_dirty_failed_and_changed_source_are_rejected(self):
        path = self.raw / "accepted" / "native-clippy" / "receipt.json"
        original = archive.read(path)
        for kind in ("dirty", "failed", "changed", "identity"):
            value = copy.deepcopy(original)
            if kind == "dirty":
                value["source"]["status"] = " M source.rs"
            elif kind == "failed":
                value["exit_code"] = 1
            elif kind == "changed":
                value["source_unchanged"] = False
            else:
                value["source"]["commit"] = "1" * 40
            archive.write(path, value)
            with self.assertRaises(ValueError):
                self.publish()
            self.assertFalse(self.output.exists())

    def test_archive_and_raw_source_mutations_are_rejected(self):
        self.publish()
        path = self.output / "results.json"
        original = path.read_text()
        path.write_text(original + " ")
        with self.assertRaises(ValueError):
            archive.verify(self.output)
        path.write_text(original)
        raw = self.raw / "screen-native-1" / "stdout.log"
        original_raw = raw.read_text()
        raw.write_text(original_raw + " ")
        with self.assertRaises(ValueError):
            archive.verify(self.output, self.raw)
        raw.write_text(original_raw)
        (self.source_root / "source.rs").write_text("// changed\n")
        with self.assertRaises(ValueError):
            archive.verify(self.output, source=self.source_root)

    def test_resealed_incomplete_receipts_and_timing_summary_are_rejected(self):
        self.publish()
        path = self.output / "validation.json"
        receipts = archive.read(path)
        archive.write(path, receipts[:-1])
        self.seal()
        with self.assertRaises(ValueError):
            archive.verify(self.output)
        archive.write(path, receipts)
        result = archive.read(self.output / "results.json")
        result["cases"][0]["summary"][0]["median_interval_ms"]["native_single"] = 99
        archive.write(self.output / "results.json", result)
        self.seal()
        with self.assertRaises(ValueError):
            archive.verify(self.output)

    def test_absolute_and_parent_paths_are_rejected(self):
        for name in ("", "/absolute", "../escape", "child/../../escape"):
            with self.assertRaises(ValueError):
                archive.safe_path(self.raw, name)


if __name__ == "__main__":
    unittest.main()

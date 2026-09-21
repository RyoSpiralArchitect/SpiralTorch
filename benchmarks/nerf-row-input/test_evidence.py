"""Exercise the shared archiver with the distinct row-input protocol."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence
from test_compare import source
from test_contract import fixture

shared = evidence.shared


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "published"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "source.rs").write_text("// synthetic fixture\n")
        identity = {"commit": "0" * 40, "status": "",
                    "files": {"source.rs": shared.sha(self.source / "source.rs")}}
        for stage in evidence.STAGES:
            folder = self.raw / "accepted" / stage
            folder.mkdir(parents=True)
            receipt = {"source": identity, "exit_code": 0, "source_unchanged": True,
                       "command": ["test-fixture"], "seconds": 0.01, "environment": {}}
            shared.write(folder / "receipt.json", receipt)
            (folder / "stdout.log").write_text("synthetic test\n")
            (folder / "stderr.log").write_text("")
        failed = self.raw / "rejected"
        failed.mkdir()
        shared.write(failed / "receipt.json", {**receipt, "exit_code": 42})
        (failed / "stderr.log").write_text("synthetic rejected attempt")
        for r in range(3):
            for family in ("native", "browser", "torch"):
                report = fixture("torch") if family == "torch" else source()
                for root, stem in [(self.raw / "accepted", f"round-{r}-{family}"),
                                   (self.raw, f"screen-{family}-{r + 1}")]:
                    path = root / (stem + ".json" if family == "browser" else stem + "/stdout.log")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shared.write(path, report)

    def test_roundtrip_is_scoped_and_retains_failed_attempts(self):
        evidence.publish(self.raw, self.output)
        self.assertFalse(evidence.verify(self.output, self.raw, self.source)["numerical_reexecution"])
        self.assertEqual(shared.read(self.output / "exploration.json")["attempts"][0]["failed_stderr"],
                         "synthetic rejected attempt")
        with self.assertRaises(ValueError):
            shared.verify(self.output)  # A row archive is not a submission archive.
        with self.assertRaises(FileExistsError):
            evidence.publish(self.raw, self.output)

    def test_dirty_source_and_resealed_missing_stage_are_rejected(self):
        path = self.raw / "accepted" / "contracts" / "receipt.json"
        original = shared.read(path)
        bad = {**original, "source": {**original["source"], "status": " M source.rs"}}
        shared.write(path, bad)
        with self.assertRaises(ValueError):
            evidence.publish(self.raw, self.output)
        self.assertFalse(self.output.exists())
        shared.write(path, original)
        evidence.publish(self.raw, self.output)
        path = self.output / "validation.json"
        shared.write(path, shared.read(path)[:-1])
        shared.write(self.output / "manifest.json", shared.files(self.output))
        with self.assertRaises(ValueError):
            evidence.verify(self.output)


if __name__ == "__main__":
    unittest.main()

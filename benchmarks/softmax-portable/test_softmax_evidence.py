"""Small synthetic fixtures exercise the existing archive with the new contract."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softmax_evidence as e
from test_softmax_protocol import fixture, TEST_KEYS


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(e.protocol, "KEYS", TEST_KEYS).start()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.output = self.root / "raw", self.root / "output"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "fixture").write_text("synthetic")
        identity = dict(commit="0" * 40, status="", files={"fixture": e.shared.sha(self.source / "fixture")})
        for stage in e.STAGES:
            folder = self.raw / "accepted" / stage
            folder.mkdir(parents=True)
            receipt = dict(source=identity, exit_code=0, source_unchanged=True, command=["fixture"])
            e.shared.write(folder / "receipt.json", receipt)
            (folder / "stdout.log").write_text("synthetic")
            (folder / "stderr.log").write_text("")
        failed = self.raw / "failed"
        failed.mkdir()
        e.shared.write(failed / "receipt.json", {**receipt, "exit_code": 42})
        (failed / "stderr.log").write_text("synthetic failed attempt")
        for r in range(3):
            for family in ("native", "browser", "torch"):
                report = fixture(family)
                for root, stem in [(self.raw / "accepted", f"round-{r}-{family}"),
                                   (self.raw, f"screen-{family}-{r+1}")]:
                    path = root / (stem + ".json" if family == "browser" else stem + "/stdout.log")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    e.shared.write(path, report)

    def test_roundtrip_keeps_failed_attempts_and_raw_fixity(self):
        e.publish(self.raw, self.output)
        self.assertFalse(e.verify(self.output, self.raw, self.source)["numerical_reexecution"])
        self.assertEqual(e.shared.read(self.output / "exploration.json")["attempts"][0]["failed_stderr"],
                         "synthetic failed attempt")
        with self.assertRaises(FileExistsError):
            e.publish(self.raw, self.output)
        with self.assertRaises(ValueError):
            e.shared.verify(self.output)
        (self.raw / "screen-native-1/stdout.log").write_text("changed")
        with self.assertRaises(ValueError):
            e.verify(self.output, self.raw)

    def test_dirty_source_missing_stage_and_tamper_rejected(self):
        path = self.raw / "accepted/backend-tests/receipt.json"
        receipt = e.shared.read(path)
        e.shared.write(path, {**receipt, "source": {**receipt["source"], "status": " M fixture"}})
        with self.assertRaises(ValueError):
            e.publish(self.raw, self.output)
        e.shared.write(path, receipt)
        e.publish(self.raw, self.output)
        path = self.output / "validation.json"
        e.shared.write(path, e.shared.read(path)[:-1])
        e.shared.write(self.output / "manifest.json", e.shared.files(self.output))
        with self.assertRaises(ValueError):
            e.verify(self.output)


if __name__ == "__main__":
    unittest.main()

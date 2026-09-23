from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evidence as e
from test_protocol_gelu import fixture, TEST_KEYS


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(e.protocol,"KEYS",TEST_KEYS).start()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.raw,self.output,self.source = root/"raw",root/"public",root/"source"
        self.source.mkdir()
        (self.source/"fixture").write_text("synthetic")
        self.identity = dict(commit="0"*40,status="",files={"fixture":e.shared.sha(self.source/"fixture")})
        receipt = dict(source=self.identity,exit_code=0,source_unchanged=True,command=["fixture"])
        for stage in e.STAGES:
            folder = self.raw/"accepted"/stage
            folder.mkdir(parents=True)
            e.shared.write(folder/"receipt.json",receipt)
            (folder/"stdout.log").write_text("synthetic")
            (folder/"stderr.log").write_text("")
        failed = self.raw/"failed"
        failed.mkdir()
        e.shared.write(failed/"receipt.json",{**receipt,"exit_code":42})
        (failed/"stderr.log").write_text("synthetic failure retained")
        for rnd in range(3):
            for family in ("native","browser","torch"):
                for root,stem in [(self.raw/"accepted",f"round-{rnd}-{family}"),(self.raw,f"screen-{family}-{rnd+1}")]:
                    path = root/(stem+".json" if family=="browser" else stem+"/stdout.log")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    e.shared.write(path,fixture(family))

    def test_roundtrip_failure_retention_and_tampering(self):
        e.publish(self.raw,self.output)
        self.assertTrue(e.verify(self.output,self.raw,self.source)["source_checked"])
        self.assertEqual(e.shared.read(self.output/"exploration.json")["attempts"][0]["failed_stderr"],"synthetic failure retained")
        with self.assertRaises(FileExistsError):
            e.publish(self.raw,self.output)
        with self.assertRaises(ValueError):
            e.shared.verify(self.output)
        (self.raw/"screen-native-1/stdout.log").write_text("tampered")
        with self.assertRaises(ValueError):
            e.verify(self.output,self.raw)

    def test_clean_source_and_complete_validation_required(self):
        path = self.raw/"accepted"/"backend-tests/receipt.json"
        receipt = e.shared.read(path)
        e.shared.write(path,{**receipt,"source":{**self.identity,"status":" M fixture"}})
        with self.assertRaises(ValueError):
            e.publish(self.raw,self.output)
        e.shared.write(path,receipt)
        e.publish(self.raw,self.output)
        e.shared.write(self.output/"validation.json",e.shared.read(self.output/"validation.json")[:-1])
        e.shared.write(self.output/"manifest.json",e.shared.files(self.output))
        with self.assertRaises(ValueError):
            e.verify(self.output)


if __name__ == "__main__":
    unittest.main()

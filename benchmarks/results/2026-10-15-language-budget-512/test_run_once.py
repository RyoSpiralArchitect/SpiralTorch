import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("run_once",Path(__file__).with_name("run_once.py"))
RUN=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ClaimTests(unittest.TestCase):
    def test_repeat_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            RUN.claim(path,"request","runner")
            with self.assertRaises(FileExistsError):
                RUN.claim(path,"request","runner")
            self.assertTrue(json.loads((path/"native-run-claim.json").read_text())["before_gpu_launch"])

    def test_existing_output_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            (path/"native-128.json").touch()
            with self.assertRaisesRegex(ValueError,"no training started"):
                RUN.claim(path,"request","runner")
            with self.assertRaises(FileExistsError):
                RUN.claim(path,"request","runner")

    def test_adoption_is_not_a_prelaunch_claim(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            RUN.claim(path,"request","runner",123)
            state=json.loads((path/"native-run-claim.json").read_text())
            self.assertFalse(state["before_gpu_launch"])
            self.assertEqual(state["adopted_existing_pid"],123)


if __name__=="__main__":
    unittest.main()

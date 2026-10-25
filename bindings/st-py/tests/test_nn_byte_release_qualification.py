"""Portable recipe admission; long real-GPU execution has a second explicit opt-in."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import byte_release_long_resume as long_resume
import spiraltorch as st


class LongResumePortable(unittest.TestCase):
    def test_exact_recipe_is_admitted_without_selecting_a_runtime(self):
        raw = long_resume.request_bytes()
        study = st.nn.ByteCorpusStudy.from_json(raw)
        self.assertEqual(study.total_updates, 128)
        self.assertEqual(study.request_sha256, long_resume.protocol.digest(raw))
        self.assertEqual(study.design_schema, "spiraltorch.byte_corpus.request.v2")
        self.assertLessEqual(study.checkpoint_size_bound(), long_resume.MAX_FILE_BYTES)
        for cursor in (*long_resume.SPLITS, long_resume.UPDATES):
            study.validate_segment(cursor)
        with self.assertRaises(ValueError):
            study.validate_segment(129)


@unittest.skipUnless(
    os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1"
    and os.environ.get("SPIRALTORCH_RUN_BYTE_RELEASE_QUALIFICATION") == "1",
    "explicit artifact-qualified long GPU study required",
)
class LongResumeGpu(unittest.TestCase):
    def test_128_updates_and_two_saved_cursors(self):
        directory = os.environ.get("SPIRALTORCH_BYTE_RELEASE_OUTPUT")
        if not directory:
            self.fail("fresh retained output directory is required")
        result = long_resume.exercise(st, directory)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["splits"], [1, 64])
        self.assertFalse(result["release_ready"])


if __name__ == "__main__":
    unittest.main()

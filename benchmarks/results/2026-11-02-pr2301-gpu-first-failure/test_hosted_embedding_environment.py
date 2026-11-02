"""The hosted preflight and adapter guards never silently widen the selection."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosted_embedding_environment as env
import run_embedding_trio as trio
from test_run_embedding_trio import FakeStudy


class HostedAdmission(unittest.TestCase):
    def fields(self):
        return dict(env.EXPECTED, run_id="17", dispatch_sha="a"*40, github_sha="a"*40, checkout_sha="a"*40)

    def test_each_declared_environment_condition_is_required(self):
        fields = self.fields()
        self.assertEqual(env.mismatches(fields), [])
        for field in fields:
            for wrong in ("", "unexpected"):
                with self.subTest(field=field, wrong=wrong):
                    self.assertTrue(env.mismatches(dict(fields, **{field: wrong})))
        for field in ("github_sha", "checkout_sha", "dispatch_sha"):
            self.assertIn("reviewed_commit", env.mismatches(dict(fields, **{field: "b"*40})))

    def test_environment_drift_writes_reason_before_import_or_training(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(env, "inspect", return_value=dict(self.fields(), os_build="old")), \
                patch.object(trio, "preflight") as load, patch.object(trio, "exercise") as train:
            root = Path(temp)
            self.assertEqual(trio.worker(root, Path("not-read.whl")), 1)
            load.assert_not_called()
            train.assert_not_called()
            self.assertEqual(json.loads((root / "host-admission.json").read_bytes())["mismatches"], ["os_build"])

    def test_other_metal_adapter_cannot_start_worker_pair(self):
        study = FakeStudy()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "historical Metal adapter"):
                trio.exercise(study, root, expected_adapter=env.ADAPTER)
            self.assertEqual(study.calls, 1)
            self.assertFalse(study.reads)
            self.assertTrue((root / "serial/run-outcome.json").is_file())
            self.assertFalse((root / "worker0/entered.json").exists())

    def test_public_fixture_supplies_the_exact_original_request_bytes(self):
        from test_byte_corpus_study import fixture
        raw = fixture()[0]
        self.assertEqual(len(raw), 47959)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), trio.SELECTION["request_sha256"])

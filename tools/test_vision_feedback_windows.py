"""Offline admission checks for reuse of immutable frozen-model evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("windows", Path(__file__).with_name("compare_vision_feedback_windows.py"))
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


class SourceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = dict(status="passed", schema="spiraltorch.vision.feedback_stationarity.v1",
                           recipe=dict(seeds=[17, 29, 43]), cases=[dict(seed=seed, checkpoint=phase)
                           for seed in (17, 29, 43) for phase in ("initial", "final")])
        self.verification = self.root / "verification.json"
        self.write_source(self.source)

    def write_source(self, source):
        (self.root / "summary.json").write_text(json.dumps(source))
        self.verification.write_text(json.dumps(dict(status="passed",
            schema="spiraltorch.vision.feedback_stationarity_verification.v1",
            summary=windows.runner.receipt(self.root / "summary.json"))))

    def test_verified_complete_case_coverage(self):
        self.assertEqual(windows.checked_source(self.root, self.verification), self.source)

    def test_changed_source_is_not_admitted_by_old_verification(self):
        changed = dict(self.source, note="changed after verification")
        (self.root / "summary.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "verification differs"):
            windows.checked_source(self.root, self.verification)

    def test_rehashed_incomplete_cases_and_wrong_identity_fail(self):
        for edit in (lambda v: v.update(status="error"), lambda v: v.update(schema="other"),
                     lambda v: v["cases"].pop(), lambda v: v["cases"][0].update(checkpoint="other"),
                     lambda v: v["recipe"].update(seeds=[17, 17, 43])):
            changed = copy.deepcopy(self.source)
            edit(changed)
            self.write_source(changed)
            with self.assertRaises(ValueError):
                windows.checked_source(self.root, self.verification)

    def test_wrong_or_failed_verification_is_not_accepted(self):
        for field, value in (("status", "error"), ("schema", "other")):
            self.write_source(self.source)
            checked = json.loads(self.verification.read_text())
            checked[field] = value
            self.verification.write_text(json.dumps(checked))
            with self.assertRaisesRegex(ValueError, "verification differs"):
                windows.checked_source(self.root, self.verification)


if __name__ == "__main__":
    unittest.main()

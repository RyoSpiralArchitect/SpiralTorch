"""No GPU/model required for exact-comparison diagnostic regressions."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

API = runpy.run_path(str(Path(__file__).with_name("byte_study_diagnostics.py")))


class DiagnosticTests(unittest.TestCase):
    def test_uninterrupted_replay_keeps_actual_reports_without_inventing_checkpoint(self):
        boundaries = mock.Mock(return_value=(b'{ "step": 1 }', '{"cursor":1}', '{"cursor":2}'))
        context = mock.Mock(return_value={"adapter": "synthetic"})
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "version-6"
            API["assert_exact_run_resume"](target, b"{}", '{"w":1}', '{"w":1}',
                                           boundaries=boundaries, runtime_context=context)
            boundaries.assert_not_called()
            context.assert_not_called()
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(AssertionError, "exact uninterrupted/resumed"):
                API["assert_exact_run_resume"](None, b"{}", '{"w":1}', '{"w":100}',
                                               boundaries=boundaries, runtime_context=context)
            boundaries.assert_not_called()
            with self.assertRaisesRegex(AssertionError, "exact uninterrupted/resumed"):
                API["assert_exact_run_resume"](target, b'{ "request": true }', '{"w":1}', '{"w":100}',
                                               boundaries=boundaries, runtime_context=context)
            boundaries.assert_called_once_with()
            context.assert_called_once_with()
            summary = json.loads((target / "summary.json").read_text())
            self.assertEqual(len(summary["files"]), 7)
            self.assertEqual(summary["full_checkpoint"], "not_returned_by_run_api")
            self.assertFalse((target / "full-checkpoint.json").exists())
            self.assertEqual((target / "first-report.json").read_bytes(), b'{ "step": 1 }')
            self.assertEqual(summary["report_differences"][0]["path"], "$.w")
            for filename, digest in summary["files"].items():
                self.assertEqual(digest, "sha256:" + hashlib.sha256((target / filename).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                API["assert_exact_run_resume"](target, b"{}", "{}", "[]", boundaries=boundaries)
            self.assertEqual((target / "request.json").read_bytes(), b'{ "request": true }')

    def test_uninterrupted_replay_rejects_invalid_payloads_before_retention(self):
        for raw in ({}, bytearray(b"{}"), b" " * (64 * 1024 * 1024 + 1)):
            with tempfile.TemporaryDirectory() as name:
                target = Path(name) / "version-8"
                with self.assertRaises(ValueError):
                    API["assert_exact_run_resume"](target, raw, "{}", "[]", boundaries=lambda: ("{}", "{}", "{}"))
                self.assertFalse(target.exists())

    def test_runtime_context_is_lazy_bounded_and_hash_bound(self):
        payloads = dict(full_report='{"ce":1}', first_report='{"ce":3}', resumed_report='{"ce":1}',
                        full_checkpoint="{}", first_checkpoint="{}", resumed_checkpoint="{}")
        collect = mock.Mock(return_value={"adapter": {"name": "synthetic GPU", "backend": "Metal"}})
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "resume"
            API["assert_exact_resume"](target, b"{}", runtime_context=collect, **payloads)
            collect.assert_not_called()
            self.assertFalse(target.exists())
            payloads["resumed_report"] = '{"ce":2}'
            with self.assertRaises(AssertionError):
                API["assert_exact_resume"](None, b"{}", runtime_context=collect, **payloads)
            collect.assert_not_called()
            with self.assertRaises(AssertionError):
                API["assert_exact_resume"](target, b"{}", runtime_context=collect, **payloads)
            collect.assert_called_once_with()
            raw = (target / "runtime.json").read_bytes()
            report = json.loads((target / "summary.json").read_text())
            self.assertEqual(json.loads(raw), collect.return_value)
            self.assertEqual(report["files"]["runtime.json"], "sha256:" + hashlib.sha256(raw).hexdigest())

    def test_context_failure_cannot_erase_original_mismatch_or_leak_exception_message(self):
        payloads = dict(full_report="{}", first_report="{}", resumed_report="[]",
                        full_checkpoint="{}", first_checkpoint="{}", resumed_checkpoint="{}")
        collectors = [mock.Mock(side_effect=RuntimeError("private-secret")),
                      lambda: {"long": "x" * (16 * 1024)}, lambda: {"x": float("nan")}, lambda: []]
        for collect in collectors:
            with self.subTest(collect=collect), tempfile.TemporaryDirectory() as name:
                target = Path(name) / "resume"
                with self.assertRaisesRegex(AssertionError, "exact contextual-mixture resume"):
                    API["assert_exact_resume"](target, b"{}", runtime_context=collect, **payloads)
                self.assertEqual((target / "resumed-report.json").read_text(), "[]")
                raw = (target / "runtime.json").read_text()
                self.assertNotIn("private-secret", raw)
                self.assertEqual(json.loads(raw)["status"], "collection_failed")

    def test_runtime_identity_selects_only_nonsecret_fields_without_tensor_execution(self):
        device = SimpleNamespace(adapter_info=lambda: {"name": "synthetic GPU", "backend": "Metal"})
        st = SimpleNamespace(wgpu=SimpleNamespace(WgpuTensorDevice=SimpleNamespace(create=lambda: device)))
        with tempfile.TemporaryDirectory() as name:
            library = Path(name) / "native.so"
            library.write_bytes(b"synthetic library")
            with mock.patch("importlib.import_module", return_value=SimpleNamespace(__file__=str(library))):
                result = API["runtime_identity"](st)
            self.assertEqual(result["native_library"], {
                "filename": "native.so", "sha256": hashlib.sha256(b"synthetic library").hexdigest()})
            self.assertEqual(result["adapter"]["name"], "synthetic GPU")
            self.assertNotIn(name, json.dumps(result))
            self.assertEqual(set(result), {"schema", "python", "system", "release", "os_version",
                                          "machine", "native_library", "adapter"})

    def test_exact_resume_gate_fails_either_difference_with_or_without_retention(self):
        base = dict(full_report='{"ce":1}', first_report='{"ce":3}', resumed_report='{"ce":1}',
                    full_checkpoint='{"moments":[1]}', first_checkpoint='{"moments":[3]}',
                    resumed_checkpoint='{"moments":[1]}')
        for changed in ({"resumed_report": '{"ce":2}'}, {"resumed_checkpoint": '{"moments":[2]}'},
                        {"resumed_report": '{"ce":2}', "resumed_checkpoint": '{"moments":[2]}'},
                        {"resumed_report": '{ "ce": 1 }'}):
            for enabled in (False, True):
                with self.subTest(changed=changed, enabled=enabled), tempfile.TemporaryDirectory() as name:
                    target = Path(name) / "resume"
                    with self.assertRaisesRegex(AssertionError, "exact contextual-mixture resume"):
                        API["assert_exact_resume"](target if enabled else None, b"{}", **dict(base, **changed))
                    self.assertEqual(target.exists(), enabled)
                    if enabled:
                        self.assertTrue((target / "summary.json").is_file())
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "resume"
            API["assert_exact_resume"](target, b"{}", **base)
            self.assertFalse(target.exists())

    def test_resume_receipt_keeps_all_boundaries_and_exact_bytes(self):
        payloads = dict(full_report='{"ce":1}', first_report='{"ce":3}',
                        resumed_report='{"ce":2}', full_checkpoint='{"moments":[1]}',
                        first_checkpoint=b'{ "moments": [3] }', resumed_checkpoint='{"moments":[2]}')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "resume"
            report = API["retain_resume_failure"](target, b'{ "fixture": true }', **payloads)
            self.assertEqual(len(report["files"]), 7)
            self.assertEqual((target / "request.json").read_bytes(), b'{ "fixture": true }')
            for name, raw in payloads.items():
                filename = name.replace("_", "-") + ".json"
                self.assertEqual((target / filename).read_bytes(),
                                 raw.encode() if isinstance(raw, str) else raw)
            for filename, expected in report["files"].items():
                self.assertEqual(expected, "sha256:" + hashlib.sha256((target / filename).read_bytes()).hexdigest())
            self.assertEqual(report["comparison"], "exact_json_bytes_no_tolerance")
            self.assertEqual(report["report_differences"][0]["path"], "$.ce")
            self.assertEqual(report["checkpoint_differences"][0]["path"], "$.moments[0]")
            self.assertEqual(json.loads((target / "summary.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                API["retain_resume_failure"](target, b"{}", **payloads)
            self.assertEqual((target / "request.json").read_bytes(), b'{ "fixture": true }')

    def test_resume_payload_validation_precedes_creating_any_files(self):
        payloads = dict(full_report="{}", first_report="{}", resumed_report="{}",
                        full_checkpoint="{}", first_checkpoint="{}", resumed_checkpoint="{}")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "resume"
            for invalid in ({}, None, bytearray(b"{}"), b" " * (64 * 1024 * 1024 + 1)):
                with self.assertRaises(ValueError):
                    API["retain_resume_failure"](target, invalid, **payloads)
                self.assertFalse(target.exists())
            with self.assertRaises(ValueError):
                API["retain_resume_failure"](target, b"{}", **dict(payloads, first_checkpoint={}))
            self.assertFalse(target.exists())

    def test_resume_capture_does_not_follow_an_existing_file_or_symlink(self):
        payloads = dict(full_report="{}", first_report="{}", resumed_report="{}",
                        full_checkpoint="{}", first_checkpoint="{}", resumed_checkpoint="{}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "existing-file"
            target.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                API["retain_resume_failure"](target, b"{}", **payloads)
            self.assertEqual(target.read_bytes(), b"preserve")
            destination = root / "elsewhere"
            destination.mkdir()
            link = root / "link"
            link.symlink_to(destination, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                API["retain_resume_failure"](link, b"{}", **payloads)
            self.assertEqual(list(destination.iterdir()), [])

    def test_small_float_differences_remain_exact_failures(self):
        before = json.dumps({"cases": [{"validation": [5.501455783843994]}]})
        after = json.dumps({"cases": [{"validation": [5.501457691192627]}]})
        diff = API["exact_report_differences"](before, after)
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0]["path"], "$.cases[0].validation[0]")
        self.assertNotEqual(diff[0]["expected"], diff[0]["actual"])
        self.assertEqual(API["exact_report_differences"](before, before), [])

    def test_structure_types_encoding_and_invalid_json_are_distinct(self):
        checks = (("{}", '{"extra":1}', "keys"), ("[]", "[1]", "length"),
                  ("true", "1", "type"), ('{"x":1}', '{ "x": 1 }', "serialization_only"),
                  ("invalid", "{}", "invalid_json"))
        for before, after, kind in checks:
            self.assertEqual(API["exact_report_differences"](before, after)[0]["kind"], kind)

    def test_difference_count_and_strings_are_bounded(self):
        diff = API["exact_report_differences"](json.dumps([0] * 100), json.dumps([1] * 100), limit=3)
        self.assertEqual(len(diff), 3)
        long = API["exact_report_differences"](json.dumps("a" * 1000), json.dumps("b" * 1000))
        self.assertEqual(len(long[0]["expected"]), 160)
        for limit in (0, 65, True):
            with self.assertRaises(ValueError):
                API["exact_report_differences"]("[]", "[]", limit=limit)

    def test_receipts_retain_original_bytes_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            report = API["retain_detached_failure"](name, b'{ "fixture": true }', '{"x":1}',
                                                     ['{"x":2}', '{"x":1}'], 0)
            target = Path(name) / "round-0"
            self.assertEqual((target / "request.json").read_bytes(), b'{ "fixture": true }')
            for filename, expected in report["files"].items():
                self.assertEqual(expected, "sha256:" + hashlib.sha256((target / filename).read_bytes()).hexdigest())
            self.assertEqual(report["comparison"], "exact_json_bytes_no_tolerance")
            self.assertEqual(report["differences"][1], [])
            self.assertEqual(json.loads((target / "summary.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                API["retain_detached_failure"](name, b"{}", "{}", ["{}", "{}"], 0)


if __name__ == "__main__":
    unittest.main()

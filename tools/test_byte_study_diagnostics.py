"""No GPU/model required for exact-comparison diagnostic regressions."""
import ast
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
    def test_detached_client_keeps_four_pairs_and_collects_only_after_failure(self):
        source = Path(__file__).resolve().parents[1] / "bindings/st-py/tests/test_nn_byte_study.py"
        tree = ast.parse(source.read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StudyGpu")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name == "test_detached_calls_share_a_recipe_not_mutable_models")
        code = compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec")
        baseline, changed = '{"x":1}', '{"x":2}'
        for fails, retain in ((False, True), (True, True), (True, False)):
            with self.subTest(fails=fails, retain=retain), tempfile.TemporaryDirectory() as directory:
                study = SimpleNamespace(request_sha256=hashlib.sha256(b"{}").hexdigest(),
                                        run_wgpu=mock.Mock(side_effect=[baseline] * 7 +
                                                          [changed if fails else baseline, baseline]))
                parse = mock.Mock(return_value=study)
                pool = mock.MagicMock()
                pool.__enter__.return_value.map.side_effect = lambda run, items: [run(i) for i in items]
                factory = mock.Mock(return_value=pool)
                collect = mock.Mock(return_value={"native_library": {"sha256": "synthetic"}})
                capture = mock.Mock(wraps=API["retain_detached_failure"])
                namespace = dict(st=SimpleNamespace(nn=SimpleNamespace(ByteCorpusStudy=SimpleNamespace(from_json=parse))),
                                 fixture=lambda: (b"{}",), retain=lambda *_: None,
                                 ThreadPoolExecutor=factory, hashlib=hashlib, json=json,
                                 os=SimpleNamespace(environ={"SPIRALTORCH_DETACHED_STUDY_DIAGNOSTICS": directory}
                                                    if retain else {}),
                                 retain_detached_failure=capture, runtime_identity=collect,
                                 exact_report_difference_summary=API["exact_report_difference_summary"])
                exec(code, namespace)
                run = namespace[method.name]
                if fails:
                    with self.assertRaisesRegex(AssertionError, "exact detached comparison round 3"):
                        run(self)
                else:
                    run(self)
                parse.assert_called_once_with(b"{}")
                self.assertEqual(study.run_wgpu.call_count, 9)
                self.assertEqual(factory.call_args_list, [mock.call(max_workers=2)] * 4)
                self.assertEqual(capture.call_count, int(fails and retain))
                self.assertEqual(collect.call_count, int(fails and retain))
                self.assertEqual((Path(directory) / "round-3" / "summary.json").exists(), fails and retain)

    def test_summary_counts_late_metric_changes_without_unbounded_examples(self):
        left = {"cases": [{"final_parameters": [[0.] * 100], "training": [{"ce": 5.5}],
                           "validation": [{"ce": 5.4}]}]}
        right = {"cases": [{"final_parameters": [[1e-7] * 100], "training": [{"ce": 5.4}],
                            "validation": [{"ce": 5.3}]}]}
        before, after = json.dumps(left), json.dumps(right)
        summary = API["exact_report_difference_summary"](before, after, limit=3)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["raw_equal"])
        self.assertEqual(summary["difference_count"], 102)
        parameters, other = (summary["groups"][key] for key in ("final_parameters", "other"))
        self.assertEqual(parameters["count"], 100)
        self.assertEqual(len(parameters["examples"]), 3)
        self.assertTrue(parameters["truncated"])
        self.assertEqual(other["count"], 2)
        self.assertFalse(other["truncated"])
        self.assertEqual([item["path"] for item in other["examples"]],
                         ["$.cases[0].training[0].ce", "$.cases[0].validation[0].ce"])
        self.assertEqual(API["exact_report_differences"](before, after, limit=3), parameters["examples"])
        self.assertEqual(API["exact_report_difference_summary"](before, before)["difference_count"], 0)

    def test_summary_keeps_structural_and_serialization_failures_distinct(self):
        for before, after, kind in (("{}", '{"x":1}', "keys"), ("[]", "[1]", "length"),
                                    ("1", "1.0", "type"), ("0.0", "-0.0", "value"),
                                    ('{"x":1}', '{ "x": 1 }', "serialization_only")):
            with self.subTest(kind=kind):
                summary = API["exact_report_difference_summary"](before, after)
                self.assertFalse(summary["raw_equal"])
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["difference_count"], 1)
                self.assertEqual(summary["groups"]["other"]["examples"][0]["kind"], kind)
        summary = API["exact_report_difference_summary"]('{"final_parameters":[0]}',
                                                          '{"final_parameters":[1]}')
        self.assertEqual(summary["groups"]["final_parameters"]["count"], 0)
        self.assertEqual(summary["groups"]["other"]["count"], 1)
        summary = API["exact_report_difference_summary"]("{}", json.dumps({str(i): 0 for i in range(100)}))
        example = summary["groups"]["other"]["examples"][0]
        self.assertEqual(example["extra_count"], 100)
        self.assertEqual(len(example["extra"]), 16)

    def test_summary_rejects_ambiguous_nonfinite_and_deep_json_without_inventing_totals(self):
        for raw in ("broken", '{"x":0,"x":1}', "NaN", "Infinity", "1e400"):
            with self.subTest(raw=raw[:32]):
                summary = API["exact_report_difference_summary"]("{}", raw)
                self.assertFalse(summary["complete"])
                self.assertIsNone(summary["difference_count"])
                self.assertIsNone(summary["groups"]["other"]["count"])
                self.assertEqual(summary["groups"]["other"]["examples"][0]["kind"], "invalid_json")
                json.dumps(summary, allow_nan=False)
        summary = API["exact_report_difference_summary"]("[" * 200 + "0" + "]" * 200,
                                                          "[" * 200 + "1" + "]" * 200)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["incomplete_reason"], "analysis_limit")
        self.assertIsNone(summary["difference_count"])
        for limit in (0, 65, True, 1.0):
            with self.assertRaises(ValueError):
                API["exact_report_difference_summary"]("[]", "[]", limit=limit)

    def test_detached_metadata_is_collected_after_original_bytes_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "round-3"

            def collect():
                self.assertEqual((target / "request.json").read_bytes(), b'{ "request": true }')
                self.assertEqual((target / "expected.json").read_bytes(), b'{"x":1}')
                self.assertEqual((target / "actual-0.json").read_bytes(), b'{"x":2}')
                self.assertEqual((target / "actual-1.json").read_bytes(), b'{"x":1}')
                return {"native_library": {"sha256": "synthetic"}}

            context = mock.Mock(side_effect=collect)
            result = API["retain_detached_failure"](directory, b'{ "request": true }', '{"x":1}',
                                                     ['{"x":2}', '{"x":1}'], 3, runtime_context=context)
            context.assert_called_once_with()
            self.assertEqual(len(result["files"]), 5)
            self.assertEqual(result["difference_summaries"][0]["difference_count"], 1)
            self.assertTrue(result["difference_summaries"][1]["raw_equal"])
            for name, digest in result["files"].items():
                self.assertEqual(digest, "sha256:" + hashlib.sha256((target / name).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                API["retain_detached_failure"](directory, b"{}", "{}", ["[]", "{}"], 3,
                                                runtime_context=context)
            context.assert_called_once_with()

    def test_detached_metadata_failure_cannot_replace_numerical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = API["retain_detached_failure"](
                directory, b"{}", '{"x":1}', ['{"x":2}', '{"x":1}'], 0,
                runtime_context=mock.Mock(side_effect=RuntimeError("private-secret")))
            target = Path(directory) / "round-0"
            self.assertEqual((target / "actual-0.json").read_bytes(), b'{"x":2}')
            context = (target / "runtime.json").read_text()
            self.assertNotIn("private-secret", context)
            self.assertEqual(json.loads(context)["status"], "collection_failed")
            self.assertEqual(result["difference_summaries"][0]["difference_count"], 1)

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

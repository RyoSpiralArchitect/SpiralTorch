import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("qualify", Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)
REPO = Path(__file__).resolve().parents[3]


def selection_fixture():
    entry = dict(name="w16-seed11", width=16, seed=11, request_sha256="request")
    request = dict(cases=[dict(name=name) for name in ("ordinary", "geometry", "geometry_frozen")])
    record = dict(**entry, decisions=[], artifacts=[])
    for index, case in enumerate(request["cases"]):
        record["decisions"].append(dict(case_name=case["name"], selected=dict(
            candidate_index=1, completed_updates=256, artifact_id=f"selected{index}")))
        for stop in qualify.STOPS:
            record["artifacts"].append(dict(case_name=case["name"], cursor=stop,
                file=f"w16-seed11-case{index}-revision{stop}.json", selected=stop == 256,
                artifact_id=f"selected{index}" if stop == 256 else "unselected"))
    return request, entry, record


class QualificationTests(unittest.TestCase):
    def test_full_case_cursor_coverage_and_selected_flags(self):
        request, entry, original = selection_fixture()
        qualify.validate_coverage(request, entry, original)
        mutations = [lambda r: r["decisions"].pop(), lambda r: r["artifacts"].pop(),
                     lambda r: r["artifacts"].reverse(), lambda r: r.__setitem__("width", 32),
                     lambda r: r["artifacts"][1].__setitem__("selected", False),
                     lambda r: r["artifacts"][1].__setitem__("file", "../foreign.json"),
                     lambda r: r["artifacts"][1].__setitem__("artifact_id", "foreign"),
                     lambda r: r["decisions"][0]["selected"].__setitem__("candidate_index", True)]
        for mutate in mutations:
            record = copy.deepcopy(original)
            mutate(record)
            with self.assertRaises(ValueError):
                qualify.validate_coverage(request, entry, record)

    def test_all_six_studies_required_before_any_probe_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared.json").write_text("{}")
            selected = dict(prepared_sha256=qualify.digest(b"{}"), sources=[], studies=[])
            with patch.object(qualify, "source_receipts", return_value=[]), patch.object(qualify, "checked_run") as run:
                with self.assertRaisesRegex(ValueError, "six"):
                    qualify.validate_selection(REPO, root, {}, None, selected, None, None)
                run.assert_not_called()

    def test_checkpoint_actual_cursor_cannot_be_mislabelled(self):
        qualify.validate_cursor(SimpleNamespace(completed_updates=512),512)
        with self.assertRaisesRegex(ValueError,"cursor"):
            qualify.validate_cursor(SimpleNamespace(completed_updates=768),512)

    def test_qualification_runtime_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"native.so"
            path.write_bytes(b"binary-a")
            st = SimpleNamespace(_rs=SimpleNamespace(__file__=str(path)))
            manifest = dict(native_extension_sha256=qualify.digest(b"binary-a"))
            self.assertEqual(qualify.validate_runtime(st,manifest),manifest["native_extension_sha256"])
            path.write_bytes(b"binary-b")
            with self.assertRaisesRegex(ValueError,"extension"):
                qualify.validate_runtime(st,manifest)

    def test_historical_mismatch_cannot_be_erased_or_relabelled_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"prepared.json").write_text("{}")
            comparison = dict(numerical_checks_passed=True)
            actual = [dict(case_name="seed11_ordinary",fields=dict(parameters=dict(exact_f32=False,max_abs_error=.01)))]
            for claimed in ([],[dict(case_name="seed11_ordinary",fields=dict(parameters=dict(exact_f32=True,max_abs_error=0.)))]):
                studies = [dict(name=name,comparison=comparison,historical_256=claimed) for name in qualify.launch.names()]
                selected = dict(studies=studies,prepared_sha256=qualify.digest(b"{}"),sources=[],native_extension_sha256="runtime")
                with patch.object(qualify,"source_receipts",return_value=[]), \
                     patch.object(qualify,"validate_runtime",return_value="runtime"), \
                     patch.object(qualify,"checked_run",return_value=(None,None,None,None,comparison)), \
                     patch.object(qualify,"historical_comparisons",return_value=actual) as recompute:
                    with self.assertRaisesRegex(ValueError,"historical"):
                        qualify.validate_selection(REPO,root,dict(entries=[{}]),None,selected,None,("old-request","old-report"))
                    recompute.assert_called_once_with(None,None,{},"old-request","old-report")

    def test_strict_comparison_is_recomputed_and_failure_retained(self):
        protocol = qualify.module(REPO / "tools/byte_corpus_study.py")
        fixtures = qualify.module(REPO / "tools/test_byte_corpus_study.py")
        raw, reference, actual = fixtures.fixture()
        passed = qualify.compare_retaining_failure(protocol, raw, reference, actual)
        self.assertTrue(passed["numerical_checks_passed"])
        self.assertEqual(passed["comparison"], protocol.compare(raw, reference, actual))
        actual["cases"][0]["final_parameters"][0][0] += 1.
        failed = qualify.compare_retaining_failure(protocol, raw, reference, actual)
        self.assertFalse(failed["numerical_checks_passed"])
        self.assertNotIn("comparison", failed)
        self.assertIn("diagnostic", failed["scope"])

    def test_historical_vectors_keep_signed_zero_and_reject_truncation(self):
        self.assertTrue(qualify.vector_comparison([1.], [1.])["exact_f32"])
        self.assertFalse(qualify.vector_comparison([0.], [-0.])["exact_f32"])
        self.assertFalse(qualify.vector_comparison([1.], [1.01])["exact_f32"])
        for left, right in (([1.], []), ([float("nan")], [1.])):
            with self.assertRaises(ValueError):
                qualify.vector_comparison(left, right)

    def test_repetition_has_no_cross_row_pairs(self):
        result = qualify.repetition([[1, 2, 1], [1, 2, 1]])
        self.assertEqual(result["adjacent_repeat_fraction"], 0.)
        self.assertEqual(result["mean_repeated_bigram_fraction"], 0.)
        self.assertEqual(result["max_constant_run"], 1)
        result = qualify.repetition([[1, 1, 1], [2, 2, 2]])
        self.assertEqual(result["adjacent_repeat_fraction"], 1.)
        self.assertEqual(result["mean_repeated_bigram_fraction"], .5)
        self.assertEqual(result["max_constant_run"], 3)

    def test_private_receipts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            qualify.write(path, dict(passed=False))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                qualify.write(path, dict(passed=True))


if __name__ == "__main__":
    unittest.main()

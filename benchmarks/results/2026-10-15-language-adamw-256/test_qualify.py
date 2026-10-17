import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("qualify", Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class QualificationTests(unittest.TestCase):
    def test_sources_must_match_ordered_declared_revision(self):
        paths = ("docs/byte_corpus_learning.md", "docs/byte_corpus_calibration.md", "docs/resident_adamw.md")
        source = {qualify.REVISION + ":" + name: bytes([i+1, 2]) for i, name in enumerate(paths)}
        request = dict(train_documents=[[1, 2], [2, 2]], validation_documents=[[3, 2]])
        with patch.object(qualify.subprocess, "check_output", side_effect=lambda args, **kwargs: source[args[-1]]):
            qualify.validate_sources(Path("."), request)
            for key in request:
                corrupted = copy.deepcopy(request)
                corrupted[key][0] = [42]
                with self.assertRaisesRegex(ValueError, "corpus"):
                    qualify.validate_sources(Path("."), corrupted)
            request["train_documents"].reverse()
            with self.assertRaises(ValueError):
                qualify.validate_sources(Path("."), request)

    def test_comparison_is_recomputed_not_trusted_by_request_hash(self):
        repo = Path(__file__).resolve().parents[3]
        protocol = qualify.load(repo, "byte_corpus_study")
        fixtures = qualify.load(repo, "test_byte_corpus_study")
        raw, reference, actual = fixtures.fixture()
        comparison = protocol.compare(raw, reference, actual)
        qualify.validate_comparison(protocol, raw, reference, actual, comparison)
        corrupted = copy.deepcopy(actual)
        corrupted["cases"][0]["final_parameters"][0][0] += 1.
        with self.assertRaises(ValueError):
            qualify.validate_comparison(protocol, raw, reference, corrupted, comparison)

    def test_coverage_rejects_missing_duplicate_or_misflagged_artifacts(self):
        request = dict(cases=[dict(name="first"), dict(name="second")])
        selection = dict(decisions=[], artifacts=[])
        for index, case in enumerate(request["cases"]):
            selection["decisions"].append(dict(case_name=case["name"], selected=dict(
                candidate_index=1, completed_updates=64, artifact_id=f"selected{index}")))
            for stop in qualify.STOPS:
                selection["artifacts"].append(dict(case_name=case["name"], cursor=stop,
                    file=f"case{index}-revision{stop}.json", selected=stop == 64,
                    artifact_id=f"selected{index}" if stop == 64 else "not-selected"))
        qualify.validate_coverage(request, selection)
        variants = [dict(decisions=[], artifacts=[])]
        for field in ("decisions", "artifacts"):
            variant = copy.deepcopy(selection)
            variant[field].pop()
            variants.append(variant)
        for change in ("selected", "file", "artifact_id"):
            variant = copy.deepcopy(selection)
            variant["artifacts"][1][change] = False if change == "selected" else "wrong"
            variants.append(variant)
        for variant in variants:
            with self.assertRaises(ValueError):
                qualify.validate_coverage(request, variant)

    def test_baselines_use_only_seen_training_targets(self):
        request = dict(config=dict(steps=2), train_documents=[[1, 2, 3, 4, 5]],
                       train_batches=[[(0, 0)], [(0, 2)]])
        windows = [bytes([1, 2, 3])]
        zero = qualify.baselines(request, 0, windows)
        self.assertEqual(zero, dict(unigram_ce=math.log(256), bigram_ce=math.log(256)))
        first = qualify.baselines(request, 1, windows)
        self.assertAlmostEqual(first["unigram_ce"], -math.log(2 / 258))
        self.assertAlmostEqual(first["bigram_ce"], -math.log(2 / 257))
        changed = copy.deepcopy(request)
        changed["train_batches"][1] = [(0, 0)]
        self.assertEqual(first, qualify.baselines(changed, 1, windows))
        self.assertNotEqual(qualify.baselines(request, 2, windows), qualify.baselines(changed, 2, windows))

    def test_writes_are_exclusive_and_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            digest = qualify.write(path, {"passed": True})
            self.assertEqual(digest, qualify.digest(path.read_bytes()))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                qualify.write(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()

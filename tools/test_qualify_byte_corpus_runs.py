import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "byte_run_qualifier", Path(__file__).with_name("qualify_byte_corpus_runs.py"))
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


class MetadataTests(unittest.TestCase):
    def test_legacy_omission_is_not_reinterpreted(self):
        self.assertEqual(qualifier.case_metadata(dict(seed=11, geometry=True)),
                         dict(seed=11, geometry=True))

    def test_explicit_policy_preserved_and_weights_not_published(self):
        for policy in ("train", "frozen"):
            case = dict(seed=11, geometry=True, geometry_update=policy,
                        name="private-case-label", parameters=["private-weights"])
            self.assertEqual(qualifier.case_metadata(case),
                             dict(seed=11, geometry=True, geometry_update=policy))

    def test_failed_requirement_is_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, "failed gate"):
            qualifier.require(False, "failed gate")


if __name__ == "__main__":
    unittest.main()

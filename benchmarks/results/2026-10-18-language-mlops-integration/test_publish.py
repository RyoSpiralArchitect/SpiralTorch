"""Publication guards; these tests do not certify execution or privacy generally."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("publish", Path(__file__).with_name("publish.py"))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class PublicationGuards(unittest.TestCase):
    def test_nested_private_payloads_are_rejected(self):
        for key in publish.FORBIDDEN:
            with self.subTest(key=key), self.assertRaises(ValueError):
                publish.inspect({"cases": [{"nested": {key: "not for publication"}}]})

    def test_weight_free_metrics_and_digests_remain_valid(self):
        publish.inspect({"request_sha256": "a" * 64, "cases": [
            {"geometry": True, "optimizer_state_errors": {"first_max_error": 1e-7}}
        ]})

    def test_checks_do_not_rely_on_assert_statements(self):
        with self.assertRaisesRegex(ValueError, "rejected"):
            publish.require(False, "rejected")


if __name__ == "__main__":
    unittest.main()

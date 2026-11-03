"""Small publication-boundary checks; no corpus, package or device required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("selection_publish", Path(__file__).with_name("publish.py"))
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationTests(unittest.TestCase):
    def test_public_identity_matches_formatted_bytes_not_source(self):
        source = b'{"loss":1.25,"passed":true}'
        raw = publisher.encoded(json.loads(source))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(raw)
            self.assertEqual(publisher.byte_identity(raw), publisher.identity(path))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), publisher.byte_identity(raw)["sha256"])
        self.assertNotEqual(publisher.byte_identity(source), publisher.byte_identity(raw))
        self.assertEqual(json.loads(raw), json.loads(source))
        self.assertTrue(raw.endswith(b"\n"))

    def test_nonfinite_json_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                publisher.encoded({"loss": value})

    def test_nested_private_payload_rejected(self):
        for key in ("model_json", "checkpoint_jsons", "train_documents", "generated_bytes",
                    "optimizer_state", "parameters", "windows"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "private payload"):
                publisher.safe({"models": [{key: []}]})

    def test_nested_private_paths_rejected(self):
        for path in ("/Users/example/private", "/tmp/private"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "private path"):
                publisher.safe({"logs": [{"location": path}]})

    def test_metrics_and_identities_allowed(self):
        publisher.safe({"models": [{"artifact_id": "0" * 64, "loss": 1.25}],
                        "rust_cpu_tests": 71, "quality_claim": False})


if __name__ == "__main__":
    unittest.main()

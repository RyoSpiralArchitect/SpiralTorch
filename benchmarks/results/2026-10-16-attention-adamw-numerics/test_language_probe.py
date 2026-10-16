import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("language_probe", Path(__file__).with_name("language_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_pinned_input_drift_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"original")
            with patch.object(probe, "PINS", {"a": probe.digest(b"original")}):
                self.assertEqual(probe.inputs(root), {"a": b"original"})
                (root / "a").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "input drift"):
                    probe.inputs(root)

    def test_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            probe.write(path, {"first": True})
            with self.assertRaises(FileExistsError):
                probe.write(path, {"second": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})

    def test_private_output_even_with_permissive_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.umask(0)
            try:
                path = Path(directory) / "record.json"
                probe.write(path, {"private": True})
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            finally:
                os.umask(old)

    def test_final_drift_check_rejects_changed_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request.json").write_bytes(b"request")
            (root / "binary").write_bytes(b"binary")
            (root / "helper.py").write_bytes(b"original helper")
            identities = {"helper.py": probe.digest(b"original helper")}
            raw = {"request.json": b"request"}
            args = (root, root, root, root / "binary", probe.digest(b"binary"), raw, identities)
            with patch.object(probe, "PINS", {"request.json": probe.digest(b"request")}):
                probe.verify_drift(*args)
                (root / "helper.py").write_bytes(b"changed during comparison")
                with self.assertRaisesRegex(ValueError, "comparison source drift"):
                    probe.verify_drift(*args)

    def test_failure_is_retained_without_parameter_bearing_error(self):
        def compare(*args):
            raise ValueError("SECRET_PARAMETER 0.12345")
        protocol = types.SimpleNamespace(decode_json=json.loads, compare=compare)
        diagnostics = types.SimpleNamespace(errors=lambda a, e, p: {"scalars": len(a)})
        request = {"cases": [{"name": "ordinary", "parameters": [{"name": "weight"}]}]}
        row = {"name": "ordinary", "final_parameters": [[0.12345]],
               "optimizer_state": {"slots": [{"first": [0.1], "second": [0.01]}]},
               "training": [{"ce": 1.0}], "validation": [{"batch_losses": [1.0]}]}
        report = {"cases": [row]}
        result = probe.summarize(protocol, diagnostics, json.dumps(request).encode(), report, report)
        self.assertFalse(result["strict_gate"]["passed"])
        self.assertEqual(len(result["cases"]), 1)
        self.assertNotIn("SECRET_PARAMETER", json.dumps(result))
        self.assertNotIn("final_parameters", json.dumps(result))
        self.assertEqual(result["strict_gate"]["error_sha256"], probe.digest(b"SECRET_PARAMETER 0.12345"))

    def test_missing_case_rejects_even_after_a_strict_failure(self):
        protocol = types.SimpleNamespace(decode_json=json.loads, compare=lambda *args: {})
        request = {"cases": [{"name": "ordinary"}]}
        with self.assertRaisesRegex(ValueError, "case coverage"):
            probe.summarize(protocol, None, json.dumps(request).encode(), {"cases": []}, {"cases": []})


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("shift_probe", Path(__file__).with_name("language_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class InputTests(unittest.TestCase):
    def test_native_build_requires_a_matching_debug_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            artifact = {"reason": "compiler-artifact", "target": {
                "name": "resident_byte_learning", "kind": ["example"]},
                "executable": str(repo / "target/debug/examples/resident_byte_learning"),
                "profile": {"opt_level": "0"}}
            self.assertEqual(probe.native_artifact(json.dumps(artifact), repo)[0],
                             Path(artifact["executable"]))
            for value in ("", json.dumps(artifact) + "\n" + json.dumps(artifact)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    probe.native_artifact(value, repo)
            artifact["profile"]["opt_level"] = "3"
            with self.assertRaisesRegex(ValueError, "debug profile"):
                probe.native_artifact(json.dumps(artifact), repo)
            artifact["profile"]["opt_level"] = "0"
            artifact["executable"] = str(repo / "unrelated")
            with self.assertRaisesRegex(ValueError, "debug profile"):
                probe.native_artifact(json.dumps(artifact), repo)

    def test_build_rechecks_sources_before_accepting_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            source = repo / "shader"
            source.write_bytes(b"before")
            helpers = probe.module(Path(__file__).resolve().parents[1] /
                                   "2026-10-16-attention-adamw-numerics/language_probe.py")
            with patch.object(probe.subprocess, "check_output", return_value=b"rustc 1.99"), \
                 patch.object(probe.subprocess, "run", side_effect=lambda *a, **k: source.write_bytes(b"after")):
                with self.assertRaisesRegex(ValueError, "source drift"):
                    probe.build_native(repo, repo, {source: probe.digest(b"before")}, helpers)

    def test_pins_bind_every_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"original")
            with patch.object(probe, "PINS", {"request": probe.digest(b"original")}):
                self.assertEqual(probe.inputs({"request": path}), {"request": b"original"})
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pinned input drift"):
                    probe.inputs({"request": path})

    def test_final_verification_rejects_each_kind_of_drift(self):
        for altered in ("input", "request", "source", "binary"):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, data in [("input", b"data"), ("request", b"data"), ("source", b"code"), ("binary", b"exe")]:
                    (root / name).write_bytes(data)
                args = ({"request": root / "input"}, {"request": b"data"},
                        {root / "source": probe.digest(b"code")}, root / "binary", probe.digest(b"exe"),
                        root / "request")
                with patch.object(probe, "PINS", {"request": probe.digest(b"data")}):
                    probe.verify(*args)
                    (root / altered).write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "drift"):
                        probe.verify(*args)


if __name__ == "__main__":
    unittest.main()

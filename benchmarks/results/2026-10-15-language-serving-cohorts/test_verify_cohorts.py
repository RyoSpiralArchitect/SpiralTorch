"""Offline selection mechanics, not a model-quality test."""
from pathlib import Path
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("verify_cohorts.py")
HELPERS = runpy.run_path(str(SCRIPT))
SPANS = HELPERS["spans"]


class SourceSelection(unittest.TestCase):
    def test_fences_preserve_original_byte_offsets_without_delimiters(self):
        data = b"prose\n```rust\nlet x = 1;\n```\nmore prose\n```text\nnot rust\n```\nend\n"
        extract = lambda kind: [data[a:b] for a, b in SPANS(data, kind)]
        self.assertEqual(extract("prose"), [b"prose\n", b"more prose\n", b"end\n"])
        self.assertEqual(extract("rust"), [b"let x = 1;\n"])
        self.assertEqual(extract("text"), [b"not rust\n"])

    def test_empty_crlf_and_utf8_bytes(self):
        self.assertEqual(SPANS(b"", "prose"), [])
        data = "\u65e5\u672c\u8a9e\r\n```rust\r\nlet x = 1;\r\n```\r\n".encode()
        a, b = SPANS(data, "prose")[0]
        self.assertEqual(data[a:b], "\u65e5\u672c\u8a9e\r\n".encode())
        a, b = SPANS(data, "rust")[0]
        self.assertEqual(data[a:b], b"let x = 1;\r\n")

    def test_optimized_python_fails_before_writing_or_importing_native_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never-created"
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-O", str(SCRIPT),
                                     "prepare", "missing-repo", "missing-archive", str(output)],
                                    text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run without Python optimization", result.stderr)
            self.assertFalse(output.exists())

    def test_changed_gate_or_source_is_rejected_before_native_import(self):
        for tamper in ("gate", "source"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "serving-gate.json").write_text("{} " if tamper == "gate" else "{}")
                (root / "source.json").write_text("changed" if tamper == "source" else "original")
                selection = dict(gate_sha256=hashlib.sha256(b"{}").hexdigest(),
                                 source_hashes={"source.json": hashlib.sha256(b"original").hexdigest()})
                (root / "selection.json").write_text(json.dumps(selection))
                with self.assertRaises(AssertionError):
                    HELPERS["run"](root, root, root / "never-created")
                self.assertFalse((root / "never-created").exists())


if __name__ == "__main__":
    unittest.main()

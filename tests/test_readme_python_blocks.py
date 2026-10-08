#!/usr/bin/env python3
"""Exercise the documentation runner without optional ML dependencies."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_readme_python_blocks.py"


class ReadmePythonBlocksTests(unittest.TestCase):
    def run_docs(self, root: Path, *names: str) -> subprocess.CompletedProcess:
        command = [sys.executable, "-I", "-S", str(RUNNER), "--cwd", str(root)]
        for name in names:
            command.extend(["--readme", str(root / name)])
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        return subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)

    def test_repeated_documents_run_in_fresh_processes_with_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("```python\nprivate = 1\nprint('first')\n```\n")
            (root / "guide.md").write_text(
                "# Guide\n\n```py\nassert 'private' not in globals()\nprint('second')\n```\n"
            )
            result = self.run_docs(root, "README.md", "guide.md")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("first", result.stdout)
            self.assertIn("second", result.stdout)
            self.assertIn("guide.md:4", result.stdout)
            self.assertIn("OK (2 blocks)", result.stdout)

    def test_default_remains_root_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("```python\nprint('default')\n```\n")
            result = self.run_docs(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("default", result.stdout)

    def test_later_document_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.md").write_text("```python\npass\n```\n")
            (root / "two.md").write_text("```python\nraise RuntimeError('second failed')\n```\n")
            result = self.run_docs(root, "one.md", "two.md")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("second failed", result.stderr)
            self.assertNotIn("OK (", result.stdout)

    def test_missing_or_empty_document_fails_before_any_execution(self):
        for content in (None, "# No Python fences\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "one.md").write_text("```python\nprint('MUST_NOT_RUN')\n```\n")
                if content is not None:
                    (root / "two.md").write_text(content)
                result = self.run_docs(root, "one.md", "two.md")
                self.assertEqual(result.returncode, 2 if content is None else 1)
                self.assertNotIn("MUST_NOT_RUN", result.stdout)

    def test_explicit_skip_still_works_but_native_gaps_fail_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "skip.md").write_text("```python\n# ST_SKIP\nraise RuntimeError('skip')\n```\n")
            skipped = self.run_docs(root, "skip.md")
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertIn("-> skipped", skipped.stdout)
            (root / "native.md").write_text(
                "```python\nraise RuntimeError('native extension is missing')\n```\n"
            )
            failed = self.run_docs(root, "native.md")
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()

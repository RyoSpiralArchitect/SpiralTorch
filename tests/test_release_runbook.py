#!/usr/bin/env python3
"""Contract checks for release runbook credential handoffs."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseRunbookTests(unittest.TestCase):
    def test_pypi_secret_setup_uses_supported_gh_stdin_path(self) -> None:
        for relative in ("README.md", "docs/ops/release.md"):
            with self.subTest(path=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")

                self.assertNotIn("--body-file", text)
                self.assertIn('"gh",', text)
                self.assertIn('"secret",', text)
                self.assertIn('"set",', text)
                self.assertIn('"PYPI_API_TOKEN",', text)
                self.assertIn('"--env",', text)
                self.assertIn('"pypi",', text)
                self.assertIn("input=token", text)
                self.assertIn("getpass.getpass", text)


if __name__ == "__main__":
    unittest.main()

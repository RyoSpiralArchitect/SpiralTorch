#!/usr/bin/env python3
"""Contract checks for release runbook credential handoffs."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseRunbookTests(unittest.TestCase):
    def test_pypi_secret_setup_uses_supported_gh_stdin_path(self) -> None:
        helper = (ROOT / "scripts" / "configure_pypi_token_secret.py").read_text(encoding="utf-8")
        self.assertIn('"secret"', helper)
        self.assertIn('"set"', helper)
        self.assertIn('secret_name,', helper)
        self.assertIn('"--env"', helper)
        self.assertIn('input=token', helper)
        self.assertIn("getpass.getpass", helper)
        self.assertNotIn("--body-file", helper)

        for relative in ("README.md", "docs/ops/release.md"):
            with self.subTest(path=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")

                self.assertNotIn("--body-file", text)
                self.assertIn("scripts/configure_pypi_token_secret.py", text)
                self.assertIn("--token-source prompt", text)
                self.assertIn("--token-source stdin", text)
                self.assertIn("stty -echo", text)
                self.assertIn("PYPI_API_TOKEN", text)
                self.assertIn("stdin", text)


if __name__ == "__main__":
    unittest.main()

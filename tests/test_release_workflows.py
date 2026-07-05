#!/usr/bin/env python3
"""Contract checks for release GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_publish_from_release_has_safe_dry_run_mode(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish_pypi_from_release.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("- dry-run", workflow)
        self.assertIn('default: "dry-run"', workflow)
        self.assertIn("publish_method=dry-run selected", workflow)
        self.assertIn("Dry-run completed without uploading to PyPI.", workflow)
        self.assertIn("if: inputs.publish_method == 'dry-run'", workflow)
        self.assertIn("if: inputs.publish_method != 'dry-run'", workflow)

    def test_publish_from_release_validates_type_payloads(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish_pypi_from_release.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("required_payloads", workflow)
        self.assertIn("spiraltorch/__init__.pyi", workflow)
        self.assertIn("spiraltorch/py.typed", workflow)
        self.assertIn("spiraltorch/spiralk.pyi", workflow)
        self.assertIn("missing required type payloads", workflow)


if __name__ == "__main__":
    unittest.main()

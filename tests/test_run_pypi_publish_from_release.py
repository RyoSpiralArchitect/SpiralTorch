#!/usr/bin/env python3
"""Unit coverage for the safe GitHub Release-to-PyPI workflow runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pypi_publish_from_release.py"
SPEC = importlib.util.spec_from_file_location("run_pypi_publish_from_release", SCRIPT)
assert SPEC and SPEC.loader
run_pypi_publish_from_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_pypi_publish_from_release)


def ready_status(*, secret_present: bool = False, published: bool = False) -> dict[str, object]:
    return {
        "release_tag": "v0.4.11",
        "expected_wheels": 3,
        "ready": {
            "local_versions": True,
            "github_release": True,
            "pypi_published": published,
            "github_token_secret": secret_present,
        },
        "commands": {
            "token_secret_setup": "python scripts/configure_pypi_token_secret.py --token-source prompt",
        },
    }


class RunPyPIPublishFromReleaseTests(unittest.TestCase):
    def test_token_publish_requires_environment_secret(self) -> None:
        with self.assertRaisesRegex(
            run_pypi_publish_from_release.PublishRunError,
            "configure_pypi_token_secret.py",
        ):
            run_pypi_publish_from_release.ensure_publish_ready(
                ready_status(secret_present=False),
                publish_method="token",
                allow_published=False,
            )

    def test_dry_run_does_not_require_environment_secret(self) -> None:
        run_pypi_publish_from_release.ensure_publish_ready(
            ready_status(secret_present=False),
            publish_method="dry-run",
            allow_published=False,
        )

    def test_refuses_already_published_version_by_default(self) -> None:
        with self.assertRaisesRegex(run_pypi_publish_from_release.PublishRunError, "already exposes"):
            run_pypi_publish_from_release.ensure_publish_ready(
                ready_status(secret_present=True, published=True),
                publish_method="token",
                allow_published=False,
            )

    def test_dry_run_allows_already_published_version(self) -> None:
        run_pypi_publish_from_release.ensure_publish_ready(
            ready_status(secret_present=False, published=True),
            publish_method="dry-run",
            allow_published=False,
        )

    def test_workflow_run_command_uses_explicit_release_inputs(self) -> None:
        command = run_pypi_publish_from_release.workflow_run_command(
            repo="RyoSpiralArchitect/SpiralTorch",
            ref="main",
            tag="v0.4.11",
            expected_wheels=3,
            publish_method="token",
            skip_existing=True,
        )

        self.assertEqual(
            command[:6],
            [
                "gh",
                "workflow",
                "run",
                "publish_pypi_from_release.yml",
                "--repo",
                "RyoSpiralArchitect/SpiralTorch",
            ],
        )
        self.assertIn("release_tag=v0.4.11", command)
        self.assertIn("expected_wheels=3", command)
        self.assertIn("publish_method=token", command)
        self.assertIn("skip_existing=true", command)

    def test_latest_run_id_reads_first_run(self) -> None:
        completed = run_pypi_publish_from_release.subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout='[{"databaseId": 123, "url": "https://example.test/run"}]',
            stderr="",
        )
        with mock.patch.object(run_pypi_publish_from_release, "run", return_value=completed):
            run_id, url = run_pypi_publish_from_release.latest_run_id(
                "RyoSpiralArchitect/SpiralTorch",
                workflow="publish_pypi_from_release.yml",
                branch="main",
            )

        self.assertEqual(run_id, 123)
        self.assertEqual(url, "https://example.test/run")


if __name__ == "__main__":
    unittest.main()

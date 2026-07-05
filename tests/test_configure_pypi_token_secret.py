#!/usr/bin/env python3
"""Unit coverage for the PyPI token GitHub secret setup helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "configure_pypi_token_secret.py"
SPEC = importlib.util.spec_from_file_location("configure_pypi_token_secret", SCRIPT)
assert SPEC and SPEC.loader
configure_pypi_token_secret = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure_pypi_token_secret)


class ConfigurePyPITokenSecretTests(unittest.TestCase):
    def test_prompt_source_trims_and_never_exposes_token_value_in_metadata(self) -> None:
        with mock.patch.object(
            configure_pypi_token_secret.getpass,
            "getpass",
            return_value=" pypi-secret-token\n",
        ):
            token, metadata = configure_pypi_token_secret.read_token("prompt", "PYPI_API_TOKEN")

        self.assertEqual(token, "pypi-secret-token")
        self.assertEqual(metadata["source"], "prompt")
        self.assertEqual(metadata["chars"], len("pypi-secret-token"))
        self.assertTrue(metadata["trimmed"])
        self.assertTrue(metadata["upload_ready"])
        self.assertNotIn("pypi-secret-token", str(metadata))

    def test_rejects_values_that_do_not_look_like_pypi_tokens(self) -> None:
        metadata = configure_pypi_token_secret.token_metadata("not-a-token", source="stdin")

        with self.assertRaisesRegex(configure_pypi_token_secret.SecretSetupError, "starts_with_pypi=False"):
            configure_pypi_token_secret.require_upload_token(metadata)

    def test_secret_set_command_uses_environment_secret_stdin_path(self) -> None:
        command = configure_pypi_token_secret.secret_set_command(
            "gh",
            repo="RyoSpiralArchitect/SpiralTorch",
            environment="pypi",
            secret_name="PYPI_API_TOKEN",
        )

        self.assertEqual(
            command,
            [
                "gh",
                "secret",
                "set",
                "PYPI_API_TOKEN",
                "--repo",
                "RyoSpiralArchitect/SpiralTorch",
                "--env",
                "pypi",
                "--app",
                "actions",
            ],
        )

    def test_set_github_secret_passes_token_via_stdin_without_shell(self) -> None:
        completed = configure_pypi_token_secret.subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with mock.patch.object(
            configure_pypi_token_secret.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = configure_pypi_token_secret.set_github_secret(
                "pypi-secret-token",
                gh="gh",
                repo="RyoSpiralArchitect/SpiralTorch",
                environment="pypi",
                secret_name="PYPI_API_TOKEN",
            )

        self.assertIs(result, completed)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["input"], "pypi-secret-token")
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertNotIn("pypi-secret-token", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

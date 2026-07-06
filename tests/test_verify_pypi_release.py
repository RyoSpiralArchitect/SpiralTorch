#!/usr/bin/env python3
"""Unit coverage for the PyPI/GitHub Release digest verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "security" / "verify_pypi_release.py"
SPEC = importlib.util.spec_from_file_location("verify_pypi_release", SCRIPT)
assert SPEC and SPEC.loader
verify_pypi_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_pypi_release)


class VerifyPyPIReleaseTests(unittest.TestCase):
    def test_parse_sha256_lines_normalizes_case_and_ignores_blank_lines(self) -> None:
        digest = "A" * 64
        parsed = verify_pypi_release.parse_sha256_lines(f"\n{digest}  spiraltorch-0.4.10.whl\n")

        self.assertEqual(parsed, {"spiraltorch-0.4.10.whl": "a" * 64})

    def test_parse_sha256_lines_rejects_duplicates(self) -> None:
        digest = "b" * 64
        text = f"{digest}  wheel.whl\n{digest}  wheel.whl\n"

        with self.assertRaisesRegex(verify_pypi_release.VerifyError, "Duplicate"):
            verify_pypi_release.parse_sha256_lines(text)

    def test_compare_digests_reports_missing_extra_and_mismatched(self) -> None:
        release = {"linux.whl": "1" * 64, "mac.whl": "2" * 64}
        pypi = {"linux.whl": "3" * 64, "win.whl": "4" * 64}

        with self.assertRaisesRegex(verify_pypi_release.VerifyError, "mismatched"):
            verify_pypi_release.compare_digests(release, pypi)

    def test_pypi_wheel_digests_filters_non_wheels(self) -> None:
        payload = {
            "urls": [
                {
                    "filename": "spiraltorch-0.4.10.whl",
                    "digests": {"sha256": "ABCDEF" + "0" * 58},
                },
                {
                    "filename": "spiraltorch-0.4.10.tar.gz",
                    "digests": {"sha256": "f" * 64},
                },
            ],
        }

        with mock.patch.object(verify_pypi_release, "download_json", return_value=payload):
            self.assertEqual(
                verify_pypi_release.pypi_wheel_digests("spiraltorch", "0.4.10"),
                {"spiraltorch-0.4.10.whl": "abcdef" + "0" * 58},
            )

    def test_wait_for_pypi_wheels_polls_until_expected_count(self) -> None:
        calls = [
            {"linux.whl": "1" * 64},
            {"linux.whl": "1" * 64, "mac.whl": "2" * 64},
        ]

        with mock.patch.object(verify_pypi_release, "pypi_wheel_digests", side_effect=calls):
            with mock.patch.object(verify_pypi_release.time, "sleep") as sleep:
                result = verify_pypi_release.wait_for_pypi_wheels(
                    "spiraltorch",
                    "0.4.10",
                    expected_wheels=2,
                    timeout=30,
                    poll_interval=0.01,
                )

        self.assertEqual(result, {"linux.whl": "1" * 64, "mac.whl": "2" * 64})
        sleep.assert_called_once_with(0.01)

    def test_main_requires_latest_when_requested(self) -> None:
        wheels = {"spiraltorch-0.4.11.whl": "1" * 64}
        argv = [
            "verify_pypi_release.py",
            "--version",
            "0.4.11",
            "--expected-wheels",
            "1",
            "--require-latest",
        ]

        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(
                verify_pypi_release,
                "github_release_wheel_digests",
                return_value=wheels,
            ):
                with mock.patch.object(verify_pypi_release, "wait_for_pypi_wheels", return_value=wheels):
                    with mock.patch.object(
                        verify_pypi_release,
                        "pypi_latest_version",
                        return_value="0.4.10",
                    ):
                        with self.assertRaisesRegex(verify_pypi_release.VerifyError, "latest version"):
                            verify_pypi_release.main()


if __name__ == "__main__":
    unittest.main()

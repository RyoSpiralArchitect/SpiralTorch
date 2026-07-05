#!/usr/bin/env python3
"""Unit coverage for the conservative PyPI wheel publisher helper."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_pypi_wheels.py"
SPEC = importlib.util.spec_from_file_location("publish_pypi_wheels", SCRIPT)
assert SPEC and SPEC.loader
publish_pypi_wheels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_pypi_wheels)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class PublishPyPIWheelsTests(unittest.TestCase):
    def test_read_token_supports_hidden_prompt_source(self) -> None:
        with mock.patch.object(publish_pypi_wheels.getpass, "getpass", return_value=" pypi-test-token\n"):
            token, metadata = publish_pypi_wheels.read_token("prompt", "PYPI_API_TOKEN")

        self.assertEqual(token, "pypi-test-token")
        self.assertEqual(metadata["source"], "prompt")
        self.assertTrue(metadata["trimmed"])
        self.assertTrue(metadata["starts_with_pypi"])
        self.assertTrue(metadata["is_ascii"])
        self.assertFalse(metadata["contains_whitespace"])

    def test_parse_sha256_lines_rejects_duplicate_entries(self) -> None:
        digest = "a" * 64
        text = f"{digest}  wheel.whl\n{digest}  wheel.whl\n"

        with self.assertRaisesRegex(publish_pypi_wheels.PublishError, "Duplicate"):
            publish_pypi_wheels.parse_sha256_lines(text)

    def test_pypi_release_wheel_digests_filters_non_wheels(self) -> None:
        payload = {
            "urls": [
                {"filename": "spiraltorch-0.4.10.whl", "digests": {"sha256": "ABC" + "0" * 61}},
                {"filename": "spiraltorch-0.4.10.tar.gz", "digests": {"sha256": "f" * 64}},
            ],
        }

        def fake_urlopen(*args: object, **kwargs: object) -> _Response:
            return _Response(json.dumps(payload).encode("utf-8"))

        with mock.patch.object(publish_pypi_wheels, "urlopen", side_effect=fake_urlopen):
            self.assertEqual(
                publish_pypi_wheels.pypi_release_wheel_digests("0.4.10"),
                {"spiraltorch-0.4.10.whl": "abc" + "0" * 61},
            )

    def test_verify_pypi_wheel_checksums_matches_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "spiraltorch-0.4.10-py3-none-any.whl"
            wheel.write_bytes(b"spiral")
            digest = publish_pypi_wheels.file_sha256(wheel)

            with mock.patch.object(
                publish_pypi_wheels,
                "pypi_release_wheel_digests",
                return_value={wheel.name: digest},
            ):
                publish_pypi_wheels.verify_pypi_wheel_checksums([wheel], "0.4.10")

    def test_verify_pypi_wheel_checksums_reports_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "spiraltorch-0.4.10-py3-none-any.whl"
            wheel.write_bytes(b"spiral")

            with mock.patch.object(
                publish_pypi_wheels,
                "pypi_release_wheel_digests",
                return_value={wheel.name: "0" * 64},
            ):
                with self.assertRaisesRegex(publish_pypi_wheels.PublishError, "mismatched"):
                    publish_pypi_wheels.verify_pypi_wheel_checksums([wheel], "0.4.10")


if __name__ == "__main__":
    unittest.main()

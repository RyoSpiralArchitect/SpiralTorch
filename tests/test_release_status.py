#!/usr/bin/env python3
"""Unit coverage for the release readiness status helper."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_status.py"
SPEC = importlib.util.spec_from_file_location("release_status", SCRIPT)
assert SPEC and SPEC.loader
release_status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_status)


class ReleaseStatusTests(unittest.TestCase):
    def write_minimal_wheel(
        self,
        root: Path,
        *,
        version: str = "0.4.11",
        include_type_payloads: bool = True,
    ) -> Path:
        wheel = root / f"spiraltorch-{version}-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"spiraltorch-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: spiraltorch\nVersion: {version}\n",
            )
            if include_type_payloads:
                for payload in release_status.REQUIRED_WHEEL_PAYLOADS:
                    archive.writestr(payload, "")
        return wheel

    def test_token_metadata_never_exposes_token_value(self) -> None:
        metadata = release_status.token_metadata(" pypi-secret-token\n", source="env:PYPI_API_TOKEN")

        self.assertEqual(metadata["chars"], len("pypi-secret-token"))
        self.assertTrue(metadata["trimmed"])
        self.assertTrue(metadata["starts_with_pypi"])
        self.assertTrue(metadata["upload_ready"])
        self.assertNotIn("token", metadata)

    def test_toml_section_value_reads_top_level_section_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Cargo.toml"
            path.write_text("[package]\nname = \"spiraltorch-py\"\nversion = \"0.4.11\"\n", encoding="utf-8")

            self.assertEqual(release_status.toml_section_value(path, "package", "version"), "0.4.11")

    def test_pypi_and_github_release_summaries(self) -> None:
        pypi_payload = {
            "info": {"version": "0.4.10"},
            "releases": {
                "0.4.11": [
                    {"filename": "spiraltorch-0.4.11-cp38-abi3-macosx.whl"},
                    {"filename": "spiraltorch-0.4.11.tar.gz"},
                ]
            },
        }
        release_payload = {
            "draft": False,
            "prerelease": False,
            "assets": [
                {"name": "spiraltorch-0.4.11-cp38-abi3-macosx.whl"},
                {"name": "spiraltorch-0.4.11-cp38-abi3-manylinux.whl"},
                {"name": "spiraltorch-0.4.11-cp38-abi3-win_amd64.whl"},
                {"name": "wheels.sha256"},
            ],
        }

        with mock.patch.object(release_status, "download_json", side_effect=[pypi_payload, release_payload]):
            pypi = release_status.pypi_status("spiraltorch", "0.4.11", 3)
            release = release_status.github_release_status(
                "RyoSpiralArchitect/SpiralTorch",
                "v0.4.11",
                "spiraltorch",
                "0.4.11",
                3,
            )

        self.assertEqual(pypi["latest"], "0.4.10")
        self.assertEqual(pypi["wheel_count"], 1)
        self.assertFalse(pypi["published"])
        self.assertTrue(release["ready"])
        self.assertEqual(release["wheel_count"], 3)

    def test_local_wheel_payload_status_requires_type_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            self.write_minimal_wheel(dist)

            status = release_status.local_wheel_payload_status(
                dist,
                package="spiraltorch",
                version="0.4.11",
                expected_wheels=1,
            )

        self.assertTrue(status["checked"])
        self.assertTrue(status["ready"])
        self.assertEqual(status["wheel_count"], 1)
        self.assertEqual(status["missing_payloads"], {})

    def test_local_wheel_payload_status_reports_missing_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            wheel = self.write_minimal_wheel(dist, include_type_payloads=False)

            status = release_status.local_wheel_payload_status(
                dist,
                package="spiraltorch",
                version="0.4.11",
                expected_wheels=1,
            )

        self.assertFalse(status["ready"])
        self.assertIn(wheel.name, status["missing_payloads"])
        self.assertIn("spiraltorch/py.typed", status["missing_payloads"][wheel.name])

    def test_build_status_points_to_auth_when_release_ready_but_unpublished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bindings" / "st-py").mkdir(parents=True)
            (root / "bindings" / "st-py" / "pyproject.toml").write_text(
                "[project]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            (root / "bindings" / "st-py" / "Cargo.toml").write_text(
                "[package]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                package="spiraltorch",
                version=None,
                release_tag=None,
                repo="RyoSpiralArchitect/SpiralTorch",
                expected_wheels=3,
                token_env="PYPI_API_TOKEN",
                github_secret_environment="pypi",
                no_clipboard=True,
                dist=None,
                json=False,
                root=root,
            )
            with mock.patch.object(
                release_status,
                "github_release_status",
                return_value={"ready": True, "exists": True, "wheel_count": 3, "has_wheels_sha256": True},
            ):
                with mock.patch.object(
                    release_status,
                    "pypi_status",
                    return_value={"published": False, "latest": "0.4.10", "file_count": 0, "wheel_count": 0},
                ):
                    with mock.patch.object(
                        release_status,
                        "github_secret_status",
                        return_value={"present": False, "available": True},
                    ):
                        with mock.patch.dict("os.environ", {"PYPI_API_TOKEN": ""}, clear=False):
                            status = release_status.build_status(args)

        self.assertEqual(status["version"], "0.4.11")
        self.assertTrue(status["ready"]["github_release"])
        self.assertFalse(status["ready"]["pypi_published"])
        self.assertIn("scripts/configure_pypi_token_secret.py", status["next_action"])
        self.assertIn("PyPI Trusted Publishing", status["next_action"])
        self.assertEqual(
            status["commands"]["token_secret_setup"],
            "python scripts/configure_pypi_token_secret.py --token-source prompt",
        )
        self.assertIn("--token-source stdin", status["commands"]["token_secret_setup_stdin"])
        self.assertIn("stty -echo", status["commands"]["token_secret_setup_stdin"])
        self.assertIn("publish_method=token", status["commands"]["publish_token_workflow"])
        self.assertEqual(
            status["trusted_publisher"]["sub"],
            "repo:RyoSpiralArchitect/SpiralTorch:environment:pypi",
        )

    def test_build_status_points_to_publish_workflow_when_secret_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bindings" / "st-py").mkdir(parents=True)
            (root / "bindings" / "st-py" / "pyproject.toml").write_text(
                "[project]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            (root / "bindings" / "st-py" / "Cargo.toml").write_text(
                "[package]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                package="spiraltorch",
                version=None,
                release_tag=None,
                repo="RyoSpiralArchitect/SpiralTorch",
                expected_wheels=3,
                token_env="PYPI_API_TOKEN",
                github_secret_environment="pypi",
                no_clipboard=True,
                dist=None,
                json=False,
                root=root,
            )
            with mock.patch.object(
                release_status,
                "github_release_status",
                return_value={"ready": True, "exists": True, "wheel_count": 3, "has_wheels_sha256": True},
            ):
                with mock.patch.object(
                    release_status,
                    "pypi_status",
                    return_value={"published": False, "latest": "0.4.10", "file_count": 0, "wheel_count": 0},
                ):
                    with mock.patch.object(
                        release_status,
                        "github_secret_status",
                        return_value={"present": True, "available": True},
                    ):
                        with mock.patch.dict("os.environ", {"PYPI_API_TOKEN": ""}, clear=False):
                            status = release_status.build_status(args)

        self.assertEqual(status["next_action"], status["commands"]["publish_token_workflow"])
        self.assertIn("publish_method=token", status["next_action"])

    def test_build_status_points_to_local_wheel_payloads_when_dist_is_bad(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            self.write_minimal_wheel(dist, include_type_payloads=False)
            (root / "bindings" / "st-py").mkdir(parents=True)
            (root / "bindings" / "st-py" / "pyproject.toml").write_text(
                "[project]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            (root / "bindings" / "st-py" / "Cargo.toml").write_text(
                "[package]\nversion = \"0.4.11\"\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                package="spiraltorch",
                version=None,
                release_tag=None,
                repo="RyoSpiralArchitect/SpiralTorch",
                expected_wheels=1,
                token_env="PYPI_API_TOKEN",
                github_secret_environment="pypi",
                no_clipboard=True,
                dist=dist,
                json=False,
                root=root,
            )
            with mock.patch.object(
                release_status,
                "github_release_status",
                return_value={"ready": True, "exists": True, "wheel_count": 1, "has_wheels_sha256": True},
            ):
                with mock.patch.object(
                    release_status,
                    "pypi_status",
                    return_value={"published": False, "latest": "0.4.10", "file_count": 0, "wheel_count": 0},
                ):
                    with mock.patch.object(
                        release_status,
                        "github_secret_status",
                        return_value={"present": False, "available": True},
                    ):
                        with mock.patch.dict("os.environ", {"PYPI_API_TOKEN": ""}, clear=False):
                            status = release_status.build_status(args)

        self.assertFalse(status["ready"]["local_wheel_payloads"])
        self.assertIn("fix local wheel metadata/type payloads", status["next_action"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Contract checks for release GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_official_actions_use_node24_generation(self) -> None:
        workflow_dir = ROOT / ".github" / "workflows"
        workflow_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(workflow_dir.glob("*.yml"))
        )

        self.assertNotIn("actions/checkout@v4", workflow_text)
        self.assertNotIn("actions/upload-artifact@v4", workflow_text)
        self.assertNotIn("actions/download-artifact@v4", workflow_text)
        self.assertNotIn("actions/cache@v4", workflow_text)
        self.assertNotIn("actions/setup-go@v5", workflow_text)
        self.assertIn("actions/checkout@v5", workflow_text)
        self.assertIn("actions/upload-artifact@v7", workflow_text)
        self.assertIn("actions/download-artifact@v7", workflow_text)
        self.assertIn("actions/cache@v5", workflow_text)
        self.assertIn("actions/setup-go@v6", workflow_text)

    def test_publish_from_release_has_safe_dry_run_mode(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "publish_pypi_from_release.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("actions/setup-python@v5", workflow)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertIn('python-version: "3.12"', workflow)
        self.assertIn("- dry-run", workflow)
        self.assertIn('default: "dry-run"', workflow)
        self.assertIn("publish_method=dry-run selected", workflow)
        self.assertIn("Dry-run completed without uploading to PyPI.", workflow)
        self.assertIn("if: inputs.publish_method == 'dry-run'", workflow)
        self.assertIn("if: inputs.publish_method != 'dry-run'", workflow)
        self.assertIn("--dist release-dist", workflow)

    def test_publish_from_release_uses_canonical_wheel_payload_gate(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "publish_pypi_from_release.yml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "from scripts.publish_pypi_wheels import validate_wheel_metadata",
            workflow,
        )
        self.assertIn("validate_wheel_metadata(wheels, expected)", workflow)
        self.assertNotIn("required_payloads = {", workflow)

    def test_release_wheels_smoke_hf_clis_and_gate_direct_publish(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release_wheels.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -P ../../tools/smoke_hf_console_scripts.py", workflow)
        self.assertIn(
            "python -P ../../tools/smoke_zspace_runtime_protocols.py",
            workflow,
        )
        self.assertIn(
            "from scripts.publish_pypi_wheels import validate_wheel_metadata",
            workflow,
        )
        self.assertIn("validate_wheel_metadata(wheels, expected)", workflow)

    def test_all_wheel_builds_smoke_rust_owned_zspace_protocols(self) -> None:
        for workflow_name in ["wheels.yml", "release_wheels.yml"]:
            with self.subTest(workflow=workflow_name):
                workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    "python -P ../../tools/smoke_zspace_runtime_protocols.py",
                    workflow,
                )

    def test_pypi_publish_verification_requires_latest_release(self) -> None:
        publish_from_release = (
            ROOT / ".github" / "workflows" / "publish_pypi_from_release.yml"
        ).read_text(encoding="utf-8")
        release_wheels = (
            ROOT / ".github" / "workflows" / "release_wheels.yml"
        ).read_text(encoding="utf-8")

        verification = (
            ROOT / ".github" / "workflows" / "verify_pypi_release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("require_latest: true", publish_from_release)
        self.assertIn("needs: publish", publish_from_release)
        self.assertIn(
            "uses: ./.github/workflows/verify_pypi_release.yml", publish_from_release
        )
        self.assertIn("--require-latest", verification)
        self.assertIn("--require-latest", release_wheels)
        self.assertIn("--require-simple-index", verification)
        self.assertIn("--require-simple-index", release_wheels)
        self.assertIn("--index-url https://pypi.org/simple", verification)

    def test_pypi_verification_can_be_repeated_without_upload_authority(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "verify_pypi_release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("workflow_call:", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("timeout-minutes: 10", workflow)
        self.assertIn("persist-credentials: false", workflow)
        job_env = workflow.partition("    env:\n")[2].partition("    steps:\n")[0]
        self.assertTrue(job_env)
        self.assertNotIn("${{ runner.", job_env)
        self.assertEqual(
            workflow.count("          PYPI_REQUIREMENTS: ${{ runner.temp }}"), 2
        )
        for upload_authority in [
            "secrets.",
            "secrets:",
            "environment:",
            "id-token: write",
            "twine",
            "pypi-publish@",
            "gh release upload",
        ]:
            with self.subTest(authority=upload_authority):
                self.assertNotIn(upload_authority, workflow)
        self.assertIn('--pip-requirements "$PYPI_REQUIREMENTS"', workflow)
        self.assertIn('--requirement "$PYPI_REQUIREMENTS"', workflow)
        self.assertIn("--require-hashes", workflow)
        self.assertIn("--no-deps", workflow)
        self.assertIn("target in Path(st.__file__).resolve().parents", workflow)
        self.assertIn("assert st._rs is not None", workflow)
        self.assertIn("target in Path(st._rs.__file__).resolve().parents", workflow)
        self.assertNotIn("|| true", workflow)

    def test_all_wheels_execute_nonlinear_training_before_upload(self) -> None:
        for name in ["wheels.yml", "release_wheels.yml"]:
            workflow = (ROOT / ".github" / "workflows" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("python -I ../../tools/smoke_autograd.py", workflow)
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("test_autograd_nonlinear.py", ci)
        self.assertIn("python -I tools/smoke_autograd.py", ci)
        self.assertIn("--test autograd_nonlinear", ci)

    def test_release_verifier_uses_isolated_python_and_quoted_arguments(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "verify-release.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn('PYTHONNOUSERSITE: "1"', workflow)
        self.assertIn('python-version: "3.12"', workflow)
        self.assertIn('ARGS=(--repo "${REPOSITORY}")', workflow)
        self.assertIn('ARGS+=(--tag "${RELEASE_TAG}")', workflow)
        self.assertIn(
            'python scripts/security/verify_release.py "${ARGS[@]}"', workflow
        )
        self.assertNotIn("TAG_ARG=", workflow)


if __name__ == "__main__":
    unittest.main()

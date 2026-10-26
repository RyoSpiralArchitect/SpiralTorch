#!/usr/bin/env python3
"""Contract checks for release GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_retained_boundary_client_is_qualified_after_original_failure_handling(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        gpu = ci.split("\n  wgpu-macos:\n", 1)[1].split("\n  wasm-build:\n", 1)[0]
        original = next(step for step in gpu.split("\n      - ")
                        if "id: byte_python\n" in step)
        commands = [line.strip() for line in original.splitlines()
                    if line.strip().startswith("python -I -B ")]
        self.assertEqual(commands, [f"python -I -B bindings/st-py/tests/{name}.py -v" for name in (
            "test_wgpu_attention", "test_nn_attention_training",
            "test_nn_residual_attention_training", "test_nn_byte_decoder",
            "test_nn_byte_study", "test_nn_byte_serving", "test_nn_byte_generation",
        )])
        name = "name: Qualify the separate Python retained-boundary client"
        diagnostic = next(step for step in gpu.split("\n      - ") if name in step)
        command = "python -I -B bindings/st-py/tests/test_nn_byte_study_boundary.py -v"
        self.assertIn("if: success()\n", diagnostic)
        self.assertIn("timeout-minutes: 10\n", diagnostic)
        self.assertIn('SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS: "1"', diagnostic)
        self.assertIn('SPIRALTORCH_STRICT_GPU: "1"', diagnostic)
        self.assertEqual(diagnostic.split("        run: ", 1)[1].strip(), command)
        self.assertEqual(gpu.count(command), 1)
        self.assertNotIn("continue-on-error", diagnostic)
        self.assertNotIn("SPIRALTORCH_PREPARED_STUDY_DIAGNOSTICS", diagnostic)
        order = [
            "id: byte_python\n",
            "name: Retain exact detached-study failure diagnostics",
            "name: Retain exact contextual-mixture resume failure diagnostics",
            "name: Retain exact prepared-study replay failure diagnostics",
            "id: byte_boundary\n",
            "name: Retain separate byte-study boundary diagnostics",
            name,
        ]
        for marker in order:
            self.assertEqual(gpu.count(marker), 1)
        positions = [gpu.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions))
        portable = re.split(r"\n  [a-zA-Z0-9_-]+:\n",
                            ci.split("\n  python-docs-smoke:\n", 1)[1], maxsplit=1)[0]
        self.assertIn("python -I -B bindings/st-py/tests/test_nn_byte_study_boundary.py BoundaryPortable -v",
                      portable)
        for scope in (ci.split("\njobs:\n", 1)[0], portable):
            self.assertNotIn("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS:", scope)
            self.assertNotIn("SPIRALTORCH_STRICT_GPU:", scope)

    def test_gpu_study_failures_retain_each_original_comparison(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        gpu = ci.split("\n  wgpu-macos:\n", 1)[1].split("\n  wasm-build:\n", 1)[0]
        self.assertIn("bindings/st-py/tests/test_nn_byte_study.py", gpu)
        self.assertNotIn("continue-on-error", gpu)
        for kind in ("detached", "resume", "prepared"):
            self.assertIn(f'SPIRALTORCH_{kind.upper()}_STUDY_DIAGNOSTICS: ${{{{ runner.temp }}}}/byte-study-{kind}', gpu)
            artifact = next(step for step in gpu.split("\n      - ")
                            if f"name: synthetic-byte-study-{kind}-failure" in step)
            self.assertIn("if: failure()", artifact)
            self.assertIn("uses: actions/upload-artifact@v7", artifact)
            self.assertIn(f"path: ${{{{ runner.temp }}}}/byte-study-{kind}/", artifact)
            self.assertIn("retention-days: 1", artifact)

    def test_gpu_failure_retains_original_wheel_without_rebuilding_or_dispatching(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        gpu = ci.split("\n  wgpu-macos:\n", 1)[1].split("\n  wasm-build:\n", 1)[0]
        artifact = next(step for step in gpu.split("\n      - ")
                        if "name: synthetic-byte-study-original-wheel" in step)
        self.assertIn("if: failure() && steps.byte_python.outcome == 'failure'\n", artifact)
        self.assertIn("uses: actions/upload-artifact@v7\n", artifact)
        self.assertIn("path: ${{ runner.temp }}/attention-wheel/*.whl\n", artifact)
        self.assertIn("retention-days: 1\n", artifact)
        self.assertIn("if-no-files-found: warn\n", artifact)
        self.assertNotIn("run:", artifact)
        self.assertNotIn("continue-on-error", artifact)
        self.assertNotIn("overwrite:", artifact)
        self.assertLess(gpu.index("name: Retain exact prepared-study replay failure diagnostics"),
                        gpu.index("name: synthetic-byte-study-original-wheel"))
        self.assertLess(gpu.index("name: synthetic-byte-study-original-wheel"),
                        gpu.index("name: Observe one retained byte-study first-update boundary"))
        self.assertEqual(gpu.count('maturin build --release --locked -m bindings/st-py/Cargo.toml'), 1)

    def test_license_metadata_is_checked_before_wheel_builds(self) -> None:
        command = "python -I -B scripts/security/generate_repo_manifest.py --check-only"
        official = (ROOT / ".github/workflows/release_wheels.yml").read_text(encoding="utf-8")
        contract = official.split("\n  release_contract:\n", 1)[1].split("\n  wheels:\n", 1)[0]
        self.assertIn(command, contract)
        self.assertIn('python-version: "3.12"', contract)
        self.assertNotIn("continue-on-error", contract)
        wheels = official.split("\n  wheels:\n", 1)[1].split("\n  pypi:\n", 1)[0]
        self.assertIn("needs: release_contract", wheels)
        manual = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        self.assertLess(manual.index(command), manual.index("- name: Build wheel"))
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        inventory = ci.split("\n  workspace-inventory:\n", 1)[1].split("\n  version-consistency:\n", 1)[0]
        self.assertIn(command.replace("python ", "python3 "), inventory)
        self.assertIn("python3 -I -B tests/test_generate_repo_manifest.py", inventory)

    def test_package_version_matches_native_manifest_and_lock(self) -> None:
        metadata = (ROOT / "bindings/st-py/pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE).group(1)
        manifest = (ROOT / "bindings/st-py/Cargo.toml").read_text(encoding="utf-8")
        native_version = re.search(r'^version = "([^"]+)"$', manifest, re.MULTILINE).group(1)
        self.assertEqual(native_version, version)
        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        locked_versions = re.findall(
            r'^name = "spiraltorch-py"\nversion = "([^"]+)"$', lock, re.MULTILINE
        )
        self.assertEqual(locked_versions, [version])

    def test_manual_publish_contract_rejects_missing_tag_before_build(self) -> None:
        workflow = (ROOT / ".github/workflows/release_wheels.yml").read_text(encoding="utf-8")
        contract = workflow.split("\n  release_contract:\n", 1)[1].split("\n  wheels:\n", 1)[0]
        script = textwrap.dedent(contract.split("        run: |\n", 1)[1].split("\n      - ", 1)[0])
        metadata = (ROOT / "bindings/st-py/pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE).group(1)
        for publish, finalize, tag, expected_error in (
            ("true", "false", "", "publish_pypi=true requires release_tag"),
            ("false", "false", "", None),
            ("false", "false", f"v{version}", None),
            ("true", "false", f"v{version}", "publish_pypi=true requires finalize_release=true"),
            ("false", "true", "", "finalize_release=true requires release_tag"),
            ("true", "true", f"v{version}", None),
        ):
            with self.subTest(publish=publish, finalize=finalize, tag=tag):
                env = dict(os.environ, EVENT_NAME="workflow_dispatch", REF_NAME="main",
                           INPUT_PUBLISH_PYPI=publish, INPUT_RELEASE_TAG=tag,
                           INPUT_CHECKOUT_REF=tag, INPUT_FINALIZE_RELEASE=finalize)
                result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1 if expected_error else 0, result.stdout + result.stderr)
                if expected_error:
                    self.assertIn(expected_error, result.stderr)
        self.assertIn("INPUT_PUBLISH_PYPI: ${{ inputs.publish_pypi }}", contract)

    def test_tag_build_retains_signed_payload_without_automatic_publication(self) -> None:
        workflow = (ROOT / ".github/workflows/release_wheels.yml").read_text()
        option = workflow.split("      finalize_release:\n", 1)[1].split("      publish_pypi:\n", 1)[0]
        self.assertIn("type: boolean", option)
        self.assertIn("default: false", option)
        gate = "if: github.event_name == 'workflow_dispatch' && inputs.finalize_release"
        finalizer = next(step for step in workflow.split("\n      - ")
                         if "name: Attach verified draft assets and publish last" in step)
        self.assertIn(gate, finalizer)
        self.assertIn("await finalize({", finalizer)
        retention = next(step for step in workflow.split("\n      - ")
                         if "name: Retain exact signed payload before publication" in step)
        self.assertNotIn("if:", retention)
        self.assertIn("actions/upload-artifact@v7", retention)
        for job in ("pypi", "verify_pypi"):
            header = workflow.split(f"\n  {job}:\n", 1)[1].split("    runs-on:", 1)[0]
            self.assertIn(gate, header)
        runbook = (ROOT / "docs/ops/byte-release-qualification.md").read_text()
        self.assertIn("recover_github_release.yml", runbook)
        self.assertIn("Do not use `finalize_release=true`", runbook)

    def test_direct_pypi_publish_waits_for_verified_github_release(self) -> None:
        workflow = (ROOT / ".github/workflows/release_wheels.yml").read_text(encoding="utf-8")
        publish = workflow.split("\n  pypi:\n", 1)[1].split("\n  verify_pypi:\n", 1)[0]
        needs = publish.split("\n    needs:", 1)[1].split("\n    if:", 1)[0]
        self.assertEqual(set(re.findall(r"^      - (\w+)$", needs, re.MULTILINE)),
                         {"wheels", "attach"})
        self.assertIn("github.event_name == 'workflow_dispatch'", publish)
        self.assertIn("inputs.publish_pypi == 'true'", publish)
        self.assertIn("inputs.finalize_release", publish)
        self.assertNotIn("always()", publish)
        self.assertNotIn("continue-on-error", publish)
        attach = workflow.split("\n  attach:\n", 1)[1]
        self.assertIn("    needs: wheels\n", attach)
        self.assertNotIn("continue-on-error", attach)
        self.assertIn("await finalize({", attach)

    def test_draft_recovery_reads_exact_retained_payload_without_regeneration(self) -> None:
        recovery = (ROOT / ".github/workflows/recover_github_release.yml").read_text()
        self.assertIn("recoveryArtifact({", recovery)
        self.assertIn("ref: ${{ github.sha }}", recovery)
        self.assertIn("ref: ${{ inputs.release_tag }}", recovery)
        self.assertIn("path: release-source", recovery)
        self.assertIn("cwd: 'release-source'", recovery)
        self.assertIn("artifact-ids: ${{ steps.source.outputs.artifact_id }}", recovery)
        self.assertIn("run-id: ${{ inputs.source_run_id }}", recovery)
        self.assertIn("dist: 'retained-dist'", recovery)
        self.assertIn("gh attestation verify", recovery)
        self.assertIn("attestations: read", recovery)
        self.assertNotIn("attestations: write", recovery)
        self.assertIn('--source-ref "refs/tags/$RELEASE_TAG"', recovery)
        self.assertIn('--source-digest "$SOURCE_SHA"', recovery)
        self.assertIn("--deny-self-hosted-runners", recovery)
        self.assertIn("--cert-identity ", recovery)
        self.assertNotIn("--signer-workflow ", recovery)
        self.assertNotIn("--signer-repo ", recovery)
        self.assertNotIn("--cert-identity-regex ", recovery)
        self.assertIn("group: release-wheels-${{ inputs.release_tag }}", recovery)
        for forbidden in [
            "cargo ", "maturin ", "generate_repo_manifest", "sbom-action",
            "gh-action-sigstore-python", "attest-build-provenance",
            "pypi-publish", "id-token: write",
        ]:
            self.assertNotIn(forbidden, recovery)

    def test_release_stages_verified_draft_before_publishing(self) -> None:
        workflow = (ROOT / ".github/workflows/release_wheels.yml").read_text()
        self.assertNotIn("softprops/action-gh-release", workflow)
        self.assertNotIn("overwrite_files", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("inputs.release_tag || github.ref_name", workflow)
        self.assertIn("await preflight({ github, repo: context.repo", workflow)
        self.assertIn("await finalize({", workflow)
        self.assertIn("execFileSync('git', ['rev-parse', 'HEAD']", workflow)
        self.assertIn("actions/github-script@v8", workflow)
        attach = workflow.partition("\n  attach:\n")[2]
        self.assertIn("pattern: wheels-*", attach)
        self.assertIn("name: signed-release-payload-", workflow)
        self.assertLess(
            workflow.index("Retain exact signed payload before publication"),
            workflow.index("Attach verified draft assets and publish last"),
        )
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("node --test tests/test_finalize_github_release.cjs", ci)
        runbook = (ROOT / "docs/ops/release.md").read_text()
        self.assertIn('--verify-tag --draft', runbook)
        self.assertIn("Do not publish an empty release", runbook)

    def test_release_docs_match_active_package_version(self) -> None:
        metadata = (ROOT / "bindings/st-py/pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE).group(1)
        for path in ["docs/ops/release.md"]:
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                versions = re.findall(r"\b(?:VERSION=|release_tag=v)(\d+\.\d+\.\d+)", text)
                self.assertTrue(versions)
                self.assertEqual(set(versions), {version})
        runbook = (ROOT / "docs/ops/release.md").read_text(encoding="utf-8")
        self.assertIn(f"Expected pre-publish shape for `{version}`", runbook)
        self.assertIn("--no-clipboard", runbook)
        package_readme = (ROOT / "bindings/st-py/README.md").read_text(encoding="utf-8")
        self.assertIn(f"Version {version}", package_readme)
        changelog = (ROOT / "bindings/st-py/CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## {version}\n", changelog)

    def test_wheel_smoke_includes_native_layer_norm_learning(self) -> None:
        smoke = (ROOT / "tools/smoke_autograd.py").read_text(encoding="utf-8")
        self.assertIn('example.with_name("autograd_layer_norm.py")', smoke)
        self.assertIn('runpy.run_path(str(layer_norm))["run"]()', smoke)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("test_layer_norm_autograd.py", ci)
        self.assertIn("--test layer_norm_autograd", ci)
        self.assertIn("node bindings/st-wasm/tests/layer_norm_autograd.cjs", ci)

    def test_native_weight_loading_is_gated_without_foreign_ml(self) -> None:
        for filename in ("ci.yml", "release_wheels.yml"):
            workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
            self.assertIn("bindings/st-py/tests/test_native_weights.py -v", workflow)
            self.assertIn("bindings/st-py/tests/test_native_causal_lm.py -v", workflow)
            self.assertIn("bindings/st-py/tests/test_native_causal_lm_evaluation.py -v", workflow)
            self.assertIn("bindings/st-py/examples/native_weights_geometry.py", workflow)
        test = (ROOT / "bindings/st-py/tests/test_native_weights.py").read_text(encoding="utf-8")
        self.assertIn("NoForeignML()", test)
        self.assertIn('"safetensors"', test)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("bindings/st-py/tests/test_native_weights_parity.py", ci)
        self.assertIn("bindings/st-py/tests/test_native_causal_lm_parity.py", ci)
        self.assertIn("bindings/st-py/tests/test_native_causal_lm_learning_parity.py", ci)
        self.assertIn("bindings/st-py/tests/test_native_causal_lm_evaluation_parity.py", ci)
        self.assertIn("python -I -B tests/test_native_gpt2_probe.py", ci)
        self.assertIn("python -I -B tests/test_native_gpt2_learning_probe.py", ci)

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

    def test_native_readout_and_evaluation_gate_each_installed_wheel_and_python_floor(self) -> None:
        scripts = ("bindings/st-py/tests/test_native_geometry_adamw.py -v",
                   "bindings/st-py/tests/test_native_causal_lm_readout.py -v",
                   "bindings/st-py/tests/test_native_causal_lm_evaluation.py -v")
        floor_step = "      - name: Verify installed wheel on CPython 3.8 (Linux)\n"
        for filename in ("ci.yml", "wheels.yml", "release_wheels.yml"):
            with self.subTest(workflow=filename):
                workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
                normal, floor = workflow.split(floor_step, 1)
                floor = floor.split("\n      - ", 1)[0]
                prefix = "" if filename == "ci.yml" else "../../"
                for script in scripts:
                    self.assertIn("python -I -B " + prefix + script, normal)
                    self.assertIn("python -I -B " + script, floor)
                    if filename != "ci.yml":
                        self.assertLess(workflow.index("python -I -B " + script),
                                        workflow.index("uses: actions/upload-artifact@"))
                self.assertIn("pip install --no-deps --force-reinstall", floor)
                self.assertNotIn("continue-on-error", floor)

    def test_native_layer_learning_is_gated_without_foreign_ml_or_stubs(self) -> None:
        script = "bindings/st-py/tests/test_nn_native_learning.py -v"
        for filename in ("ci.yml", "wheels.yml", "release_wheels.yml"):
            workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
            normal, floor = workflow.split("      - name: Verify installed wheel on CPython 3.8 (Linux)\n", 1)
            floor = floor.split("\n      - ", 1)[0]
            prefix = "" if filename == "ci.yml" else "../../"
            with self.subTest(workflow=filename):
                self.assertIn("python -I -B " + prefix + script, normal)
                self.assertIn("python -I -B " + script, floor)
        smoke = (ROOT / "bindings/st-py/tests/test_nn_native_learning.py").read_text(encoding="utf-8")
        self.assertIn("sys.meta_path.insert(0, NativeOnly())", smoke)
        self.assertIn('"torch", "transformers", "peft", "accelerate", "numpy", "safetensors"', smoke)
        legacy = (ROOT / "bindings/st-py/tests/test_nn_linear_sequential.py").read_text(encoding="utf-8")
        self.assertNotIn("_ensure_torch_stub", legacy)

    def test_manual_and_official_smokes_identify_external_gradient_basis(self) -> None:
        for filename in ("wheels.yml", "release_wheels.yml"):
            with self.subTest(workflow=filename):
                workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
                invocation = workflow.split("partial_loss = trainer.step_partial(", 1)[1].split(")", 1)[0]
                self.assertIn('gradient_basis="spiraltorch.release.smoke.control.v1"', invocation)

    def test_native_repetition_is_gated_with_forbidden_imports_and_optional_reference(self) -> None:
        script = "bindings/st-py/tests/test_native_causal_lm_repetition.py -v"
        for filename in ("ci.yml", "wheels.yml", "release_wheels.yml"):
            workflow = (ROOT / ".github/workflows" / filename).read_text()
            normal, floor = workflow.split("      - name: Verify installed wheel on CPython 3.8 (Linux)\n", 1)
            prefix = "" if filename == "ci.yml" else "../../"
            self.assertIn("python -I -B " + prefix + script, normal)
            self.assertIn("python -I -B " + script, floor)
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("bindings/st-py/tests/test_native_causal_lm_repetition_parity.py", ci)
        test = (ROOT / "bindings/st-py/tests/test_native_causal_lm_repetition.py").read_text()
        self.assertIn("NoForeignML()", test)

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

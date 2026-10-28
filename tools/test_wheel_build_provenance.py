"""Synthetic CPU-only orchestration checks: no compiler, native import or GPU."""
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wheel_build_provenance as api
from test_byte_release_artifacts import files, install, wheel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"commit": "a" * 40, "tree": "b" * 40, "package_version": "0.4.33",
          "files": {"Cargo.lock": "c" * 64}}


def inputs():
    return {"source": SOURCE, "observed_tools": {"rustc": "synthetic rustc identity"}}


class BuildProvenanceTests(unittest.TestCase):
    def test_original_native_and_public_build_roles_remain_distinct(self):
        for role, system, extra in (
            ("ci-gpu", "Darwin", []),
            ("ci-portable", "Linux", ["--features", "logic,kdsl"]),
            ("release", "Darwin", ["--target", "universal2-apple-darwin", "--features", "logic,kdsl"]),
            ("manual", "Linux", ["--compatibility", "manylinux2014", "--zig", "--features", "logic,kdsl"]),
            ("release", "Windows", ["--features", "logic,kdsl"]),
        ):
            with self.subTest(role=role, system=system):
                plan = api.build_plan(role, system, "/synthetic/out")
                self.assertEqual(plan["command"], ["maturin", "build", "--release", "--locked", "-m",
                                                  "bindings/st-py/Cargo.toml", "--out", "/synthetic/out", *extra])
                self.assertTrue(plan["default_features"])
                self.assertNotIn("--no-default-features", plan["command"])
        for role, system in (("ci-gpu", "Linux"), ("ci-portable", "Darwin"), ("other", "Darwin")):
            with self.assertRaises(ValueError):
                api.build_plan(role, system, "out")

    def test_archive_inspection_does_not_weaken_release_admission(self):
        for target, role, system in (("macosx_11_0_arm64", "ci-gpu", "Darwin"),
                                     ("linux_x86_64", "ci-portable", "Linux")):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                path = wheel(temporary, target=target)
                inventory = api.artifacts.wheel_archive_inventory(path)
                api.validate_wheel_role(inventory, api.build_plan(role, system, "out"))
                with self.assertRaises(ValueError):
                    api.artifacts.wheel_inventory(path)
        with tempfile.TemporaryDirectory() as temporary:
            inventory = api.artifacts.wheel_inventory(wheel(temporary))
            with self.assertRaisesRegex(ValueError, "cannot substitute"):
                api.validate_wheel_role(inventory, api.build_plan("ci-gpu", "Darwin", "out"))
            inventory = api.artifacts.wheel_archive_inventory(wheel(temporary, target="macosx_11_0_arm64"))
            with self.assertRaisesRegex(ValueError, "cannot substitute"):
                api.validate_wheel_role(inventory, api.build_plan("release", "Darwin", "out"))

    def test_public_floor_and_compatibility_cannot_silently_change(self):
        for target, system in (("macosx_13_0_universal2", "Darwin"), ("manylinux_2_28_x86_64", "Linux")):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(target=target):
                inventory = api.artifacts.wheel_inventory(wheel(temporary, target=target))
                with self.assertRaises(ValueError):
                    api.validate_wheel_role(inventory, api.build_plan("release", system, "out"))

    def build_fixture(self, root, *, behavior="success"):
        root = root.resolve()
        output, evidence = root / "out", root / "evidence"

        def fake_build(command, working_directory, timeout_seconds):
            self.assertEqual(command, api.build_plan("ci-gpu", "Darwin", output)["command"])
            self.assertEqual(working_directory, root)
            self.assertEqual(timeout_seconds, api.POSIX_BUILD_SECONDS)
            outcome = behavior.removesuffix("_with_output")
            if behavior.endswith("_with_output") or outcome not in ("failure", "missing", "timeout", "cancel"):
                wheel(output, target="macosx_11_0_arm64")
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 7200)
            if outcome == "cancel":
                raise KeyboardInterrupt
            if behavior == "multiple":
                wheel(output, target="macosx_14_0_universal2")
            if behavior == "invalid":
                next(output.glob("*.whl")).write_bytes(b"invalid original bytes")
            return subprocess.CompletedProcess(command, 19 if outcome == "failure" else 0)

        with mock.patch.object(api.platform, "system", return_value="Darwin"), \
                mock.patch.object(api, "build_inputs", return_value=inputs()), \
                mock.patch.object(api, "source_identity", return_value=SOURCE), \
                mock.patch.object(api, "run_build_command", side_effect=fake_build) as run:
            try:
                result = api.run_build(root, "ci-gpu", output, evidence)
            finally:
                self.assertEqual(run.call_count, 1)
        return result, output, evidence

    def test_retained_original_and_build_inputs_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt, output, evidence = self.build_fixture(Path(temporary))
            self.assertFalse(receipt["release_ready"])
            self.assertEqual(receipt["wheel_sha256"], api.artifacts.sha256(output / receipt["wheel"]["filename"]))
            self.assertEqual(receipt["wheel"], api.artifacts.wheel_archive_inventory(evidence / receipt["wheel"]["filename"]))
            self.assertEqual(receipt["inputs_sha256"], api.artifacts.sha256(evidence / "inputs.json"))
            self.assertEqual(json.loads((evidence / "outcome.json").read_text())["status"],
                             "built_and_retained_not_qualified")
            with self.assertRaises(FileExistsError):
                api.run_build(Path(temporary), "ci-gpu", output, evidence)

    def test_first_build_error_timeout_cancel_or_missing_output_never_retries(self):
        for behavior, error, status in (("failure", RuntimeError, "build_failed"),
                                        ("timeout", subprocess.TimeoutExpired, "build_timeout"),
                                        ("cancel", KeyboardInterrupt, "cancelled"),
                                        ("missing", ValueError, "retention_failed"),
                                        ("multiple", ValueError, "retention_failed")):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(error):
                    self.build_fixture(Path(temporary), behavior=behavior)
                evidence = Path(temporary) / "evidence"
                outcome = json.loads((evidence / "outcome.json").read_text())
                self.assertEqual(outcome["status"], status)
                self.assertFalse(outcome["release_ready"])
                self.assertFalse((evidence / "build.json").exists())
                self.assertTrue((evidence / "inputs.json").is_file())

    def test_existing_output_is_not_rebuilt_or_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            output.mkdir()
            original = wheel(output)
            digest = api.artifacts.sha256(original)
            with mock.patch.object(api.subprocess, "run") as run, \
                    mock.patch.object(api.platform, "system", return_value="Darwin"):
                with self.assertRaisesRegex(ValueError, "already contains"):
                    api.run_build(root, "ci-gpu", output, root / "evidence")
                run.assert_not_called()
            self.assertEqual(api.artifacts.sha256(original), digest)

    def test_outputs_written_before_failure_timeout_or_cancel_are_retained(self):
        for behavior, error, status in (("failure", RuntimeError, "build_failed"),
                                        ("timeout", subprocess.TimeoutExpired, "build_timeout"),
                                        ("cancel", KeyboardInterrupt, "cancelled")):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(error):
                    self.build_fixture(root, behavior=behavior + "_with_output")
                original = next((root / "out").glob("*.whl"))
                retained = root / "evidence" / original.name
                self.assertTrue(retained.is_file(), "the first failed build output must not be lost")
                self.assertEqual(retained.read_bytes(), original.read_bytes())
                outputs = json.loads((root / "evidence/outputs.json").read_text())
                self.assertEqual(outputs["files"][0]["sha256"], api.artifacts.sha256(original))
                self.assertFalse(outputs["release_ready"])
                outcome = json.loads((root / "evidence/outcome.json").read_text())
                self.assertEqual(outcome["status"], status)
                self.assertFalse((root / "evidence/build.json").exists())

    def test_multiple_outputs_are_preserved_without_selecting_a_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "exactly one original wheel"):
                self.build_fixture(root, behavior="multiple")
            outputs = json.loads((root / "evidence/outputs.json").read_text())
            self.assertEqual(len(outputs["files"]), 2)
            for record in outputs["files"]:
                self.assertEqual((root / "evidence" / record["filename"]).read_bytes(),
                                 (root / "out" / record["filename"]).read_bytes())
            self.assertFalse((root / "evidence/build.json").exists())

    def test_retention_record_error_never_replaces_the_first_build_error(self):
        for behavior, error, status in (("failure_with_output", RuntimeError, "build_failed"),
                                        ("timeout_with_output", subprocess.TimeoutExpired, "build_timeout"),
                                        ("cancel_with_output", KeyboardInterrupt, "cancelled"),
                                        ("success", OSError, "retention_failed")):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temporary:
                original_write = api.write_json

                def fail_retention_record(path, value):
                    if Path(path).name == "outputs.json":
                        raise OSError("synthetic retention write failure")
                    original_write(path, value)

                with mock.patch.object(api, "write_json", side_effect=fail_retention_record):
                    with self.assertRaises(error):
                        self.build_fixture(Path(temporary), behavior=behavior)
                evidence = Path(temporary) / "evidence"
                self.assertEqual(len(list(evidence.glob("*.whl"))), 1)
                outcome = json.loads((evidence / "outcome.json").read_text())
                self.assertEqual(outcome["status"], status)
                if behavior != "success":
                    self.assertEqual(outcome["retention_error_type"], "OSError")
                self.assertFalse((evidence / "build.json").exists())

    def test_unsafe_output_is_not_read_and_other_raw_outputs_are_preserved(self):
        for kind in ("symlink", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, evidence = root / "out", root / "evidence"
                output.mkdir()
                evidence.mkdir()
                unsafe, safe = output / "a.whl", output / "b.whl"
                safe.write_bytes(b"synthetic raw output")
                if kind == "symlink":
                    private = root / "private"
                    private.write_bytes(b"must not be read or uploaded")
                    unsafe.symlink_to(private)
                else:
                    unsafe.mkdir()
                original_digest = api.artifacts.sha256

                def digest(path):
                    self.assertNotEqual(path, unsafe)
                    return original_digest(path)

                with mock.patch.object(api.artifacts, "sha256", side_effect=digest):
                    with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                        api.retain_build_outputs(output, evidence)
                self.assertFalse((evidence / unsafe.name).exists())
                self.assertEqual((evidence / safe.name).read_bytes(), safe.read_bytes())
                record = json.loads((evidence / "outputs.json").read_text())
                self.assertFalse(record["complete"])
                self.assertEqual([entry["status"] for entry in record["files"]],
                                 ["not_retained", "retained"])

    def test_output_count_and_aggregate_copy_bytes_are_bounded(self):
        for kind in ("count", "bytes"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, evidence = root / "out", root / "evidence"
                output.mkdir()
                evidence.mkdir()
                count = api.MAX_RETAINED_OUTPUTS + 1 if kind == "count" else 2
                for index in range(count):
                    (output / f"output-{index}.whl").write_bytes(b"1234")
                with mock.patch.object(api.artifacts, "MAX_TOTAL", 6):
                    with self.assertRaisesRegex(ValueError, "retention.*bound"):
                        api.retain_build_outputs(output, evidence)
                self.assertLessEqual(sum(path.stat().st_size for path in evidence.glob("*.whl")), 6)
                record = json.loads((evidence / "outputs.json").read_text())
                self.assertEqual(record["output_count"], count)
                self.assertFalse(record["complete"])

    def test_growing_output_does_not_extend_the_original_copy_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, evidence = root / "out", root / "evidence"
            output.mkdir()
            evidence.mkdir()
            original = output / "growing.whl"
            original.write_bytes(b"1234")
            original_digest = api.artifacts.sha256

            def grow(path):
                digest = original_digest(path)
                if path == original:
                    with path.open("ab") as stream:
                        stream.write(b"extra bytes")
                return digest

            with mock.patch.object(api.artifacts, "sha256", side_effect=grow):
                with self.assertRaisesRegex(ValueError, "grew during retention"):
                    api.retain_build_outputs(output, evidence)
            self.assertEqual((evidence / original.name).stat().st_size, 4)
            self.assertFalse(json.loads((evidence / "outputs.json").read_text())["complete"])
            self.assertFalse((evidence / "original.json").exists())

    def test_invalid_archive_is_retained_without_approving_or_rebuilding_it(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(zipfile.BadZipFile):
                self.build_fixture(Path(temporary), behavior="invalid")
            evidence = Path(temporary) / "evidence"
            record = json.loads((evidence / "original.json").read_text())
            self.assertEqual((evidence / record["filename"]).read_bytes(), b"invalid original bytes")
            self.assertFalse(record["release_ready"])
            self.assertFalse((evidence / "build.json").exists())

    def test_changed_source_or_package_version_retains_bytes_without_a_build_receipt(self):
        for defect in ("changed_source", "version"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                output, evidence = root / "out", root / "evidence"
                initial = inputs()
                after = SOURCE
                if defect == "version":
                    initial["source"] = dict(SOURCE, package_version="0.4.34")
                else:
                    after = dict(SOURCE, tree="d" * 40)

                def build(command, *_args):
                    wheel(output, target="macosx_11_0_arm64")
                    return subprocess.CompletedProcess(command, 0)

                with mock.patch.object(api.platform, "system", return_value="Darwin"), \
                        mock.patch.object(api, "build_inputs", return_value=initial), \
                        mock.patch.object(api, "source_identity", return_value=after), \
                        mock.patch.object(api, "run_build_command", side_effect=build) as run:
                    with self.assertRaises(ValueError):
                        api.run_build(root, "ci-gpu", output, evidence)
                    run.assert_called_once()
                self.assertTrue((evidence / "original.json").is_file())
                self.assertFalse((evidence / "build.json").exists())

    def test_installed_files_checked_without_loading_native_and_never_qualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, output, evidence = self.build_fixture(root)
            site = root / "site"
            install(site, files("macosx_11_0_arm64"))
            def identity(inventory):
                return api.artifacts.verify_installed_files(
                    inventory, site, site / "spiraltorch/__init__.py", "0.4.33")
            with mock.patch.object(api.artifacts, "installed_identity", side_effect=identity):
                result = api.verify_install(evidence, output)
            self.assertEqual(result["native_sha256"], receipt["wheel"]["native"]["sha256"])
            self.assertFalse(json.loads((evidence / "installed.json").read_text())["release_ready"])

    def test_changed_original_retained_input_or_installed_native_is_rejected(self):
        for changed in ("original", "retained", "inputs", "native"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                receipt, output, evidence = self.build_fixture(root)
                site = root / "site"
                install(site, files("macosx_11_0_arm64"))
                paths = {"original": output / receipt["wheel"]["filename"],
                         "retained": evidence / receipt["wheel"]["filename"],
                         "inputs": evidence / "inputs.json", "native": site / receipt["wheel"]["native"]["path"]}
                with paths[changed].open("ab") as stream:
                    stream.write(b" ")
                def identity(inventory):
                    return api.artifacts.verify_installed_files(
                        inventory, site, site / "spiraltorch/__init__.py", "0.4.33")
                with mock.patch.object(api.artifacts, "installed_identity", side_effect=identity):
                    with self.assertRaises(ValueError):
                        api.verify_install(evidence, output)
                self.assertFalse((evidence / "installed.json").exists())
                outcome = json.loads((evidence / "installation-outcome.json").read_text())
                self.assertEqual(outcome["status"], "installation_rejected")

    def test_outcome_write_failure_does_not_hide_original_build_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_write = api.write_json

            def fail_outcome(path, value):
                if Path(path).name == "outcome.json":
                    raise OSError("private-metadata-error")
                original_write(path, value)

            with mock.patch.object(api, "write_json", side_effect=fail_outcome), \
                    mock.patch.object(api.sys, "stderr"):
                with self.assertRaisesRegex(RuntimeError, "original build command failed"):
                    self.build_fixture(Path(temporary), behavior="failure")

    def test_compiler_process_ownership_is_explicit(self):
        self.assertLessEqual(api.POSIX_BUILD_SECONDS, api.WINDOWS_STEP_SECONDS - 600)
        with mock.patch.object(api.os, "name", "posix"), mock.patch.object(api.owned_process, "run") as owned:
            api.run_build_command(["synthetic"], "source", 7200)
            owned.assert_called_once_with(["synthetic"], cwd="source", timeout=7200, stdout=None, stderr=None)
        with mock.patch.object(api.os, "name", "nt"), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(api.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                api.run_build_command(["synthetic"], "source", 7200)
            run.assert_not_called()
        with mock.patch.object(api.os, "name", "nt"), \
                mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True), \
                mock.patch.object(api.subprocess, "run") as run:
            api.run_build_command(["synthetic"], "source", 7200)
            run.assert_called_once_with(["synthetic"], cwd="source", check=False)

    def test_post_step_collection_preserves_an_interrupted_first_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, evidence = root / "out", root / "evidence"
            output.mkdir()
            evidence.mkdir()
            original = wheel(output, target="win_amd64")
            first_copy = evidence / original.name
            first_copy.write_bytes(b"interrupted first copy")
            (evidence / "outputs.json").write_text('{"complete":')
            with mock.patch.object(api, "run_build_command") as build:
                retained = api.collect_after_step(output, evidence)
                build.assert_not_called()
                with self.assertRaises(FileExistsError):
                    api.collect_after_step(output, evidence)
            self.assertEqual(first_copy.read_bytes(), b"interrupted first copy")
            self.assertEqual((evidence / "outputs.json").read_text(), '{"complete":')
            self.assertEqual((evidence / "after-step" / original.name).read_bytes(), original.read_bytes())
            self.assertEqual(retained[0]["sha256"], api.artifacts.sha256(original))
            outcome = json.loads((evidence / "after-step/outcome.json").read_text())
            self.assertFalse(outcome["release_ready"])
            self.assertEqual(outcome["status"], "raw_outputs_retained_not_qualified")
            self.assertFalse((evidence / "after-step/build.json").exists())

    def test_post_step_missing_output_is_not_successful_build_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(api, "run_build_command") as build:
                self.assertEqual(api.collect_after_step(root / "missing", root / "evidence"), [])
                build.assert_not_called()
            outcome = json.loads((root / "evidence/after-step/outcome.json").read_text())
            self.assertEqual(outcome["status"], "no_outputs_found")
            self.assertFalse(outcome["release_ready"])

    def test_actual_tools_must_match_pins_without_dumping_secrets(self):
        for rust, maturin, env, accepted in (
            (api.RUST, api.MATURIN, {}, True), ("1.97.0", api.MATURIN, {}, False),
            (api.RUST, "1.11.5", {}, False),
            (api.RUST, api.MATURIN, {"RUSTFLAGS": "private-secret"}, False),
            (api.RUST, api.MATURIN, {"MACOSX_DEPLOYMENT_TARGET": "14.0"}, False),
        ):
            with self.subTest(rust=rust, maturin=maturin, env=bool(env)), tempfile.TemporaryDirectory() as temporary:
                # Cargo's patch version need not equal the Rust release patch.
                outputs = [f"rustc {rust}\nrelease: {rust}", "cargo 1.99.0 (synthetic)",
                           f"maturin {maturin}", "cargo 1.99.0 (synthetic)", "26.0"]
                with mock.patch.dict(os.environ, {"RUSTUP_TOOLCHAIN": api.RUST, **env}, clear=True), \
                        mock.patch.object(api, "source_identity", return_value=SOURCE), \
                        mock.patch.object(api, "probe", side_effect=outputs):
                    if accepted:
                        value = api.build_inputs(Path(temporary), api.build_plan("ci-gpu", "Darwin", "out"))
                        self.assertEqual(value["requested_pins"], {"rust": api.RUST, "maturin": api.MATURIN})
                    else:
                        with self.assertRaises(ValueError) as error:
                            api.build_inputs(Path(temporary), api.build_plan("ci-gpu", "Darwin", "out"))
                        self.assertNotIn("private-secret", str(error.exception))

    def test_build_package_metadata_is_visible_under_no_site_without_imports(self):
        distribution = mock.Mock(metadata={"Name": "maturin"}, version=api.MATURIN)
        with mock.patch.object(api.site, "getsitepackages", return_value=["synthetic-site"]), \
                mock.patch.object(api.importlib.metadata, "distributions", return_value=[distribution]) as discover:
            versions = api.installed_build_packages()
            self.assertEqual(versions["maturin"], api.MATURIN)
            self.assertIsNone(versions["ziglang"])
            discover.assert_called_once_with(path=["synthetic-site"])
        with mock.patch.object(api.importlib.metadata, "distributions", return_value=[distribution, distribution]):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                api.installed_build_packages()

    def test_no_site_venv_metadata_uses_the_lexical_executable_not_its_symlink_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            executable = root / "bin/python"
            executable.symlink_to(sys.executable)
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            site = Path(api.site.getsitepackages([str(root)])[0])
            metadata = site / "maturin-1.15.0.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text("Name: maturin\nVersion: 1.15.0\n")
            (site / "maturin.py").write_text("raise RuntimeError('package code must not be imported')\n")
            (site / "startup.pth").write_text("import sys; raise RuntimeError('site hooks must not run')\n")
            original_path = list(sys.path)
            with mock.patch.object(api.sys, "executable", str(executable)):
                self.assertIn(str(site), api.build_package_locations())
                versions = api.installed_build_packages()
            self.assertEqual(versions["maturin"], api.MATURIN)
            self.assertIsNone(versions["pip"])
            self.assertNotIn("maturin", sys.modules)
            self.assertEqual(sys.path, original_path)

    def test_system_site_metadata_has_lower_precedence_than_venv_metadata(self):
        venv_maturin = mock.Mock(metadata={"Name": "maturin"}, version=api.MATURIN)
        base_maturin = mock.Mock(metadata={"Name": "maturin"}, version="1.11.5")
        base_pip = mock.Mock(metadata={"Name": "pip"}, version="synthetic-pip")
        with mock.patch.object(api, "build_package_locations", return_value=["venv-site", "base-site"]), \
                mock.patch.object(api.importlib.metadata, "distributions",
                                  side_effect=[[venv_maturin], [base_maturin, base_pip]]):
            versions = api.installed_build_packages()
        self.assertEqual(versions["maturin"], api.MATURIN)
        self.assertEqual(versions["pip"], "synthetic-pip")

    def test_venv_config_controls_system_site_visibility_without_loading_site_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for enabled in ("true", "false"):
                (root / "pyvenv.cfg").write_text("include-system-site-packages = " + enabled + "\n")
                with mock.patch.object(api.sys, "executable", str(root / "bin/python")):
                    locations = api.build_package_locations()
                    bases = api.site.getsitepackages([sys.base_prefix, sys.base_exec_prefix])
                    self.assertEqual(all(base in locations for base in bases), enabled == "true")

    def test_distinct_system_lib64_and_lib_roots_follow_cpython_metadata_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv = root / "venv"
            venv.mkdir()
            (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
            base = root / "base"
            for libdir, version in (("lib64", "1.15.0"), ("lib", "1.11.5")):
                path = base / libdir / ("python%d.%d" % sys.version_info[:2]) / "site-packages"
                metadata = path / ("maturin-" + version + ".dist-info")
                metadata.mkdir(parents=True)
                (metadata / "METADATA").write_text("Name: maturin\nVersion: " + version + "\n")
            with mock.patch.object(api.sys, "executable", str(venv / "bin/python")), \
                    mock.patch.object(api.sys, "base_prefix", str(base)), \
                    mock.patch.object(api.sys, "base_exec_prefix", str(base)), \
                    mock.patch.object(api.sys, "platlibdir", "lib64"):
                locations = api.build_package_locations()
                self.assertEqual(locations, api.site.getsitepackages([str(venv), str(base)]))
                self.assertIn("lib64", locations[2])
                self.assertEqual(api.installed_build_packages()["maturin"], "1.15.0")

    def test_all_wheel_jobs_pin_tools_and_preserve_bytes_before_execution(self):
        for filename, job, role in (("ci.yml", "wgpu-macos", "ci-gpu"),
                                    ("ci.yml", "python-docs-smoke", "ci-portable"),
                                    ("wheels.yml", "build-wheels", "manual"),
                                    ("release_wheels.yml", "wheels", "release")):
            with self.subTest(role=role):
                text = (ROOT / ".github/workflows" / filename).read_text()
                import re
                block = re.split(r"\n  [\w-]+:\n", text.split(f"\n  {job}:\n", 1)[1], maxsplit=1)[0]
                self.assertIn("wheel_build_provenance.py pins", block)
                self.assertIn("toolchain: ${{ steps.wheel_pins.outputs.rust }}", block)
                self.assertIn("RUSTUP_TOOLCHAIN: ${{ steps.wheel_pins.outputs.rust }}", block)
                self.assertIn("maturin==${{ steps.wheel_pins.outputs.maturin }}", block)
                self.assertEqual(block.count(f"wheel_build_provenance.py build --role {role}"), 1)
                steps = block.split("\n      - ")
                build = next(step for step in steps if "id: wheel_build" in step)
                self.assertIn("timeout-minutes: 120", build)
                self.assertNotIn("pip install", build)
                preparation = next(step for step in steps if "name: Install pinned wheel build tools" in step)
                self.assertLess(block.index(preparation), block.index(build))
                recovery = next(step for step in steps if "wheel_build_provenance.py collect-after-step" in step)
                self.assertIn("if: always() && (steps.wheel_build.outcome == 'failure' || "
                              "steps.wheel_build.outcome == 'cancelled')", recovery)
                self.assertIn("timeout-minutes: 10", recovery)
                self.assertNotIn("build --role", recovery)
                self.assertLess(block.index(build), block.index(recovery))
                upload = next(step for step in steps if f"name: unqualified-wheel-{role}" in step)
                self.assertLess(block.index(recovery), block.index(upload))
                self.assertIn("if: always() && steps.wheel_build.outcome != 'skipped'", upload)
                self.assertIn("if-no-files-found: error", upload)
                self.assertNotIn("continue-on-error", block)
                self.assertLess(block.index("id: wheel_build"), block.index(upload))
                self.assertLess(block.index(upload), block.index("wheel_build_provenance.py verify-install"))
                name = upload.split("name: unqualified-", 1)[1].splitlines()[0]
                self.assertFalse(fnmatch.fnmatch("unqualified-" + name, "wheels-*"))
                self.assertIn("--no-compile", block)
                if role in ("manual", "release"):
                    qualified = next(step for step in steps if "path: target/wheels/*.whl" in step)
                    self.assertNotIn("if: always()", qualified)
                    self.assertNotIn("if: failure()", qualified)
                    self.assertLess(block.index("Verify installed wheel on CPython 3.8"), block.index(qualified))


if __name__ == "__main__":
    unittest.main()

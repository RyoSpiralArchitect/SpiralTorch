#!/usr/bin/env python3
"""License preflight and byte-preserving historical fixture declarations."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/security/generate_repo_manifest.py"
SPEC = importlib.util.spec_from_file_location("repo_license_manifest", SCRIPT)
manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest
SPEC.loader.exec_module(manifest)
FIXTURE = Path("benchmarks/results/2000-01-01-fixture/reproduction/shim/Cargo.toml")


class RepositoryLicenseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.tracked = set()
        self.write("LICENSE .txt", "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n")
        self.write("NOTICE", "Project license: AGPL-3.0-or-later\n")

    def write(self, name: str | Path, text: str) -> Path:
        relative = Path(name)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.tracked.add(relative)
        return path

    def fixture(self, extra: str = "") -> dict:
        path = self.write(FIXTURE, '[package]\nname = "fixture"\nversion = "0.0.0"\n'
                          'publish = false\n' + extra)
        return {"manifest": FIXTURE.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "license": "AGPL-3.0-or-later"}

    def catalog(self, entries: list) -> None:
        self.write(manifest.FROZEN_LICENSE_DECLARATIONS,
                   json.dumps({"schema_version": 1, "declarations": entries}))

    def gather(self) -> dict:
        return manifest.gather_compliance_metadata(self.root, iter(self.tracked))

    def git_index(self) -> None:
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True,
                       capture_output=True)
        subprocess.run(["git", "add", "--", *sorted(str(p) for p in self.tracked)],
                       cwd=self.root, check=True, capture_output=True)

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT),
                               "--repo-root", str(self.root), *args],
                              capture_output=True, text=True, timeout=15)

    def test_project_and_vendor_declarations_keep_their_scopes(self) -> None:
        self.write("Cargo.toml", '[workspace.package]\nlicense = "AGPL-3.0-or-later"\n')
        self.write("vendor/third/Cargo.toml", '[package]\nname = "third"\nlicense = "MIT"\n')
        self.write("bindings/pyproject.toml", '[project]\nname = "py"\nlicense = {text = "AGPL-3.0-or-later"}\n')
        records = self.gather()
        self.assertEqual([r["license_scope"] for r in records["cargo"]], ["project", "third_party"])
        self.assertEqual(len(records["python"]), 1)
        self.assertNotIn("license_source", records["cargo"][0])

    def test_real_frozen_declarations_survive_windows_checkout_conversion(self) -> None:
        catalog = json.loads((ROOT / manifest.FROZEN_LICENSE_DECLARATIONS).read_text(encoding="utf-8"))
        for entry in catalog["declarations"]:
            with self.subTest(manifest=entry["manifest"]):
                result = subprocess.run(
                    ["git", "-c", "core.autocrlf=true", "cat-file", "--filters", "HEAD:" + entry["manifest"]],
                    cwd=ROOT, check=True, capture_output=True, timeout=15,
                )
                self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), entry["sha256"])

    def test_active_missing_or_conflicting_license_still_fails(self) -> None:
        for extra in ["", 'license = "MIT"\n']:
            with self.subTest(extra=extra):
                self.write("benchmarks/active/Cargo.toml", '[package]\nname = "active"\n' + extra)
                with self.assertRaises(SystemExit):
                    self.gather()

    def test_exact_frozen_bytes_are_supplemented_and_catalog_is_hashed(self) -> None:
        self.catalog([self.fixture()])
        before = (self.root / FIXTURE).read_bytes()
        records = self.gather()
        self.assertEqual(records["cargo"][0]["license"], "AGPL-3.0-or-later")
        self.assertEqual(records["cargo"][0]["license_scope"], "project")
        self.assertEqual(records["cargo"][0]["license_source"],
                         manifest.FROZEN_LICENSE_DECLARATIONS.as_posix())
        full = manifest.build_manifest(self.root, sorted(self.tracked), {})
        by_path = {r["path"]: r for r in full["files"]}
        self.assertIn(manifest.FROZEN_LICENSE_DECLARATIONS.as_posix(), by_path)
        self.assertEqual(by_path[FIXTURE.as_posix()]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual((self.root / FIXTURE).read_bytes(), before)

    def test_no_directory_wide_exemption(self) -> None:
        self.catalog([self.fixture()])
        self.write(FIXTURE.with_name("extra") / "Cargo.toml", '[package]\nname = "new"\n')
        with self.assertRaisesRegex(SystemExit, "missing an AGPL license declaration"):
            self.gather()

    def test_changed_historical_bytes_fail(self) -> None:
        self.catalog([self.fixture()])
        path = self.root / FIXTURE
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
            self.gather()

    def test_duplicate_wrong_license_or_wrong_digest_fail(self) -> None:
        entry = self.fixture()
        for entries in [[entry, entry], [{**entry, "license": "MIT"}],
                        [{**entry, "sha256": "0" * 64}]]:
            with self.subTest(entries=entries):
                self.catalog(entries)
                with self.assertRaises(SystemExit):
                    self.gather()

    def test_catalog_cannot_override_manifest_license_or_publication(self) -> None:
        for extra in ['license = "MIT"\n', 'license = "AGPL-3.0-or-later"\n',
                      'license-file = "LICENSE"\n']:
            with self.subTest(extra=extra):
                self.catalog([self.fixture(extra)])
                with self.assertRaisesRegex(SystemExit, "only for"):
                    self.gather()
        entry = self.fixture()
        path = self.root / FIXTURE
        for value in ['true', '["private"]']:
            path.write_text('[package]\nname = "fixture"\npublish = ' + value + '\n')
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.catalog([entry])
            with self.assertRaisesRegex(SystemExit, "only for"):
                self.gather()

    def test_catalog_rejects_untracked_and_out_of_scope_paths(self) -> None:
        entry = self.fixture()
        for name in ["Cargo.toml", "vendor/pkg/Cargo.toml", "/tmp/Cargo.toml",
                     "benchmarks/results/../escape/reproduction/shim/Cargo.toml",
                     "benchmarks/results/2000-01-01-fixture/active/shim/Cargo.toml",
                     "benchmarks/results/2000-01-01-fixture/reproduction/other/Cargo.toml"]:
            with self.subTest(name=name):
                self.catalog([{**entry, "manifest": name}])
                with self.assertRaisesRegex(SystemExit, "Not a tracked frozen"):
                    self.gather()

    def test_missing_and_symlinked_fixture_fail(self) -> None:
        self.catalog([self.fixture()])
        path = self.root / FIXTURE
        contents = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(SystemExit, "regular repository file"):
            self.gather()
        other = self.write("other", contents.decode())
        path.symlink_to(other)
        with self.assertRaisesRegex(SystemExit, "regular repository file"):
            self.gather()

    def test_untracked_catalog_cannot_authorize_declarations(self) -> None:
        self.catalog([self.fixture()])
        self.tracked.remove(manifest.FROZEN_LICENSE_DECLARATIONS)
        with self.assertRaisesRegex(SystemExit, "must be tracked"):
            self.gather()

    def test_bad_catalog_schema_fails(self) -> None:
        for value in [[], {}, {"schema_version": 2, "declarations": []},
                      {"schema_version": 1, "declarations": []},
                      {"schema_version": 1, "declarations": [None]}]:
            with self.subTest(value=value):
                self.write(manifest.FROZEN_LICENSE_DECLARATIONS, json.dumps(value))
                with self.assertRaises(SystemExit):
                    self.gather()

    def test_preflight_is_read_only_but_normal_generation_rejects_dirty_tree(self) -> None:
        self.catalog([self.fixture()])
        self.git_index()
        self.write("untracked-weight.bin", "not part of the license scan")
        before = {p: (self.root / p).read_bytes() for p in self.tracked}
        result = self.cli("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No release manifest written", result.stdout)
        self.assertFalse((self.root / "spiraltorch-repo-license-manifest.json").exists())
        self.assertEqual(before, {p: (self.root / p).read_bytes() for p in self.tracked})
        result = self.cli("--allow-untracked")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Working tree has modifications", result.stderr)

    def test_preflight_does_not_construct_or_hash_full_manifest(self) -> None:
        self.catalog([self.fixture()])
        self.git_index()
        argv = [str(SCRIPT), "--repo-root", str(self.root), "--check-only"]
        with patch.object(sys, "argv", argv), patch.object(manifest, "build_manifest") as build:
            with contextlib.redirect_stdout(io.StringIO()):
                manifest.main()
        build.assert_not_called()

    def test_preflight_output_flag_and_missing_license_fail(self) -> None:
        self.fixture()
        self.git_index()
        result = self.cli("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing an AGPL license declaration", result.stderr)
        result = self.cli("--check-only", "--output", str(self.root / "out.json"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "out.json").exists())

    def test_preflight_preserves_canonical_and_notice_checks(self) -> None:
        self.catalog([self.fixture()])
        self.git_index()
        self.write("NOTICE", "no license mentioned")
        self.assertNotEqual(self.cli("--check-only").returncode, 0)
        self.write("NOTICE", "AGPL")
        self.write("LICENSE .txt", "not the license")
        self.assertNotEqual(self.cli("--check-only").returncode, 0)

    def test_normal_manifest_seal_and_clone_verifier_round_trip(self) -> None:
        self.catalog([self.fixture()])
        self.git_index()
        subprocess.run(["git", "-c", "user.name=License Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                        "commit", "--quiet", "-m", "fixture"], cwd=self.root,
                       check=True, capture_output=True)
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "spiraltorch-repo-license-manifest.json"
        seal = self.root / "spiraltorch-compliance-seal.json"
        result = subprocess.run([sys.executable, "-I", "-B",
                                 str(ROOT / "scripts/security/generate_compliance_seal.py"),
                                 "--repo-root", str(self.root), "--manifest", str(output)],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        verify = [sys.executable, "-I", "-B", str(ROOT / "scripts/security/verify_repo_clone.py"),
                  "--repo-root", str(self.root), "--manifest", str(output), "--seal", str(seal)]
        result = subprocess.run(verify, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source = self.root / manifest.FROZEN_LICENSE_DECLARATIONS
        source.write_text(source.read_text() + "\n")
        result = subprocess.run(verify, text=True, capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch", (result.stdout + result.stderr).lower())


if __name__ == "__main__":
    unittest.main()

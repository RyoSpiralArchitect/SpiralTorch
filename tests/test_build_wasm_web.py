"""Build-profile tests with a fake wasm-pack; no Rust toolchain is required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildWasmWebTests(unittest.TestCase):
    def run_profile(self, *arguments):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture = root / "capture"
            fake = fake_bin / "wasm-pack"
            fake.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf 'rust=%s\\nencoded=%s\\nbuild=%s\\ntarget=%s\\nlibrary=%s\\npkg_config=%s\\nargs=%s\\n' \\
  "${RUSTFLAGS-unset}" "${CARGO_ENCODED_RUSTFLAGS-unset}" \\
  "${CARGO_BUILD_RUSTFLAGS-unset}" \\
  "${CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS-unset}" \\
  "${LIBRARY_PATH-unset}" "${PKG_CONFIG_PATH-unset}" "$*" > "$CAPTURE"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--out-dir" ]]; then out="$2"; shift 2; else shift; fi
done
mkdir -p "$out"
touch "$out/spiraltorch_wasm.js"
"""
            )
            fake.chmod(0o755)
            crate = root / "crate"
            output = root / "pkg"
            examples = root / "examples"
            crate.mkdir()
            (examples / "demo").mkdir(parents=True)
            env = os.environ.copy()
            env.update(
                PATH=f"{fake_bin}:{env['PATH']}",
                CAPTURE=str(capture),
                CRATE_DIR=str(crate),
                OUT_DIR=str(output),
                EXAMPLES_DIR=str(examples),
                RUSTFLAGS="host-rustflags",
                CARGO_ENCODED_RUSTFLAGS="host-encoded-flags",
                CARGO_BUILD_RUSTFLAGS="build-wide-flags",
                CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="ambient-target-flags",
                LIBRARY_PATH="host-libraries",
                PKG_CONFIG_PATH="host-pkg-config",
            )
            subprocess.run(
                ["bash", str(ROOT / "scripts/build_wasm_web.sh"), *arguments],
                check=True,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            return capture.read_text()

    def test_portable_profile_strips_all_rust_flag_sources(self):
        capture = self.run_profile("--dev")
        self.assertIn(
            "rust=unset\nencoded=unset\nbuild=unset\ntarget=unset\nlibrary=unset\npkg_config=unset\n",
            capture,
        )
        self.assertIn("--dev", capture)

    def test_portable_profile_accepts_no_wasm_pack_options(self):
        capture = self.run_profile()
        self.assertIn("target=unset", capture)
        self.assertIn("args=build ", capture)

    def test_simd128_profile_is_explicit_and_not_forwarded_to_wasm_pack(self):
        capture = self.run_profile("--release", "--simd128")
        self.assertIn("target=-Ctarget-feature=+simd128", capture)
        self.assertIn("--release", capture)
        self.assertNotIn("--simd128", capture)


if __name__ == "__main__":
    unittest.main()

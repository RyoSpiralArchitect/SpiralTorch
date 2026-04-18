#!/usr/bin/env bash
# Verify that the version string in bindings/st-py/pyproject.toml matches
# bindings/st-py/Cargo.toml. The wheel published to PyPI is built from
# pyproject.toml, but the underlying Rust crate version comes from
# Cargo.toml; if the two drift the wheel metadata silently disagrees
# with the compiled extension's reported crate version.
#
# Run from the repository root: `bash scripts/check_version_consistency.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
pyproject="${repo_root}/bindings/st-py/pyproject.toml"
cargo_toml="${repo_root}/bindings/st-py/Cargo.toml"

extract_version() {
  # First top-level `version = "x.y.z"` line in the file.
  grep -E '^version = "' "$1" | head -1 | sed -E 's/^version = "([^"]+)".*/\1/'
}

py_version="$(extract_version "$pyproject")"
rs_version="$(extract_version "$cargo_toml")"

if [[ -z "$py_version" || -z "$rs_version" ]]; then
  echo "::error::could not parse version from one of:"
  echo "  $pyproject -> '${py_version}'"
  echo "  $cargo_toml -> '${rs_version}'"
  exit 2
fi

if [[ "$py_version" != "$rs_version" ]]; then
  echo "::error::bindings/st-py version mismatch"
  echo "  pyproject.toml -> ${py_version}"
  echo "  Cargo.toml     -> ${rs_version}"
  echo
  echo "Bump both files together; the wheel built by maturin embeds"
  echo "the pyproject.toml version while the crate metadata reports"
  echo "the Cargo.toml version."
  exit 1
fi

echo "OK: bindings/st-py is at ${py_version} (pyproject.toml and Cargo.toml agree)"

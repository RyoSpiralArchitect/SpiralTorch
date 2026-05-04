#!/usr/bin/env bash
set -euo pipefail

NIGHTLY_TOOLCHAIN="${NIGHTLY_TOOLCHAIN:-nightly-2026-04-15}"
STRICT_LIST="${1:-${CLIPPY_STRICT_LIST_FILE:-configs/clippy_strict_packages.txt}}"

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup is required for CI-parity linting because CI tracks the stable channel." >&2
  exit 1
fi

if ! command -v protoc >/dev/null 2>&1 && [[ -x ".buildenv/protoc-bin/protoc" ]]; then
  export PATH="$PWD/.buildenv/protoc-bin:$PATH"
fi

if ! command -v protoc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
protoc was not found on PATH.

Install protobuf-compiler, or place a protoc shim at .buildenv/protoc-bin/protoc.
CI installs protobuf-compiler before running the lint job.
EOF
  exit 1
fi

echo "==> sync Rust stable toolchain"
rustup update stable
rustup component add clippy --toolchain stable

echo "==> install rustfmt toolchain: ${NIGHTLY_TOOLCHAIN}"
rustup toolchain install "$NIGHTLY_TOOLCHAIN" --profile minimal --component rustfmt

echo "==> rustfmt (workspace)"
cargo +"$NIGHTLY_TOOLCHAIN" fmt --all -- --check

echo "==> clippy (workspace, stable)"
RUSTUP_TOOLCHAIN=stable cargo clippy --workspace --all-targets

echo "==> clippy (strict subset, stable)"
RUSTUP_TOOLCHAIN=stable bash scripts/run_rust_clippy_strict.sh "$STRICT_LIST"

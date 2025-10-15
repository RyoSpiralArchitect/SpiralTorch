#!/usr/bin/env bash
set -euo pipefail
cargo build -p st-nn --release --offline
cargo build --workspace --features wgpu --release --offline
maturin build -m bindings/st-py/Cargo.toml --release --features mps

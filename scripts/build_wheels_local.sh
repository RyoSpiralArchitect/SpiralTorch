#!/usr/bin/env bash
set -euo pipefail
# Default: WGPU
maturin build -m bindings/st-py/pyproject.toml --release --features wgpu --out dist

# macOS MPS (universal2)
if [[ "$OSTYPE" == "darwin"* ]]; then
  maturin build -m bindings/st-py/pyproject.toml --release --features mps --universal2 --out dist
fi

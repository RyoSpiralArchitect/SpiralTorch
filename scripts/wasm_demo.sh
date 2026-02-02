#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

example="${1:-mellin-log-grid}"
mode="${2:-dev}"

example_dir="$ROOT/bindings/st-wasm/examples/$example"

if [[ ! -d "$example_dir" ]]; then
  echo "[SpiralTorch] unknown wasm example: $example"
  echo "[SpiralTorch] available: mellin-log-grid | canvas-hypertrain | cobol-console"
  exit 1
fi

echo "[SpiralTorch] building spiraltorch-wasm package..."
"$ROOT/scripts/build_wasm_web.sh" --dev

pushd "$example_dir" >/dev/null

if [[ ! -d node_modules ]]; then
  echo "[SpiralTorch] installing frontend dependencies..."
  npm install
fi

case "$mode" in
  dev)
    echo "[SpiralTorch] starting dev server..."
    npm run dev
    ;;
  preview)
    echo "[SpiralTorch] building + previewing static bundle..."
    npm run build
    npm run preview
    ;;
  build)
    echo "[SpiralTorch] building static bundle..."
    npm run build
    echo "[SpiralTorch] built: $example_dir/dist"
    ;;
  *)
    echo "[SpiralTorch] usage:"
    echo "  bash scripts/wasm_demo.sh <example> [dev|build|preview]"
    exit 2
    ;;
esac


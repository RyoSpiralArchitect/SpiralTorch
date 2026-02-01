#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CRATE_DIR="${CRATE_DIR:-"$ROOT/bindings/st-wasm"}"
OUT_DIR="${OUT_DIR:-"$ROOT/bindings/st-wasm/examples/pkg"}"
EXAMPLES_DIR="${EXAMPLES_DIR:-"$ROOT/bindings/st-wasm/examples"}"

echo "[SpiralTorch] building spiraltorch-wasm (web) ..."
echo "  crate:   $CRATE_DIR"
echo "  out_dir: $OUT_DIR"
echo
echo "[SpiralTorch] note: sanitising env (RUSTFLAGS/LIBRARY_PATH/PKG_CONFIG_PATH) for wasm builds"

env -u RUSTFLAGS -u LIBRARY_PATH -u PKG_CONFIG_PATH \
  wasm-pack build "$CRATE_DIR" --target web --out-dir "$OUT_DIR" "$@"

echo "[SpiralTorch] copying TypeScript declarations..."
cp "$ROOT/bindings/st-wasm/types/spiraltorch-wasm.d.ts" "$OUT_DIR/"

echo "[SpiralTorch] syncing package into Vite examples..."
for example in "$EXAMPLES_DIR"/cobol-console "$EXAMPLES_DIR"/mellin-log-grid; do
  dest="$example/pkg"
  mkdir -p "$dest"
  cp -f "$OUT_DIR"/spiraltorch* "$dest"/
done

echo "[SpiralTorch] done."

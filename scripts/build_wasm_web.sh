#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CRATE_DIR="${CRATE_DIR:-"$ROOT/bindings/st-wasm"}"
OUT_DIR="${OUT_DIR:-"$ROOT/bindings/st-wasm/examples/pkg"}"
EXAMPLES_DIR="${EXAMPLES_DIR:-"$ROOT/bindings/st-wasm/examples"}"

simd128=0
wasm_pack_args=()
for arg in "$@"; do
  if [[ "$arg" == "--simd128" ]]; then
    simd128=1
  else
    wasm_pack_args+=("$arg")
  fi
done

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "[SpiralTorch] error: wasm-pack was not found on PATH." >&2
  echo "Install it with: cargo install wasm-pack" >&2
  exit 1
fi

echo "[SpiralTorch] building spiraltorch-wasm (web) ..."
echo "  crate:   $CRATE_DIR"
echo "  out_dir: $OUT_DIR"
if [[ "$simd128" == 1 ]]; then
  echo "  profile: simd128 (requires WebAssembly SIMD)"
else
  echo "  profile: portable"
fi
echo
echo "[SpiralTorch] note: sanitising host Rust/linker flags for wasm builds"

build_env=(env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
  -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS \
  -u LIBRARY_PATH -u PKG_CONFIG_PATH)
if [[ "$simd128" == 1 ]]; then
  build_env+=(CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS=-Ctarget-feature=+simd128)
fi
wasm_pack_command=(wasm-pack build "$CRATE_DIR" --target web --out-dir "$OUT_DIR")
if [[ "${#wasm_pack_args[@]}" -gt 0 ]]; then
  wasm_pack_command+=("${wasm_pack_args[@]}")
fi
"${build_env[@]}" "${wasm_pack_command[@]}"

echo "[SpiralTorch] copying TypeScript declarations..."
cp "$ROOT/bindings/st-wasm/types/spiraltorch-wasm.d.ts" "$OUT_DIR/"

echo "[SpiralTorch] syncing package into Vite examples..."
for example in "$EXAMPLES_DIR"/*; do
  if [[ ! -d "$example" ]]; then
    continue
  fi
  if [[ "$(basename "$example")" == "pkg" ]]; then
    continue
  fi

  dest="$example/pkg"
  mkdir -p "$dest"
  cp -f "$OUT_DIR"/spiraltorch* "$dest"/
done

echo "[SpiralTorch] done."

# SpiralTorch Mellin log grid demo (WASM)

This example exposes `st-frac`'s Mellin log-lattice tooling in the browser via the
`spiraltorch-wasm` bindings. It builds a log-uniform sample grid, evaluates the Mellin
transform at many complex points (`evaluateMany`), and plots the magnitude.

## Quickstart

From the repo root:

```bash
bash scripts/wasm_demo.sh mellin-log-grid
```

## Prerequisites

1. Build the `spiraltorch-wasm` bindings once so the example can import them.

   ```bash
   ./scripts/build_wasm_web.sh --dev
   ```

   (If you prefer calling `wasm-pack` directly, make sure to unset any host-only linker
   flags like `RUSTFLAGS` / `LIBRARY_PATH` / `PKG_CONFIG_PATH` that point at native
   `vcpkg` archives.)

2. Install the frontend dependencies:

   ```bash
   cd bindings/st-wasm/examples/mellin-log-grid
   npm install
   ```

## Running

```bash
npm run dev
# or: npm run build && npm run preview
```

Notes:

- The demo bootstraps WASM by fetching `spiraltorch_wasm_bg.wasm` as bytes (avoids strict
  `application/wasm` MIME requirements).
- If your server still trips MIME-type issues, use `python scripts/serve_wasm_demo.py <dist>`.

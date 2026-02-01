# SpiralTorch Mellin log grid demo (WASM)

This example exposes `st-frac`'s Mellin log-lattice tooling in the browser via the
`spiraltorch-wasm` bindings. It builds a log-uniform sample grid, evaluates the Mellin
transform at many complex points (`evaluateMany`), and plots the magnitude.

## Prerequisites

1. Build the `spiraltorch-wasm` bindings once so the example can import them:

   ```bash
   wasm-pack build bindings/st-wasm --target web --out-dir bindings/st-wasm/examples/pkg
   ```

   Copy the TypeScript declarations next to the generated glue for editor hints:

   ```bash
   cp bindings/st-wasm/types/spiraltorch-wasm.d.ts bindings/st-wasm/examples/pkg/
   ```

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


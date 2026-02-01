# SpiralTorch Canvas hypertrain (WASM)

This example demonstrates running a tiny “learning loop” in the browser using
`spiraltorch-wasm`'s `FractalCanvas` bindings:

- Render the fractal canvas to an HTML `<canvas>`
- Sample the hyperbolic trail buffer (`emitWasmTrail`) for visualisation
- Update the relation tensor using `hypergradWave` / `realgradWave` each step

## Prerequisites

Build the WASM package (and sync it into each example’s `pkg/` directory):

```bash
./scripts/build_wasm_web.sh --dev
```

Install frontend deps:

```bash
cd bindings/st-wasm/examples/canvas-hypertrain
npm install
```

## Running

```bash
npm run dev
```


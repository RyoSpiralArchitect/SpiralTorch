# SpiralTorch Canvas hypertrain (WASM)

This example demonstrates running a tiny “learning loop” in the browser using
`spiraltorch-wasm`'s `FractalCanvas` bindings:

- `FractalCanvas.framePacket()` returns pixels + relation + trail + Desire metrics in one call.
- CPU loop: update relation via `hypergradWaveCurrent` / `realgradWaveCurrent`.
- WebGPU: optional 3D trail renderer + FFT row probe + “hypergrad operator” compute step.

## Quickstart

From the repo root:

```bash
bash scripts/wasm_demo.sh canvas-hypertrain
```

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
